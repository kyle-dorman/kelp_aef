"""Build pooled Phase 2 context diagnostics for binary, ridge, and hurdle surfaces."""
# mypy: disable-error-code="no-untyped-call,no-any-return"

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from pyarrow.lib import ArrowInvalid

from kelp_aef.config import require_mapping, require_string
from kelp_aef.evaluation.baselines import (
    KELPWATCH_PIXEL_AREA_M2,
    percent_bias,
    precision_recall_f1,
    root_mean_squared_error,
    safe_ratio,
)
from kelp_aef.evaluation.binary_presence import BinaryCalibrator, apply_binary_calibrator
from kelp_aef.evaluation.component_failure import (
    ANNUAL_MAX_10PCT_AREA_M2,
    ANNUAL_MAX_50PCT_AREA_M2,
    ComponentFailureConfig,
    ComponentFailureInput,
    add_failure_flags,
    binary_outcomes,
    component_failure_classes,
    margin_bin_labels,
    parquet_safe_cache_frame,
    probability_bin_labels,
    read_component_failure_frame,
)

DEFAULT_TOLERANCE_M2 = 90.0
DEFAULT_GRID_CELL_SIZE_M = 30.0
DEFAULT_OBSERVED_AREA_BINS = (0.0, 1.0, 90.0, 225.0, 450.0, 810.0, 900.0)
DEFAULT_THRESHOLD_POLICY = "validation_max_f1_calibrated"
RIDGE_SURFACE = "ridge"
HURDLE_SURFACE = "hurdle_expected_value"
BINARY_SURFACE = "binary"
CONTEXT_BASE_COLUMNS = (
    "context_id",
    "evaluation_region",
    "training_regime",
    "model_origin_region",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "context_type",
    "context_value",
)
BINARY_CONTEXT_FIELDS = CONTEXT_BASE_COLUMNS + (
    "model_surface",
    "target_label",
    "probability_source",
    "threshold_policy",
    "probability_threshold",
    "row_count",
    "observed_positive_count",
    "observed_positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "true_negative_count",
    "probability_mean",
    "probability_p05",
    "probability_p25",
    "probability_p50",
    "probability_p75",
    "probability_p90",
    "probability_p95",
    "probability_p99",
    "threshold_margin_mean",
    "threshold_margin_p05",
    "threshold_margin_p50",
    "threshold_margin_p95",
)
AMOUNT_CONTEXT_FIELDS = CONTEXT_BASE_COLUMNS + (
    "model_surface",
    "row_count",
    "observed_positive_count",
    "observed_positive_rate",
    "observed_mean",
    "predicted_mean",
    "observed_total_area",
    "predicted_total_area",
    "area_bias",
    "area_pct_bias",
    "mean_residual",
    "median_residual",
    "mae",
    "rmse",
    "prediction_p50",
    "prediction_p90",
    "prediction_p95",
    "prediction_p99",
    "prediction_max",
    "prediction_low_positive_count",
    "prediction_low_positive_rate",
    "prediction_ge_90m2_count",
    "prediction_ge_450m2_count",
    "prediction_ge_810m2_count",
    "prediction_clipped_zero_count",
    "prediction_clipped_upper_count",
    "amount_rate_denominator_count",
    "amount_under_count",
    "amount_under_rate",
    "composition_shrink_count",
    "composition_shrink_rate",
)
PREDICTION_DISTRIBUTION_FIELDS = CONTEXT_BASE_COLUMNS + (
    "model_surface",
    "prediction_units",
    "row_count",
    "observed_mean",
    "observed_p50",
    "observed_p90",
    "observed_p95",
    "observed_p99",
    "predicted_mean",
    "predicted_p50",
    "predicted_p90",
    "predicted_p95",
    "predicted_p99",
    "predicted_max",
    "observed_ge_450m2_count",
    "observed_ge_450m2_prediction_mean",
    "observed_ge_450m2_prediction_p95",
    "observed_ge_810m2_count",
    "observed_ge_810m2_prediction_mean",
    "observed_ge_810m2_prediction_p95",
    "low_positive_prediction_count",
    "low_positive_prediction_rate",
    "compression_ratio_observed_ge_450m2_mean",
    "compression_ratio_observed_ge_810m2_mean",
)
PERFORMANCE_FIELDS = CONTEXT_BASE_COLUMNS + (
    "model_surface",
    "row_count",
    "observed_positive_count",
    "observed_positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "observed_total_area",
    "predicted_total_area",
    "area_bias",
    "area_pct_bias",
    "mae",
    "rmse",
    "amount_rate_denominator_count",
    "amount_under_count",
    "amount_under_rate",
    "composition_shrink_count",
    "composition_shrink_rate",
)
CONTEXT_COLUMNS = (
    ("overall", None),
    ("observed_annual_max_bin", "observed_annual_max_bin"),
    ("temporal_label_class", "pooled_temporal_label_class"),
    ("previous_year_class", "previous_year_class"),
    ("crm_depth_m_bin", "crm_depth_m_bin"),
    ("elevation_bin", "elevation_bin"),
    ("binary_outcome", "binary_outcome"),
    ("component_failure_class", "component_failure_class"),
)


@dataclass(frozen=True)
class PooledContextInput:
    """Configured pooled evaluation context for one target region."""

    context_id: str
    baseline_predictions_path: Path
    binary_predictions_path: Path
    binary_calibration_model_path: Path
    hurdle_predictions_path: Path
    label_path: Path | None
    config_path: Path | None
    training_regime: str
    model_origin_region: str
    evaluation_region: str
    threshold_policy: str
    required: bool


@dataclass(frozen=True)
class PooledContextDiagnosticsConfig:
    """Resolved settings and outputs for pooled Phase 2 context diagnostics."""

    inputs: tuple[PooledContextInput, ...]
    performance_path: Path
    binary_context_path: Path
    amount_context_path: Path
    prediction_distribution_path: Path
    manifest_path: Path
    primary_split: str
    primary_year: int
    primary_mask_status: str
    primary_evaluation_scope: str
    primary_label_source: str
    observed_area_bins: tuple[float, ...]
    tolerance_m2: float
    grid_cell_size_m: float


@dataclass(frozen=True)
class PooledContextDiagnosticsTables:
    """All pooled-context diagnostic rows emitted by the analysis pass."""

    performance: list[dict[str, object]]
    binary_context: list[dict[str, object]]
    amount_context: list[dict[str, object]]
    prediction_distribution: list[dict[str, object]]


def load_pooled_context_config(
    comparison: dict[str, Any],
    outputs: dict[str, Any],
    tables_dir: Path,
    config_path: Path,
    *,
    primary_split: str,
    primary_year: int,
    primary_mask_status: str,
    primary_evaluation_scope: str,
    primary_label_source: str,
) -> PooledContextDiagnosticsConfig | None:
    """Load optional pooled-context diagnostics from the Phase 2 config."""
    settings = optional_mapping(
        comparison.get("pooled_context"),
        "training_regime_comparison.pooled_context",
    )
    if not settings:
        return None
    inputs = require_mapping(
        settings.get("inputs"),
        "training_regime_comparison.pooled_context.inputs",
    )
    return PooledContextDiagnosticsConfig(
        inputs=tuple(
            pooled_context_input(name, value, config_path)
            for name, value in sorted(inputs.items(), key=lambda item: str(item[0]))
        ),
        performance_path=pooled_output_path(
            settings,
            outputs,
            "performance",
            "model_analysis_phase2_pooled_context_model_performance",
            tables_dir / "monterey_big_sur_pooled_context_model_performance.csv",
            config_path,
        ),
        binary_context_path=pooled_output_path(
            settings,
            outputs,
            "binary_context",
            "model_analysis_phase2_pooled_binary_context_diagnostics",
            tables_dir / "monterey_big_sur_pooled_binary_context_diagnostics.csv",
            config_path,
        ),
        amount_context_path=pooled_output_path(
            settings,
            outputs,
            "amount_context",
            "model_analysis_phase2_pooled_amount_context_diagnostics",
            tables_dir / "monterey_big_sur_pooled_amount_context_diagnostics.csv",
            config_path,
        ),
        prediction_distribution_path=pooled_output_path(
            settings,
            outputs,
            "prediction_distribution",
            "model_analysis_phase2_pooled_prediction_distribution_by_context",
            tables_dir / "monterey_big_sur_pooled_prediction_distribution_by_context.csv",
            config_path,
        ),
        manifest_path=pooled_output_path(
            settings,
            outputs,
            "manifest",
            "model_analysis_phase2_pooled_context_diagnostics_manifest",
            tables_dir.parent.parent
            / "interim/monterey_big_sur_pooled_context_diagnostics_manifest.json",
            config_path,
        ),
        primary_split=primary_split,
        primary_year=primary_year,
        primary_mask_status=primary_mask_status,
        primary_evaluation_scope=primary_evaluation_scope,
        primary_label_source=primary_label_source,
        observed_area_bins=read_float_tuple(
            settings.get("observed_area_bins"),
            DEFAULT_OBSERVED_AREA_BINS,
        ),
        tolerance_m2=float(settings.get("tolerance_m2", DEFAULT_TOLERANCE_M2)),
        grid_cell_size_m=float(settings.get("grid_cell_size_m", DEFAULT_GRID_CELL_SIZE_M)),
    )


def pooled_context_input(
    name: object,
    value: object,
    config_path: Path,
) -> PooledContextInput:
    """Load one pooled-context input entry."""
    context_id = str(name)
    entry = require_mapping(
        value,
        f"training_regime_comparison.pooled_context.inputs.{context_id}",
    )
    label_value = entry.get("label_path")
    context_config = entry.get("config_path")
    return PooledContextInput(
        context_id=context_id,
        baseline_predictions_path=config_relative_path(
            entry.get("baseline_predictions"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.baseline_predictions",
            config_path,
        ),
        binary_predictions_path=config_relative_path(
            entry.get("binary_predictions"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.binary_predictions",
            config_path,
        ),
        binary_calibration_model_path=config_relative_path(
            entry.get("binary_calibration_model"),
            "training_regime_comparison.pooled_context.inputs."
            f"{context_id}.binary_calibration_model",
            config_path,
        ),
        hurdle_predictions_path=config_relative_path(
            entry.get("hurdle_predictions"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.hurdle_predictions",
            config_path,
        ),
        label_path=config_relative_path(
            label_value,
            f"training_regime_comparison.pooled_context.inputs.{context_id}.label_path",
            config_path,
        )
        if label_value is not None
        else None,
        config_path=config_relative_path(
            context_config,
            f"training_regime_comparison.pooled_context.inputs.{context_id}.config_path",
            config_path,
        )
        if context_config is not None
        else None,
        training_regime=require_string(
            entry.get("training_regime"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.training_regime",
        ),
        model_origin_region=require_string(
            entry.get("model_origin_region"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.model_origin_region",
        ),
        evaluation_region=require_string(
            entry.get("evaluation_region"),
            f"training_regime_comparison.pooled_context.inputs.{context_id}.evaluation_region",
        ),
        threshold_policy=str(entry.get("threshold_policy", DEFAULT_THRESHOLD_POLICY)),
        required=bool(entry.get("required", True)),
    )


def pooled_output_path(
    settings: dict[str, Any],
    outputs: dict[str, Any],
    setting_key: str,
    output_key: str,
    default: Path,
    config_path: Path,
) -> Path:
    """Resolve one pooled-context output path from local or report outputs."""
    value = settings.get(setting_key, outputs.get(output_key))
    if value is None:
        return default
    return config_relative_path(value, setting_key, config_path)


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return a dynamic mapping or an empty mapping when omitted."""
    if value is None:
        return {}
    return require_mapping(value, name)


def config_relative_path(value: object, name: str, config_path: Path) -> Path:
    """Read and resolve one path-like config value."""
    path = Path(require_string(value, name))
    if path.is_absolute():
        return path
    config_relative = config_path.parent / path
    if path.exists() and not config_relative.exists():
        return path
    return config_relative


def read_float_tuple(value: object, default: tuple[float, ...]) -> tuple[float, ...]:
    """Read a numeric tuple setting with a default."""
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        msg = "pooled-context observed_area_bins must be a non-empty list"
        raise ValueError(msg)
    return tuple(float(item) for item in value)


def build_pooled_context_tables(
    config: PooledContextDiagnosticsConfig,
) -> PooledContextDiagnosticsTables:
    """Build all pooled-context diagnostic tables."""
    frames = [read_pooled_context_frame(input_config, config) for input_config in config.inputs]
    return build_pooled_context_tables_from_frames(frames, config)


def build_pooled_context_tables_from_frames(
    frames: list[pd.DataFrame],
    config: PooledContextDiagnosticsConfig,
) -> PooledContextDiagnosticsTables:
    """Build pooled-context diagnostics from annotated context frames."""
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return PooledContextDiagnosticsTables([], [], [], [])
    combined = cast(pd.DataFrame, pd.concat(frames, ignore_index=True, sort=False))
    binary_rows = build_binary_context_rows(combined, config)
    amount_rows = build_amount_context_rows(combined, config)
    distribution_rows = build_prediction_distribution_rows(combined, config)
    return PooledContextDiagnosticsTables(
        performance=build_context_performance_rows(binary_rows, amount_rows),
        binary_context=binary_rows,
        amount_context=amount_rows,
        prediction_distribution=distribution_rows,
    )


def read_pooled_context_frame(
    input_config: PooledContextInput,
    config: PooledContextDiagnosticsConfig,
    *,
    component_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read, filter, and align one pooled context across all model surfaces."""
    component_input = ComponentFailureInput(
        context_id=input_config.context_id,
        hurdle_predictions_path=input_config.hurdle_predictions_path,
        binary_predictions_path=input_config.binary_predictions_path,
        label_path=input_config.label_path,
        config_path=input_config.config_path,
        training_regime=input_config.training_regime,
        model_origin_region=input_config.model_origin_region,
        evaluation_region=input_config.evaluation_region,
        required=input_config.required,
    )
    component_config = component_config_from_pooled(config, component_input)
    frame = (
        read_component_failure_frame(component_input, component_config)
        if component_frame is None
        else component_frame.copy()
    )
    if frame.empty:
        return frame
    ridge = read_ridge_predictions(input_config, config)
    binary = read_binary_predictions(input_config, config)
    merged = frame.merge(
        ridge,
        on=["year", "aef_grid_cell_id"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        binary,
        on=["year", "aef_grid_cell_id"],
        how="left",
        validate="one_to_one",
    )
    missing_ridge = merged["ridge_predicted_area_m2"].isna()
    missing_binary = merged["binary_calibrated_probability"].isna()
    if input_config.required and (bool(missing_ridge.any()) or bool(missing_binary.any())):
        msg = (
            "pooled-context input did not align all ridge/binary rows for "
            f"{input_config.context_id}"
        )
        raise ValueError(msg)
    return annotate_aligned_pooled_context(merged, config)


def pooled_context_frame_cache_path(cache_dir: Path, input_config: PooledContextInput) -> Path:
    """Return the cache file path for one pooled-context frame."""
    return cache_dir / f"{input_config.context_id}.parquet"


def write_pooled_context_frame_cache(
    config: PooledContextDiagnosticsConfig,
    cache_dir: Path,
    *,
    component_frames_by_context: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Write annotated pooled-context frames and return frame metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for input_config in config.inputs:
        component_frame = (
            component_frames_by_context.get(input_config.context_id)
            if component_frames_by_context is not None
            else None
        )
        frame = read_pooled_context_frame(
            input_config,
            config,
            component_frame=component_frame,
        )
        frames.append(frame)
        output_path = pooled_context_frame_cache_path(cache_dir, input_config)
        wrote_frame = not frame.empty
        if wrote_frame:
            parquet_safe_cache_frame(frame).to_parquet(output_path, index=False)
        metadata.append(
            {
                "context_id": input_config.context_id,
                "output_path": str(output_path),
                "row_count": int(len(frame)),
                "wrote_frame": wrote_frame,
            }
        )
    return frames, metadata


def read_pooled_context_frame_cache(
    config: PooledContextDiagnosticsConfig,
    cache_dir: Path,
) -> list[pd.DataFrame]:
    """Read annotated pooled-context frames from the configured cache."""
    frames: list[pd.DataFrame] = []
    for input_config in config.inputs:
        path = pooled_context_frame_cache_path(cache_dir, input_config)
        if not path.exists():
            if input_config.required:
                msg = f"required pooled-context cache frame is missing: {path}"
                raise FileNotFoundError(msg)
            frames.append(pd.DataFrame())
            continue
        frames.append(cast(pd.DataFrame, pd.read_parquet(path)))
    return frames


def component_config_from_pooled(
    config: PooledContextDiagnosticsConfig,
    input_config: ComponentFailureInput,
) -> ComponentFailureConfig:
    """Build the component-frame reader config needed for one pooled context."""
    return ComponentFailureConfig(
        inputs=(input_config,),
        summary_path=config.amount_context_path,
        by_label_context_path=config.amount_context_path,
        by_domain_context_path=config.amount_context_path,
        by_spatial_context_path=config.amount_context_path,
        by_model_context_path=config.amount_context_path,
        edge_effect_path=config.amount_context_path,
        temporal_label_context_path=config.amount_context_path,
        manifest_path=config.manifest_path,
        primary_split=config.primary_split,
        primary_year=config.primary_year,
        primary_mask_status=config.primary_mask_status,
        primary_evaluation_scope=config.primary_evaluation_scope,
        primary_label_source=config.primary_label_source,
        observed_area_bins=config.observed_area_bins,
        tolerance_m2=config.tolerance_m2,
        grid_cell_size_m=config.grid_cell_size_m,
    )


def read_ridge_predictions(
    input_config: PooledContextInput,
    config: PooledContextDiagnosticsConfig,
) -> pd.DataFrame:
    """Read primary-year pooled ridge predictions for one context."""
    path = input_config.baseline_predictions_path
    if not path.exists():
        if input_config.required:
            msg = f"required pooled-context ridge input is missing: {path}"
            raise FileNotFoundError(msg)
        return pd.DataFrame(columns=["year", "aef_grid_cell_id", "ridge_predicted_area_m2"])
    columns = dataset_columns(
        path,
        (
            "year",
            "split",
            "aef_grid_cell_id",
            "model_name",
            "pred_kelp_max_y",
            "pred_kelp_fraction_y_clipped",
        ),
    )
    frame = read_primary_parquet(path, columns, config.primary_year, config.primary_split)
    if "model_name" in frame.columns:
        frame = frame.loc[frame["model_name"].astype(str) == "ridge_regression"].copy()
    required = {"year", "aef_grid_cell_id"}
    if "pred_kelp_max_y" not in frame.columns and "pred_kelp_fraction_y_clipped" in frame.columns:
        frame["pred_kelp_max_y"] = (
            frame["pred_kelp_fraction_y_clipped"].astype(float) * KELPWATCH_PIXEL_AREA_M2
        )
    if not required.union({"pred_kelp_max_y"}).issubset(frame.columns):
        msg = f"pooled-context ridge input is missing required columns: {path}"
        raise ValueError(msg)
    output = frame.loc[:, ["year", "aef_grid_cell_id", "pred_kelp_max_y"]].copy()
    output = output.rename(columns={"pred_kelp_max_y": "ridge_predicted_area_m2"})
    return output.drop_duplicates(["year", "aef_grid_cell_id"]).reset_index(drop=True)


def read_binary_predictions(
    input_config: PooledContextInput,
    config: PooledContextDiagnosticsConfig,
) -> pd.DataFrame:
    """Read primary-year pooled binary predictions and apply calibration policy."""
    path = input_config.binary_predictions_path
    if not path.exists():
        if input_config.required:
            msg = f"required pooled-context binary input is missing: {path}"
            raise FileNotFoundError(msg)
        return pd.DataFrame(columns=["year", "aef_grid_cell_id", "binary_calibrated_probability"])
    columns = dataset_columns(
        path,
        (
            "year",
            "split",
            "aef_grid_cell_id",
            "binary_observed_y",
            "pred_binary_probability",
            "target_label",
        ),
    )
    frame = read_primary_parquet(path, columns, config.primary_year, config.primary_split)
    required = {"year", "aef_grid_cell_id", "binary_observed_y", "pred_binary_probability"}
    if not required.issubset(frame.columns):
        msg = f"pooled-context binary input is missing required columns: {path}"
        raise ValueError(msg)
    payload = cast(dict[str, Any], joblib.load(input_config.binary_calibration_model_path))
    probability_source, threshold = binary_policy_threshold(payload, input_config.threshold_policy)
    raw_probabilities = frame["pred_binary_probability"].to_numpy(dtype=float)
    probabilities = (
        raw_probabilities
        if probability_source == "raw_logistic"
        else apply_binary_calibrator(binary_calibrator_from_payload(payload), raw_probabilities)
    )
    target_label = first_string_value(frame, "target_label", "annual_max_ge_10pct")
    output = frame.loc[:, ["year", "aef_grid_cell_id", "binary_observed_y"]].copy()
    output["binary_calibrated_probability"] = probabilities
    output["binary_probability_source"] = probability_source
    output["binary_probability_threshold"] = float(threshold)
    output["binary_threshold_policy"] = input_config.threshold_policy
    output["binary_target_label"] = target_label
    output["binary_predicted_positive"] = probabilities >= float(threshold)
    return output.drop_duplicates(["year", "aef_grid_cell_id"]).reset_index(drop=True)


def dataset_columns(path: Path, candidates: tuple[str, ...]) -> list[str]:
    """Return candidate parquet columns that are present in a dataset."""
    available = set(ds.dataset(path, format="parquet").schema.names)
    return [column for column in candidates if column in available]


def read_primary_parquet(
    path: Path,
    columns: list[str],
    year: int,
    split: str,
) -> pd.DataFrame:
    """Read primary split/year rows with a parquet filter fallback."""
    try:
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[("year", "==", year), ("split", "==", split)],
        )
    except (ArrowInvalid, NotImplementedError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[
            (frame["year"].astype(int) == year) & (frame["split"].astype(str) == split)
        ].copy()
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def binary_policy_threshold(
    payload: dict[str, Any],
    threshold_policy: str,
) -> tuple[str, float]:
    """Return the probability source and threshold for a calibration policy."""
    policy_thresholds = cast(dict[str, tuple[str, float]], payload.get("policy_thresholds", {}))
    if threshold_policy not in policy_thresholds:
        msg = f"binary calibration payload is missing threshold policy {threshold_policy}"
        raise ValueError(msg)
    source, threshold = policy_thresholds[threshold_policy]
    return str(source), float(threshold)


def binary_calibrator_from_payload(payload: dict[str, Any]) -> BinaryCalibrator:
    """Build a binary calibrator object from a serialized calibration payload."""
    return BinaryCalibrator(
        method=str(payload.get("calibration_method", "platt")),
        model=payload.get("calibrator"),
        status=str(payload.get("calibration_status", "")),
        coefficient=object_to_float(payload.get("coefficient")),
        intercept=object_to_float(payload.get("intercept")),
    )


def object_to_float(value: object) -> float:
    """Convert a config or payload scalar to float, returning NaN when missing."""
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return math.nan


def first_string_value(dataframe: pd.DataFrame, column: str, default: str) -> str:
    """Return the first non-null string from a dataframe column."""
    if column not in dataframe.columns:
        return default
    values = dataframe[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else default


def annotate_aligned_pooled_context(
    dataframe: pd.DataFrame,
    config: PooledContextDiagnosticsConfig,
) -> pd.DataFrame:
    """Refresh binary/failure fields after aligning the explicit binary surface."""
    frame = dataframe.copy()
    probability = frame["binary_calibrated_probability"].to_numpy(dtype=float)
    threshold = frame["binary_probability_threshold"].to_numpy(dtype=float)
    predicted = probability >= threshold
    frame["calibrated_presence_probability"] = probability
    frame["presence_probability_threshold"] = threshold
    frame["predicted_binary_positive"] = predicted
    frame["pred_presence_class"] = predicted
    frame["binary_outcome"] = binary_outcomes(
        frame["observed_binary_positive"].to_numpy(dtype=bool),
        predicted,
    )
    frame["binary_probability_bin"] = probability_bin_labels(probability)
    frame["binary_threshold_margin"] = probability - threshold
    frame["binary_threshold_margin_bin"] = margin_bin_labels(frame["binary_threshold_margin"])
    add_failure_flags(frame, config.tolerance_m2)
    frame["component_failure_class"] = component_failure_classes(frame, config.tolerance_m2)
    frame["pooled_temporal_label_class"] = pooled_temporal_label_classes(frame)
    return frame


def pooled_temporal_label_classes(dataframe: pd.DataFrame) -> pd.Series:
    """Map station quarterly context and assumed background into report bins."""
    labels: list[str] = []
    for label_source, seasonality, observed in zip(
        dataframe["label_source"].astype(str),
        dataframe["seasonality_class"].astype(str),
        dataframe["kelp_max_y"].to_numpy(dtype=float),
        strict=True,
    ):
        if label_source == "assumed_background":
            labels.append("assumed_background")
        elif seasonality == "no_quarter_present" and observed > 0:
            labels.append("no_quarter_positive")
        elif seasonality == "one_quarter_spike":
            labels.append("one_quarter_spike")
        elif seasonality == "intermittent_two_or_three_quarters":
            labels.append("intermittent")
        elif seasonality == "persistent_all_valid_quarters":
            labels.append("persistent")
        else:
            labels.append(seasonality)
    return pd.Series(labels, index=dataframe.index, dtype="object")


def build_binary_context_rows(
    dataframe: pd.DataFrame,
    config: PooledContextDiagnosticsConfig,
) -> list[dict[str, object]]:
    """Build calibrated binary-support diagnostic rows by context bin."""
    rows: list[dict[str, object]] = []
    for values, group in iter_context_groups(dataframe, config):
        rows.append(binary_context_row(group, values))
    return sorted(rows, key=context_sort_key)


def binary_context_row(
    group: pd.DataFrame,
    values: Mapping[str, object],
) -> dict[str, object]:
    """Summarize calibrated binary support for one context group."""
    observed = bool_array(group, "observed_binary_positive")
    predicted = bool_array(group, "predicted_binary_positive")
    probabilities = float_array(group, "calibrated_presence_probability")
    margins = float_array(group, "binary_threshold_margin")
    positive_count = int(np.count_nonzero(observed))
    predicted_positive_count = int(np.count_nonzero(predicted))
    true_positive = observed & predicted
    false_positive = ~observed & predicted
    false_negative = observed & ~predicted
    true_negative = ~observed & ~predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    row = base_context_row(values)
    row.update(
        {
            "model_surface": BINARY_SURFACE,
            "target_label": first_group_scalar(group, "binary_target_label", "annual_max_ge_10pct"),
            "probability_source": first_group_scalar(
                group, "binary_probability_source", "calibrated"
            ),
            "threshold_policy": first_group_scalar(
                group, "binary_threshold_policy", DEFAULT_THRESHOLD_POLICY
            ),
            "probability_threshold": first_group_number(
                group, "binary_probability_threshold", math.nan
            ),
            "row_count": int(len(group)),
            "observed_positive_count": positive_count,
            "observed_positive_rate": safe_ratio(positive_count, int(len(group))),
            "predicted_positive_count": predicted_positive_count,
            "predicted_positive_rate": safe_ratio(predicted_positive_count, int(len(group))),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive_count": int(np.count_nonzero(true_positive)),
            "false_positive_count": int(np.count_nonzero(false_positive)),
            "false_positive_rate": safe_ratio(
                int(np.count_nonzero(false_positive)),
                int(len(group) - positive_count),
            ),
            "false_negative_count": int(np.count_nonzero(false_negative)),
            "false_negative_rate": safe_ratio(
                int(np.count_nonzero(false_negative)),
                positive_count,
            ),
            "true_negative_count": int(np.count_nonzero(true_negative)),
            "probability_mean": safe_mean(probabilities),
            "probability_p05": safe_quantile(probabilities, 0.05),
            "probability_p25": safe_quantile(probabilities, 0.25),
            "probability_p50": safe_quantile(probabilities, 0.50),
            "probability_p75": safe_quantile(probabilities, 0.75),
            "probability_p90": safe_quantile(probabilities, 0.90),
            "probability_p95": safe_quantile(probabilities, 0.95),
            "probability_p99": safe_quantile(probabilities, 0.99),
            "threshold_margin_mean": safe_mean(margins),
            "threshold_margin_p05": safe_quantile(margins, 0.05),
            "threshold_margin_p50": safe_quantile(margins, 0.50),
            "threshold_margin_p95": safe_quantile(margins, 0.95),
        }
    )
    return {field: row.get(field, "") for field in BINARY_CONTEXT_FIELDS}


def build_amount_context_rows(
    dataframe: pd.DataFrame,
    config: PooledContextDiagnosticsConfig,
) -> list[dict[str, object]]:
    """Build ridge and hurdle amount diagnostic rows by context bin."""
    rows: list[dict[str, object]] = []
    for values, group in iter_context_groups(dataframe, config):
        rows.append(
            amount_context_row(
                group,
                values,
                model_surface=RIDGE_SURFACE,
                prediction_column="ridge_predicted_area_m2",
                config=config,
            )
        )
        rows.append(
            amount_context_row(
                group,
                values,
                model_surface=HURDLE_SURFACE,
                prediction_column="pred_expected_value_area_m2",
                config=config,
            )
        )
    return sorted(rows, key=context_sort_key)


def amount_context_row(
    group: pd.DataFrame,
    values: Mapping[str, object],
    *,
    model_surface: str,
    prediction_column: str,
    config: PooledContextDiagnosticsConfig,
) -> dict[str, object]:
    """Summarize one amount-prediction surface for a context group."""
    observed = float_array(group, "kelp_max_y")
    predicted = float_array(group, prediction_column)
    residual = observed - predicted
    observed_positive = bool_array(group, "observed_binary_positive")
    detected_positive = observed_positive & bool_array(group, "predicted_binary_positive")
    denominator_count = int(np.count_nonzero(detected_positive))
    amount_under_count = (
        int(np.count_nonzero(detected_positive & (residual > config.tolerance_m2)))
        if model_surface == HURDLE_SURFACE
        else math.nan
    )
    composition_shrink_count = (
        int(group["composition_shrinkage"].sum()) if model_surface == HURDLE_SURFACE else math.nan
    )
    predicted_total = float(np.nansum(predicted))
    observed_total = float(np.nansum(observed))
    row = base_context_row(values)
    row.update(
        {
            "model_surface": model_surface,
            "row_count": int(len(group)),
            "observed_positive_count": int(np.count_nonzero(observed_positive)),
            "observed_positive_rate": safe_ratio(
                int(np.count_nonzero(observed_positive)), int(len(group))
            ),
            "observed_mean": safe_mean(observed),
            "predicted_mean": safe_mean(predicted),
            "observed_total_area": observed_total,
            "predicted_total_area": predicted_total,
            "area_bias": predicted_total - observed_total,
            "area_pct_bias": percent_bias(predicted_total, observed_total),
            "mean_residual": safe_mean(residual),
            "median_residual": safe_quantile(residual, 0.50),
            "mae": safe_mean(np.abs(residual)),
            "rmse": root_mean_squared_error(observed, predicted),
            "prediction_p50": safe_quantile(predicted, 0.50),
            "prediction_p90": safe_quantile(predicted, 0.90),
            "prediction_p95": safe_quantile(predicted, 0.95),
            "prediction_p99": safe_quantile(predicted, 0.99),
            "prediction_max": safe_max(predicted),
            "prediction_low_positive_count": int(
                np.count_nonzero((predicted > 0) & (predicted < ANNUAL_MAX_10PCT_AREA_M2))
            ),
            "prediction_low_positive_rate": safe_ratio(
                int(np.count_nonzero((predicted > 0) & (predicted < ANNUAL_MAX_10PCT_AREA_M2))),
                int(len(group)),
            ),
            "prediction_ge_90m2_count": int(
                np.count_nonzero(predicted >= ANNUAL_MAX_10PCT_AREA_M2)
            ),
            "prediction_ge_450m2_count": int(
                np.count_nonzero(predicted >= ANNUAL_MAX_50PCT_AREA_M2)
            ),
            "prediction_ge_810m2_count": int(np.count_nonzero(predicted >= 810.0)),
            "prediction_clipped_zero_count": int(np.count_nonzero(predicted <= 0.0)),
            "prediction_clipped_upper_count": int(
                np.count_nonzero(predicted >= KELPWATCH_PIXEL_AREA_M2)
            ),
            "amount_rate_denominator_count": denominator_count
            if model_surface == HURDLE_SURFACE
            else math.nan,
            "amount_under_count": amount_under_count,
            "amount_under_rate": safe_ratio(cast(int, amount_under_count), denominator_count)
            if model_surface == HURDLE_SURFACE
            else math.nan,
            "composition_shrink_count": composition_shrink_count,
            "composition_shrink_rate": safe_ratio(
                cast(int, composition_shrink_count), denominator_count
            )
            if model_surface == HURDLE_SURFACE
            else math.nan,
        }
    )
    return {field: row.get(field, "") for field in AMOUNT_CONTEXT_FIELDS}


def build_prediction_distribution_rows(
    dataframe: pd.DataFrame,
    config: PooledContextDiagnosticsConfig,
) -> list[dict[str, object]]:
    """Build ridge and hurdle prediction-distribution diagnostics by context."""
    rows: list[dict[str, object]] = []
    for values, group in iter_context_groups(dataframe, config):
        rows.append(
            prediction_distribution_row(
                group,
                values,
                model_surface=RIDGE_SURFACE,
                prediction_column="ridge_predicted_area_m2",
            )
        )
        rows.append(
            prediction_distribution_row(
                group,
                values,
                model_surface=HURDLE_SURFACE,
                prediction_column="pred_expected_value_area_m2",
            )
        )
    return sorted(rows, key=context_sort_key)


def prediction_distribution_row(
    group: pd.DataFrame,
    values: Mapping[str, object],
    *,
    model_surface: str,
    prediction_column: str,
) -> dict[str, object]:
    """Summarize prediction distributions for one amount surface and context."""
    observed = float_array(group, "kelp_max_y")
    predicted = float_array(group, prediction_column)
    high = observed >= ANNUAL_MAX_50PCT_AREA_M2
    near_saturated = observed >= 810.0
    high_pred = predicted[high]
    near_saturated_pred = predicted[near_saturated]
    row = base_context_row(values)
    row.update(
        {
            "model_surface": model_surface,
            "prediction_units": "m2_canopy_area",
            "row_count": int(len(group)),
            "observed_mean": safe_mean(observed),
            "observed_p50": safe_quantile(observed, 0.50),
            "observed_p90": safe_quantile(observed, 0.90),
            "observed_p95": safe_quantile(observed, 0.95),
            "observed_p99": safe_quantile(observed, 0.99),
            "predicted_mean": safe_mean(predicted),
            "predicted_p50": safe_quantile(predicted, 0.50),
            "predicted_p90": safe_quantile(predicted, 0.90),
            "predicted_p95": safe_quantile(predicted, 0.95),
            "predicted_p99": safe_quantile(predicted, 0.99),
            "predicted_max": safe_max(predicted),
            "observed_ge_450m2_count": int(np.count_nonzero(high)),
            "observed_ge_450m2_prediction_mean": safe_mean(high_pred),
            "observed_ge_450m2_prediction_p95": safe_quantile(high_pred, 0.95),
            "observed_ge_810m2_count": int(np.count_nonzero(near_saturated)),
            "observed_ge_810m2_prediction_mean": safe_mean(near_saturated_pred),
            "observed_ge_810m2_prediction_p95": safe_quantile(near_saturated_pred, 0.95),
            "low_positive_prediction_count": int(
                np.count_nonzero((predicted > 0) & (predicted < ANNUAL_MAX_10PCT_AREA_M2))
            ),
            "low_positive_prediction_rate": safe_ratio(
                int(np.count_nonzero((predicted > 0) & (predicted < ANNUAL_MAX_10PCT_AREA_M2))),
                int(len(group)),
            ),
            "compression_ratio_observed_ge_450m2_mean": safe_ratio_float(
                safe_mean(high_pred), safe_mean(observed[high])
            ),
            "compression_ratio_observed_ge_810m2_mean": safe_ratio_float(
                safe_mean(near_saturated_pred), safe_mean(observed[near_saturated])
            ),
        }
    )
    return {field: row.get(field, "") for field in PREDICTION_DISTRIBUTION_FIELDS}


def build_context_performance_rows(
    binary_rows: list[dict[str, object]],
    amount_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Align binary and amount surface headline metrics into one table."""
    rows: list[dict[str, object]] = []
    for row in binary_rows:
        output = base_context_row(row)
        output.update(
            {
                "model_surface": BINARY_SURFACE,
                "row_count": row.get("row_count", ""),
                "observed_positive_count": row.get("observed_positive_count", ""),
                "observed_positive_rate": row.get("observed_positive_rate", ""),
                "predicted_positive_count": row.get("predicted_positive_count", ""),
                "predicted_positive_rate": row.get("predicted_positive_rate", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1": row.get("f1", ""),
                "false_positive_count": row.get("false_positive_count", ""),
                "false_positive_rate": row.get("false_positive_rate", ""),
                "false_negative_count": row.get("false_negative_count", ""),
                "false_negative_rate": row.get("false_negative_rate", ""),
            }
        )
        rows.append({field: output.get(field, "") for field in PERFORMANCE_FIELDS})
    for row in amount_rows:
        output = base_context_row(row)
        output.update(
            {
                "model_surface": row.get("model_surface", ""),
                "row_count": row.get("row_count", ""),
                "observed_positive_count": row.get("observed_positive_count", ""),
                "observed_positive_rate": row.get("observed_positive_rate", ""),
                "observed_total_area": row.get("observed_total_area", ""),
                "predicted_total_area": row.get("predicted_total_area", ""),
                "area_bias": row.get("area_bias", ""),
                "area_pct_bias": row.get("area_pct_bias", ""),
                "mae": row.get("mae", ""),
                "rmse": row.get("rmse", ""),
                "amount_rate_denominator_count": row.get("amount_rate_denominator_count", ""),
                "amount_under_count": row.get("amount_under_count", ""),
                "amount_under_rate": row.get("amount_under_rate", ""),
                "composition_shrink_count": row.get("composition_shrink_count", ""),
                "composition_shrink_rate": row.get("composition_shrink_rate", ""),
            }
        )
        rows.append({field: output.get(field, "") for field in PERFORMANCE_FIELDS})
    return sorted(rows, key=context_sort_key)


def iter_context_groups(
    dataframe: pd.DataFrame,
    config: PooledContextDiagnosticsConfig,
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    """Return all base-context and context-bin groups for diagnostics."""
    base_columns = [
        "context_id",
        "evaluation_region",
        "training_regime",
        "model_origin_region",
        "split",
        "year",
        "mask_status",
        "evaluation_scope",
    ]
    groups: list[tuple[dict[str, object], pd.DataFrame]] = []
    for context_type, context_column in CONTEXT_COLUMNS:
        grouping = base_columns if context_column is None else [*base_columns, context_column]
        for keys, group in dataframe.groupby(grouping, sort=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            values: dict[str, object] = {
                column: normalized_group_value(value)
                for column, value in zip(grouping, key_tuple, strict=True)
            }
            values["label_source"] = config.primary_label_source
            values["context_type"] = context_type
            values["context_value"] = (
                "all" if context_column is None else values.get(context_column, "missing")
            )
            groups.append((values, group.copy()))
    return groups


def base_context_row(values: Mapping[str, object]) -> dict[str, object]:
    """Return common context fields for one output row."""
    return {column: values.get(column, "") for column in CONTEXT_BASE_COLUMNS}


def normalized_group_value(value: object) -> str:
    """Normalize a group key to a stable string."""
    if pd.isna(value):
        return "missing"
    return str(value)


def bool_array(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    """Return a boolean numpy array from a dataframe column."""
    if column not in dataframe.columns:
        return np.zeros(len(dataframe), dtype=bool)
    values = dataframe[column].fillna(False).astype(bool).to_numpy()
    return cast(np.ndarray, values)


def float_array(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    """Return a float numpy array from a dataframe column."""
    if column not in dataframe.columns:
        return np.full(len(dataframe), math.nan, dtype=float)
    values = dataframe[column].to_numpy(dtype=float)
    return cast(np.ndarray, values)


def safe_mean(values: np.ndarray) -> float:
    """Return a finite mean or NaN when no finite values exist."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(finite))


def safe_quantile(values: np.ndarray, quantile: float) -> float:
    """Return a finite quantile or NaN when no finite values exist."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanquantile(finite, quantile))


def safe_max(values: np.ndarray) -> float:
    """Return a finite max or NaN when no finite values exist."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmax(finite))


def safe_ratio_float(numerator: float, denominator: float) -> float:
    """Return a numeric ratio while preserving NaN for missing denominators."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)


def first_group_scalar(group: pd.DataFrame, column: str, default: str) -> str:
    """Return the first string value in a group column."""
    if group.empty or column not in group.columns:
        return default
    values = group[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else default


def first_group_number(group: pd.DataFrame, column: str, default: float) -> float:
    """Return the first numeric value in a group column."""
    if group.empty or column not in group.columns:
        return default
    values = group[column].dropna().to_numpy(dtype=float)
    return float(values[0]) if len(values) else default


def context_sort_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Sort rows by context, region, and model surface."""
    return (
        str(row.get("context_type", "")),
        str(row.get("evaluation_region", "")),
        str(row.get("context_value", "")),
        str(row.get("model_surface", "")),
    )


def write_pooled_context_outputs(
    tables: PooledContextDiagnosticsTables,
    config: PooledContextDiagnosticsConfig,
) -> None:
    """Write pooled-context CSV tables and sidecar manifest."""
    write_csv(tables.performance, config.performance_path, PERFORMANCE_FIELDS)
    write_csv(tables.binary_context, config.binary_context_path, BINARY_CONTEXT_FIELDS)
    write_csv(tables.amount_context, config.amount_context_path, AMOUNT_CONTEXT_FIELDS)
    write_csv(
        tables.prediction_distribution,
        config.prediction_distribution_path,
        PREDICTION_DISTRIBUTION_FIELDS,
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w") as file:
        json.dump(pooled_context_manifest(tables, config), file, indent=2)
        file.write("\n")


def write_csv(rows: list[dict[str, object]], path: Path, fields: tuple[str, ...]) -> None:
    """Write dictionaries to a CSV file with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled_context_manifest(
    tables: PooledContextDiagnosticsTables,
    config: PooledContextDiagnosticsConfig,
) -> dict[str, object]:
    """Build manifest metadata for pooled-context diagnostics."""
    return {
        "command": "analyze-model",
        "diagnostic": "phase2_pooled_context",
        "primary_filters": {
            "split": config.primary_split,
            "year": config.primary_year,
            "mask_status": config.primary_mask_status,
            "evaluation_scope": config.primary_evaluation_scope,
            "label_source": config.primary_label_source,
        },
        "definitions": {
            "target": "Kelpwatch-style annual maximum canopy reproduction",
            "binary_target": "annual_max_ge_10pct",
            "binary_positive_area_m2": ANNUAL_MAX_10PCT_AREA_M2,
            "amount_tolerance_m2": config.tolerance_m2,
            "amount_rate_denominator": (
                "observed-positive rows where calibrated binary support is detected"
            ),
            "amount_under_rate": (
                "share of denominator rows whose expected-value hurdle prediction is "
                "more than tolerance_m2 below observed annual max"
            ),
            "composition_shrink_rate": (
                "share of the same denominator where conditional prediction exceeds "
                "expected-value hurdle by at least tolerance_m2 and conditional "
                "prediction is at least 225 m2"
            ),
            "model_surface_alignment": (
                "context rows are anchored on expected-value hurdle retained-domain rows "
                "and joined to same-cell pooled ridge and calibrated binary predictions"
            ),
        },
        "inputs": [
            {
                "context_id": input_config.context_id,
                "training_regime": input_config.training_regime,
                "model_origin_region": input_config.model_origin_region,
                "evaluation_region": input_config.evaluation_region,
                "baseline_predictions": str(input_config.baseline_predictions_path),
                "binary_predictions": str(input_config.binary_predictions_path),
                "binary_calibration_model": str(input_config.binary_calibration_model_path),
                "hurdle_predictions": str(input_config.hurdle_predictions_path),
                "label_path": str(input_config.label_path)
                if input_config.label_path is not None
                else None,
                "required": input_config.required,
            }
            for input_config in config.inputs
        ],
        "outputs": {
            "performance": str(config.performance_path),
            "binary_context": str(config.binary_context_path),
            "amount_context": str(config.amount_context_path),
            "prediction_distribution": str(config.prediction_distribution_path),
            "manifest": str(config.manifest_path),
        },
        "row_counts": pooled_context_row_counts(tables),
    }


def pooled_context_row_counts(
    tables: PooledContextDiagnosticsTables | None,
) -> dict[str, int]:
    """Return pooled-context row counts for sidecar and model-analysis manifests."""
    if tables is None:
        return {
            "pooled_context_model_performance": 0,
            "pooled_binary_context_diagnostics": 0,
            "pooled_amount_context_diagnostics": 0,
            "pooled_prediction_distribution_by_context": 0,
        }
    return {
        "pooled_context_model_performance": len(tables.performance),
        "pooled_binary_context_diagnostics": len(tables.binary_context),
        "pooled_amount_context_diagnostics": len(tables.amount_context),
        "pooled_prediction_distribution_by_context": len(tables.prediction_distribution),
    }


def pooled_context_report_markdown(
    tables: PooledContextDiagnosticsTables,
    config: PooledContextDiagnosticsConfig,
) -> str:
    """Build compact report prose for pooled Phase 2 context diagnostics."""
    if not tables.performance:
        return "Pooled context diagnostics are configured, but no primary rows were available."
    overall = [row for row in tables.performance if str(row.get("context_type")) == "overall"]
    binary_rows = {
        str(row.get("evaluation_region")): row
        for row in overall
        if row.get("model_surface") == BINARY_SURFACE
    }
    ridge_rows = {
        str(row.get("evaluation_region")): row
        for row in overall
        if row.get("model_surface") == RIDGE_SURFACE
    }
    hurdle_rows = {
        str(row.get("evaluation_region")): row
        for row in overall
        if row.get("model_surface") == HURDLE_SURFACE
    }
    lines = [
        "These diagnostics keep the six-context train/apply comparison as a gate and focus "
        "the deeper analysis on pooled Monterey+Big Sur models evaluated separately on each "
        "region. They are diagnostic held-out `2022` rows only; no thresholds, sample quotas, "
        "masks, labels, features, or model policy are tuned from these summaries.",
        "",
        (
            "| Evaluation region | Rows | Binary F1 | Binary recall | Ridge area bias | "
            "Hurdle area bias | Hurdle amount under | Composition shrink |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region in ("big_sur", "monterey"):
        binary = binary_rows.get(region, {})
        ridge = ridge_rows.get(region, {})
        hurdle = hurdle_rows.get(region, {})
        lines.append(
            "| "
            f"{region} | "
            f"{format_int(binary.get('row_count'))} | "
            f"{format_float(binary.get('f1'))} | "
            f"{format_float(binary.get('recall'))} | "
            f"{format_percent(ridge.get('area_pct_bias'))} | "
            f"{format_percent(hurdle.get('area_pct_bias'))} | "
            f"{format_count_rate(hurdle, 'amount_under_count', 'amount_under_rate')} | "
            f"{format_count_rate(hurdle, 'composition_shrink_count', 'composition_shrink_rate')} |"
        )
    lines.extend(
        [
            "",
            "The shared denominator for `amount_under_rate` and `composition_shrink_rate` "
            "is observed-positive rows where calibrated binary support is detected. Prediction "
            "distribution rows summarize ridge and expected-value hurdle mass by observed annual "
            "max, temporal-label class, fine CRM depth, elevation, binary outcome, and component "
            "failure class without changing the model policy.",
            "",
            "Artifact tables:",
            f"- Model performance: `{config.performance_path}`",
            f"- Binary context diagnostics: `{config.binary_context_path}`",
            f"- Amount context diagnostics: `{config.amount_context_path}`",
            f"- Prediction distributions: `{config.prediction_distribution_path}`",
            f"- Manifest: `{config.manifest_path}`",
        ]
    )
    return "\n".join(lines)


def format_int(value: object) -> str:
    """Format an integer-like report value."""
    try:
        return f"{int(float(cast(Any, value))):,}"
    except (TypeError, ValueError):
        return "n/a"


def format_float(value: object) -> str:
    """Format a float report value."""
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.3f}"


def format_percent(value: object) -> str:
    """Format a fractional report value as a percentage."""
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.1%}"


def format_count_rate(row: Mapping[str, object], count_key: str, rate_key: str) -> str:
    """Format a count and rate for a report table."""
    if not row:
        return "n/a"
    return f"{format_int(row.get(count_key))} ({format_percent(row.get(rate_key))})"
