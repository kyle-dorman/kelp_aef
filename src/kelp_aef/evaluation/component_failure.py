"""Diagnose Phase 2 model-component failures across regions and regimes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from pyarrow.lib import ArrowInvalid
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from kelp_aef.config import require_mapping, require_string
from kelp_aef.evaluation.baselines import (
    percent_bias,
    root_mean_squared_error,
    safe_ratio,
)

DEFAULT_COMPONENT_TOLERANCE_M2 = 90.0
DEFAULT_GRID_CELL_SIZE_M = 30.0
DEFAULT_OBSERVED_AREA_BINS = (0.0, 1.0, 90.0, 225.0, 450.0, 810.0, 900.0)
EXPECTED_VALUE_MODEL_NAME = "calibrated_probability_x_conditional_canopy"
HARD_GATE_MODEL_NAME = "calibrated_hard_gate_conditional_canopy"
EXPECTED_VALUE_POLICY = "expected_value"
HARD_GATE_POLICY = "hard_gate"
ANNUAL_MAX_10PCT_AREA_M2 = 90.0
ANNUAL_MAX_50PCT_AREA_M2 = 450.0
NEAR_SATURATED_AREA_M2 = 810.0
PRIMARY_CONTEXT_IDS = {
    "big_sur_local",
    "monterey_local",
    "pooled_monterey_big_sur_on_big_sur",
    "pooled_monterey_big_sur_on_monterey",
}
COMPONENT_FAILURE_CONTEXT_COLUMNS = (
    "context_id",
    "evaluation_region",
    "training_regime",
    "model_origin_region",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "observed_annual_max_bin",
    "binary_target_status",
    "high_canopy_status",
    "near_saturated_status",
    "previous_year_class",
    "seasonality_class",
    "domain_mask_reason",
    "domain_mask_detail",
    "depth_bin",
    "crm_depth_m_bin",
    "elevation_bin",
    "coast_depth_context",
    "edge_class",
    "exterior_ring_class",
    "binary_outcome",
    "binary_probability_bin",
    "binary_threshold_margin_bin",
    "conditional_prediction_bin",
    "expected_value_residual_bin",
    "hard_gate_residual_bin",
    "component_failure_class",
)
COMPONENT_FAILURE_METRIC_COLUMNS = (
    "row_count",
    "observed_positive_count",
    "observed_zero_count",
    "predicted_positive_count",
    "detected_observed_positive_count",
    "observed_canopy_area",
    "conditional_predicted_area",
    "expected_value_predicted_area",
    "hard_gate_predicted_area",
    "expected_value_area_bias",
    "expected_value_area_pct_bias",
    "hard_gate_area_bias",
    "hard_gate_area_pct_bias",
    "expected_value_mean_residual",
    "expected_value_mae",
    "expected_value_rmse",
    "hard_gate_mean_residual",
    "hard_gate_mae",
    "hard_gate_rmse",
    "mean_calibrated_probability",
    "mean_probability_margin",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "true_negative_count",
    "support_miss_positive_count",
    "support_miss_positive_rate",
    "support_leakage_zero_count",
    "support_leakage_zero_rate",
    "amount_underprediction_detected_positive_count",
    "amount_underprediction_detected_positive_rate",
    "amount_overprediction_low_or_zero_count",
    "amount_overprediction_low_or_zero_rate",
    "composition_shrinkage_count",
    "composition_shrinkage_rate",
    "high_confidence_wrong_count",
    "high_confidence_wrong_rate",
    "near_correct_count",
    "near_correct_rate",
    "near_correct_positive_count",
    "near_correct_positive_rate",
    "positive_edge_count",
    "positive_interior_count",
    "zero_adjacent_to_positive_count",
    "near_positive_exterior_count",
    "mean_observed_positive_distance_m",
    "mean_predicted_positive_distance_m",
)
COMPONENT_FAILURE_FIELDS = COMPONENT_FAILURE_CONTEXT_COLUMNS + COMPONENT_FAILURE_METRIC_COLUMNS
COMPONENT_EDGE_FIELDS = (
    "context_id",
    "evaluation_region",
    "training_regime",
    "model_origin_region",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "observed_positive_count",
    "false_positive_count",
    "false_negative_count",
    "fp_isolated_predicted_positive_count",
    "fp_isolated_predicted_positive_rate",
    "fp_predicted_edge_count",
    "fp_predicted_edge_rate",
    "fp_predicted_interior_count",
    "fp_predicted_interior_rate",
    "fp_adjacent_observed_count",
    "fp_adjacent_observed_rate",
    "fp_near_observed_count",
    "fp_near_observed_rate",
    "fp_far_from_observed_count",
    "fp_far_from_observed_rate",
    "fp_adjacent_or_near_positive_count",
    "fp_adjacent_or_near_positive_rate",
    "fn_isolated_positive_count",
    "fn_isolated_positive_rate",
    "fn_positive_edge_count",
    "fn_positive_edge_rate",
    "fn_positive_edge_or_isolated_count",
    "fn_positive_edge_or_isolated_rate",
    "fn_positive_interior_count",
    "fn_positive_interior_rate",
    "high_confidence_fp_adjacent_or_near_count",
    "high_confidence_fn_edge_or_isolated_count",
    "expected_value_positive_edge_mean_residual",
    "expected_value_positive_interior_mean_residual",
    "hard_gate_positive_edge_mean_residual",
    "hard_gate_positive_interior_mean_residual",
    "composition_shrinkage_edge_count",
    "composition_shrinkage_edge_rate",
)
COMPONENT_TEMPORAL_FIELDS = COMPONENT_FAILURE_FIELDS + (
    "positive_quarter_count",
    "annual_mean_canopy_area_bin",
    "annual_max_to_mean_ratio_bin",
    "first_positive_quarter",
    "last_positive_quarter",
)


@dataclass(frozen=True)
class ComponentFailureInput:
    """Configured component-failure input for one train/evaluate context."""

    context_id: str
    hurdle_predictions_path: Path
    binary_predictions_path: Path | None
    label_path: Path | None
    config_path: Path | None
    training_regime: str
    model_origin_region: str
    evaluation_region: str
    required: bool


@dataclass(frozen=True)
class ComponentFailureConfig:
    """Resolved component-failure diagnostic settings and output paths."""

    inputs: tuple[ComponentFailureInput, ...]
    summary_path: Path
    by_label_context_path: Path
    by_domain_context_path: Path
    by_spatial_context_path: Path
    by_model_context_path: Path
    edge_effect_path: Path
    temporal_label_context_path: Path
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
class ComponentFailureTables:
    """All component-failure table rows produced by the diagnostic pass."""

    summary: list[dict[str, object]]
    by_label_context: list[dict[str, object]]
    by_domain_context: list[dict[str, object]]
    by_spatial_context: list[dict[str, object]]
    by_model_context: list[dict[str, object]]
    edge_effect: list[dict[str, object]]
    temporal_label_context: list[dict[str, object]]


def load_component_failure_config(
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
) -> ComponentFailureConfig | None:
    """Load optional Phase 2 component-failure settings from config."""
    settings = optional_mapping(
        comparison.get("component_failure"),
        "training_regime_comparison.component_failure",
    )
    if not settings:
        return None
    inputs = require_mapping(
        settings.get("inputs"),
        "training_regime_comparison.component_failure.inputs",
    )
    return ComponentFailureConfig(
        inputs=tuple(
            component_failure_input(name, value, config_path)
            for name, value in sorted(inputs.items(), key=lambda item: str(item[0]))
        ),
        summary_path=component_output_path(
            settings,
            outputs,
            "summary",
            "model_analysis_phase2_component_failure_summary",
            tables_dir / "monterey_big_sur_component_failure_summary.csv",
            config_path,
        ),
        by_label_context_path=component_output_path(
            settings,
            outputs,
            "by_label_context",
            "model_analysis_phase2_component_failure_by_label_context",
            tables_dir / "monterey_big_sur_component_failure_by_label_context.csv",
            config_path,
        ),
        by_domain_context_path=component_output_path(
            settings,
            outputs,
            "by_domain_context",
            "model_analysis_phase2_component_failure_by_domain_context",
            tables_dir / "monterey_big_sur_component_failure_by_domain_context.csv",
            config_path,
        ),
        by_spatial_context_path=component_output_path(
            settings,
            outputs,
            "by_spatial_context",
            "model_analysis_phase2_component_failure_by_spatial_context",
            tables_dir / "monterey_big_sur_component_failure_by_spatial_context.csv",
            config_path,
        ),
        by_model_context_path=component_output_path(
            settings,
            outputs,
            "by_model_context",
            "model_analysis_phase2_component_failure_by_model_context",
            tables_dir / "monterey_big_sur_component_failure_by_model_context.csv",
            config_path,
        ),
        edge_effect_path=component_output_path(
            settings,
            outputs,
            "edge_effect_diagnostics",
            "model_analysis_phase2_edge_effect_diagnostics",
            tables_dir / "monterey_big_sur_edge_effect_diagnostics.csv",
            config_path,
        ),
        temporal_label_context_path=component_output_path(
            settings,
            outputs,
            "temporal_label_context",
            "model_analysis_phase2_temporal_label_context",
            tables_dir / "monterey_big_sur_temporal_label_context.csv",
            config_path,
        ),
        manifest_path=component_output_path(
            settings,
            outputs,
            "manifest",
            "model_analysis_phase2_component_failure_manifest",
            tables_dir.parent / "interim/monterey_big_sur_component_failure_manifest.json",
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
        tolerance_m2=float(settings.get("tolerance_m2", DEFAULT_COMPONENT_TOLERANCE_M2)),
        grid_cell_size_m=float(settings.get("grid_cell_size_m", DEFAULT_GRID_CELL_SIZE_M)),
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return a dynamic mapping or an empty mapping when omitted."""
    if value is None:
        return {}
    return require_mapping(value, name)


def component_failure_input(
    name: object,
    value: object,
    config_path: Path,
) -> ComponentFailureInput:
    """Load one configured component-failure context."""
    context_id = str(name)
    entry = require_mapping(
        value,
        f"training_regime_comparison.component_failure.inputs.{context_id}",
    )
    binary_value = entry.get("binary_predictions")
    label_value = entry.get("label_path")
    context_config = entry.get("config_path")
    return ComponentFailureInput(
        context_id=context_id,
        hurdle_predictions_path=config_relative_path(
            entry.get("hurdle_predictions"),
            f"training_regime_comparison.component_failure.inputs.{context_id}.hurdle_predictions",
            config_path,
        ),
        binary_predictions_path=config_relative_path(
            binary_value,
            f"training_regime_comparison.component_failure.inputs.{context_id}.binary_predictions",
            config_path,
        )
        if binary_value is not None
        else None,
        label_path=config_relative_path(
            label_value,
            f"training_regime_comparison.component_failure.inputs.{context_id}.label_path",
            config_path,
        )
        if label_value is not None
        else None,
        config_path=config_relative_path(
            context_config,
            f"training_regime_comparison.component_failure.inputs.{context_id}.config_path",
            config_path,
        )
        if context_config is not None
        else None,
        training_regime=require_string(
            entry.get("training_regime"),
            f"training_regime_comparison.component_failure.inputs.{context_id}.training_regime",
        ),
        model_origin_region=require_string(
            entry.get("model_origin_region"),
            f"training_regime_comparison.component_failure.inputs.{context_id}.model_origin_region",
        ),
        evaluation_region=require_string(
            entry.get("evaluation_region"),
            f"training_regime_comparison.component_failure.inputs.{context_id}.evaluation_region",
        ),
        required=bool(entry.get("required", context_id in PRIMARY_CONTEXT_IDS)),
    )


def component_output_path(
    settings: dict[str, Any],
    outputs: dict[str, Any],
    setting_key: str,
    output_key: str,
    default: Path,
    config_path: Path,
) -> Path:
    """Resolve one component-failure output path from local or report outputs."""
    value = settings.get(setting_key, outputs.get(output_key))
    if value is None:
        return default
    return config_relative_path(value, setting_key, config_path)


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
        msg = "component-failure observed_area_bins must be a non-empty list"
        raise ValueError(msg)
    return tuple(float(item) for item in value)


def build_component_failure_tables(config: ComponentFailureConfig) -> ComponentFailureTables:
    """Build all component-failure aggregates for configured contexts."""
    frames = [read_component_failure_frame(input_config, config) for input_config in config.inputs]
    return build_component_failure_tables_from_frames(frames, config)


def build_component_failure_tables_from_frames(
    frames: list[pd.DataFrame],
    config: ComponentFailureConfig,
) -> ComponentFailureTables:
    """Build component-failure aggregates from annotated context frames."""
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return ComponentFailureTables([], [], [], [], [], [], [])
    combined = cast(pd.DataFrame, pd.concat(frames, ignore_index=True))
    return ComponentFailureTables(
        summary=aggregate_component_failures(combined, [], config),
        by_label_context=aggregate_component_failures(
            combined,
            [
                "label_source",
                "observed_annual_max_bin",
                "binary_target_status",
                "high_canopy_status",
                "near_saturated_status",
                "previous_year_class",
                "seasonality_class",
            ],
            config,
        ),
        by_domain_context=aggregate_component_failures(
            combined,
            [
                "domain_mask_reason",
                "domain_mask_detail",
                "depth_bin",
                "crm_depth_m_bin",
                "elevation_bin",
                "coast_depth_context",
            ],
            config,
        ),
        by_spatial_context=aggregate_component_failures(
            combined,
            ["edge_class", "exterior_ring_class"],
            config,
        ),
        by_model_context=aggregate_component_failures(
            combined,
            [
                "binary_outcome",
                "binary_probability_bin",
                "binary_threshold_margin_bin",
                "conditional_prediction_bin",
                "expected_value_residual_bin",
                "hard_gate_residual_bin",
                "component_failure_class",
            ],
            config,
        ),
        edge_effect=build_edge_effect_rows(combined, config),
        temporal_label_context=aggregate_component_failures(
            combined.loc[combined["seasonality_class"] != "missing_station_temporal_context"],
            [
                "seasonality_class",
                "positive_quarter_count",
                "annual_mean_canopy_area_bin",
                "annual_max_to_mean_ratio_bin",
                "first_positive_quarter",
                "last_positive_quarter",
                "previous_year_class",
            ],
            config,
            fields=COMPONENT_TEMPORAL_FIELDS,
        ),
    )


def component_failure_frame_cache_path(
    cache_dir: Path, input_config: ComponentFailureInput
) -> Path:
    """Return the cache file path for one component-failure context frame."""
    return cache_dir / f"{input_config.context_id}.parquet"


def write_component_failure_frame_cache(
    config: ComponentFailureConfig,
    cache_dir: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Write annotated component-failure frames and return frame metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for input_config in config.inputs:
        frame = read_component_failure_frame(input_config, config)
        frames.append(frame)
        output_path = component_failure_frame_cache_path(cache_dir, input_config)
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


def read_component_failure_frame_cache(
    config: ComponentFailureConfig,
    cache_dir: Path,
) -> list[pd.DataFrame]:
    """Read annotated component-failure frames from the configured cache."""
    frames: list[pd.DataFrame] = []
    for input_config in config.inputs:
        path = component_failure_frame_cache_path(cache_dir, input_config)
        if not path.exists():
            if input_config.required:
                msg = f"required component-failure cache frame is missing: {path}"
                raise FileNotFoundError(msg)
            frames.append(pd.DataFrame())
            continue
        frames.append(cast(pd.DataFrame, pd.read_parquet(path)))
    return frames


def parquet_safe_cache_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a frame whose mixed object columns can be serialized to Parquet."""
    output = dataframe.copy()
    for column in output.columns:
        if not pd.api.types.is_object_dtype(output[column]):
            continue
        values = output[column].dropna()
        value_types = {type(value) for value in values}
        if len(value_types) > 1:
            output[column] = output[column].map(
                lambda value: None if pd.isna(value) else str(value)
            )
    return output


def read_component_failure_frame(
    input_config: ComponentFailureInput,
    config: ComponentFailureConfig,
) -> pd.DataFrame:
    """Read and annotate primary rows for one component-failure context."""
    if not input_config.hurdle_predictions_path.exists():
        if input_config.required:
            msg = (
                "required component-failure input is missing: "
                f"{input_config.hurdle_predictions_path}"
            )
            raise FileNotFoundError(msg)
        return pd.DataFrame()
    raw = read_hurdle_context_rows(input_config.hurdle_predictions_path, config)
    if raw.empty:
        if input_config.required:
            msg = (
                "required component-failure input has no primary rows: "
                f"{input_config.hurdle_predictions_path}"
            )
            raise ValueError(msg)
        return pd.DataFrame()
    current = expected_value_rows(raw)
    current = current.loc[
        (current["year"].astype(int) == config.primary_year)
        & (current["split"].astype(str) == config.primary_split)
        & (current["mask_status"].astype(str) == config.primary_mask_status)
        & (current["evaluation_scope"].astype(str) == config.primary_evaluation_scope)
    ].copy()
    current = current.reset_index(drop=True)
    if current.empty:
        if input_config.required:
            msg = (
                "required component-failure input has no filtered rows: "
                f"{input_config.hurdle_predictions_path}"
            )
            raise ValueError(msg)
        return current
    current = attach_previous_year_context(current, raw, config)
    current = attach_temporal_label_context(current, input_config.label_path)
    current = annotate_component_failure_frame(current, input_config, config)
    return add_component_spatial_context(current, config.grid_cell_size_m)


def read_hurdle_context_rows(path: Path, config: ComponentFailureConfig) -> pd.DataFrame:
    """Read current and previous-year hurdle rows needed for diagnostics."""
    available = set(ds.dataset(path, format="parquet").schema.names)  # type: ignore[no-untyped-call]
    columns = [column for column in hurdle_context_columns() if column in available]
    years = [config.primary_year - 1, config.primary_year]
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("year", "in", years)])
    except (ArrowInvalid, NotImplementedError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["year"].astype(int).isin(years)].copy()
    required = {"year", "split", "model_name", "aef_grid_cell_id", "kelp_max_y"}
    if not required.issubset(frame.columns):
        msg = f"component-failure hurdle input is missing required columns: {path}"
        raise ValueError(msg)
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def hurdle_context_columns() -> tuple[str, ...]:
    """Return hurdle prediction columns used by component-failure diagnostics."""
    return (
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        "kelp_fraction_y",
        "kelp_max_y",
        "aef_grid_cell_id",
        "aef_grid_row",
        "aef_grid_col",
        "label_source",
        "is_kelpwatch_observed",
        "kelpwatch_station_count",
        "is_plausible_kelp_domain",
        "domain_mask_reason",
        "domain_mask_detail",
        "domain_mask_version",
        "crm_elevation_m",
        "crm_depth_m",
        "depth_bin",
        "elevation_bin",
        "mask_status",
        "evaluation_scope",
        "model_name",
        "composition_policy",
        "presence_probability_threshold",
        "calibrated_presence_probability",
        "pred_presence_class",
        "pred_conditional_area_m2",
        "pred_expected_value_area_m2",
        "pred_hard_gate_area_m2",
        "pred_hurdle_area_m2",
        "pred_kelp_max_y",
        "residual_kelp_max_y",
    )


def expected_value_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return expected-value hurdle rows from a mixed-policy prediction frame."""
    model_match = dataframe["model_name"].astype(str) == EXPECTED_VALUE_MODEL_NAME
    if "composition_policy" in dataframe.columns:
        policy_match = dataframe["composition_policy"].astype(str) == EXPECTED_VALUE_POLICY
        return dataframe.loc[model_match | policy_match].copy()
    return dataframe.loc[model_match].copy()


def attach_previous_year_context(
    current: pd.DataFrame,
    raw: pd.DataFrame,
    config: ComponentFailureConfig,
) -> pd.DataFrame:
    """Join previous-year observed annual max to primary rows by grid cell."""
    previous = expected_value_rows(raw)
    previous = previous.loc[previous["year"].astype(int) == config.primary_year - 1]
    if previous.empty:
        current["previous_year_kelp_max_y"] = math.nan
        current["previous_year_class"] = "missing_previous_year"
        return current
    lookup = previous.drop_duplicates("aef_grid_cell_id").set_index("aef_grid_cell_id")[
        "kelp_max_y"
    ]
    output = current.copy()
    output["previous_year_kelp_max_y"] = output["aef_grid_cell_id"].map(lookup)
    output["previous_year_class"] = previous_year_classes(
        output["previous_year_kelp_max_y"].to_numpy(dtype=float),
        output["kelp_max_y"].to_numpy(dtype=float),
    ).to_numpy()
    return output


def previous_year_classes(previous: np.ndarray, current: np.ndarray) -> pd.Series:
    """Classify current rows by one-year annual-max persistence."""
    labels: list[str] = []
    for previous_area, current_area in zip(previous, current, strict=True):
        if not np.isfinite(previous_area):
            labels.append("missing_previous_year")
        elif previous_area >= ANNUAL_MAX_10PCT_AREA_M2 and current_area >= ANNUAL_MAX_10PCT_AREA_M2:
            labels.append("persistent_ge_10pct")
        elif previous_area < ANNUAL_MAX_10PCT_AREA_M2 and current_area >= ANNUAL_MAX_10PCT_AREA_M2:
            labels.append("new_ge_10pct")
        elif previous_area >= ANNUAL_MAX_10PCT_AREA_M2 and current_area < ANNUAL_MAX_10PCT_AREA_M2:
            labels.append("lost_ge_10pct")
        elif previous_area > 0:
            labels.append("previous_low_canopy")
        else:
            labels.append("stable_zero_or_background")
    return pd.Series(labels, dtype="object")


def attach_temporal_label_context(dataframe: pd.DataFrame, label_path: Path | None) -> pd.DataFrame:
    """Join quarterly station-label context when annual labels are available."""
    output = dataframe.copy()
    for column in temporal_label_columns():
        output[column] = default_temporal_value(column)
    if label_path is None or not label_path.exists() or "kelpwatch_station_id" not in output:
        return output
    labels = pd.read_parquet(label_path)
    required = {"year", "kelpwatch_station_id", "area_q1", "area_q2", "area_q3", "area_q4"}
    if not required.issubset(labels.columns):
        return output
    temporal = labels.loc[:, sorted(required)].copy()
    temporal = temporal.drop_duplicates(["year", "kelpwatch_station_id"])
    merged = output.merge(
        temporal,
        on=["year", "kelpwatch_station_id"],
        how="left",
        validate="many_to_one",
    )
    return add_temporal_fields(merged)


def temporal_label_columns() -> tuple[str, ...]:
    """Return derived temporal context columns added to component rows."""
    return (
        "positive_quarter_count",
        "annual_mean_canopy_area",
        "annual_max_to_mean_ratio",
        "seasonality_class",
        "first_positive_quarter",
        "last_positive_quarter",
        "annual_mean_canopy_area_bin",
        "annual_max_to_mean_ratio_bin",
    )


def default_temporal_value(column: str) -> object:
    """Return the missing-context default for one temporal column."""
    if column in {"positive_quarter_count", "first_positive_quarter", "last_positive_quarter"}:
        return "missing"
    if column == "seasonality_class":
        return "missing_station_temporal_context"
    if column.endswith("_bin"):
        return "missing"
    return math.nan


def add_temporal_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive quarterly persistence fields from joined annual-label quarters."""
    frame = dataframe.copy()
    quarter_columns = ["area_q1", "area_q2", "area_q3", "area_q4"]
    quarter_values = frame[quarter_columns].to_numpy(dtype=float)
    positive = quarter_values > 0
    valid = np.isfinite(quarter_values)
    positive_counts = positive.sum(axis=1)
    valid_counts = valid.sum(axis=1)
    quarter_sums = np.nansum(quarter_values, axis=1)
    means = np.divide(
        quarter_sums,
        valid_counts,
        out=np.full(len(frame), math.nan, dtype=float),
        where=valid_counts > 0,
    )
    frame["positive_quarter_count"] = np.where(valid_counts > 0, positive_counts, "missing")
    frame["annual_mean_canopy_area"] = means
    frame["annual_max_to_mean_ratio"] = np.divide(
        frame["kelp_max_y"].to_numpy(dtype=float),
        means,
        out=np.full(len(frame), math.nan, dtype=float),
        where=means > 0,
    )
    frame["seasonality_class"] = seasonality_classes(positive_counts, valid_counts)
    first, last = positive_quarter_bounds(positive, valid_counts)
    frame["first_positive_quarter"] = first
    frame["last_positive_quarter"] = last
    frame["annual_mean_canopy_area_bin"] = area_bin_labels(
        pd.Series(frame["annual_mean_canopy_area"]), DEFAULT_OBSERVED_AREA_BINS
    ).astype(str)
    frame["annual_max_to_mean_ratio_bin"] = ratio_bin_labels(frame["annual_max_to_mean_ratio"])
    return frame.drop(columns=[column for column in quarter_columns if column in frame.columns])


def seasonality_classes(positive_counts: np.ndarray, valid_counts: np.ndarray) -> pd.Series:
    """Classify station rows by count of positive quarterly labels."""
    labels: list[str] = []
    for positive_count, valid_count in zip(positive_counts, valid_counts, strict=True):
        if valid_count == 0:
            labels.append("missing_quarters")
        elif positive_count == 0:
            labels.append("no_quarter_present")
        elif positive_count == 1:
            labels.append("one_quarter_spike")
        elif positive_count == valid_count:
            labels.append("persistent_all_valid_quarters")
        else:
            labels.append("intermittent_two_or_three_quarters")
    return pd.Series(labels, dtype="object")


def positive_quarter_bounds(
    positive: np.ndarray,
    valid_counts: np.ndarray,
) -> tuple[list[object], list[object]]:
    """Return first and last one-based positive quarter labels."""
    first: list[object] = []
    last: list[object] = []
    for row_positive, valid_count in zip(positive, valid_counts, strict=True):
        indices = np.flatnonzero(row_positive) + 1
        if valid_count == 0:
            first.append("missing")
            last.append("missing")
        elif len(indices) == 0:
            first.append("none")
            last.append("none")
        else:
            first.append(int(indices[0]))
            last.append(int(indices[-1]))
    return first, last


def ratio_bin_labels(values: pd.Series) -> pd.Series:
    """Bin annual max divided by annual mean into stable labels."""
    labels: list[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing")
        elif value <= 1.25:
            labels.append("[1, 1.25]")
        elif value <= 2.0:
            labels.append("(1.25, 2]")
        elif value <= 4.0:
            labels.append("(2, 4]")
        else:
            labels.append(">4")
    return pd.Series(labels, index=values.index, dtype="object")


def annotate_component_failure_frame(
    dataframe: pd.DataFrame,
    input_config: ComponentFailureInput,
    config: ComponentFailureConfig,
) -> pd.DataFrame:
    """Add binary, amount, domain, label, and failure taxonomy fields."""
    frame = dataframe.copy()
    frame["context_id"] = input_config.context_id
    frame["evaluation_region"] = input_config.evaluation_region
    frame["training_regime"] = input_config.training_regime
    frame["model_origin_region"] = input_config.model_origin_region
    ensure_component_columns(frame)
    observed_area = frame["kelp_max_y"].to_numpy(dtype=float)
    observed_positive = observed_area >= ANNUAL_MAX_10PCT_AREA_M2
    probability = frame["calibrated_presence_probability"].to_numpy(dtype=float)
    threshold = frame["presence_probability_threshold"].to_numpy(dtype=float)
    predicted_positive = predicted_presence_mask(frame, probability, threshold)
    expected_predicted = frame["pred_expected_value_area_m2"].to_numpy(dtype=float)
    hard_predicted = frame["pred_hard_gate_area_m2"].to_numpy(dtype=float)
    conditional_predicted = frame["pred_conditional_area_m2"].to_numpy(dtype=float)
    frame["observed_binary_positive"] = observed_positive
    frame["predicted_binary_positive"] = predicted_positive
    frame["binary_outcome"] = binary_outcomes(observed_positive, predicted_positive)
    frame["binary_probability_bin"] = probability_bin_labels(probability)
    frame["binary_threshold_margin"] = probability - threshold
    frame["binary_threshold_margin_bin"] = margin_bin_labels(frame["binary_threshold_margin"])
    frame["conditional_prediction_bin"] = area_bin_labels(
        pd.Series(conditional_predicted), config.observed_area_bins
    ).astype(str)
    frame["expected_value_residual"] = observed_area - expected_predicted
    frame["hard_gate_residual"] = observed_area - hard_predicted
    frame["expected_value_residual_bin"] = residual_bin_labels(frame["expected_value_residual"])
    frame["hard_gate_residual_bin"] = residual_bin_labels(frame["hard_gate_residual"])
    frame["observed_annual_max_bin"] = area_bin_labels(
        frame["kelp_max_y"], config.observed_area_bins
    ).astype(str)
    frame["binary_target_status"] = np.where(observed_positive, "annual_max_ge_10pct", "negative")
    frame["high_canopy_status"] = np.where(
        observed_area >= ANNUAL_MAX_50PCT_AREA_M2,
        "annual_max_ge_50pct",
        "below_50pct",
    )
    frame["near_saturated_status"] = np.where(
        observed_area >= NEAR_SATURATED_AREA_M2,
        "near_saturated_ge_810m2",
        "below_810m2",
    )
    frame["crm_depth_m_bin"] = depth_value_bins(frame["crm_depth_m"])
    frame["coast_depth_context"] = coast_depth_contexts(frame)
    add_failure_flags(frame, config.tolerance_m2)
    frame["component_failure_class"] = component_failure_classes(frame, config.tolerance_m2)
    return frame


def ensure_component_columns(frame: pd.DataFrame) -> None:
    """Fill optional component-failure input columns with explicit defaults."""
    defaults: dict[str, object] = {
        "label_source": "unknown",
        "domain_mask_reason": "missing_domain_reason",
        "domain_mask_detail": "missing_domain_detail",
        "depth_bin": "missing_depth_bin",
        "elevation_bin": "missing_elevation_bin",
        "mask_status": "unknown_mask_status",
        "evaluation_scope": "unknown_evaluation_scope",
        "crm_depth_m": math.nan,
        "crm_elevation_m": math.nan,
        "pred_conditional_area_m2": math.nan,
        "pred_expected_value_area_m2": math.nan,
        "pred_hard_gate_area_m2": math.nan,
        "presence_probability_threshold": math.nan,
        "calibrated_presence_probability": math.nan,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default


def predicted_presence_mask(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    """Return binary support decisions from saved class or threshold fields."""
    if "pred_presence_class" in frame.columns:
        mask = frame["pred_presence_class"].fillna(False).astype(bool).to_numpy()
        return cast(np.ndarray, mask)
    return cast(np.ndarray, probability >= threshold)


def binary_outcomes(observed: np.ndarray, predicted: np.ndarray) -> pd.Series:
    """Classify binary support as TP, FP, FN, or TN."""
    labels = np.full(observed.shape, "TN", dtype=object)
    labels[observed & predicted] = "TP"
    labels[~observed & predicted] = "FP"
    labels[observed & ~predicted] = "FN"
    return pd.Series(labels, dtype="object")


def probability_bin_labels(probabilities: np.ndarray) -> pd.Series:
    """Bin calibrated probabilities for boundary-margin diagnostics."""
    labels = []
    for value in probabilities:
        if not np.isfinite(value):
            labels.append("missing")
        elif value < 0.10:
            labels.append("[0, .1)")
        elif value < 0.25:
            labels.append("[.1, .25)")
        elif value < 0.50:
            labels.append("[.25, .5)")
        elif value < 0.75:
            labels.append("[.5, .75)")
        elif value < 0.90:
            labels.append("[.75, .9)")
        else:
            labels.append("[.9, 1]")
    return pd.Series(labels, dtype="object")


def margin_bin_labels(margins: pd.Series) -> pd.Series:
    """Bin calibrated probability minus selected threshold."""
    labels: list[str] = []
    for value in margins.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing")
        elif value < -0.50:
            labels.append("<-.50")
        elif value < -0.25:
            labels.append("[-.50, -.25)")
        elif value < -0.10:
            labels.append("[-.25, -.10)")
        elif value < 0:
            labels.append("[-.10, 0)")
        elif value < 0.10:
            labels.append("[0, .10)")
        elif value < 0.25:
            labels.append("[.10, .25)")
        elif value < 0.50:
            labels.append("[.25, .50)")
        else:
            labels.append(">=.50")
    return pd.Series(labels, index=margins.index, dtype="object")


def area_bin_labels(values: pd.Series, bins: tuple[float, ...]) -> pd.Series:
    """Assign annual canopy area values to stable labels with zero isolated."""
    sorted_bins = sorted(set(float(value) for value in bins if value > 0))
    labels: list[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing")
        elif value == 0:
            labels.append("000_zero")
        else:
            lower = 0.0
            selected = f">{sorted_bins[-1]:g}" if sorted_bins else ">0"
            for upper in sorted_bins:
                if value <= upper:
                    selected = f"({lower:g}, {upper:g}]"
                    break
                lower = upper
            labels.append(selected)
    return pd.Series(labels, index=values.index, dtype="object")


def residual_bin_labels(values: pd.Series) -> pd.Series:
    """Bin observed minus predicted residuals into diagnosis-scale buckets."""
    labels: list[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing")
        elif value < -450:
            labels.append("<-450")
        elif value < -90:
            labels.append("[-450, -90)")
        elif value <= 90:
            labels.append("[-90, 90]")
        elif value <= 450:
            labels.append("(90, 450]")
        else:
            labels.append(">450")
    return pd.Series(labels, index=values.index, dtype="object")


def depth_value_bins(values: pd.Series) -> pd.Series:
    """Bin continuous CRM depth values for retained-domain summaries."""
    labels: list[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing_depth")
        elif value <= 0:
            labels.append("not_subtidal_or_zero")
        elif value <= 10:
            labels.append("(0, 10m]")
        elif value <= 20:
            labels.append("(10, 20m]")
        elif value <= 40:
            labels.append("(20, 40m]")
        elif value <= 60:
            labels.append("(40, 60m]")
        else:
            labels.append(">60m")
    return pd.Series(labels, index=values.index, dtype="object")


def coast_depth_contexts(frame: pd.DataFrame) -> pd.Series:
    """Classify rows as ambiguous coast, subtidal, landward, or missing depth."""
    labels: list[str] = []
    for reason, depth, elevation in zip(
        frame["domain_mask_reason"].astype(str),
        frame["crm_depth_m"].to_numpy(dtype=float),
        frame["crm_elevation_m"].to_numpy(dtype=float),
        strict=True,
    ):
        if "ambiguous" in reason:
            labels.append("ambiguous_coast")
        elif np.isfinite(depth) and depth > 0:
            labels.append("subtidal_ocean")
        elif np.isfinite(elevation) and elevation > 0:
            labels.append("landward_or_intertidal")
        else:
            labels.append("missing_depth_context")
    return pd.Series(labels, index=frame.index, dtype="object")


def add_failure_flags(frame: pd.DataFrame, tolerance_m2: float) -> None:
    """Add boolean component-failure indicator columns to a frame in-place."""
    observed = frame["kelp_max_y"].to_numpy(dtype=float)
    observed_positive = frame["observed_binary_positive"].to_numpy(dtype=bool)
    predicted_positive = frame["predicted_binary_positive"].to_numpy(dtype=bool)
    probability = frame["calibrated_presence_probability"].to_numpy(dtype=float)
    conditional = frame["pred_conditional_area_m2"].to_numpy(dtype=float)
    expected = frame["pred_expected_value_area_m2"].to_numpy(dtype=float)
    expected_residual = frame["expected_value_residual"].to_numpy(dtype=float)
    detected_observed_positive = observed_positive & predicted_positive
    frame["support_miss_positive"] = observed_positive & ~predicted_positive
    frame["support_leakage_zero"] = (observed == 0) & predicted_positive
    frame["amount_underprediction_detected_positive"] = detected_observed_positive & (
        expected_residual > tolerance_m2
    )
    frame["amount_overprediction_low_or_zero"] = (observed < ANNUAL_MAX_10PCT_AREA_M2) & (
        (expected - observed) > tolerance_m2
    )
    frame["composition_shrinkage"] = (
        detected_observed_positive
        & (conditional - expected >= tolerance_m2)
        & (conditional >= ANNUAL_MAX_50PCT_AREA_M2 / 2)
    )
    frame["high_confidence_wrong"] = ((probability >= 0.90) & ~observed_positive) | (
        (probability < 0.10) & observed_positive
    )
    frame["near_correct"] = frame["expected_value_residual"].abs() <= tolerance_m2


def component_failure_classes(frame: pd.DataFrame, tolerance_m2: float) -> pd.Series:
    """Assign one primary component-failure class per retained row."""
    classes: list[str] = []
    for row in frame.to_dict("records"):
        if bool(row["high_confidence_wrong"]):
            classes.append("high_confidence_wrong")
        elif bool(row["support_miss_positive"]):
            classes.append("support_miss_positive")
        elif bool(row["support_leakage_zero"]):
            classes.append("support_leakage_zero")
        elif bool(row["amount_overprediction_low_or_zero"]):
            classes.append("amount_overprediction_low_or_zero")
        elif bool(row["composition_shrinkage"]):
            classes.append("composition_shrinkage")
        elif bool(row["amount_underprediction_detected_positive"]):
            classes.append("amount_underprediction_detected_positive")
        elif abs(float(row["expected_value_residual"])) <= tolerance_m2:
            classes.append("near_correct")
        else:
            classes.append("other_residual_or_context_mismatch")
    return pd.Series(classes, index=frame.index, dtype="object")


def add_component_spatial_context(dataframe: pd.DataFrame, grid_cell_size_m: float) -> pd.DataFrame:
    """Compute grid-neighborhood, distance, and edge/interior context fields."""
    if dataframe.empty:
        return dataframe.copy()
    frame = dataframe.copy()
    required = {
        "aef_grid_row",
        "aef_grid_col",
        "observed_binary_positive",
        "predicted_binary_positive",
    }
    if not required.issubset(frame.columns):
        return add_missing_spatial_context(frame)
    rows = frame["aef_grid_row"].astype(int).to_numpy()
    cols = frame["aef_grid_col"].astype(int).to_numpy()
    min_row = int(rows.min())
    min_col = int(cols.min())
    row_offsets = rows - min_row
    col_offsets = cols - min_col
    shape = (int(row_offsets.max()) + 1, int(col_offsets.max()) + 1)
    retained_grid = np.zeros(shape, dtype=bool)
    observed_grid = np.zeros(shape, dtype=bool)
    predicted_grid = np.zeros(shape, dtype=bool)
    retained_grid[row_offsets, col_offsets] = True
    observed_grid[row_offsets, col_offsets] = frame["observed_binary_positive"].to_numpy(dtype=bool)
    predicted_grid[row_offsets, col_offsets] = frame["predicted_binary_positive"].to_numpy(
        dtype=bool
    )
    observed_3 = window_sum(observed_grid, 1)[row_offsets, col_offsets]
    observed_5 = window_sum(observed_grid, 2)[row_offsets, col_offsets]
    predicted_3 = window_sum(predicted_grid, 1)[row_offsets, col_offsets]
    predicted_5 = window_sum(predicted_grid, 2)[row_offsets, col_offsets]
    retained_3 = window_sum(retained_grid, 1)[row_offsets, col_offsets]
    retained_5 = window_sum(retained_grid, 2)[row_offsets, col_offsets]
    frame["observed_positive_count_3x3"] = observed_3
    frame["observed_positive_fraction_3x3"] = safe_divide_array(observed_3, retained_3)
    frame["observed_positive_count_5x5"] = observed_5
    frame["observed_positive_fraction_5x5"] = safe_divide_array(observed_5, retained_5)
    frame["predicted_positive_count_3x3"] = predicted_3
    frame["predicted_positive_fraction_3x3"] = safe_divide_array(predicted_3, retained_3)
    frame["predicted_positive_count_5x5"] = predicted_5
    frame["predicted_positive_fraction_5x5"] = safe_divide_array(predicted_5, retained_5)
    coordinates = np.column_stack([row_offsets, col_offsets]).astype(float)
    frame["distance_to_observed_positive_cells"] = nearest_grid_distances(
        coordinates,
        coordinates[frame["observed_binary_positive"].to_numpy(dtype=bool)],
    )
    frame["distance_to_observed_positive_m"] = (
        frame["distance_to_observed_positive_cells"].to_numpy(dtype=float) * grid_cell_size_m
    )
    frame["distance_to_predicted_positive_cells"] = nearest_grid_distances(
        coordinates,
        coordinates[frame["predicted_binary_positive"].to_numpy(dtype=bool)],
    )
    frame["distance_to_predicted_positive_m"] = (
        frame["distance_to_predicted_positive_cells"].to_numpy(dtype=float) * grid_cell_size_m
    )
    component_ids, component_areas = observed_component_context(
        row_offsets,
        col_offsets,
        frame["observed_binary_positive"].to_numpy(dtype=bool),
        grid_cell_size_m,
    )
    frame["observed_component_id"] = component_ids
    frame["observed_component_area_m2"] = component_areas
    frame["edge_class"] = edge_classes(frame)
    frame["exterior_ring_class"] = exterior_ring_classes(frame)
    return frame


def add_missing_spatial_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill spatial context with missing labels when grid coordinates are unavailable."""
    output = frame.copy()
    output["observed_positive_count_3x3"] = math.nan
    output["observed_positive_fraction_3x3"] = math.nan
    output["observed_positive_count_5x5"] = math.nan
    output["observed_positive_fraction_5x5"] = math.nan
    output["predicted_positive_count_3x3"] = math.nan
    output["predicted_positive_fraction_3x3"] = math.nan
    output["predicted_positive_count_5x5"] = math.nan
    output["predicted_positive_fraction_5x5"] = math.nan
    output["distance_to_observed_positive_cells"] = math.nan
    output["distance_to_observed_positive_m"] = math.nan
    output["distance_to_predicted_positive_cells"] = math.nan
    output["distance_to_predicted_positive_m"] = math.nan
    output["observed_component_id"] = "missing"
    output["observed_component_area_m2"] = math.nan
    output["edge_class"] = "missing_spatial_context"
    output["exterior_ring_class"] = "missing_spatial_context"
    return output


def window_sum(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return square-window sums over a boolean grid using cumulative sums."""
    values = mask.astype(np.int32)
    padded = np.pad(values, radius, mode="constant", constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return cast(
        np.ndarray,
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size],
    )


def safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide integer arrays while returning NaN when the denominator is zero."""
    result = np.divide(
        numerator.astype(float),
        denominator.astype(float),
        out=np.full(numerator.shape, math.nan, dtype=float),
        where=denominator > 0,
    )
    return cast(np.ndarray, result)


def nearest_grid_distances(coordinates: np.ndarray, positive_coordinates: np.ndarray) -> np.ndarray:
    """Compute Euclidean grid-cell distance to the nearest positive coordinate."""
    if positive_coordinates.size == 0:
        return np.full(coordinates.shape[0], math.nan, dtype=float)
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    neighbors.fit(positive_coordinates)
    distances, _ = neighbors.kneighbors(coordinates, return_distance=True)
    return cast(np.ndarray, distances[:, 0])


def observed_component_context(
    row_offsets: np.ndarray,
    col_offsets: np.ndarray,
    observed_positive: np.ndarray,
    grid_cell_size_m: float,
) -> tuple[list[object], list[float]]:
    """Compute observed positive connected-component ids and component areas."""
    positive_indices = np.flatnonzero(observed_positive)
    component_ids: list[object] = ["none"] * len(observed_positive)
    component_areas = [math.nan] * len(observed_positive)
    if len(positive_indices) == 0:
        return component_ids, component_areas
    parent = {int(index): int(index) for index in positive_indices}
    coordinate_to_index = {
        (int(row_offsets[index]), int(col_offsets[index])): int(index) for index in positive_indices
    }
    for index in positive_indices:
        row = int(row_offsets[index])
        col = int(col_offsets[index])
        for d_row, d_col in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            neighbor = coordinate_to_index.get((row + d_row, col + d_col))
            if neighbor is not None:
                union_components(parent, int(index), neighbor)
    root_to_id: dict[int, int] = {}
    root_sizes: dict[int, int] = {}
    for index in positive_indices:
        root = find_component(parent, int(index))
        root_sizes[root] = root_sizes.get(root, 0) + 1
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
    cell_area = grid_cell_size_m * grid_cell_size_m
    for index in positive_indices:
        root = find_component(parent, int(index))
        component_ids[int(index)] = root_to_id[root]
        component_areas[int(index)] = root_sizes[root] * cell_area
    return component_ids, component_areas


def find_component(parent: dict[int, int], index: int) -> int:
    """Find a union-find component root with path compression."""
    root = index
    while parent[root] != root:
        root = parent[root]
    while parent[index] != index:
        next_index = parent[index]
        parent[index] = root
        index = next_index
    return root


def union_components(parent: dict[int, int], left: int, right: int) -> None:
    """Union two observed-positive component roots."""
    left_root = find_component(parent, left)
    right_root = find_component(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def edge_classes(frame: pd.DataFrame) -> pd.Series:
    """Classify rows as positive interior/edge or exterior rings around positives."""
    labels: list[str] = []
    observed_positive = frame["observed_binary_positive"].to_numpy(dtype=bool)
    observed_3 = frame["observed_positive_count_3x3"].to_numpy(dtype=float)
    observed_5 = frame["observed_positive_count_5x5"].to_numpy(dtype=float)
    for positive, count_3, count_5 in zip(observed_positive, observed_3, observed_5, strict=True):
        if positive and count_3 <= 1:
            labels.append("isolated_positive")
        elif positive and count_3 >= 9:
            labels.append("positive_interior")
        elif positive:
            labels.append("positive_edge")
        elif count_3 > 0:
            labels.append("zero_adjacent_to_positive")
        elif count_5 > 0:
            labels.append("near_positive_exterior")
        else:
            labels.append("far_zero_exterior")
    return pd.Series(labels, index=frame.index, dtype="object")


def exterior_ring_classes(frame: pd.DataFrame) -> pd.Series:
    """Classify non-positive exterior rows by proximity ring to observed positives."""
    labels: list[str] = []
    for edge_class in frame["edge_class"].astype(str):
        if edge_class in {"positive_interior", "positive_edge", "isolated_positive"}:
            labels.append("observed_positive")
        elif edge_class == "zero_adjacent_to_positive":
            labels.append("adjacent_8_neighbor_ring")
        elif edge_class == "near_positive_exterior":
            labels.append("near_5x5_ring")
        else:
            labels.append("far_exterior")
    return pd.Series(labels, index=frame.index, dtype="object")


def aggregate_component_failures(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    config: ComponentFailureConfig,
    *,
    fields: tuple[str, ...] = COMPONENT_FAILURE_FIELDS,
) -> list[dict[str, object]]:
    """Aggregate component-failure rows over configured context columns."""
    if dataframe.empty:
        return []
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
    grouping = base_columns + group_columns
    rows: list[dict[str, object]] = []
    for keys, group in dataframe.groupby(grouping, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = {
            column: normalized_group_value(value)
            for column, value in zip(grouping, key_tuple, strict=True)
        }
        rows.append(component_aggregate_row(group, values, fields, config))
    return rows


def component_aggregate_row(
    group: pd.DataFrame,
    values: Mapping[str, object],
    fields: tuple[str, ...],
    config: ComponentFailureConfig,
) -> dict[str, object]:
    """Build one component-failure aggregate row."""
    row = {column: values.get(column, "all") for column in COMPONENT_FAILURE_CONTEXT_COLUMNS}
    for temporal_column in (
        "positive_quarter_count",
        "annual_mean_canopy_area_bin",
        "annual_max_to_mean_ratio_bin",
        "first_positive_quarter",
        "last_positive_quarter",
    ):
        row[temporal_column] = values.get(temporal_column, "all")
    observed = group["kelp_max_y"].to_numpy(dtype=float)
    conditional = group["pred_conditional_area_m2"].to_numpy(dtype=float)
    expected = group["pred_expected_value_area_m2"].to_numpy(dtype=float)
    hard_gate = group["pred_hard_gate_area_m2"].to_numpy(dtype=float)
    expected_residual = group["expected_value_residual"].to_numpy(dtype=float)
    hard_residual = group["hard_gate_residual"].to_numpy(dtype=float)
    observed_positive = group["observed_binary_positive"].to_numpy(dtype=bool)
    predicted_positive = group["predicted_binary_positive"].to_numpy(dtype=bool)
    detected_observed_positive = observed_positive & predicted_positive
    observed_area = float(np.nansum(observed))
    expected_area = float(np.nansum(expected))
    hard_area = float(np.nansum(hard_gate))
    row.update(
        {
            "row_count": int(len(group)),
            "observed_positive_count": int(np.count_nonzero(observed_positive)),
            "observed_zero_count": int(np.count_nonzero(observed == 0)),
            "predicted_positive_count": int(np.count_nonzero(predicted_positive)),
            "detected_observed_positive_count": int(np.count_nonzero(detected_observed_positive)),
            "observed_canopy_area": observed_area,
            "conditional_predicted_area": float(np.nansum(conditional)),
            "expected_value_predicted_area": expected_area,
            "hard_gate_predicted_area": hard_area,
            "expected_value_area_bias": expected_area - observed_area,
            "expected_value_area_pct_bias": percent_bias(expected_area, observed_area),
            "hard_gate_area_bias": hard_area - observed_area,
            "hard_gate_area_pct_bias": percent_bias(hard_area, observed_area),
            "expected_value_mean_residual": safe_mean(expected_residual),
            "expected_value_mae": safe_mae(expected_residual),
            "expected_value_rmse": root_mean_squared_error(observed, expected),
            "hard_gate_mean_residual": safe_mean(hard_residual),
            "hard_gate_mae": safe_mae(hard_residual),
            "hard_gate_rmse": root_mean_squared_error(observed, hard_gate),
            "mean_calibrated_probability": safe_mean(
                group["calibrated_presence_probability"].to_numpy(dtype=float)
            ),
            "mean_probability_margin": safe_mean(
                group["binary_threshold_margin"].to_numpy(dtype=float)
            ),
            "true_positive_count": int(np.count_nonzero(group["binary_outcome"] == "TP")),
            "false_positive_count": int(np.count_nonzero(group["binary_outcome"] == "FP")),
            "false_negative_count": int(np.count_nonzero(group["binary_outcome"] == "FN")),
            "true_negative_count": int(np.count_nonzero(group["binary_outcome"] == "TN")),
            "support_miss_positive_count": int(group["support_miss_positive"].sum()),
            "support_miss_positive_rate": safe_ratio(
                int(group["support_miss_positive"].sum()), int(np.count_nonzero(observed_positive))
            ),
            "support_leakage_zero_count": int(group["support_leakage_zero"].sum()),
            "support_leakage_zero_rate": safe_ratio(
                int(group["support_leakage_zero"].sum()), int(np.count_nonzero(observed == 0))
            ),
            "amount_underprediction_detected_positive_count": int(
                group["amount_underprediction_detected_positive"].sum()
            ),
            "amount_underprediction_detected_positive_rate": safe_ratio(
                int(group["amount_underprediction_detected_positive"].sum()),
                int(np.count_nonzero(detected_observed_positive)),
            ),
            "amount_overprediction_low_or_zero_count": int(
                group["amount_overprediction_low_or_zero"].sum()
            ),
            "amount_overprediction_low_or_zero_rate": safe_ratio(
                int(group["amount_overprediction_low_or_zero"].sum()),
                int(np.count_nonzero(observed < ANNUAL_MAX_10PCT_AREA_M2)),
            ),
            "composition_shrinkage_count": int(group["composition_shrinkage"].sum()),
            "composition_shrinkage_rate": safe_ratio(
                int(group["composition_shrinkage"].sum()),
                int(np.count_nonzero(detected_observed_positive)),
            ),
            "high_confidence_wrong_count": int(group["high_confidence_wrong"].sum()),
            "high_confidence_wrong_rate": safe_ratio(
                int(group["high_confidence_wrong"].sum()), int(len(group))
            ),
            "near_correct_count": int(group["near_correct"].sum()),
            "near_correct_rate": safe_ratio(int(group["near_correct"].sum()), int(len(group))),
            "near_correct_positive_count": int(
                (group["near_correct"] & group["observed_binary_positive"]).sum()
            ),
            "near_correct_positive_rate": safe_ratio(
                int((group["near_correct"] & group["observed_binary_positive"]).sum()),
                int(np.count_nonzero(observed_positive)),
            ),
            "positive_edge_count": int(np.count_nonzero(group["edge_class"] == "positive_edge")),
            "positive_interior_count": int(
                np.count_nonzero(group["edge_class"] == "positive_interior")
            ),
            "zero_adjacent_to_positive_count": int(
                np.count_nonzero(group["edge_class"] == "zero_adjacent_to_positive")
            ),
            "near_positive_exterior_count": int(
                np.count_nonzero(group["edge_class"] == "near_positive_exterior")
            ),
            "mean_observed_positive_distance_m": safe_mean(
                group["distance_to_observed_positive_m"].to_numpy(dtype=float)
            ),
            "mean_predicted_positive_distance_m": safe_mean(
                group["distance_to_predicted_positive_m"].to_numpy(dtype=float)
            ),
        }
    )
    if row["label_source"] == "all":
        row["label_source"] = config.primary_label_source
    return {field: row.get(field, "") for field in fields}


def normalized_group_value(value: object) -> str:
    """Normalize a group key to a stable string."""
    if pd.isna(value):
        return "missing"
    return str(value)


def safe_mean(values: np.ndarray) -> float:
    """Return a finite mean or NaN when no finite values exist."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(finite))


def safe_mae(residual: np.ndarray) -> float:
    """Return mean absolute residual for finite residual values."""
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(np.abs(finite)))


def build_edge_effect_rows(
    dataframe: pd.DataFrame,
    config: ComponentFailureConfig,
) -> list[dict[str, object]]:
    """Build one edge-effect hypothesis row per model context."""
    if dataframe.empty:
        return []
    rows: list[dict[str, object]] = []
    group_columns = [
        "context_id",
        "evaluation_region",
        "training_regime",
        "model_origin_region",
        "split",
        "year",
        "mask_status",
        "evaluation_scope",
    ]
    for keys, group in dataframe.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = {
            column: normalized_group_value(value)
            for column, value in zip(group_columns, key_tuple, strict=True)
        }
        rows.append(edge_effect_row(group, values, config))
    return rows


def edge_effect_row(
    group: pd.DataFrame,
    values: Mapping[str, object],
    config: ComponentFailureConfig,
) -> dict[str, object]:
    """Summarize edge/interior structure of FP, FN, and composition shrinkage."""
    false_positive = group["binary_outcome"] == "FP"
    false_negative = group["binary_outcome"] == "FN"
    adjacent_observed = group["edge_class"] == "zero_adjacent_to_positive"
    near_observed = group["edge_class"] == "near_positive_exterior"
    adjacent_or_near = adjacent_observed | near_observed
    far_from_observed = ~(adjacent_observed | near_observed)
    isolated_positive = group["edge_class"] == "isolated_positive"
    positive_edge = group["edge_class"] == "positive_edge"
    edge_or_isolated = isolated_positive | positive_edge
    interior = group["edge_class"] == "positive_interior"
    predicted_3x3 = group["predicted_positive_count_3x3"].to_numpy(dtype=float)
    fp_isolated = false_positive & (predicted_3x3 <= 1)
    fp_predicted_edge = false_positive & (predicted_3x3 > 1) & (predicted_3x3 < 9)
    fp_predicted_interior = false_positive & (predicted_3x3 >= 9)
    edge_positive = positive_edge | isolated_positive
    interior_positive = group["edge_class"] == "positive_interior"
    composition_edge = group["composition_shrinkage"] & edge_positive
    return {
        "context_id": values["context_id"],
        "evaluation_region": values["evaluation_region"],
        "training_regime": values["training_regime"],
        "model_origin_region": values["model_origin_region"],
        "split": values["split"],
        "year": values["year"],
        "mask_status": values["mask_status"],
        "evaluation_scope": values["evaluation_scope"],
        "row_count": int(len(group)),
        "observed_positive_count": int(group["observed_binary_positive"].sum()),
        "false_positive_count": int(false_positive.sum()),
        "false_negative_count": int(false_negative.sum()),
        "fp_isolated_predicted_positive_count": int(fp_isolated.sum()),
        "fp_isolated_predicted_positive_rate": safe_ratio(
            int(fp_isolated.sum()), int(false_positive.sum())
        ),
        "fp_predicted_edge_count": int(fp_predicted_edge.sum()),
        "fp_predicted_edge_rate": safe_ratio(
            int(fp_predicted_edge.sum()), int(false_positive.sum())
        ),
        "fp_predicted_interior_count": int(fp_predicted_interior.sum()),
        "fp_predicted_interior_rate": safe_ratio(
            int(fp_predicted_interior.sum()), int(false_positive.sum())
        ),
        "fp_adjacent_observed_count": int((false_positive & adjacent_observed).sum()),
        "fp_adjacent_observed_rate": safe_ratio(
            int((false_positive & adjacent_observed).sum()), int(false_positive.sum())
        ),
        "fp_near_observed_count": int((false_positive & near_observed).sum()),
        "fp_near_observed_rate": safe_ratio(
            int((false_positive & near_observed).sum()), int(false_positive.sum())
        ),
        "fp_far_from_observed_count": int((false_positive & far_from_observed).sum()),
        "fp_far_from_observed_rate": safe_ratio(
            int((false_positive & far_from_observed).sum()), int(false_positive.sum())
        ),
        "fp_adjacent_or_near_positive_count": int((false_positive & adjacent_or_near).sum()),
        "fp_adjacent_or_near_positive_rate": safe_ratio(
            int((false_positive & adjacent_or_near).sum()), int(false_positive.sum())
        ),
        "fn_isolated_positive_count": int((false_negative & isolated_positive).sum()),
        "fn_isolated_positive_rate": safe_ratio(
            int((false_negative & isolated_positive).sum()), int(false_negative.sum())
        ),
        "fn_positive_edge_count": int((false_negative & positive_edge).sum()),
        "fn_positive_edge_rate": safe_ratio(
            int((false_negative & positive_edge).sum()), int(false_negative.sum())
        ),
        "fn_positive_edge_or_isolated_count": int((false_negative & edge_or_isolated).sum()),
        "fn_positive_edge_or_isolated_rate": safe_ratio(
            int((false_negative & edge_or_isolated).sum()), int(false_negative.sum())
        ),
        "fn_positive_interior_count": int((false_negative & interior).sum()),
        "fn_positive_interior_rate": safe_ratio(
            int((false_negative & interior).sum()), int(false_negative.sum())
        ),
        "high_confidence_fp_adjacent_or_near_count": int(
            (false_positive & adjacent_or_near & group["high_confidence_wrong"]).sum()
        ),
        "high_confidence_fn_edge_or_isolated_count": int(
            (false_negative & edge_or_isolated & group["high_confidence_wrong"]).sum()
        ),
        "expected_value_positive_edge_mean_residual": safe_mean(
            group.loc[edge_positive, "expected_value_residual"].to_numpy(dtype=float)
        ),
        "expected_value_positive_interior_mean_residual": safe_mean(
            group.loc[interior_positive, "expected_value_residual"].to_numpy(dtype=float)
        ),
        "hard_gate_positive_edge_mean_residual": safe_mean(
            group.loc[edge_positive, "hard_gate_residual"].to_numpy(dtype=float)
        ),
        "hard_gate_positive_interior_mean_residual": safe_mean(
            group.loc[interior_positive, "hard_gate_residual"].to_numpy(dtype=float)
        ),
        "composition_shrinkage_edge_count": int(composition_edge.sum()),
        "composition_shrinkage_edge_rate": safe_ratio(
            int(composition_edge.sum()), int(edge_positive.sum())
        ),
    }


def write_component_failure_outputs(
    tables: ComponentFailureTables,
    config: ComponentFailureConfig,
) -> None:
    """Write component-failure CSV tables and the sidecar manifest."""
    write_csv(tables.summary, config.summary_path, COMPONENT_FAILURE_FIELDS)
    write_csv(tables.by_label_context, config.by_label_context_path, COMPONENT_FAILURE_FIELDS)
    write_csv(tables.by_domain_context, config.by_domain_context_path, COMPONENT_FAILURE_FIELDS)
    write_csv(tables.by_spatial_context, config.by_spatial_context_path, COMPONENT_FAILURE_FIELDS)
    write_csv(tables.by_model_context, config.by_model_context_path, COMPONENT_FAILURE_FIELDS)
    write_csv(tables.edge_effect, config.edge_effect_path, COMPONENT_EDGE_FIELDS)
    write_csv(
        tables.temporal_label_context,
        config.temporal_label_context_path,
        COMPONENT_TEMPORAL_FIELDS,
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w") as file:
        json.dump(component_failure_manifest(tables, config), file, indent=2)
        file.write("\n")


def write_csv(rows: list[dict[str, object]], path: Path, fields: tuple[str, ...]) -> None:
    """Write dictionaries to a CSV file with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for field in fields:
        if field not in frame.columns:
            frame[field] = []
    frame.loc[:, list(fields)].to_csv(path, index=False)


def component_failure_manifest(
    tables: ComponentFailureTables,
    config: ComponentFailureConfig,
) -> dict[str, object]:
    """Build manifest metadata for component-failure diagnostics."""
    return {
        "command": "analyze-model",
        "diagnostic": "phase2_component_failure",
        "primary_filters": {
            "split": config.primary_split,
            "year": config.primary_year,
            "mask_status": config.primary_mask_status,
            "evaluation_scope": config.primary_evaluation_scope,
            "label_source": config.primary_label_source,
        },
        "definitions": {
            "binary_target": "annual_max_ge_10pct",
            "binary_positive_area_m2": ANNUAL_MAX_10PCT_AREA_M2,
            "near_correct_tolerance_m2": config.tolerance_m2,
            "grid_cell_size_m": config.grid_cell_size_m,
            "edge_class": (
                "derived from retained-grid 3x3 and 5x5 observed annual_max_ge_10pct neighborhoods"
            ),
            "distance": (
                "Euclidean grid-cell distance to nearest observed or predicted "
                "annual_max_ge_10pct positive"
            ),
        },
        "inputs": [
            {
                "context_id": input_config.context_id,
                "training_regime": input_config.training_regime,
                "model_origin_region": input_config.model_origin_region,
                "evaluation_region": input_config.evaluation_region,
                "hurdle_predictions": str(input_config.hurdle_predictions_path),
                "binary_predictions": str(input_config.binary_predictions_path)
                if input_config.binary_predictions_path is not None
                else None,
                "label_path": str(input_config.label_path)
                if input_config.label_path is not None
                else None,
                "config_path": str(input_config.config_path)
                if input_config.config_path is not None
                else None,
                "required": input_config.required,
            }
            for input_config in config.inputs
        ],
        "outputs": {
            "summary": str(config.summary_path),
            "by_label_context": str(config.by_label_context_path),
            "by_domain_context": str(config.by_domain_context_path),
            "by_spatial_context": str(config.by_spatial_context_path),
            "by_model_context": str(config.by_model_context_path),
            "edge_effect_diagnostics": str(config.edge_effect_path),
            "temporal_label_context": str(config.temporal_label_context_path),
            "manifest": str(config.manifest_path),
        },
        "row_counts": {
            "summary": len(tables.summary),
            "by_label_context": len(tables.by_label_context),
            "by_domain_context": len(tables.by_domain_context),
            "by_spatial_context": len(tables.by_spatial_context),
            "by_model_context": len(tables.by_model_context),
            "edge_effect_diagnostics": len(tables.edge_effect),
            "temporal_label_context": len(tables.temporal_label_context),
        },
    }


def component_failure_report_markdown(
    tables: ComponentFailureTables,
    config: ComponentFailureConfig,
) -> str:
    """Build compact report prose for Phase 2 component-failure diagnostics."""
    if not tables.summary:
        return "Component-failure diagnostics are configured, but no primary rows were available."
    required_rows = [
        row for row in tables.summary if str(row.get("context_id")) in PRIMARY_CONTEXT_IDS
    ]
    rows = required_rows if required_rows else tables.summary
    dominant = sorted(
        rows,
        key=lambda row: (
            row_number(row, "amount_underprediction_detected_positive_count")
            + row_number(row, "support_miss_positive_count")
            + row_number(row, "support_leakage_zero_count")
        ),
        reverse=True,
    )
    edge_rows = {str(row.get("context_id")): row for row in tables.edge_effect}
    lines = [
        "Primary rows remain Kelpwatch-style annual maximum reproduction diagnostics, not "
        "independent field biomass validation. The tables decompose each retained 2022 row into "
        "binary support, calibrated probability, conditional amount, expected-value hurdle, "
        "hard-gated hurdle, label, domain, temporal, and edge/interior context.",
        "",
        "See [component-failure column definitions](#component-failure-column-definitions) "
        "for denominators and spatial definitions.",
        "",
        (
            "| Context | Rows | FN | FP | Amount under | Composition shrink | "
            "Positive near correct | FN isolated | FN edge | FP isolated | Adjacent/near FP |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=component_report_sort_key):
        edge = edge_rows.get(str(row.get("context_id")), {})
        amount_under = format_count_rate(
            row,
            "amount_underprediction_detected_positive_count",
            "amount_underprediction_detected_positive_rate",
        )
        composition_shrink = format_count_rate(
            row,
            "composition_shrinkage_count",
            "composition_shrinkage_rate",
        )
        lines.append(
            "| "
            f"{component_context_label(row)} | "
            f"{int(row_number(row, 'row_count')):,} | "
            f"{int(row_number(row, 'false_negative_count')):,} | "
            f"{int(row_number(row, 'false_positive_count')):,} | "
            f"{amount_under} | "
            f"{composition_shrink} | "
            f"{format_percent(row.get('near_correct_positive_rate'))} | "
            f"{format_percent(edge.get('fn_isolated_positive_rate'))} | "
            f"{format_percent(edge.get('fn_positive_edge_rate'))} | "
            f"{format_percent(edge.get('fp_isolated_predicted_positive_rate'))} | "
            f"{format_percent(edge.get('fp_adjacent_or_near_positive_rate'))} |"
        )
    leading = dominant[0] if dominant else rows[0]
    lines.extend(
        [
            "",
            "The largest flagged count in the required contexts is in "
            f"`{component_context_label(leading)}`: "
            f"{int(row_number(leading, 'amount_underprediction_detected_positive_count')):,} "
            "detected-positive rows are still underpredicted by the expected-value hurdle, "
            f"{int(row_number(leading, 'support_miss_positive_count')):,} observed positives "
            "are missed by binary support, and "
            f"{int(row_number(leading, 'support_leakage_zero_count')):,} observed-zero rows "
            "leak through binary support.",
            "",
            "Artifact tables:",
            f"- Summary: `{config.summary_path}`",
            f"- Label context: `{config.by_label_context_path}`",
            f"- Domain context: `{config.by_domain_context_path}`",
            f"- Spatial context: `{config.by_spatial_context_path}`",
            f"- Model context: `{config.by_model_context_path}`",
            f"- Edge diagnostics: `{config.edge_effect_path}`",
            f"- Temporal label context: `{config.temporal_label_context_path}`",
            "",
            "#### Component-Failure Column Definitions",
            "",
            "- `Context`: `evaluation_region / training_regime (model_origin_region)`.",
            "- `Rows`: retained 2022 test rows in `full_grid_masked` and "
            "`plausible_kelp_domain` for the context.",
            "- `FN`: observed annual max is at least `90 m2`, but calibrated binary "
            "support predicts negative.",
            "- `FP`: observed annual max is below `90 m2`, but calibrated binary "
            "support predicts positive.",
            "- `Amount under`: count and percent of detected observed positives where "
            "the expected-value hurdle underpredicts observed annual max by more than "
            "`90 m2`; the denominator is observed positive and predicted positive rows.",
            "- `Composition shrink`: count and percent using the same detected-observed-"
            "positive denominator as `Amount under`; a row counts when the conditional "
            "canopy prediction exceeds the expected-value hurdle by at least `90 m2` "
            "and the conditional prediction is at least `225 m2`.",
            "- `Positive near correct`: among observed positives, the expected-value "
            "hurdle is within `90 m2` of observed annual max.",
            "- `FN isolated`: among FNs, the observed positive cell has no other "
            "observed positive in its 3x3 retained-grid neighborhood.",
            "- `FN edge`: among FNs, the observed positive cell is on a positive patch "
            "edge, not isolated and not a full 3x3 positive interior.",
            "- `FP isolated`: among FPs, the predicted positive cell has no other "
            "predicted positive in its 3x3 retained-grid neighborhood.",
            "- `Adjacent/near FP`: among FPs, the observed-zero cell is adjacent to an "
            "observed positive in the 3x3 neighborhood or near one in the 5x5 "
            "neighborhood. On the 30 m grid, that means within two row/column cells, "
            "roughly up to 60 m horizontally or vertically.",
        ]
    )
    return "\n".join(lines)


def component_report_sort_key(row: dict[str, object]) -> tuple[int, int, str]:
    """Sort component report rows by region and training regime."""
    region_order = {"big_sur": 0, "monterey": 1}
    regime_order = {"monterey_only": 0, "big_sur_only": 1, "pooled_monterey_big_sur": 2}
    return (
        region_order.get(str(row.get("evaluation_region")), 99),
        regime_order.get(str(row.get("training_regime")), 99),
        str(row.get("context_id")),
    )


def component_context_label(row: dict[str, object]) -> str:
    """Return a compact context display label."""
    return (
        f"{row.get('evaluation_region')} / {row.get('training_regime')} "
        f"({row.get('model_origin_region')})"
    )


def format_percent(value: object) -> str:
    """Format a numeric fraction as a report percentage."""
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.1%}"


def format_count_rate(row: Mapping[str, object], count_key: str, rate_key: str) -> str:
    """Format a count plus percentage for a component-failure report cell."""
    count = int(row_number(row, count_key))
    return f"{count:,} ({format_percent(row.get(rate_key))})"


def row_number(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    """Return a numeric report value from a dictionary-like output row."""
    try:
        return float(cast(Any, row.get(key, default)))
    except (TypeError, ValueError):
        return default
