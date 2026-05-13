"""Train and evaluate the Phase 1 positive-only conditional canopy model."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    MASK_RETAIN_COLUMN,
    ReportingDomainMask,
    load_reporting_domain_mask,
    mask_status,
)
from kelp_aef.evaluation.baselines import (
    KELPWATCH_PIXEL_AREA_M2,
    SPLIT_ORDER,
    correlation,
    mean_absolute_error,
    parse_bands,
    percent_bias,
    precision_recall_f1,
    r2_score,
    root_mean_squared_error,
    safe_ratio,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

CONDITIONAL_MODEL_NAME = "ridge_positive_annual_max"
CONDITIONAL_MODEL_FAMILY = "conditional_canopy"
DEFAULT_TARGET_COLUMN = "kelp_fraction_y"
DEFAULT_TARGET_AREA_COLUMN = "kelp_max_y"
DEFAULT_POSITIVE_TARGET_LABEL = "annual_max_ge_10pct"
DEFAULT_POSITIVE_TARGET_THRESHOLD_FRACTION = 0.10
DEFAULT_POSITIVE_TARGET_THRESHOLD_AREA = 90.0
DEFAULT_POSITIVE_SUPPORT_POLICY = "observed_positive_train_validation_test"
DEFAULT_LIKELY_POSITIVE_POLICY = "calibrated_probability_ge_validation_max_f1"
DEFAULT_LIKELY_POSITIVE_THRESHOLD_POLICY = "validation_max_f1_calibrated"
DEFAULT_RIDGE_BASELINE_MODEL_NAME = "ridge_regression"
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SELECTION_SPLIT = "validation"
TEST_SPLIT = "test"
OPTIONAL_ID_COLUMNS = (
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "label_source",
    "is_kelpwatch_observed",
    "kelpwatch_station_count",
    "sample_weight",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "crm_elevation_m",
    "crm_depth_m",
    "depth_bin",
    "elevation_bin",
)
REQUIRED_INPUT_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TARGET_AREA_COLUMN,
)
CONDITIONAL_PREDICTION_FIELDS = (
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
    "sample_weight",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "crm_elevation_m",
    "crm_depth_m",
    "depth_bin",
    "elevation_bin",
    "model_name",
    "model_family",
    "target",
    "target_area_column",
    "positive_target_label",
    "positive_target_threshold_fraction",
    "positive_target_threshold_area",
    "positive_support_policy",
    "likely_positive_policy",
    "likely_positive_threshold_policy",
    "selection_split",
    "selection_year",
    "test_split",
    "test_year",
    "selected_alpha",
    "observed_positive_support",
    "likely_positive_support",
    "calibrated_binary_probability",
    "calibrated_probability_threshold",
    "evaluation_scope",
    "pred_conditional_fraction",
    "pred_conditional_fraction_clipped",
    "pred_conditional_area_m2",
    "residual_conditional_fraction",
    "residual_conditional_area_m2",
)
CONDITIONAL_METRIC_FIELDS = (
    "model_name",
    "model_family",
    "target",
    "target_area_column",
    "positive_target_label",
    "positive_target_threshold_fraction",
    "positive_target_threshold_area",
    "positive_support_policy",
    "selection_split",
    "selection_year",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "selected_alpha",
    "observed_mean_fraction",
    "predicted_mean_fraction",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mae_fraction",
    "rmse_fraction",
    "mae_area",
    "rmse_area",
    "r2_fraction",
    "spearman_fraction",
    "mean_residual_area",
    "median_residual_area",
    "underprediction_count",
    "overprediction_count",
    "high_canopy_count",
    "near_saturated_count",
)
CONDITIONAL_RESIDUAL_FIELDS = (
    "model_name",
    "model_family",
    "target",
    "target_area_column",
    "positive_target_label",
    "positive_support_policy",
    "likely_positive_policy",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "observed_bin",
    "row_count",
    "observed_mean_area",
    "predicted_mean_area",
    "mean_residual_area",
    "median_residual_area",
    "mae_area",
    "rmse_area",
    "underprediction_count",
    "overprediction_count",
)
CONDITIONAL_COMPARISON_FIELDS = (
    "model_name",
    "model_family",
    "comparison_support",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "selected_alpha",
    "mae_fraction",
    "rmse_fraction",
    "mae_area",
    "rmse_area",
    "r2_fraction",
    "spearman_fraction",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mean_residual_area",
    "underprediction_count",
    "overprediction_count",
    "f1_ge_10pct",
)
CONDITIONAL_FULL_GRID_SUMMARY_FIELDS = (
    "conditional_model_name",
    "model_family",
    "likely_positive_policy",
    "likely_positive_threshold_policy",
    "probability_source",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "probability_threshold",
    "row_count",
    "likely_positive_count",
    "likely_positive_rate",
    "likely_positive_cell_area_m2",
    "observed_positive_count",
    "observed_positive_rate",
    "assumed_background_count",
    "assumed_background_likely_positive_count",
    "assumed_background_likely_positive_rate",
    "diagnostic_note",
)


@dataclass(frozen=True)
class ConditionalCanopyConfig:
    """Resolved config values for conditional canopy training."""

    config_path: Path
    input_table_path: Path
    split_manifest_path: Path
    model_output_path: Path
    sample_predictions_path: Path
    metrics_path: Path
    positive_residuals_path: Path
    model_comparison_path: Path
    manifest_path: Path
    residual_figure_path: Path | None
    full_grid_likely_positive_summary_path: Path | None
    baseline_sample_predictions_path: Path | None
    calibrated_sample_predictions_path: Path | None
    calibrated_full_grid_area_summary_path: Path | None
    target_column: str
    target_area_column: str
    positive_target_label: str
    positive_target_threshold_fraction: float
    positive_target_threshold_area: float
    positive_support_policy: str
    likely_positive_policy: str
    likely_positive_threshold_policy: str
    ridge_baseline_model_name: str
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    drop_missing_features: bool
    reporting_domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class ConditionalCanopySidecarConfig:
    """Resolved config for a conditional-canopy sidecar reuse decision."""

    name: str
    sample_policy: str
    conditional_config: ConditionalCanopyConfig
    reuse_manifest_path: Path


@dataclass(frozen=True)
class PreparedConditionalData:
    """Prepared sample rows plus split and support diagnostics."""

    retained_rows: pd.DataFrame
    dropped_counts_by_split: dict[str, int]
    train_positive_count: int
    validation_positive_count: int
    test_positive_count: int


@dataclass(frozen=True)
class ConditionalModelSelection:
    """Selected conditional ridge model and validation diagnostics."""

    model: Any
    selected_alpha: float
    validation_rows: list[dict[str, object]]


def train_conditional_canopy(config_path: Path) -> int:
    """Train the positive-only conditional canopy model and write artifacts."""
    conditional_config = load_conditional_canopy_config(config_path)
    LOGGER.info("Loading conditional-canopy model input: %s", conditional_config.input_table_path)
    sample = pd.read_parquet(conditional_config.input_table_path)
    split_manifest = pd.read_parquet(conditional_config.split_manifest_path)
    prepared = prepare_conditional_model_frame(sample, split_manifest, conditional_config)
    train_rows = observed_positive_rows(prepared.retained_rows, "train", conditional_config)
    validation_rows = observed_positive_rows(
        prepared.retained_rows,
        "validation",
        conditional_config,
    )
    selection = fit_select_conditional_ridge(train_rows, validation_rows, conditional_config)
    likely_flags = read_likely_positive_flags(prepared.retained_rows, conditional_config)
    sample_predictions = conditional_sample_prediction_frame(
        prepared.retained_rows,
        likely_flags,
        selection,
        conditional_config,
    )
    primary_predictions = primary_observed_positive_predictions(sample_predictions)
    metrics = build_conditional_metric_rows(primary_predictions, conditional_config)
    residuals = build_positive_residual_rows(primary_predictions, conditional_config)
    comparison = build_conditional_comparison_rows(
        primary_predictions,
        conditional_config,
        selection.selected_alpha,
    )
    full_grid_summary = build_full_grid_likely_positive_summary(conditional_config)
    write_conditional_predictions(
        sample_predictions,
        conditional_config.sample_predictions_path,
    )
    write_csv_rows(metrics, conditional_config.metrics_path, CONDITIONAL_METRIC_FIELDS)
    write_csv_rows(
        residuals,
        conditional_config.positive_residuals_path,
        CONDITIONAL_RESIDUAL_FIELDS,
    )
    write_csv_rows(
        comparison,
        conditional_config.model_comparison_path,
        CONDITIONAL_COMPARISON_FIELDS,
    )
    if conditional_config.full_grid_likely_positive_summary_path is not None:
        write_csv_rows(
            full_grid_summary,
            conditional_config.full_grid_likely_positive_summary_path,
            CONDITIONAL_FULL_GRID_SUMMARY_FIELDS,
        )
    if conditional_config.residual_figure_path is not None:
        write_positive_residual_figure(residuals, conditional_config.residual_figure_path)
    write_conditional_model(selection, conditional_config, prepared)
    write_conditional_manifest(
        prepared,
        selection,
        sample_predictions,
        metrics,
        residuals,
        comparison,
        full_grid_summary,
        conditional_config,
    )
    LOGGER.info("Wrote conditional-canopy model: %s", conditional_config.model_output_path)
    LOGGER.info(
        "Wrote conditional sample predictions: %s",
        conditional_config.sample_predictions_path,
    )
    LOGGER.info("Wrote conditional metrics: %s", conditional_config.metrics_path)
    for sidecar in load_conditional_canopy_sidecar_configs(config_path, conditional_config):
        LOGGER.info("Writing conditional-canopy sidecar reuse decision: %s", sidecar.name)
        write_conditional_sidecar_reuse_artifacts(
            base_config=conditional_config,
            sidecar=sidecar,
            base_prepared=prepared,
        )
    return 0


def load_conditional_canopy_config(config_path: Path) -> ConditionalCanopyConfig:
    """Load conditional canopy settings from the workflow config."""
    config = load_yaml_config(config_path)
    alignment = require_mapping(config.get("alignment"), "alignment")
    splits = require_mapping(config.get("splits"), "splits")
    features = require_mapping(config.get("features"), "features")
    models = require_mapping(config.get("models"), "models")
    conditional = require_mapping(
        models.get("conditional_canopy"),
        "models.conditional_canopy",
    )
    baselines = optional_mapping(models.get("baselines"), "models.baselines")
    binary_presence = optional_mapping(models.get("binary_presence"), "models.binary_presence")
    binary_calibration = optional_mapping(
        binary_presence.get("calibration"),
        "models.binary_presence.calibration",
    )
    reporting_domain_mask = load_reporting_domain_mask(config)
    positive_support_policy = str(
        conditional.get("positive_support_policy", DEFAULT_POSITIVE_SUPPORT_POLICY)
    )
    if positive_support_policy != DEFAULT_POSITIVE_SUPPORT_POLICY:
        msg = (
            "models.conditional_canopy.positive_support_policy currently supports only "
            f"{DEFAULT_POSITIVE_SUPPORT_POLICY!r}"
        )
        raise ValueError(msg)
    return ConditionalCanopyConfig(
        config_path=config_path,
        input_table_path=Path(
            require_string(
                conditional.get("input_table") or alignment.get("output_table"),
                "models.conditional_canopy.input_table or alignment.output_table",
            )
        ),
        split_manifest_path=Path(
            require_string(splits.get("output_manifest"), "splits.output_manifest")
        ),
        model_output_path=Path(
            require_string(conditional.get("model"), "models.conditional_canopy.model")
        ),
        sample_predictions_path=Path(
            require_string(
                conditional.get("sample_predictions"),
                "models.conditional_canopy.sample_predictions",
            )
        ),
        metrics_path=Path(
            require_string(conditional.get("metrics"), "models.conditional_canopy.metrics")
        ),
        positive_residuals_path=Path(
            require_string(
                conditional.get("positive_residuals"),
                "models.conditional_canopy.positive_residuals",
            )
        ),
        model_comparison_path=Path(
            require_string(
                conditional.get("model_comparison"),
                "models.conditional_canopy.model_comparison",
            )
        ),
        manifest_path=Path(
            require_string(conditional.get("manifest"), "models.conditional_canopy.manifest")
        ),
        residual_figure_path=optional_path(conditional.get("residual_figure")),
        full_grid_likely_positive_summary_path=optional_path(
            conditional.get("full_grid_likely_positive_summary")
        ),
        baseline_sample_predictions_path=optional_path(
            conditional.get("baseline_sample_predictions") or baselines.get("sample_predictions")
        ),
        calibrated_sample_predictions_path=optional_path(
            conditional.get("calibrated_binary_sample_predictions")
            or binary_calibration.get("calibrated_sample_predictions")
        ),
        calibrated_full_grid_area_summary_path=optional_path(
            conditional.get("calibrated_binary_full_grid_area_summary")
            or binary_calibration.get("full_grid_area_summary")
        ),
        target_column=str(conditional.get("target", DEFAULT_TARGET_COLUMN)),
        target_area_column=str(conditional.get("target_area_column", DEFAULT_TARGET_AREA_COLUMN)),
        positive_target_label=str(
            conditional.get("positive_target_label", DEFAULT_POSITIVE_TARGET_LABEL)
        ),
        positive_target_threshold_fraction=optional_float(
            conditional.get("positive_target_threshold_fraction"),
            "models.conditional_canopy.positive_target_threshold_fraction",
            DEFAULT_POSITIVE_TARGET_THRESHOLD_FRACTION,
        ),
        positive_target_threshold_area=optional_float(
            conditional.get("positive_target_threshold_area"),
            "models.conditional_canopy.positive_target_threshold_area",
            DEFAULT_POSITIVE_TARGET_THRESHOLD_AREA,
        ),
        positive_support_policy=positive_support_policy,
        likely_positive_policy=str(
            conditional.get("likely_positive_policy", DEFAULT_LIKELY_POSITIVE_POLICY)
        ),
        likely_positive_threshold_policy=str(
            conditional.get(
                "likely_positive_threshold_policy",
                DEFAULT_LIKELY_POSITIVE_THRESHOLD_POLICY,
            )
        ),
        ridge_baseline_model_name=str(
            conditional.get("ridge_baseline_model_name", DEFAULT_RIDGE_BASELINE_MODEL_NAME)
        ),
        feature_columns=parse_bands(conditional.get("features") or features.get("bands")),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
        alpha_grid=read_alpha_grid(conditional.get("alpha_grid")),
        drop_missing_features=read_bool(
            conditional.get("drop_missing_features"),
            "models.conditional_canopy.drop_missing_features",
            default=True,
        ),
        reporting_domain_mask=reporting_domain_mask,
    )


def load_conditional_canopy_sidecar_configs(
    config_path: Path,
    base_config: ConditionalCanopyConfig,
) -> tuple[ConditionalCanopySidecarConfig, ...]:
    """Load optional conditional sidecars that reuse the positive-only model."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    conditional = require_mapping(
        models.get("conditional_canopy"),
        "models.conditional_canopy",
    )
    sidecars = optional_mapping(
        conditional.get("sidecars"),
        "models.conditional_canopy.sidecars",
    )
    output: list[ConditionalCanopySidecarConfig] = []
    for name, value in sidecars.items():
        sidecar_name = str(name)
        sidecar = require_mapping(value, f"models.conditional_canopy.sidecars.{sidecar_name}")
        if not read_bool(
            sidecar.get("enabled"),
            f"models.conditional_canopy.sidecars.{sidecar_name}.enabled",
            default=True,
        ):
            continue
        if not read_bool(
            sidecar.get("reuse_model"),
            f"models.conditional_canopy.sidecars.{sidecar_name}.reuse_model",
            default=True,
        ):
            msg = (
                "conditional canopy sidecars currently support only reuse_model=true; "
                f"got models.conditional_canopy.sidecars.{sidecar_name}.reuse_model=false"
            )
            raise ValueError(msg)
        output.append(
            ConditionalCanopySidecarConfig(
                name=sidecar_name,
                sample_policy=str(sidecar.get("sample_policy", sidecar_name)),
                conditional_config=replace(
                    base_config,
                    input_table_path=conditional_sidecar_path(sidecar, sidecar_name, "input_table"),
                    calibrated_sample_predictions_path=conditional_sidecar_optional_path(
                        sidecar,
                        sidecar_name,
                        "calibrated_binary_sample_predictions",
                        base_config.calibrated_sample_predictions_path,
                    ),
                    calibrated_full_grid_area_summary_path=conditional_sidecar_optional_path(
                        sidecar,
                        sidecar_name,
                        "calibrated_binary_full_grid_area_summary",
                        base_config.calibrated_full_grid_area_summary_path,
                    ),
                    full_grid_likely_positive_summary_path=conditional_sidecar_optional_path(
                        sidecar,
                        sidecar_name,
                        "full_grid_likely_positive_summary",
                        None,
                    ),
                ),
                reuse_manifest_path=conditional_sidecar_path(
                    sidecar, sidecar_name, "reuse_manifest"
                ),
            )
        )
    return tuple(output)


def conditional_sidecar_path(config: dict[str, Any], sidecar_name: str, key: str) -> Path:
    """Read a required conditional sidecar path from config."""
    return Path(
        require_string(
            config.get(key),
            f"models.conditional_canopy.sidecars.{sidecar_name}.{key}",
        )
    )


def conditional_sidecar_optional_path(
    config: dict[str, Any],
    sidecar_name: str,
    key: str,
    default: Path | None,
) -> Path | None:
    """Read an optional conditional sidecar path with a configured default."""
    value = config.get(key)
    if value is None:
        return default
    return Path(
        require_string(
            value,
            f"models.conditional_canopy.sidecars.{sidecar_name}.{key}",
        )
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating missing values as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def optional_path(value: object) -> Path | None:
    """Return an optional path from a config value."""
    if value is None:
        return None
    return Path(require_string(value, "optional path"))


def optional_float(value: object, name: str, default: float) -> float:
    """Read an optional floating-point config value."""
    if value is None:
        return default
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    return float(cast(Any, value))


def read_bool(value: object, name: str, *, default: bool) -> bool:
    """Read an optional boolean config value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"config field must be a boolean: {name}"
        raise ValueError(msg)
    return value


def read_year_list(config: dict[str, Any], key: str) -> tuple[int, ...]:
    """Read a non-empty split year list from config."""
    values = config.get(key)
    if not isinstance(values, list) or not values:
        msg = f"config field must be a non-empty list of years: splits.{key}"
        raise ValueError(msg)
    if any(isinstance(value, bool) for value in values):
        msg = f"split years must be integers, not booleans: splits.{key}"
        raise ValueError(msg)
    return tuple(operator_index(value) for value in values)


def read_alpha_grid(value: object) -> tuple[float, ...]:
    """Read conditional ridge alpha candidates from config."""
    if value is None:
        return DEFAULT_ALPHA_GRID
    if not isinstance(value, list) or not value:
        msg = "models.conditional_canopy.alpha_grid must be a non-empty list"
        raise ValueError(msg)
    parsed = tuple(float(item) for item in value)
    if any(item <= 0 for item in parsed):
        msg = "models.conditional_canopy.alpha_grid values must be positive"
        raise ValueError(msg)
    if len(set(parsed)) != len(parsed):
        msg = "models.conditional_canopy.alpha_grid contains duplicate values"
        raise ValueError(msg)
    return parsed


def operator_index(value: object) -> int:
    """Return an index-style integer for mypy-friendly dynamic config parsing."""
    return cast(SupportsIndex, value).__index__()


def prepare_conditional_model_frame(
    dataframe: pd.DataFrame,
    split_manifest: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> PreparedConditionalData:
    """Attach splits, derive positive support, and drop unusable rows."""
    validate_input_columns(dataframe, conditional_config)
    frame = attach_split_membership(dataframe, split_manifest, conditional_config)
    frame["observed_positive_support"] = (
        frame[conditional_config.target_column].astype(float)
        >= conditional_config.positive_target_threshold_fraction
    )
    feature_complete = frame.loc[:, list(conditional_config.feature_columns)].notna().all(axis=1)
    target_complete = frame[conditional_config.target_column].notna()
    manifest_used = frame.get("used_for_training_eval")
    manifest_mask = (
        manifest_used.fillna(False).astype(bool)
        if isinstance(manifest_used, pd.Series)
        else pd.Series(True, index=frame.index)
    )
    frame["has_complete_features"] = feature_complete.to_numpy(dtype=bool)
    frame["has_conditional_target"] = target_complete.to_numpy(dtype=bool)
    frame["used_for_conditional_model"] = (
        manifest_mask & frame["has_complete_features"] & frame["has_conditional_target"]
    )
    retained = frame.loc[frame["used_for_conditional_model"]].copy()
    if not conditional_config.drop_missing_features and len(retained) != len(frame):
        msg = "configured to keep missing features, but ridge model cannot fit missing rows"
        raise ValueError(msg)
    ensure_required_splits_present(retained)
    train_positive_count = support_count(retained, "train")
    validation_positive_count = support_count(retained, "validation")
    test_positive_count = support_count(retained, "test")
    if train_positive_count == 0:
        msg = "conditional model requires observed-positive training rows"
        raise ValueError(msg)
    if validation_positive_count == 0:
        msg = "conditional model requires observed-positive validation rows for alpha selection"
        raise ValueError(msg)
    LOGGER.info(
        "Retained %s conditional rows; observed-positive counts train=%s validation=%s test=%s",
        len(retained),
        train_positive_count,
        validation_positive_count,
        test_positive_count,
    )
    return PreparedConditionalData(
        retained_rows=retained,
        dropped_counts_by_split=dropped_counts_by_split(frame),
        train_positive_count=train_positive_count,
        validation_positive_count=validation_positive_count,
        test_positive_count=test_positive_count,
    )


def validate_input_columns(
    dataframe: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> None:
    """Validate sample input columns needed for conditional modeling."""
    required = [
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        conditional_config.target_column,
        conditional_config.target_area_column,
        *conditional_config.feature_columns,
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        msg = f"conditional model input is missing required columns: {missing}"
        raise ValueError(msg)


def attach_split_membership(
    dataframe: pd.DataFrame,
    split_manifest: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> pd.DataFrame:
    """Attach split labels from the configured split manifest."""
    key_columns = split_join_columns(dataframe, split_manifest)
    if not key_columns:
        frame = dataframe.copy()
        frame["split"] = assign_splits_by_year(frame["year"], conditional_config)
        return frame
    manifest_columns = [
        *key_columns,
        *[
            column
            for column in ("split", "used_for_training_eval", "drop_reason")
            if column in split_manifest.columns
        ],
    ]
    if "split" not in manifest_columns:
        msg = "split manifest is missing required column: split"
        raise ValueError(msg)
    manifest = split_manifest[manifest_columns].drop_duplicates()
    if manifest.duplicated(key_columns).any():
        msg = f"split manifest has duplicate conditional-model keys: {key_columns}"
        raise ValueError(msg)
    frame = dataframe.merge(manifest, on=key_columns, how="left", validate="many_to_one")
    if frame["split"].isna().any():
        missing_count = int(frame["split"].isna().sum())
        msg = f"split manifest is missing {missing_count} conditional model rows"
        raise ValueError(msg)
    return frame


def split_join_columns(dataframe: pd.DataFrame, split_manifest: pd.DataFrame) -> list[str]:
    """Return the strongest available key for joining the split manifest."""
    candidates = (
        ("year", "aef_grid_cell_id"),
        ("year", "aef_grid_row", "aef_grid_col"),
        ("year", "kelpwatch_station_id", "longitude", "latitude"),
    )
    dataframe_columns = set(dataframe.columns)
    manifest_columns = set(split_manifest.columns)
    for candidate in candidates:
        if set(candidate).issubset(dataframe_columns) and set(candidate).issubset(manifest_columns):
            return list(candidate)
    return []


def assign_splits_by_year(
    years: pd.Series,
    conditional_config: ConditionalCanopyConfig,
) -> pd.Series:
    """Assign split labels from configured year lists."""
    train_years = set(conditional_config.train_years)
    validation_years = set(conditional_config.validation_years)
    test_years = set(conditional_config.test_years)
    overlap = (
        (train_years & validation_years)
        | (train_years & test_years)
        | (validation_years & test_years)
    )
    if overlap:
        msg = f"split years overlap: {sorted(overlap)}"
        raise ValueError(msg)
    split = pd.Series(index=years.index, data="unassigned", dtype="object")
    split.loc[years.isin(train_years)] = "train"
    split.loc[years.isin(validation_years)] = "validation"
    split.loc[years.isin(test_years)] = "test"
    if (split == "unassigned").any():
        missing = sorted(set(int(year) for year in years.loc[split == "unassigned"].unique()))
        msg = f"conditional model rows contain years not assigned to a split: {missing}"
        raise ValueError(msg)
    return split


def dropped_counts_by_split(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count dropped conditional-model rows by split."""
    counts = {split: 0 for split in SPLIT_ORDER}
    dropped = dataframe.loc[~dataframe["used_for_conditional_model"]]
    for split, group in dropped.groupby("split", sort=False, dropna=False):
        counts[str(split)] = int(len(group))
    return counts


def ensure_required_splits_present(dataframe: pd.DataFrame) -> None:
    """Validate that retained rows cover train, validation, and test splits."""
    missing = [split for split in SPLIT_ORDER if split not in set(dataframe["split"])]
    if missing:
        msg = f"retained conditional model rows are missing splits: {missing}"
        raise ValueError(msg)


def support_count(dataframe: pd.DataFrame, split: str) -> int:
    """Count observed-positive support rows for one split."""
    rows = dataframe.loc[dataframe["split"] == split]
    return int(rows["observed_positive_support"].fillna(False).astype(bool).sum())


def observed_positive_rows(
    dataframe: pd.DataFrame,
    split: str,
    conditional_config: ConditionalCanopyConfig,
) -> pd.DataFrame:
    """Return observed-positive rows for one split."""
    rows = dataframe.loc[
        (dataframe["split"] == split)
        & (
            dataframe[conditional_config.target_column].astype(float)
            >= conditional_config.positive_target_threshold_fraction
        )
    ].copy()
    if rows.empty:
        msg = f"no observed-positive conditional rows for split: {split}"
        raise ValueError(msg)
    return rows


def fit_select_conditional_ridge(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> ConditionalModelSelection:
    """Fit positive-only ridge candidates and select alpha on validation RMSE."""
    x_train = feature_matrix(train_rows, conditional_config.feature_columns)
    y_train = target_vector(train_rows, conditional_config.target_column)
    x_validation = feature_matrix(validation_rows, conditional_config.feature_columns)
    y_validation = target_vector(validation_rows, conditional_config.target_column)
    validation_rows_out: list[dict[str, object]] = []
    candidates: list[tuple[float, float, Any]] = []
    for alpha in conditional_config.alpha_grid:
        model = make_ridge_pipeline(alpha)
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        clipped = np.clip(predictions, 0.0, 1.0)
        rmse = root_mean_squared_error(y_validation, clipped)
        mae = mean_absolute_error(y_validation, clipped)
        validation_rows_out.append({"alpha": alpha, "validation_rmse": rmse, "validation_mae": mae})
        candidates.append((rmse, alpha, model))
        LOGGER.info("Conditional ridge alpha=%s validation positive RMSE=%s", alpha, rmse)
    selected_rmse, selected_alpha, selected_model = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    LOGGER.info(
        "Selected conditional ridge alpha=%s with validation positive RMSE=%s",
        selected_alpha,
        selected_rmse,
    )
    return ConditionalModelSelection(
        model=selected_model,
        selected_alpha=selected_alpha,
        validation_rows=validation_rows_out,
    )


def make_ridge_pipeline(alpha: float) -> Any:
    """Build the standard scaler plus ridge pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def feature_matrix(dataframe: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Return configured feature columns as a floating-point matrix."""
    return cast(np.ndarray, dataframe.loc[:, list(feature_columns)].to_numpy(dtype=float))


def target_vector(dataframe: pd.DataFrame, target_column: str) -> np.ndarray:
    """Return the configured target column as a floating-point vector."""
    return cast(np.ndarray, dataframe[target_column].to_numpy(dtype=float))


def read_likely_positive_flags(
    retained_rows: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> pd.DataFrame:
    """Read calibrated binary sample flags used for likely-positive diagnostics."""
    path = conditional_config.calibrated_sample_predictions_path
    if path is None or not path.exists():
        LOGGER.info("Skipping likely-positive sample diagnostics; calibrated predictions missing")
        return pd.DataFrame()
    predictions = pd.read_parquet(path)
    key_columns = row_key_columns(retained_rows, predictions)
    if not key_columns:
        LOGGER.info("Skipping likely-positive sample diagnostics; no shared row keys")
        return pd.DataFrame()
    required = [
        "calibrated_binary_probability",
        "calibrated_probability_threshold",
        "calibrated_threshold_policy",
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        LOGGER.info("Skipping likely-positive sample diagnostics; missing columns %s", missing)
        return pd.DataFrame()
    flags = predictions[
        [
            *key_columns,
            "calibrated_binary_probability",
            "calibrated_probability_threshold",
            "calibrated_threshold_policy",
            *[
                column
                for column in ("calibrated_pred_binary_class",)
                if column in predictions.columns
            ],
        ]
    ].copy()
    if "calibrated_pred_binary_class" not in flags.columns:
        flags["calibrated_pred_binary_class"] = flags["calibrated_binary_probability"].astype(
            float
        ) >= flags["calibrated_probability_threshold"].astype(float)
    flags["likely_positive_support"] = flags["calibrated_pred_binary_class"].fillna(False).astype(
        bool
    ) & (
        flags["calibrated_threshold_policy"].astype(str)
        == conditional_config.likely_positive_threshold_policy
    )
    flags = flags.drop_duplicates(key_columns)
    if flags.duplicated(key_columns).any():
        msg = f"calibrated sample predictions have duplicate keys: {key_columns}"
        raise ValueError(msg)
    return flags


def row_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Return stable row key columns shared by two prediction-like frames."""
    candidates = (
        ("year", "aef_grid_cell_id"),
        ("year", "aef_grid_row", "aef_grid_col"),
        ("year", "kelpwatch_station_id", "longitude", "latitude"),
    )
    left_columns = set(left.columns)
    right_columns = set(right.columns)
    for candidate in candidates:
        if set(candidate).issubset(left_columns) and set(candidate).issubset(right_columns):
            return list(candidate)
    return []


def conditional_sample_prediction_frame(
    retained_rows: pd.DataFrame,
    likely_flags: pd.DataFrame,
    selection: ConditionalModelSelection,
    conditional_config: ConditionalCanopyConfig,
) -> pd.DataFrame:
    """Build row-level conditional sample predictions for supported rows."""
    frame = attach_likely_positive_flags(retained_rows, likely_flags)
    support_mask = frame["observed_positive_support"].astype(bool) | frame[
        "likely_positive_support"
    ].astype(bool)
    supported = frame.loc[support_mask].copy()
    predictions = np.asarray(
        selection.model.predict(feature_matrix(supported, conditional_config.feature_columns)),
        dtype=float,
    )
    clipped = np.clip(predictions, 0.0, 1.0)
    output = supported[prediction_identity_columns(supported, conditional_config)].copy()
    output["model_name"] = CONDITIONAL_MODEL_NAME
    output["model_family"] = CONDITIONAL_MODEL_FAMILY
    output["target"] = conditional_config.target_column
    output["target_area_column"] = conditional_config.target_area_column
    output["positive_target_label"] = conditional_config.positive_target_label
    output["positive_target_threshold_fraction"] = (
        conditional_config.positive_target_threshold_fraction
    )
    output["positive_target_threshold_area"] = conditional_config.positive_target_threshold_area
    output["positive_support_policy"] = conditional_config.positive_support_policy
    output["likely_positive_policy"] = conditional_config.likely_positive_policy
    output["likely_positive_threshold_policy"] = conditional_config.likely_positive_threshold_policy
    output["selection_split"] = SELECTION_SPLIT
    output["selection_year"] = primary_selection_year(conditional_config)
    output["test_split"] = TEST_SPLIT
    output["test_year"] = primary_test_year(conditional_config)
    output["selected_alpha"] = selection.selected_alpha
    output["observed_positive_support"] = supported["observed_positive_support"].to_numpy(
        dtype=bool
    )
    output["likely_positive_support"] = supported["likely_positive_support"].to_numpy(dtype=bool)
    output["calibrated_binary_probability"] = supported["calibrated_binary_probability"].to_numpy(
        dtype=float
    )
    output["calibrated_probability_threshold"] = supported[
        "calibrated_probability_threshold"
    ].to_numpy(dtype=float)
    output["evaluation_scope"] = np.where(
        output["observed_positive_support"].to_numpy(dtype=bool),
        "conditional_observed_positive_sample",
        "conditional_likely_positive_diagnostic",
    )
    output["pred_conditional_fraction"] = predictions
    output["pred_conditional_fraction_clipped"] = clipped
    output["pred_conditional_area_m2"] = clipped * KELPWATCH_PIXEL_AREA_M2
    target_values = output[conditional_config.target_column].to_numpy(dtype=float)
    target_area = output[conditional_config.target_area_column].to_numpy(dtype=float)
    output["residual_conditional_fraction"] = target_values - clipped
    output["residual_conditional_area_m2"] = target_area - output[
        "pred_conditional_area_m2"
    ].to_numpy(dtype=float)
    return output.loc[:, conditional_prediction_output_columns(output)]


def attach_likely_positive_flags(
    retained_rows: pd.DataFrame,
    likely_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Attach calibrated binary likely-positive flags to retained sample rows."""
    frame = retained_rows.copy()
    default_values = {
        "likely_positive_support": False,
        "calibrated_binary_probability": math.nan,
        "calibrated_probability_threshold": math.nan,
        "calibrated_threshold_policy": "",
    }
    if likely_flags.empty:
        for column, value in default_values.items():
            frame[column] = value
        return frame
    key_columns = row_key_columns(frame, likely_flags)
    if not key_columns:
        for column, value in default_values.items():
            frame[column] = value
        return frame
    flag_columns = [
        *key_columns,
        "likely_positive_support",
        "calibrated_binary_probability",
        "calibrated_probability_threshold",
        "calibrated_threshold_policy",
    ]
    merged = frame.merge(
        likely_flags[flag_columns],
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    for column, value in default_values.items():
        if column in {"likely_positive_support"}:
            merged[column] = merged[column].fillna(value).astype(bool)
        elif column in merged.columns:
            merged[column] = merged[column].fillna(value)
    return merged


def prediction_identity_columns(
    dataframe: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> list[str]:
    """Return identity, provenance, and target columns for prediction outputs."""
    columns = [
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        conditional_config.target_column,
        conditional_config.target_area_column,
    ]
    for column in OPTIONAL_ID_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    return columns


def conditional_prediction_output_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return row-level conditional prediction columns in a stable order."""
    return [column for column in CONDITIONAL_PREDICTION_FIELDS if column in dataframe.columns]


def primary_selection_year(conditional_config: ConditionalCanopyConfig) -> int | str:
    """Return the validation year label used for alpha selection."""
    if len(conditional_config.validation_years) == 1:
        return conditional_config.validation_years[0]
    return "all_validation_years"


def primary_test_year(conditional_config: ConditionalCanopyConfig) -> int | str:
    """Return the test year label used for held-out audit reporting."""
    if len(conditional_config.test_years) == 1:
        return conditional_config.test_years[0]
    return "all_test_years"


def primary_observed_positive_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return row-level predictions for the primary observed-positive support."""
    return predictions.loc[
        predictions["evaluation_scope"] == "conditional_observed_positive_sample"
    ].copy()


def build_conditional_metric_rows(
    predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> list[dict[str, object]]:
    """Build conditional model metrics grouped by split, year, and label source."""
    rows: list[dict[str, object]] = []
    rows.extend(grouped_conditional_metric_rows(predictions, conditional_config, ["split", "year"]))
    rows.extend(
        grouped_conditional_metric_rows(
            predictions,
            conditional_config,
            ["split", "year", "label_source"],
        )
    )
    return rows


def grouped_conditional_metric_rows(
    predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate conditional metrics for one grouping layout."""
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return rows
    frame = ensure_label_source(predictions)
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(conditional_metric_row(group, group_values, conditional_config))
    return rows


def conditional_metric_row(
    group: pd.DataFrame,
    group_values: dict[str, object],
    conditional_config: ConditionalCanopyConfig,
) -> dict[str, object]:
    """Build one grouped conditional model metric row."""
    observed = group[conditional_config.target_column].to_numpy(dtype=float)
    predicted = group["pred_conditional_fraction_clipped"].to_numpy(dtype=float)
    observed_area = group[conditional_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group["pred_conditional_area_m2"].to_numpy(dtype=float)
    residual_area = group["residual_conditional_area_m2"].to_numpy(dtype=float)
    observed_total = float(np.nansum(observed_area))
    predicted_total = float(np.nansum(predicted_area))
    return {
        "model_name": CONDITIONAL_MODEL_NAME,
        "model_family": CONDITIONAL_MODEL_FAMILY,
        "target": conditional_config.target_column,
        "target_area_column": conditional_config.target_area_column,
        "positive_target_label": conditional_config.positive_target_label,
        "positive_target_threshold_fraction": (
            conditional_config.positive_target_threshold_fraction
        ),
        "positive_target_threshold_area": conditional_config.positive_target_threshold_area,
        "positive_support_policy": conditional_config.positive_support_policy,
        "selection_split": SELECTION_SPLIT,
        "selection_year": primary_selection_year(conditional_config),
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, conditional_config),
        "evaluation_scope": "conditional_observed_positive_sample",
        "row_count": int(len(group)),
        "selected_alpha": first_selected_alpha(group),
        "observed_mean_fraction": safe_mean(observed),
        "predicted_mean_fraction": safe_mean(predicted),
        "observed_canopy_area": observed_total,
        "predicted_canopy_area": predicted_total,
        "area_bias": predicted_total - observed_total,
        "area_pct_bias": percent_bias(predicted_total, observed_total),
        "mae_fraction": mean_absolute_error(observed, predicted),
        "rmse_fraction": root_mean_squared_error(observed, predicted),
        "mae_area": mean_absolute_error(observed_area, predicted_area),
        "rmse_area": root_mean_squared_error(observed_area, predicted_area),
        "r2_fraction": r2_score(observed, predicted),
        "spearman_fraction": correlation(observed, predicted, method="spearman"),
        "mean_residual_area": safe_mean(residual_area),
        "median_residual_area": safe_percentile(residual_area, 50),
        "underprediction_count": int(np.count_nonzero(residual_area > 0)),
        "overprediction_count": int(np.count_nonzero(residual_area < 0)),
        "high_canopy_count": int(np.count_nonzero(observed >= 0.50)),
        "near_saturated_count": int(np.count_nonzero(observed_area >= 810.0)),
    }


def ensure_label_source(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a stable label_source column."""
    frame = dataframe.copy()
    if "label_source" not in frame.columns:
        if "is_kelpwatch_observed" in frame.columns:
            observed = frame["is_kelpwatch_observed"].fillna(False).astype(bool)
            frame["label_source"] = np.where(observed, "kelpwatch_station", "assumed_background")
        else:
            frame["label_source"] = "all"
    return frame


def input_mask_status(
    group: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> str:
    """Infer sample mask status for conditional metric rows."""
    if conditional_config.reporting_domain_mask is None or MASK_RETAIN_COLUMN not in group.columns:
        return "unmasked"
    retained = group[MASK_RETAIN_COLUMN].dropna().astype(bool)
    if retained.empty or not bool(retained.all()):
        return "unmasked"
    return mask_status(conditional_config.reporting_domain_mask)


def first_selected_alpha(group: pd.DataFrame) -> float:
    """Return the selected alpha recorded on prediction rows."""
    values = pd.to_numeric(group["selected_alpha"], errors="coerce").dropna().unique()
    return float(values[0]) if len(values) else math.nan


def safe_mean(values: np.ndarray) -> float:
    """Return a NaN-aware mean or NaN for empty arrays."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(finite))


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a NaN-aware percentile or NaN for empty arrays."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanpercentile(finite, percentile))


def normalized_group_value(value: object) -> str:
    """Normalize a pandas group key to a stable string label."""
    if pd.isna(value):
        return "missing"
    return str(value)


def build_positive_residual_rows(
    predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> list[dict[str, object]]:
    """Build positive-cell residual rows for configured high-canopy bins."""
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return rows
    frame = ensure_label_source(predictions)
    rows.extend(
        grouped_positive_residual_rows(
            frame,
            conditional_config,
            ["split", "year", "evaluation_scope"],
        )
    )
    rows.extend(
        grouped_positive_residual_rows(
            frame,
            conditional_config,
            ["split", "year", "label_source", "evaluation_scope"],
        )
    )
    return rows


def grouped_positive_residual_rows(
    frame: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate positive-cell residual rows for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {
            "split": "all",
            "year": "all",
            "label_source": "all",
            "evaluation_scope": "all",
        }
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        for observed_bin, subset in observed_bin_subsets(group, conditional_config):
            rows.append(
                positive_residual_row(
                    subset,
                    group_values,
                    observed_bin,
                    conditional_config,
                )
            )
    return rows


def observed_bin_subsets(
    group: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> list[tuple[str, pd.DataFrame]]:
    """Return overlapping observed-support bins required for conditional diagnostics."""
    fraction = group[conditional_config.target_column].astype(float)
    area = group[conditional_config.target_area_column].astype(float)
    return [
        ("observed_positive_all", group),
        (
            "annual_max_ge_10pct",
            group.loc[fraction >= conditional_config.positive_target_threshold_fraction],
        ),
        ("annual_max_ge_50pct", group.loc[fraction >= 0.50]),
        ("annual_max_ge_90pct", group.loc[fraction >= 0.90]),
        ("near_saturated_ge_810m2", group.loc[area >= 810.0]),
    ]


def positive_residual_row(
    group: pd.DataFrame,
    group_values: dict[str, object],
    observed_bin: str,
    conditional_config: ConditionalCanopyConfig,
) -> dict[str, object]:
    """Build one grouped positive-cell residual row."""
    observed = group[conditional_config.target_area_column].to_numpy(dtype=float)
    predicted = group["pred_conditional_area_m2"].to_numpy(dtype=float)
    residual = group["residual_conditional_area_m2"].to_numpy(dtype=float)
    return {
        "model_name": CONDITIONAL_MODEL_NAME,
        "model_family": CONDITIONAL_MODEL_FAMILY,
        "target": conditional_config.target_column,
        "target_area_column": conditional_config.target_area_column,
        "positive_target_label": conditional_config.positive_target_label,
        "positive_support_policy": conditional_config.positive_support_policy,
        "likely_positive_policy": conditional_config.likely_positive_policy,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, conditional_config),
        "evaluation_scope": group_values["evaluation_scope"],
        "observed_bin": observed_bin,
        "row_count": int(len(group)),
        "observed_mean_area": safe_mean(observed),
        "predicted_mean_area": safe_mean(predicted),
        "mean_residual_area": safe_mean(residual),
        "median_residual_area": safe_percentile(residual, 50),
        "mae_area": mean_absolute_error(observed, predicted),
        "rmse_area": root_mean_squared_error(observed, predicted),
        "underprediction_count": int(np.count_nonzero(residual > 0)),
        "overprediction_count": int(np.count_nonzero(residual < 0)),
    }


def build_conditional_comparison_rows(
    primary_predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    selected_alpha: float,
) -> list[dict[str, object]]:
    """Compare conditional predictions against ridge on identical positive rows."""
    rows: list[dict[str, object]] = []
    rows.extend(
        comparison_rows_for_predictions(
            primary_predictions,
            conditional_config,
            model_name=CONDITIONAL_MODEL_NAME,
            model_family=CONDITIONAL_MODEL_FAMILY,
            prediction_fraction_column="pred_conditional_fraction_clipped",
            prediction_area_column="pred_conditional_area_m2",
            selected_alpha=selected_alpha,
        )
    )
    baseline = read_baseline_positive_support(primary_predictions, conditional_config)
    if baseline.empty:
        return rows
    rows.extend(
        comparison_rows_for_predictions(
            baseline,
            conditional_config,
            model_name=conditional_config.ridge_baseline_model_name,
            model_family="continuous_baseline",
            prediction_fraction_column="pred_kelp_fraction_y_clipped",
            prediction_area_column="pred_kelp_max_y",
            selected_alpha=math.nan,
        )
    )
    return rows


def read_baseline_positive_support(
    primary_predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
) -> pd.DataFrame:
    """Read ridge baseline rows for the exact conditional positive support."""
    path = conditional_config.baseline_sample_predictions_path
    if path is None or not path.exists():
        LOGGER.info("Skipping ridge comparison; baseline sample predictions missing")
        return pd.DataFrame()
    baseline = pd.read_parquet(path)
    baseline = baseline.loc[
        baseline["model_name"].astype(str) == conditional_config.ridge_baseline_model_name
    ].copy()
    required = {"pred_kelp_fraction_y_clipped", "pred_kelp_max_y"}
    if baseline.empty or not required.issubset(baseline.columns):
        LOGGER.info("Skipping ridge comparison; ridge prediction columns missing")
        return pd.DataFrame()
    key_columns = row_key_columns(primary_predictions, baseline)
    if not key_columns:
        LOGGER.info("Skipping ridge comparison; no shared row keys")
        return pd.DataFrame()
    support_keys = primary_predictions[key_columns].drop_duplicates()
    return baseline.merge(support_keys, on=key_columns, how="inner", validate="one_to_one")


def comparison_rows_for_predictions(
    predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    *,
    model_name: str,
    model_family: str,
    prediction_fraction_column: str,
    prediction_area_column: str,
    selected_alpha: float,
) -> list[dict[str, object]]:
    """Build comparison rows for a prediction frame."""
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return rows
    frame = ensure_label_source(predictions)
    rows.extend(
        grouped_comparison_rows(
            frame,
            conditional_config,
            model_name=model_name,
            model_family=model_family,
            prediction_fraction_column=prediction_fraction_column,
            prediction_area_column=prediction_area_column,
            selected_alpha=selected_alpha,
            group_columns=["split", "year"],
        )
    )
    rows.extend(
        grouped_comparison_rows(
            frame,
            conditional_config,
            model_name=model_name,
            model_family=model_family,
            prediction_fraction_column=prediction_fraction_column,
            prediction_area_column=prediction_area_column,
            selected_alpha=selected_alpha,
            group_columns=["split", "year", "label_source"],
        )
    )
    return rows


def grouped_comparison_rows(
    predictions: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    *,
    model_name: str,
    model_family: str,
    prediction_fraction_column: str,
    prediction_area_column: str,
    selected_alpha: float,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate comparison metrics for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(
            comparison_row(
                group,
                conditional_config,
                model_name=model_name,
                model_family=model_family,
                prediction_fraction_column=prediction_fraction_column,
                prediction_area_column=prediction_area_column,
                selected_alpha=selected_alpha,
                group_values=group_values,
            )
        )
    return rows


def comparison_row(
    group: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    *,
    model_name: str,
    model_family: str,
    prediction_fraction_column: str,
    prediction_area_column: str,
    selected_alpha: float,
    group_values: dict[str, object],
) -> dict[str, object]:
    """Build one apples-to-apples positive-support comparison row."""
    observed = group[conditional_config.target_column].to_numpy(dtype=float)
    predicted = group[prediction_fraction_column].to_numpy(dtype=float)
    observed_area = group[conditional_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group[prediction_area_column].to_numpy(dtype=float)
    residual_area = observed_area - predicted_area
    observed_positive = observed >= conditional_config.positive_target_threshold_fraction
    predicted_positive = predicted >= conditional_config.positive_target_threshold_fraction
    _precision, _recall, f1 = precision_recall_f1(observed_positive, predicted_positive)
    observed_total = float(np.nansum(observed_area))
    predicted_total = float(np.nansum(predicted_area))
    return {
        "model_name": model_name,
        "model_family": model_family,
        "comparison_support": "observed_positive_same_rows",
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, conditional_config),
        "evaluation_scope": "conditional_observed_positive_sample",
        "row_count": int(len(group)),
        "selected_alpha": selected_alpha,
        "mae_fraction": mean_absolute_error(observed, predicted),
        "rmse_fraction": root_mean_squared_error(observed, predicted),
        "mae_area": mean_absolute_error(observed_area, predicted_area),
        "rmse_area": root_mean_squared_error(observed_area, predicted_area),
        "r2_fraction": r2_score(observed, predicted),
        "spearman_fraction": correlation(observed, predicted, method="spearman"),
        "observed_canopy_area": observed_total,
        "predicted_canopy_area": predicted_total,
        "area_bias": predicted_total - observed_total,
        "area_pct_bias": percent_bias(predicted_total, observed_total),
        "mean_residual_area": safe_mean(residual_area),
        "underprediction_count": int(np.count_nonzero(residual_area > 0)),
        "overprediction_count": int(np.count_nonzero(residual_area < 0)),
        "f1_ge_10pct": f1,
    }


def build_full_grid_likely_positive_summary(
    conditional_config: ConditionalCanopyConfig,
) -> list[dict[str, object]]:
    """Build compact full-grid likely-positive diagnostic rows from binary calibration."""
    path = conditional_config.calibrated_full_grid_area_summary_path
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    if "threshold_policy" in frame.columns:
        frame = frame.loc[
            frame["threshold_policy"].astype(str)
            == conditional_config.likely_positive_threshold_policy
        ].copy()
    if "probability_source" in frame.columns:
        calibrated = frame.loc[frame["probability_source"].astype(str) == "platt_calibrated"]
        if not calibrated.empty:
            frame = calibrated.copy()
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        row_dict = cast(dict[str, object], row)
        likely_count = object_to_int(row_dict.get("predicted_positive_count"))
        row_count = object_to_int(row_dict.get("row_count"))
        assumed_background_count = object_to_int(row_dict.get("assumed_background_count"))
        assumed_background_likely = object_to_int(
            row_dict.get("assumed_background_predicted_positive_count")
        )
        rows.append(
            {
                "conditional_model_name": CONDITIONAL_MODEL_NAME,
                "model_family": CONDITIONAL_MODEL_FAMILY,
                "likely_positive_policy": conditional_config.likely_positive_policy,
                "likely_positive_threshold_policy": (
                    conditional_config.likely_positive_threshold_policy
                ),
                "probability_source": row_dict.get("probability_source", ""),
                "split": row_dict.get("split", ""),
                "year": row_dict.get("year", ""),
                "label_source": row_dict.get("label_source", ""),
                "mask_status": row_dict.get("mask_status", mask_status(None)),
                "evaluation_scope": "conditional_likely_positive_diagnostic",
                "probability_threshold": row_dict.get("probability_threshold", math.nan),
                "row_count": row_count,
                "likely_positive_count": likely_count,
                "likely_positive_rate": safe_ratio(likely_count, row_count),
                "likely_positive_cell_area_m2": row_dict.get(
                    "predicted_positive_area_m2",
                    math.nan,
                ),
                "observed_positive_count": row_dict.get("observed_positive_count", 0),
                "observed_positive_rate": row_dict.get("observed_positive_rate", math.nan),
                "assumed_background_count": assumed_background_count,
                "assumed_background_likely_positive_count": assumed_background_likely,
                "assumed_background_likely_positive_rate": safe_ratio(
                    assumed_background_likely,
                    assumed_background_count,
                ),
                "diagnostic_note": (
                    "compact likely-positive count only; this task does not compose "
                    "full-grid hurdle canopy predictions"
                ),
            }
        )
    return rows


def write_conditional_sidecar_reuse_artifacts(
    *,
    base_config: ConditionalCanopyConfig,
    sidecar: ConditionalCanopySidecarConfig,
    base_prepared: PreparedConditionalData,
) -> None:
    """Write sidecar diagnostics proving the conditional model support is shared."""
    sidecar_sample = pd.read_parquet(sidecar.conditional_config.input_table_path)
    support = compare_observed_positive_support(
        base_prepared.retained_rows,
        sidecar_sample,
        base_config,
        sidecar.conditional_config,
    )
    full_grid_summary = build_full_grid_likely_positive_summary(sidecar.conditional_config)
    if sidecar.conditional_config.full_grid_likely_positive_summary_path is not None:
        write_csv_rows(
            full_grid_summary,
            sidecar.conditional_config.full_grid_likely_positive_summary_path,
            CONDITIONAL_FULL_GRID_SUMMARY_FIELDS,
        )
    manifest = {
        "command": "train-conditional-canopy",
        "sidecar": sidecar.name,
        "sample_policy": sidecar.sample_policy,
        "conditional_model_reused": True,
        "reuse_reason": (
            "configured support policy trains only observed-positive rows, and the "
            "sidecar sample has the same observed-positive support keys"
        ),
        "positive_support_policy": base_config.positive_support_policy,
        "base_input_table": str(base_config.input_table_path),
        "sidecar_input_table": str(sidecar.conditional_config.input_table_path),
        "model": str(base_config.model_output_path),
        "support_key_columns": support["key_columns"],
        "base_observed_positive_support_count": support["base_count"],
        "sidecar_observed_positive_support_count": support["sidecar_count"],
        "support_sets_match": support["sets_match"],
        "missing_sidecar_support_examples": support["missing_examples"],
        "extra_sidecar_support_examples": support["extra_examples"],
        "full_grid_likely_positive_summary_rows": len(full_grid_summary),
        "outputs": {
            "reuse_manifest": str(sidecar.reuse_manifest_path),
            "full_grid_likely_positive_summary": (
                str(sidecar.conditional_config.full_grid_likely_positive_summary_path)
                if sidecar.conditional_config.full_grid_likely_positive_summary_path is not None
                else None
            ),
        },
    }
    if not support["sets_match"]:
        msg = (
            "conditional sidecar cannot reuse the current model because observed-positive "
            f"support differs for sidecar {sidecar.name}"
        )
        raise ValueError(msg)
    sidecar.reuse_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.reuse_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def compare_observed_positive_support(
    base_rows: pd.DataFrame,
    sidecar_rows: pd.DataFrame,
    base_config: ConditionalCanopyConfig,
    sidecar_config: ConditionalCanopyConfig,
) -> dict[str, object]:
    """Compare observed-positive support keys between current and sidecar samples."""
    key_columns = conditional_support_key_columns(base_rows, sidecar_rows)
    base_keys = observed_positive_support_keys(base_rows, base_config, key_columns)
    sidecar_keys = observed_positive_support_keys(sidecar_rows, sidecar_config, key_columns)
    missing = sorted(base_keys - sidecar_keys)
    extra = sorted(sidecar_keys - base_keys)
    return {
        "key_columns": list(key_columns),
        "base_count": len(base_keys),
        "sidecar_count": len(sidecar_keys),
        "sets_match": not missing and not extra,
        "missing_examples": [list(item) for item in missing[:10]],
        "extra_examples": [list(item) for item in extra[:10]],
    }


def conditional_support_key_columns(
    base_rows: pd.DataFrame,
    sidecar_rows: pd.DataFrame,
) -> tuple[str, ...]:
    """Return stable key columns available in both conditional samples."""
    candidates = (
        ("year", "aef_grid_cell_id"),
        ("year", "aef_grid_row", "aef_grid_col"),
        ("year", "kelpwatch_station_id", "longitude", "latitude"),
    )
    base_columns = set(base_rows.columns)
    sidecar_columns = set(sidecar_rows.columns)
    for candidate in candidates:
        if set(candidate).issubset(base_columns) and set(candidate).issubset(sidecar_columns):
            return candidate
    msg = "conditional sidecar support comparison requires a shared row key"
    raise ValueError(msg)


def observed_positive_support_keys(
    dataframe: pd.DataFrame,
    conditional_config: ConditionalCanopyConfig,
    key_columns: tuple[str, ...],
) -> set[tuple[object, ...]]:
    """Return observed-positive support keys after target and feature completeness checks."""
    validate_input_columns(dataframe, conditional_config)
    target = dataframe[conditional_config.target_column].astype(float)
    feature_complete = (
        dataframe.loc[:, list(conditional_config.feature_columns)].notna().all(axis=1)
    )
    support = dataframe.loc[
        (target >= conditional_config.positive_target_threshold_fraction) & feature_complete,
        list(key_columns),
    ].copy()
    support = support.sort_values(list(key_columns)).drop_duplicates()
    return {tuple(row[column] for column in key_columns) for row in support.to_dict("records")}


def object_to_int(value: object) -> int:
    """Convert a generic row value to int with a zero fallback."""
    if value is None or pd.isna(value):
        return 0
    return int(cast(Any, value))


def write_conditional_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write row-level conditional predictions to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)


def write_csv_rows(
    rows: list[dict[str, object]], output_path: Path, fields: tuple[str, ...]
) -> None:
    """Write rows to CSV with a stable schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_positive_residual_figure(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write a compact residual diagnostic figure for high-canopy bins."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No conditional residual rows", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = frame.loc[
        (frame["label_source"].astype(str) == "all")
        & (frame["observed_bin"].astype(str).isin(["annual_max_ge_10pct", "annual_max_ge_50pct"]))
    ].copy()
    if plot_frame.empty:
        plot_frame = frame.copy()
    plot_frame["x_label"] = (
        plot_frame["split"].astype(str)
        + " "
        + plot_frame["year"].astype(str)
        + "\n"
        + plot_frame["observed_bin"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(max(6, len(plot_frame) * 0.5), 3.5))
    ax.bar(plot_frame["x_label"], plot_frame["mean_residual_area"].astype(float), color="#3a6ea5")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean observed - predicted area (m2)")
    ax.set_title("Conditional canopy residuals on observed-positive rows")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_conditional_model(
    selection: ConditionalModelSelection,
    conditional_config: ConditionalCanopyConfig,
    prepared: PreparedConditionalData,
) -> None:
    """Write the selected conditional ridge model and metadata with joblib."""
    conditional_config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": selection.model,
        "model_name": CONDITIONAL_MODEL_NAME,
        "model_family": CONDITIONAL_MODEL_FAMILY,
        "selected_alpha": selection.selected_alpha,
        "target_column": conditional_config.target_column,
        "target_area_column": conditional_config.target_area_column,
        "feature_columns": list(conditional_config.feature_columns),
        "positive_target_label": conditional_config.positive_target_label,
        "positive_target_threshold_fraction": conditional_config.positive_target_threshold_fraction,
        "positive_target_threshold_area": conditional_config.positive_target_threshold_area,
        "positive_support_policy": conditional_config.positive_support_policy,
        "selection_split": SELECTION_SPLIT,
        "selection_year": primary_selection_year(conditional_config),
        "validation_rows": selection.validation_rows,
        "train_positive_count": prepared.train_positive_count,
        "validation_positive_count": prepared.validation_positive_count,
        "test_positive_count": prepared.test_positive_count,
    }
    joblib.dump(payload, conditional_config.model_output_path)


def write_conditional_manifest(
    prepared: PreparedConditionalData,
    selection: ConditionalModelSelection,
    sample_predictions: pd.DataFrame,
    metrics: list[dict[str, object]],
    residuals: list[dict[str, object]],
    comparison: list[dict[str, object]],
    full_grid_summary: list[dict[str, object]],
    conditional_config: ConditionalCanopyConfig,
) -> None:
    """Write the conditional canopy JSON manifest."""
    conditional_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "train-conditional-canopy",
        "config_path": str(conditional_config.config_path),
        "model_name": CONDITIONAL_MODEL_NAME,
        "model_family": CONDITIONAL_MODEL_FAMILY,
        "target": conditional_config.target_column,
        "target_area_column": conditional_config.target_area_column,
        "positive_target_label": conditional_config.positive_target_label,
        "positive_target_threshold_fraction": conditional_config.positive_target_threshold_fraction,
        "positive_target_threshold_area": conditional_config.positive_target_threshold_area,
        "positive_support_policy": conditional_config.positive_support_policy,
        "likely_positive_policy": conditional_config.likely_positive_policy,
        "likely_positive_threshold_policy": conditional_config.likely_positive_threshold_policy,
        "selection_split": SELECTION_SPLIT,
        "selection_year": primary_selection_year(conditional_config),
        "test_split": TEST_SPLIT,
        "test_year": primary_test_year(conditional_config),
        "selected_alpha": selection.selected_alpha,
        "validation_grid": selection.validation_rows,
        "inputs": {
            "sample_table": str(conditional_config.input_table_path),
            "split_manifest": str(conditional_config.split_manifest_path),
            "baseline_sample_predictions": str(conditional_config.baseline_sample_predictions_path)
            if conditional_config.baseline_sample_predictions_path is not None
            else None,
            "calibrated_sample_predictions": str(
                conditional_config.calibrated_sample_predictions_path
            )
            if conditional_config.calibrated_sample_predictions_path is not None
            else None,
            "calibrated_full_grid_area_summary": str(
                conditional_config.calibrated_full_grid_area_summary_path
            )
            if conditional_config.calibrated_full_grid_area_summary_path is not None
            else None,
        },
        "outputs": {
            "model": str(conditional_config.model_output_path),
            "sample_predictions": str(conditional_config.sample_predictions_path),
            "metrics": str(conditional_config.metrics_path),
            "positive_residuals": str(conditional_config.positive_residuals_path),
            "model_comparison": str(conditional_config.model_comparison_path),
            "full_grid_likely_positive_summary": str(
                conditional_config.full_grid_likely_positive_summary_path
            )
            if conditional_config.full_grid_likely_positive_summary_path is not None
            else None,
            "residual_figure": str(conditional_config.residual_figure_path)
            if conditional_config.residual_figure_path is not None
            else None,
            "manifest": str(conditional_config.manifest_path),
        },
        "row_counts": {
            "retained_sample_rows": int(len(prepared.retained_rows)),
            "train_observed_positive_rows": prepared.train_positive_count,
            "validation_observed_positive_rows": prepared.validation_positive_count,
            "test_observed_positive_rows": prepared.test_positive_count,
            "sample_prediction_rows": int(len(sample_predictions)),
            "metric_rows": len(metrics),
            "positive_residual_rows": len(residuals),
            "comparison_rows": len(comparison),
            "full_grid_likely_positive_summary_rows": len(full_grid_summary),
        },
        "dropped_counts_by_split": prepared.dropped_counts_by_split,
        "test_rows_used_for_training_or_selection": False,
        "full_grid_hurdle_predictions_written": False,
        "qa_notes": [
            (
                "Conditional canopy skill is evaluated on Kelpwatch-style observed-positive "
                "annual-max rows only; likely-positive rows are diagnostics and no final "
                "hurdle prediction is composed in this task."
            )
        ],
    }
    conditional_config.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
