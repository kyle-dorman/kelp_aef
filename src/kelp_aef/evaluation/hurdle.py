"""Compose calibrated binary presence and conditional canopy amount predictions."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    ReportingDomainMask,
    apply_reporting_domain_mask,
    evaluation_scope,
    load_reporting_domain_mask,
    mask_status,
)
from kelp_aef.evaluation.baselines import (
    FULL_GRID_PREDICTION_BATCH_SIZE,
    KELPWATCH_PIXEL_AREA_M2,
    iter_parquet_batches,
    parse_bands,
    percent_bias,
    reset_output_path,
    safe_ratio,
    write_prediction_part,
)
from kelp_aef.evaluation.binary_presence import (
    PLATT_PROBABILITY_SOURCE,
    BinaryCalibrator,
    apply_binary_calibrator,
)
from kelp_aef.evaluation.conditional_canopy import CONDITIONAL_MODEL_NAME

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

HURDLE_MODEL_FAMILY = "hurdle"
EXPECTED_VALUE_POLICY = "expected_value"
HARD_GATE_POLICY = "hard_gate"
DEFAULT_MODEL_NAME = "calibrated_probability_x_conditional_canopy"
DEFAULT_HARD_GATE_MODEL_NAME = "calibrated_hard_gate_conditional_canopy"
DEFAULT_PRESENCE_MODEL_NAME = "logistic_annual_max_ge_10pct"
DEFAULT_PRESENCE_THRESHOLD_POLICY = "validation_max_f1_calibrated"
DEFAULT_PRESENCE_TARGET_LABEL = "annual_max_ge_10pct"
DEFAULT_PRESENCE_TARGET_THRESHOLD_FRACTION = 0.10
DEFAULT_TARGET_COLUMN = "kelp_fraction_y"
DEFAULT_TARGET_AREA_COLUMN = "kelp_max_y"
DEFAULT_SELECTION_SPLIT = "validation"
DEFAULT_TEST_SPLIT = "test"
DEFAULT_REPORT_YEAR = 2022
DEFAULT_MAP_ROW_LIMIT = 1_250_000
DEFAULT_OBSERVED_AREA_BINS = (0.0, 1.0, 90.0, 225.0, 450.0, 810.0, 900.0)
REQUIRED_INPUT_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_fraction_y",
    "kelp_max_y",
)
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
HURDLE_PREDICTION_FIELDS = (
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
    "mask_status",
    "evaluation_scope",
    "model_name",
    "model_family",
    "composition_policy",
    "presence_model_name",
    "presence_probability_source",
    "presence_threshold_policy",
    "presence_target_label",
    "presence_target_threshold_fraction",
    "presence_probability_threshold",
    "raw_presence_probability",
    "calibrated_presence_probability",
    "pred_presence_class",
    "conditional_model_name",
    "conditional_target",
    "target",
    "target_area_column",
    "cell_area_m2",
    "selection_split",
    "selection_year",
    "test_split",
    "test_year",
    "pred_conditional_fraction",
    "pred_conditional_fraction_clipped",
    "pred_conditional_area_m2",
    "pred_expected_value_fraction",
    "pred_expected_value_area_m2",
    "pred_hard_gate_fraction",
    "pred_hard_gate_area_m2",
    "pred_hurdle_fraction",
    "pred_hurdle_area_m2",
    "pred_kelp_fraction_y_clipped",
    "pred_kelp_max_y",
    "residual_hurdle_fraction",
    "residual_hurdle_area_m2",
    "residual_kelp_max_y",
)
HURDLE_METRIC_FIELDS = (
    "model_name",
    "model_family",
    "composition_policy",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "presence_probability_threshold",
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
    "precision_ge_10pct",
    "recall_ge_10pct",
    "f1_ge_10pct",
    "positive_count",
    "predicted_positive_count",
    "predicted_positive_rate",
    "assumed_background_count",
    "assumed_background_predicted_positive_count",
    "assumed_background_predicted_positive_rate",
    "assumed_background_predicted_area_m2",
    "assumed_background_max_predicted_area_m2",
    "mean_residual_area",
    "high_canopy_count",
    "high_canopy_mean_residual_area",
)
HURDLE_AREA_CALIBRATION_FIELDS = (
    "model_name",
    "model_family",
    "composition_policy",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "row_count",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mae",
    "rmse",
    "r2",
    "f1_ge_10pct",
)
HURDLE_MODEL_COMPARISON_FIELDS = (
    "model_name",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "row_count",
    "mae",
    "rmse",
    "r2",
    "spearman",
    "f1_ge_10pct",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_pct_bias",
)
HURDLE_RESIDUAL_BIN_FIELDS = (
    "model_name",
    "model_family",
    "composition_policy",
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
    "saturated_count",
)
HURDLE_LEAKAGE_FIELDS = (
    "model_name",
    "model_family",
    "composition_policy",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "assumed_background_count",
    "assumed_background_predicted_area_m2",
    "assumed_background_mean_predicted_area_m2",
    "assumed_background_max_predicted_area_m2",
    "assumed_background_predicted_positive_count",
    "assumed_background_predicted_positive_rate",
    "presence_probability_threshold",
)

MetricKey = tuple[str, str, str, int, str]
ResidualKey = tuple[str, str, str, int, str, str]


@dataclass(frozen=True)
class HurdleConfig:
    """Resolved config values for composing the first hurdle model."""

    config_path: Path
    sample_policy: str
    inference_table_path: Path
    binary_full_grid_predictions_path: Path
    binary_calibration_model_path: Path
    binary_threshold_selection_path: Path
    conditional_model_path: Path
    reference_area_calibration_path: Path | None
    predictions_path: Path
    manifest_path: Path
    metrics_path: Path
    area_calibration_path: Path
    model_comparison_path: Path
    residual_by_observed_bin_path: Path
    assumed_background_leakage_path: Path
    map_figure_path: Path | None
    model_name: str
    hard_gate_model_name: str
    presence_model_name: str
    presence_probability_source: str
    presence_threshold_policy: str
    presence_threshold: float
    presence_target_label: str
    presence_target_threshold_fraction: float
    conditional_model_name: str
    target_column: str
    target_area_column: str
    cell_area_m2: float
    composition_policies: tuple[str, ...]
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    selection_split: str
    test_split: str
    analysis_year: int
    observed_area_bins: tuple[float, ...]
    batch_size: int
    drop_missing_features: bool
    reporting_domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class HurdleSidecarConfig:
    """Resolved config values for an optional hurdle sidecar composition."""

    name: str
    hurdle_config: HurdleConfig


@dataclass(frozen=True)
class LoadedHurdleModels:
    """Loaded model payloads needed for composition only."""

    calibrator: BinaryCalibrator
    conditional_model: Any
    conditional_payload: dict[str, Any]


def compose_hurdle_model(config_path: Path) -> int:
    """Compose calibrated presence and conditional amount full-grid predictions."""
    hurdle_config = load_hurdle_config(config_path)
    compose_hurdle_config(hurdle_config)
    for sidecar in load_hurdle_sidecar_configs(config_path, hurdle_config):
        LOGGER.info("Composing hurdle sidecar: %s", sidecar.name)
        compose_hurdle_config(sidecar.hurdle_config)
    return 0


def compose_hurdle_config(hurdle_config: HurdleConfig) -> None:
    """Compose calibrated presence and conditional amount for one resolved config."""
    models = load_hurdle_models(hurdle_config)
    binary_lookup = load_binary_probability_lookup(hurdle_config, models.calibrator)
    reset_output_path(hurdle_config.predictions_path)

    metric_totals: dict[MetricKey, dict[str, float]] = {}
    residual_totals: dict[ResidualKey, dict[str, float | list[float]]] = {}
    map_frames: list[pd.DataFrame] = []
    total_rows = 0
    part_count = 0
    dropped_missing_feature_rows = 0
    columns = full_grid_input_columns(hurdle_config)
    LOGGER.info("Streaming hurdle composition from %s", hurdle_config.inference_table_path)
    for batch in iter_parquet_batches(
        hurdle_config.inference_table_path,
        columns,
        hurdle_config.batch_size,
    ):
        batch["split"] = split_series_by_year(batch["year"], hurdle_config)
        masked = apply_reporting_domain_mask(batch, hurdle_config.reporting_domain_mask)
        complete = feature_complete_mask(masked, hurdle_config.feature_columns)
        dropped_missing_feature_rows += int((~complete).sum())
        retained = masked.loc[complete].copy()
        if retained.empty:
            continue
        with_probability = attach_calibrated_probabilities(retained, binary_lookup)
        composed = compose_batch_predictions(with_probability, models, hurdle_config)
        write_prediction_part(composed, hurdle_config.predictions_path, part_count)
        update_metric_accumulators(metric_totals, composed, hurdle_config)
        update_residual_bin_accumulators(residual_totals, composed, hurdle_config)
        collect_map_rows(map_frames, composed, hurdle_config)
        total_rows += len(composed)
        part_count += 1
        LOGGER.info("Wrote hurdle prediction part %s with %s rows", part_count, len(composed))

    metric_rows = metric_rows_from_accumulators(metric_totals, hurdle_config)
    area_rows = area_calibration_rows(metric_rows)
    residual_rows = residual_rows_from_accumulators(residual_totals, hurdle_config)
    leakage_rows = assumed_background_leakage_rows(metric_rows)
    comparison_rows = model_comparison_rows(area_rows, hurdle_config)
    write_csv_rows(metric_rows, hurdle_config.metrics_path, HURDLE_METRIC_FIELDS)
    write_csv_rows(area_rows, hurdle_config.area_calibration_path, HURDLE_AREA_CALIBRATION_FIELDS)
    write_csv_rows(
        comparison_rows,
        hurdle_config.model_comparison_path,
        HURDLE_MODEL_COMPARISON_FIELDS,
    )
    write_csv_rows(
        residual_rows,
        hurdle_config.residual_by_observed_bin_path,
        HURDLE_RESIDUAL_BIN_FIELDS,
    )
    write_csv_rows(
        leakage_rows,
        hurdle_config.assumed_background_leakage_path,
        HURDLE_LEAKAGE_FIELDS,
    )
    if hurdle_config.map_figure_path is not None:
        write_hurdle_map_figure(map_frames, hurdle_config.map_figure_path)
    write_hurdle_manifest(
        total_rows=total_rows,
        part_count=part_count,
        dropped_missing_feature_rows=dropped_missing_feature_rows,
        metric_rows=metric_rows,
        area_rows=area_rows,
        residual_rows=residual_rows,
        leakage_rows=leakage_rows,
        comparison_rows=comparison_rows,
        hurdle_config=hurdle_config,
    )
    LOGGER.info("Wrote hurdle full-grid predictions: %s", hurdle_config.predictions_path)
    LOGGER.info("Wrote hurdle metrics: %s", hurdle_config.metrics_path)


def load_hurdle_config(config_path: Path) -> HurdleConfig:
    """Load first-hurdle composition settings from the workflow config."""
    config = load_yaml_config(config_path)
    features = require_mapping(config.get("features"), "features")
    splits = require_mapping(config.get("splits"), "splits")
    models = require_mapping(config.get("models"), "models")
    baselines = optional_mapping(models.get("baselines"), "models.baselines")
    binary = optional_mapping(models.get("binary_presence"), "models.binary_presence")
    binary_calibration = optional_mapping(
        binary.get("calibration"),
        "models.binary_presence.calibration",
    )
    conditional = optional_mapping(models.get("conditional_canopy"), "models.conditional_canopy")
    hurdle = require_mapping(models.get("hurdle"), "models.hurdle")
    reports = optional_mapping(config.get("reports"), "reports")
    report_settings = optional_mapping(reports.get("model_analysis"), "reports.model_analysis")
    report_outputs = optional_mapping(reports.get("outputs"), "reports.outputs")
    reporting_domain_mask = load_reporting_domain_mask(config)
    configured_threshold = optional_float(
        hurdle.get("presence_threshold"),
        "models.hurdle.presence_threshold",
        None,
    )
    threshold_policy = str(
        hurdle.get("presence_threshold_policy", DEFAULT_PRESENCE_THRESHOLD_POLICY)
    )
    threshold_selection_path = Path(
        require_string(
            hurdle.get("binary_calibrated_threshold_selection")
            or binary_calibration.get("threshold_selection"),
            "models.hurdle.binary_calibrated_threshold_selection",
        )
    )
    return HurdleConfig(
        config_path=config_path,
        sample_policy=str(hurdle.get("sample_policy", "current_masked_sample")),
        inference_table_path=Path(
            require_string(
                hurdle.get("inference_table")
                or binary.get("inference_table")
                or baselines.get("inference_table"),
                "models.hurdle.inference_table",
            )
        ),
        binary_full_grid_predictions_path=Path(
            require_string(
                hurdle.get("binary_full_grid_predictions") or binary.get("full_grid_predictions"),
                "models.hurdle.binary_full_grid_predictions",
            )
        ),
        binary_calibration_model_path=Path(
            require_string(
                hurdle.get("binary_calibration_model") or binary_calibration.get("model"),
                "models.hurdle.binary_calibration_model",
            )
        ),
        binary_threshold_selection_path=threshold_selection_path,
        conditional_model_path=Path(
            require_string(
                hurdle.get("conditional_model") or conditional.get("model"),
                "models.hurdle.conditional_model",
            )
        ),
        reference_area_calibration_path=optional_path(
            hurdle.get("reference_area_calibration")
            or report_outputs.get("reference_baseline_area_calibration_masked")
            or report_outputs.get("reference_baseline_area_calibration")
        ),
        predictions_path=Path(
            require_string(hurdle.get("predictions"), "models.hurdle.predictions")
        ),
        manifest_path=Path(
            require_string(
                hurdle.get("prediction_manifest") or hurdle.get("manifest"),
                "models.hurdle.prediction_manifest",
            )
        ),
        metrics_path=Path(require_string(hurdle.get("metrics"), "models.hurdle.metrics")),
        area_calibration_path=Path(
            require_string(hurdle.get("area_calibration"), "models.hurdle.area_calibration")
        ),
        model_comparison_path=Path(
            require_string(hurdle.get("model_comparison"), "models.hurdle.model_comparison")
        ),
        residual_by_observed_bin_path=Path(
            require_string(
                hurdle.get("residual_by_observed_bin"),
                "models.hurdle.residual_by_observed_bin",
            )
        ),
        assumed_background_leakage_path=Path(
            require_string(
                hurdle.get("assumed_background_leakage"),
                "models.hurdle.assumed_background_leakage",
            )
        ),
        map_figure_path=optional_path(hurdle.get("map_figure")),
        model_name=str(hurdle.get("model_name", DEFAULT_MODEL_NAME)),
        hard_gate_model_name=str(hurdle.get("hard_gate_model_name", DEFAULT_HARD_GATE_MODEL_NAME)),
        presence_model_name=str(hurdle.get("presence_model_name", DEFAULT_PRESENCE_MODEL_NAME)),
        presence_probability_source=str(
            hurdle.get("presence_probability_source", PLATT_PROBABILITY_SOURCE)
        ),
        presence_threshold_policy=threshold_policy,
        presence_threshold=resolve_presence_threshold(
            threshold_selection_path,
            threshold_policy,
            configured_threshold,
        ),
        presence_target_label=str(
            hurdle.get("presence_target_label", DEFAULT_PRESENCE_TARGET_LABEL)
        ),
        presence_target_threshold_fraction=optional_float(
            hurdle.get("presence_target_threshold_fraction"),
            "models.hurdle.presence_target_threshold_fraction",
            DEFAULT_PRESENCE_TARGET_THRESHOLD_FRACTION,
        ),
        conditional_model_name=str(hurdle.get("conditional_model_name", CONDITIONAL_MODEL_NAME)),
        target_column=str(hurdle.get("target", DEFAULT_TARGET_COLUMN)),
        target_area_column=str(hurdle.get("target_area_column", DEFAULT_TARGET_AREA_COLUMN)),
        cell_area_m2=optional_float(
            hurdle.get("cell_area_m2"),
            "models.hurdle.cell_area_m2",
            KELPWATCH_PIXEL_AREA_M2,
        ),
        composition_policies=read_composition_policies(hurdle.get("composition_policies")),
        feature_columns=parse_bands(
            hurdle.get("features") or conditional.get("features") or features.get("bands")
        ),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
        selection_split=str(hurdle.get("selection_split", DEFAULT_SELECTION_SPLIT)),
        test_split=str(hurdle.get("test_split", DEFAULT_TEST_SPLIT)),
        analysis_year=optional_int(
            hurdle.get("analysis_year", report_settings.get("year")),
            "models.hurdle.analysis_year",
            DEFAULT_REPORT_YEAR,
        ),
        observed_area_bins=read_float_tuple(
            hurdle.get("observed_area_bins") or report_settings.get("observed_area_bins"),
            "models.hurdle.observed_area_bins",
            DEFAULT_OBSERVED_AREA_BINS,
        ),
        batch_size=optional_positive_int(
            hurdle.get("batch_size"),
            "models.hurdle.batch_size",
            FULL_GRID_PREDICTION_BATCH_SIZE,
        ),
        drop_missing_features=read_bool(
            hurdle.get("drop_missing_features"),
            "models.hurdle.drop_missing_features",
            default=True,
        ),
        reporting_domain_mask=reporting_domain_mask,
    )


def load_hurdle_sidecar_configs(
    config_path: Path,
    base_config: HurdleConfig,
) -> tuple[HurdleSidecarConfig, ...]:
    """Load optional hurdle sidecars from the workflow config."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    hurdle = require_mapping(models.get("hurdle"), "models.hurdle")
    sidecars = optional_mapping(hurdle.get("sidecars"), "models.hurdle.sidecars")
    output: list[HurdleSidecarConfig] = []
    for name, value in sidecars.items():
        sidecar_name = str(name)
        sidecar = require_mapping(value, f"models.hurdle.sidecars.{sidecar_name}")
        if not read_bool(
            sidecar.get("enabled"),
            f"models.hurdle.sidecars.{sidecar_name}.enabled",
            default=True,
        ):
            continue
        threshold_selection_path = hurdle_sidecar_path(
            sidecar,
            sidecar_name,
            "binary_calibrated_threshold_selection",
        )
        threshold_value = sidecar.get("presence_threshold")
        configured_threshold = (
            None
            if threshold_value is None
            else optional_float(
                threshold_value,
                f"models.hurdle.sidecars.{sidecar_name}.presence_threshold",
                base_config.presence_threshold,
            )
        )
        threshold_policy = str(
            sidecar.get("presence_threshold_policy", base_config.presence_threshold_policy)
        )
        output.append(
            HurdleSidecarConfig(
                name=sidecar_name,
                hurdle_config=replace(
                    base_config,
                    sample_policy=str(sidecar.get("sample_policy", sidecar_name)),
                    binary_full_grid_predictions_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "binary_full_grid_predictions"
                    ),
                    binary_calibration_model_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "binary_calibration_model"
                    ),
                    binary_threshold_selection_path=threshold_selection_path,
                    reference_area_calibration_path=hurdle_sidecar_optional_path(
                        sidecar,
                        sidecar_name,
                        "reference_area_calibration",
                        base_config.reference_area_calibration_path,
                    ),
                    predictions_path=hurdle_sidecar_path(sidecar, sidecar_name, "predictions"),
                    manifest_path=hurdle_sidecar_path(sidecar, sidecar_name, "prediction_manifest"),
                    metrics_path=hurdle_sidecar_path(sidecar, sidecar_name, "metrics"),
                    area_calibration_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "area_calibration"
                    ),
                    model_comparison_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "model_comparison"
                    ),
                    residual_by_observed_bin_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "residual_by_observed_bin"
                    ),
                    assumed_background_leakage_path=hurdle_sidecar_path(
                        sidecar, sidecar_name, "assumed_background_leakage"
                    ),
                    map_figure_path=hurdle_sidecar_optional_path(
                        sidecar, sidecar_name, "map_figure", None
                    ),
                    presence_threshold_policy=threshold_policy,
                    presence_threshold=resolve_presence_threshold(
                        threshold_selection_path,
                        threshold_policy,
                        configured_threshold,
                    ),
                ),
            )
        )
    return tuple(output)


def hurdle_sidecar_path(config: dict[str, Any], sidecar_name: str, key: str) -> Path:
    """Read a required hurdle sidecar path from config."""
    return Path(
        require_string(
            config.get(key),
            f"models.hurdle.sidecars.{sidecar_name}.{key}",
        )
    )


def hurdle_sidecar_optional_path(
    config: dict[str, Any],
    sidecar_name: str,
    key: str,
    default: Path | None,
) -> Path | None:
    """Read an optional hurdle sidecar path from config."""
    value = config.get(key)
    if value is None:
        return default
    return Path(
        require_string(
            value,
            f"models.hurdle.sidecars.{sidecar_name}.{key}",
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


def optional_float(value: object, name: str, default: float | None) -> float:
    """Read an optional floating-point config value."""
    if value is None:
        if default is None:
            msg = f"missing required numeric field: {name}"
            raise ValueError(msg)
        return default
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    return float(cast(Any, value))


def optional_int(value: object, name: str, default: int) -> int:
    """Read an optional integer config value."""
    if value is None:
        return default
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Read an optional positive integer config value."""
    parsed = optional_int(value, name, default)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


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
    return tuple(operator.index(cast(SupportsIndex, value)) for value in values)


def read_float_tuple(value: object, name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Read an optional non-empty numeric list from config."""
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        msg = f"config field must be a non-empty list of numbers: {name}"
        raise ValueError(msg)
    return tuple(float(item) for item in value)


def read_composition_policies(value: object) -> tuple[str, ...]:
    """Read and validate configured hurdle composition policies."""
    if value is None:
        return (EXPECTED_VALUE_POLICY, HARD_GATE_POLICY)
    if not isinstance(value, list) or not value:
        msg = "models.hurdle.composition_policies must be a non-empty list"
        raise ValueError(msg)
    policies = tuple(str(item) for item in value)
    allowed = {EXPECTED_VALUE_POLICY, HARD_GATE_POLICY}
    unexpected = [policy for policy in policies if policy not in allowed]
    if unexpected:
        msg = f"unsupported hurdle composition policies: {unexpected}"
        raise ValueError(msg)
    if EXPECTED_VALUE_POLICY not in policies:
        msg = "models.hurdle.composition_policies must include expected_value"
        raise ValueError(msg)
    return policies


def resolve_presence_threshold(
    threshold_selection_path: Path,
    threshold_policy: str,
    configured_threshold: float | None,
) -> float:
    """Resolve the validation-selected calibrated probability threshold."""
    table_threshold = selected_threshold_from_table(threshold_selection_path, threshold_policy)
    if configured_threshold is not None and table_threshold is not None:
        if not math.isclose(configured_threshold, table_threshold, rel_tol=0.0, abs_tol=1e-9):
            msg = (
                "configured hurdle presence threshold does not match selected "
                f"{threshold_policy} threshold: {configured_threshold} vs {table_threshold}"
            )
            raise ValueError(msg)
        return configured_threshold
    if configured_threshold is not None:
        return configured_threshold
    if table_threshold is not None:
        return table_threshold
    msg = f"could not resolve calibrated threshold from {threshold_selection_path}"
    raise ValueError(msg)


def selected_threshold_from_table(path: Path, threshold_policy: str) -> float | None:
    """Read the recommended validation threshold from the calibration table."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "threshold_policy" not in frame.columns or "probability_threshold" not in frame.columns:
        return None
    selected = frame.loc[frame["threshold_policy"].astype(str) == threshold_policy].copy()
    if selected.empty:
        return None
    if "probability_source" in selected.columns:
        platt = selected.loc[selected["probability_source"].astype(str) == PLATT_PROBABILITY_SOURCE]
        if not platt.empty:
            selected = platt.copy()
    preferred = selected
    if "selected_threshold" in selected.columns:
        flagged = selected.loc[selected["selected_threshold"].map(boolish)]
        if not flagged.empty:
            preferred = flagged.copy()
    if "recommended_policy" in preferred.columns:
        recommended = preferred.loc[preferred["recommended_policy"].map(boolish)]
        if not recommended.empty:
            preferred = recommended.copy()
    if preferred.empty:
        return None
    return float(preferred.iloc[0]["probability_threshold"])


def boolish(value: object) -> bool:
    """Return whether a dynamic CSV value should be treated as true."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_hurdle_models(hurdle_config: HurdleConfig) -> LoadedHurdleModels:
    """Load saved calibration and conditional model payloads without refitting."""
    calibration_payload = cast(
        dict[str, Any], joblib.load(hurdle_config.binary_calibration_model_path)
    )
    conditional_payload = cast(dict[str, Any], joblib.load(hurdle_config.conditional_model_path))
    calibrator = BinaryCalibrator(
        method=str(calibration_payload.get("calibration_method", "platt")),
        model=calibration_payload.get("calibrator"),
        status=str(calibration_payload.get("calibration_status", "loaded")),
        coefficient=float(calibration_payload.get("coefficient", math.nan)),
        intercept=float(calibration_payload.get("intercept", math.nan)),
    )
    conditional_model = conditional_payload.get("model")
    if conditional_model is None:
        msg = f"conditional canopy payload is missing model: {hurdle_config.conditional_model_path}"
        raise ValueError(msg)
    payload_features = tuple(str(item) for item in conditional_payload.get("feature_columns", ()))
    if payload_features and payload_features != hurdle_config.feature_columns:
        msg = (
            "configured hurdle features do not match conditional model payload: "
            f"{hurdle_config.feature_columns} vs {payload_features}"
        )
        raise ValueError(msg)
    return LoadedHurdleModels(
        calibrator=calibrator,
        conditional_model=conditional_model,
        conditional_payload=conditional_payload,
    )


def load_binary_probability_lookup(
    hurdle_config: HurdleConfig,
    calibrator: BinaryCalibrator,
) -> pd.DataFrame:
    """Load raw binary probabilities and attach calibrated probabilities by row key."""
    columns = ["year", "aef_grid_cell_id", "pred_binary_probability"]
    frame = pd.read_parquet(hurdle_config.binary_full_grid_predictions_path, columns=columns)
    if frame.duplicated(["year", "aef_grid_cell_id"]).any():
        duplicates = int(frame.duplicated(["year", "aef_grid_cell_id"]).sum())
        msg = f"binary full-grid predictions have duplicate year/cell keys: {duplicates}"
        raise ValueError(msg)
    raw = frame["pred_binary_probability"].to_numpy(dtype=float)
    frame["calibrated_presence_probability"] = apply_binary_calibrator(calibrator, raw)
    frame = frame.rename(columns={"pred_binary_probability": "raw_presence_probability"})
    indexed = frame.set_index(["year", "aef_grid_cell_id"]).sort_index()
    LOGGER.info("Loaded %s binary probability rows for hurdle composition", len(indexed))
    return cast(pd.DataFrame, indexed)


def full_grid_input_columns(hurdle_config: HurdleConfig) -> list[str]:
    """Return full-grid columns needed for conditional prediction and output identity."""
    available = available_schema_names(hurdle_config.inference_table_path)
    candidates = [
        *REQUIRED_INPUT_COLUMNS,
        *hurdle_config.feature_columns,
        *OPTIONAL_ID_COLUMNS,
    ]
    return [column for column in dict.fromkeys(candidates) if column in available]


def available_schema_names(path: Path) -> set[str]:
    """Return available column names from a Parquet file or dataset path."""
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    return set(dataset.schema.names)


def split_series_by_year(years: pd.Series, hurdle_config: HurdleConfig) -> pd.Series:
    """Assign split labels from configured year lists."""
    split_by_year = {
        **{year: "train" for year in hurdle_config.train_years},
        **{year: "validation" for year in hurdle_config.validation_years},
        **{year: "test" for year in hurdle_config.test_years},
    }
    return years.astype(int).map(split_by_year).fillna("unassigned").astype(str)


def feature_complete_mask(dataframe: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.Series:
    """Return rows with complete conditional model features."""
    missing = [column for column in feature_columns if column not in dataframe.columns]
    if missing:
        msg = f"full-grid inference table is missing hurdle features: {missing}"
        raise ValueError(msg)
    return cast(pd.Series, dataframe.loc[:, list(feature_columns)].notna().all(axis=1))


def attach_calibrated_probabilities(
    retained: pd.DataFrame,
    binary_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach raw and calibrated binary probabilities to retained full-grid rows."""
    keys = pd.MultiIndex.from_frame(retained[["year", "aef_grid_cell_id"]])
    probabilities = binary_lookup.reindex(keys)
    missing = probabilities["calibrated_presence_probability"].isna()
    if bool(missing.any()):
        msg = f"missing binary probability rows for {int(missing.sum())} hurdle rows"
        raise ValueError(msg)
    output = retained.copy()
    output["raw_presence_probability"] = probabilities["raw_presence_probability"].to_numpy(
        dtype=float
    )
    output["calibrated_presence_probability"] = probabilities[
        "calibrated_presence_probability"
    ].to_numpy(dtype=float)
    return output


def compose_batch_predictions(
    retained: pd.DataFrame,
    models: LoadedHurdleModels,
    hurdle_config: HurdleConfig,
) -> pd.DataFrame:
    """Predict conditional amounts and return policy-specific hurdle rows."""
    features = retained.loc[:, list(hurdle_config.feature_columns)].to_numpy(dtype=float)
    conditional = np.asarray(models.conditional_model.predict(features), dtype=float)
    clipped = np.clip(conditional, 0.0, 1.0)
    base = retained[prediction_identity_columns(retained, hurdle_config)].copy()
    base["mask_status"] = mask_status(hurdle_config.reporting_domain_mask)
    base["evaluation_scope"] = evaluation_scope(hurdle_config.reporting_domain_mask)
    base["model_family"] = HURDLE_MODEL_FAMILY
    base["presence_model_name"] = hurdle_config.presence_model_name
    base["presence_probability_source"] = hurdle_config.presence_probability_source
    base["presence_threshold_policy"] = hurdle_config.presence_threshold_policy
    base["presence_target_label"] = hurdle_config.presence_target_label
    base["presence_target_threshold_fraction"] = hurdle_config.presence_target_threshold_fraction
    base["presence_probability_threshold"] = hurdle_config.presence_threshold
    base["raw_presence_probability"] = retained["raw_presence_probability"].to_numpy(dtype=float)
    base["calibrated_presence_probability"] = retained["calibrated_presence_probability"].to_numpy(
        dtype=float
    )
    base["pred_presence_class"] = (
        base["calibrated_presence_probability"].to_numpy(dtype=float)
        >= hurdle_config.presence_threshold
    )
    base["conditional_model_name"] = hurdle_config.conditional_model_name
    base["conditional_target"] = hurdle_config.target_column
    base["target"] = hurdle_config.target_column
    base["target_area_column"] = hurdle_config.target_area_column
    base["cell_area_m2"] = hurdle_config.cell_area_m2
    base["selection_split"] = hurdle_config.selection_split
    base["selection_year"] = selection_year(hurdle_config)
    base["test_split"] = hurdle_config.test_split
    base["test_year"] = test_year(hurdle_config)
    base["pred_conditional_fraction"] = conditional
    base["pred_conditional_fraction_clipped"] = clipped
    base["pred_conditional_area_m2"] = clipped * hurdle_config.cell_area_m2
    base["pred_expected_value_fraction"] = (
        base["calibrated_presence_probability"].to_numpy(dtype=float) * clipped
    )
    base["pred_expected_value_area_m2"] = (
        base["pred_expected_value_fraction"].to_numpy(dtype=float) * hurdle_config.cell_area_m2
    )
    base["pred_hard_gate_fraction"] = np.where(
        base["pred_presence_class"].to_numpy(dtype=bool),
        clipped,
        0.0,
    )
    base["pred_hard_gate_area_m2"] = (
        base["pred_hard_gate_fraction"].to_numpy(dtype=float) * hurdle_config.cell_area_m2
    )
    policy_frames = [
        policy_prediction_frame(base, policy, hurdle_config)
        for policy in hurdle_config.composition_policies
    ]
    combined = pd.concat(policy_frames, ignore_index=True)
    return cast(pd.DataFrame, combined.loc[:, hurdle_prediction_output_columns(combined)])


def prediction_identity_columns(dataframe: pd.DataFrame, hurdle_config: HurdleConfig) -> list[str]:
    """Return identity, provenance, and target columns for hurdle outputs."""
    columns = [
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        hurdle_config.target_column,
        hurdle_config.target_area_column,
    ]
    for column in OPTIONAL_ID_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    return columns


def policy_prediction_frame(
    base: pd.DataFrame,
    policy: str,
    hurdle_config: HurdleConfig,
) -> pd.DataFrame:
    """Build one set of rows for a configured hurdle composition policy."""
    output = base.copy()
    output["composition_policy"] = policy
    if policy == EXPECTED_VALUE_POLICY:
        output["model_name"] = hurdle_config.model_name
        selected_fraction = output["pred_expected_value_fraction"].to_numpy(dtype=float)
        selected_area = output["pred_expected_value_area_m2"].to_numpy(dtype=float)
    elif policy == HARD_GATE_POLICY:
        output["model_name"] = hurdle_config.hard_gate_model_name
        selected_fraction = output["pred_hard_gate_fraction"].to_numpy(dtype=float)
        selected_area = output["pred_hard_gate_area_m2"].to_numpy(dtype=float)
    else:
        msg = f"unsupported composition policy: {policy}"
        raise ValueError(msg)
    observed_fraction = output[hurdle_config.target_column].to_numpy(dtype=float)
    observed_area = output[hurdle_config.target_area_column].to_numpy(dtype=float)
    output["pred_hurdle_fraction"] = selected_fraction
    output["pred_hurdle_area_m2"] = selected_area
    output["pred_kelp_fraction_y_clipped"] = selected_fraction
    output["pred_kelp_max_y"] = selected_area
    output["residual_hurdle_fraction"] = observed_fraction - selected_fraction
    output["residual_hurdle_area_m2"] = observed_area - selected_area
    output["residual_kelp_max_y"] = observed_area - selected_area
    return output


def hurdle_prediction_output_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return row-level hurdle prediction columns in a stable order."""
    return [column for column in HURDLE_PREDICTION_FIELDS if column in dataframe.columns]


def selection_year(hurdle_config: HurdleConfig) -> int | str:
    """Return the configured validation selection-year label."""
    if len(hurdle_config.validation_years) == 1:
        return hurdle_config.validation_years[0]
    return "all_validation_years"


def test_year(hurdle_config: HurdleConfig) -> int | str:
    """Return the configured held-out test-year label."""
    if len(hurdle_config.test_years) == 1:
        return hurdle_config.test_years[0]
    return "all_test_years"


def update_metric_accumulators(
    totals: dict[MetricKey, dict[str, float]],
    predictions: pd.DataFrame,
    hurdle_config: HurdleConfig,
) -> None:
    """Update metric totals from one composed prediction batch."""
    group_sets = [
        ["model_name", "composition_policy", "split", "year"],
        ["model_name", "composition_policy", "split", "year", "label_source"],
    ]
    for group_columns in group_sets:
        for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            label_source = "all" if "label_source" not in group_columns else str(key_tuple[-1])
            key = (
                str(key_tuple[0]),
                str(key_tuple[1]),
                str(key_tuple[2]),
                int(key_tuple[3]),
                label_source,
            )
            update_metric_state(totals.setdefault(key, empty_metric_state()), group, hurdle_config)


def empty_metric_state() -> dict[str, float]:
    """Return an initialized mutable metric accumulator."""
    return {
        "row_count": 0.0,
        "observed_fraction_sum": 0.0,
        "predicted_fraction_sum": 0.0,
        "observed_fraction_square_sum": 0.0,
        "squared_error_fraction_sum": 0.0,
        "absolute_error_fraction_sum": 0.0,
        "observed_area_sum": 0.0,
        "predicted_area_sum": 0.0,
        "squared_error_area_sum": 0.0,
        "absolute_error_area_sum": 0.0,
        "residual_area_sum": 0.0,
        "positive_count": 0.0,
        "predicted_positive_count": 0.0,
        "true_positive_count": 0.0,
        "false_positive_count": 0.0,
        "false_negative_count": 0.0,
        "assumed_background_count": 0.0,
        "assumed_background_predicted_positive_count": 0.0,
        "assumed_background_predicted_area_m2": 0.0,
        "assumed_background_max_predicted_area_m2": 0.0,
        "high_canopy_count": 0.0,
        "high_canopy_residual_area_sum": 0.0,
    }


def update_metric_state(
    state: dict[str, float],
    group: pd.DataFrame,
    hurdle_config: HurdleConfig,
) -> None:
    """Accumulate scalar metrics for one grouped prediction frame."""
    observed = group[hurdle_config.target_column].to_numpy(dtype=float)
    predicted = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = group[hurdle_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    residual_area = observed_area - predicted_area
    row_count = float(len(group))
    observed_positive = observed >= hurdle_config.presence_target_threshold_fraction
    predicted_positive = predicted >= hurdle_config.presence_target_threshold_fraction
    label_sources = label_source_array(group)
    assumed_background = label_sources == "assumed_background"
    high_canopy = observed >= 0.50
    state["row_count"] += row_count
    state["observed_fraction_sum"] += float(np.nansum(observed))
    state["predicted_fraction_sum"] += float(np.nansum(predicted))
    state["observed_fraction_square_sum"] += float(np.nansum(observed**2))
    state["squared_error_fraction_sum"] += float(np.nansum((observed - predicted) ** 2))
    state["absolute_error_fraction_sum"] += float(np.nansum(np.abs(observed - predicted)))
    state["observed_area_sum"] += float(np.nansum(observed_area))
    state["predicted_area_sum"] += float(np.nansum(predicted_area))
    state["squared_error_area_sum"] += float(np.nansum((observed_area - predicted_area) ** 2))
    state["absolute_error_area_sum"] += float(np.nansum(np.abs(observed_area - predicted_area)))
    state["residual_area_sum"] += float(np.nansum(residual_area))
    state["positive_count"] += float(np.count_nonzero(observed_positive))
    state["predicted_positive_count"] += float(np.count_nonzero(predicted_positive))
    state["true_positive_count"] += float(np.count_nonzero(observed_positive & predicted_positive))
    state["false_positive_count"] += float(
        np.count_nonzero(~observed_positive & predicted_positive)
    )
    state["false_negative_count"] += float(
        np.count_nonzero(observed_positive & ~predicted_positive)
    )
    state["assumed_background_count"] += float(np.count_nonzero(assumed_background))
    state["assumed_background_predicted_positive_count"] += float(
        np.count_nonzero(assumed_background & predicted_positive)
    )
    background_predicted_area = predicted_area[assumed_background]
    state["assumed_background_predicted_area_m2"] += float(np.nansum(background_predicted_area))
    if background_predicted_area.size:
        state["assumed_background_max_predicted_area_m2"] = max(
            state["assumed_background_max_predicted_area_m2"],
            float(np.nanmax(background_predicted_area)),
        )
    state["high_canopy_count"] += float(np.count_nonzero(high_canopy))
    state["high_canopy_residual_area_sum"] += float(np.nansum(residual_area[high_canopy]))


def label_source_array(dataframe: pd.DataFrame) -> np.ndarray:
    """Return a label-source array with a stable fallback."""
    if "label_source" in dataframe.columns:
        return cast(np.ndarray, dataframe["label_source"].astype(str).to_numpy(dtype=object))
    return np.full(len(dataframe), "all", dtype=object)


def metric_rows_from_accumulators(
    totals: dict[MetricKey, dict[str, float]],
    hurdle_config: HurdleConfig,
) -> list[dict[str, object]]:
    """Convert accumulated metric totals into stable output rows."""
    rows = [
        metric_row_from_state(key, state, hurdle_config)
        for key, state in sorted(totals.items(), key=lambda item: item[0])
    ]
    return rows


def metric_row_from_state(
    key: MetricKey,
    state: dict[str, float],
    hurdle_config: HurdleConfig,
) -> dict[str, object]:
    """Build one hurdle metric row from an accumulated state."""
    model_name, policy, split, year, label_source = key
    row_count = state["row_count"]
    observed_area = state["observed_area_sum"]
    predicted_area = state["predicted_area_sum"]
    precision = safe_ratio(
        state["true_positive_count"],
        state["true_positive_count"] + state["false_positive_count"],
    )
    recall = safe_ratio(
        state["true_positive_count"],
        state["true_positive_count"] + state["false_negative_count"],
    )
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    total_sum_squares = (
        state["observed_fraction_square_sum"] - (state["observed_fraction_sum"] ** 2) / row_count
        if row_count
        else math.nan
    )
    r2 = (
        1.0 - state["squared_error_fraction_sum"] / total_sum_squares
        if total_sum_squares and total_sum_squares > 0
        else math.nan
    )
    return {
        "model_name": model_name,
        "model_family": HURDLE_MODEL_FAMILY,
        "composition_policy": policy,
        "split": split,
        "year": year,
        "label_source": label_source,
        "mask_status": mask_status(hurdle_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(hurdle_config.reporting_domain_mask),
        "row_count": int(row_count),
        "presence_probability_threshold": hurdle_config.presence_threshold,
        "observed_mean_fraction": safe_ratio(state["observed_fraction_sum"], row_count),
        "predicted_mean_fraction": safe_ratio(state["predicted_fraction_sum"], row_count),
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "area_bias": predicted_area - observed_area,
        "area_pct_bias": percent_bias(predicted_area, observed_area),
        "mae_fraction": safe_ratio(state["absolute_error_fraction_sum"], row_count),
        "rmse_fraction": math.sqrt(safe_ratio(state["squared_error_fraction_sum"], row_count)),
        "mae_area": safe_ratio(state["absolute_error_area_sum"], row_count),
        "rmse_area": math.sqrt(safe_ratio(state["squared_error_area_sum"], row_count)),
        "r2_fraction": r2,
        "spearman_fraction": math.nan,
        "precision_ge_10pct": precision,
        "recall_ge_10pct": recall,
        "f1_ge_10pct": f1,
        "positive_count": int(state["positive_count"]),
        "predicted_positive_count": int(state["predicted_positive_count"]),
        "predicted_positive_rate": safe_ratio(state["predicted_positive_count"], row_count),
        "assumed_background_count": int(state["assumed_background_count"]),
        "assumed_background_predicted_positive_count": int(
            state["assumed_background_predicted_positive_count"]
        ),
        "assumed_background_predicted_positive_rate": safe_ratio(
            state["assumed_background_predicted_positive_count"],
            state["assumed_background_count"],
        ),
        "assumed_background_predicted_area_m2": state["assumed_background_predicted_area_m2"],
        "assumed_background_max_predicted_area_m2": state[
            "assumed_background_max_predicted_area_m2"
        ],
        "mean_residual_area": safe_ratio(state["residual_area_sum"], row_count),
        "high_canopy_count": int(state["high_canopy_count"]),
        "high_canopy_mean_residual_area": safe_ratio(
            state["high_canopy_residual_area_sum"],
            state["high_canopy_count"],
        ),
    }


def area_calibration_rows(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Convert metric rows into the compact hurdle area-calibration schema."""
    rows: list[dict[str, object]] = []
    for row in metric_rows:
        rows.append(
            {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "composition_policy": row["composition_policy"],
                "split": row["split"],
                "year": row["year"],
                "mask_status": row["mask_status"],
                "evaluation_scope": row["evaluation_scope"],
                "label_source": row["label_source"],
                "row_count": row["row_count"],
                "observed_canopy_area": row["observed_canopy_area"],
                "predicted_canopy_area": row["predicted_canopy_area"],
                "area_bias": row["area_bias"],
                "area_pct_bias": row["area_pct_bias"],
                "mae": row["mae_fraction"],
                "rmse": row["rmse_fraction"],
                "r2": row["r2_fraction"],
                "f1_ge_10pct": row["f1_ge_10pct"],
            }
        )
    return rows


def update_residual_bin_accumulators(
    totals: dict[ResidualKey, dict[str, float | list[float]]],
    predictions: pd.DataFrame,
    hurdle_config: HurdleConfig,
) -> None:
    """Update observed-area-bin residual totals from one prediction batch."""
    frame = predictions.copy()
    frame["observed_bin"] = observed_area_bin_labels(
        frame[hurdle_config.target_area_column].to_numpy(dtype=float),
        hurdle_config.observed_area_bins,
    )
    group_sets = [
        ["model_name", "composition_policy", "split", "year", "observed_bin"],
        ["model_name", "composition_policy", "split", "year", "label_source", "observed_bin"],
    ]
    for group_columns in group_sets:
        for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            label_source = "all" if "label_source" not in group_columns else str(key_tuple[-2])
            observed_bin = str(key_tuple[-1])
            key = (
                str(key_tuple[0]),
                str(key_tuple[1]),
                str(key_tuple[2]),
                int(key_tuple[3]),
                label_source,
                observed_bin,
            )
            update_residual_state(
                totals.setdefault(key, empty_residual_state()),
                group,
                hurdle_config,
            )


def observed_area_bin_labels(values: np.ndarray, bins: tuple[float, ...]) -> np.ndarray:
    """Return observed-area bin labels matching model-analysis residual bins."""
    labels = np.full(values.shape, "missing", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values == 0)] = "000_zero"
    positive = finite & (values > 0)
    previous = 0.0
    for upper in bins[1:]:
        current = positive & (values > previous) & (values <= upper)
        labels[current] = f"({previous:g}, {upper:g}]"
        previous = upper
    labels[positive & (values > bins[-1])] = f">{bins[-1]:g}"
    return labels


def empty_residual_state() -> dict[str, float | list[float]]:
    """Return an initialized residual-bin accumulator."""
    return {
        "row_count": 0.0,
        "observed_area_sum": 0.0,
        "predicted_area_sum": 0.0,
        "residual_area_sum": 0.0,
        "absolute_error_area_sum": 0.0,
        "squared_error_area_sum": 0.0,
        "underprediction_count": 0.0,
        "overprediction_count": 0.0,
        "saturated_count": 0.0,
        "residual_values": [],
    }


def update_residual_state(
    state: dict[str, float | list[float]],
    group: pd.DataFrame,
    hurdle_config: HurdleConfig,
) -> None:
    """Accumulate residual-bin metrics for one grouped prediction frame."""
    observed_area = group[hurdle_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    residual = observed_area - predicted_area
    row_count = float(len(group))
    state["row_count"] = residual_state_float(state, "row_count") + row_count
    state["observed_area_sum"] = residual_state_float(state, "observed_area_sum") + float(
        np.nansum(observed_area)
    )
    state["predicted_area_sum"] = residual_state_float(state, "predicted_area_sum") + float(
        np.nansum(predicted_area)
    )
    state["residual_area_sum"] = residual_state_float(state, "residual_area_sum") + float(
        np.nansum(residual)
    )
    state["absolute_error_area_sum"] = residual_state_float(
        state, "absolute_error_area_sum"
    ) + float(np.nansum(np.abs(residual)))
    state["squared_error_area_sum"] = residual_state_float(state, "squared_error_area_sum") + float(
        np.nansum(residual**2)
    )
    state["underprediction_count"] = residual_state_float(state, "underprediction_count") + float(
        np.count_nonzero(residual > 0)
    )
    state["overprediction_count"] = residual_state_float(state, "overprediction_count") + float(
        np.count_nonzero(residual < 0)
    )
    state["saturated_count"] = residual_state_float(state, "saturated_count") + float(
        np.count_nonzero(observed_area >= KELPWATCH_PIXEL_AREA_M2)
    )
    cast(list[float], state["residual_values"]).extend(
        float(value) for value in residual[np.isfinite(residual)]
    )


def residual_rows_from_accumulators(
    totals: dict[ResidualKey, dict[str, float | list[float]]],
    hurdle_config: HurdleConfig,
) -> list[dict[str, object]]:
    """Convert residual-bin totals into stable output rows."""
    rows: list[dict[str, object]] = []
    for key, state in sorted(totals.items(), key=lambda item: item[0]):
        model_name, policy, split, year, label_source, observed_bin = key
        row_count = residual_state_float(state, "row_count")
        residual_values = cast(list[float], state["residual_values"])
        rows.append(
            {
                "model_name": model_name,
                "model_family": HURDLE_MODEL_FAMILY,
                "composition_policy": policy,
                "split": split,
                "year": year,
                "label_source": label_source,
                "mask_status": mask_status(hurdle_config.reporting_domain_mask),
                "evaluation_scope": evaluation_scope(hurdle_config.reporting_domain_mask),
                "observed_bin": observed_bin,
                "row_count": int(row_count),
                "observed_mean_area": safe_ratio(
                    residual_state_float(state, "observed_area_sum"),
                    row_count,
                ),
                "predicted_mean_area": safe_ratio(
                    residual_state_float(state, "predicted_area_sum"),
                    row_count,
                ),
                "mean_residual_area": safe_ratio(
                    residual_state_float(state, "residual_area_sum"),
                    row_count,
                ),
                "median_residual_area": float(np.median(residual_values))
                if residual_values
                else math.nan,
                "mae_area": safe_ratio(
                    residual_state_float(state, "absolute_error_area_sum"),
                    row_count,
                ),
                "rmse_area": math.sqrt(
                    safe_ratio(residual_state_float(state, "squared_error_area_sum"), row_count)
                ),
                "underprediction_count": int(residual_state_float(state, "underprediction_count")),
                "overprediction_count": int(residual_state_float(state, "overprediction_count")),
                "saturated_count": int(residual_state_float(state, "saturated_count")),
            }
        )
    return rows


def residual_state_float(state: dict[str, float | list[float]], key: str) -> float:
    """Return a numeric residual-state value with a typed cast."""
    return float(cast(float, state[key]))


def row_object_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    """Return a numeric value from a dynamic output row."""
    value = row.get(key, default)
    if value is None:
        return default
    return float(cast(Any, value))


def assumed_background_leakage_rows(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build assumed-background leakage diagnostics from metric rows."""
    rows: list[dict[str, object]] = []
    for row in metric_rows:
        if row.get("label_source") != "assumed_background":
            continue
        background_count = row_object_float(row, "assumed_background_count")
        rows.append(
            {
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "composition_policy": row["composition_policy"],
                "split": row["split"],
                "year": row["year"],
                "mask_status": row["mask_status"],
                "evaluation_scope": row["evaluation_scope"],
                "label_source": row["label_source"],
                "assumed_background_count": int(background_count),
                "assumed_background_predicted_area_m2": row["assumed_background_predicted_area_m2"],
                "assumed_background_mean_predicted_area_m2": safe_ratio(
                    row_object_float(row, "assumed_background_predicted_area_m2"),
                    background_count,
                ),
                "assumed_background_max_predicted_area_m2": row_object_float(
                    row,
                    "assumed_background_max_predicted_area_m2",
                    default=math.nan,
                ),
                "assumed_background_predicted_positive_count": row[
                    "assumed_background_predicted_positive_count"
                ],
                "assumed_background_predicted_positive_rate": row[
                    "assumed_background_predicted_positive_rate"
                ],
                "presence_probability_threshold": row["presence_probability_threshold"],
            }
        )
    return rows


def model_comparison_rows(
    area_rows: list[dict[str, object]],
    hurdle_config: HurdleConfig,
) -> list[dict[str, object]]:
    """Build a hurdle-vs-reference comparison table in the Phase 1 schema."""
    rows = reference_comparison_rows(hurdle_config.reference_area_calibration_path)
    for row in area_rows:
        rows.append(
            {
                "model_name": row["model_name"],
                "split": row["split"],
                "year": row["year"],
                "mask_status": row["mask_status"],
                "evaluation_scope": row["evaluation_scope"],
                "label_source": row["label_source"],
                "row_count": row["row_count"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "spearman": math.nan,
                "f1_ge_10pct": row["f1_ge_10pct"],
                "observed_canopy_area": row["observed_canopy_area"],
                "predicted_canopy_area": row["predicted_canopy_area"],
                "area_pct_bias": row["area_pct_bias"],
            }
        )
    return rows


def reference_comparison_rows(path: Path | None) -> list[dict[str, object]]:
    """Read existing ridge/reference rows in the Phase 1 comparison schema."""
    if path is None or not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        row_dict = cast(dict[str, object], row)
        rows.append(
            {
                "model_name": row_dict.get("model_name", ""),
                "split": row_dict.get("split", ""),
                "year": row_dict.get("year", ""),
                "mask_status": row_dict.get("mask_status", ""),
                "evaluation_scope": row_dict.get("evaluation_scope", ""),
                "label_source": row_dict.get("label_source", ""),
                "row_count": row_dict.get("row_count", ""),
                "mae": row_dict.get("mae", math.nan),
                "rmse": row_dict.get("rmse", math.nan),
                "r2": row_dict.get("r2", math.nan),
                "spearman": row_dict.get("spearman", math.nan),
                "f1_ge_10pct": row_dict.get("f1_ge_10pct", math.nan),
                "observed_canopy_area": row_dict.get("observed_canopy_area", math.nan),
                "predicted_canopy_area": row_dict.get("predicted_canopy_area", math.nan),
                "area_pct_bias": row_dict.get("area_pct_bias", math.nan),
            }
        )
    return rows


def collect_map_rows(
    map_frames: list[pd.DataFrame],
    predictions: pd.DataFrame,
    hurdle_config: HurdleConfig,
) -> None:
    """Collect expected-value report-year rows for the hurdle map figure."""
    if len(map_frames) >= 1 and sum(len(frame) for frame in map_frames) >= DEFAULT_MAP_ROW_LIMIT:
        return
    selected = predictions.loc[
        (predictions["model_name"] == hurdle_config.model_name)
        & (predictions["split"] == hurdle_config.test_split)
        & (predictions["year"] == hurdle_config.analysis_year),
        [
            "longitude",
            "latitude",
            "kelp_max_y",
            "pred_kelp_max_y",
            "residual_kelp_max_y",
        ],
    ].copy()
    if not selected.empty:
        map_frames.append(selected)


def write_hurdle_map_figure(map_frames: list[pd.DataFrame], output_path: Path) -> None:
    """Write a three-panel observed, expected-value prediction, and residual map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(map_frames, ignore_index=True) if map_frames else pd.DataFrame()
    if frame.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No hurdle map rows", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    if len(frame) > DEFAULT_MAP_ROW_LIMIT:
        frame = frame.sample(DEFAULT_MAP_ROW_LIMIT, random_state=13)
    observed = frame["kelp_max_y"].to_numpy(dtype=float)
    predicted = frame["pred_kelp_max_y"].to_numpy(dtype=float)
    residual = frame["residual_kelp_max_y"].to_numpy(dtype=float)
    area_max = finite_percentile(np.concatenate([observed, predicted]), 99.5, default=900.0)
    residual_abs = finite_percentile(np.abs(residual), 99.5, default=900.0)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 12.5), sharex=True, sharey=True)
    panels = [
        ("Observed area", observed, "viridis", 0.0, area_max),
        ("Hurdle expected area", predicted, "viridis", 0.0, area_max),
        ("Observed - predicted", residual, "coolwarm", -residual_abs, residual_abs),
    ]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        points = ax.scatter(
            frame["longitude"],
            frame["latitude"],
            c=values,
            s=0.35,
            linewidths=0,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(points, ax=ax, fraction=0.036, pad=0.02)
    fig.suptitle("First hurdle model: expected-value 2022 retained-domain map")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finite_percentile(values: np.ndarray, percentile: float, *, default: float) -> float:
    """Return a finite percentile with a default for empty vectors."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    value = float(np.percentile(finite, percentile))
    if value <= 0:
        return default
    return value


def write_hurdle_manifest(
    *,
    total_rows: int,
    part_count: int,
    dropped_missing_feature_rows: int,
    metric_rows: list[dict[str, object]],
    area_rows: list[dict[str, object]],
    residual_rows: list[dict[str, object]],
    leakage_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    hurdle_config: HurdleConfig,
) -> None:
    """Write a compact JSON manifest for hurdle composition outputs."""
    hurdle_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": "compose-hurdle-model",
        "config_path": str(hurdle_config.config_path),
        "model_family": HURDLE_MODEL_FAMILY,
        "sample_policy": hurdle_config.sample_policy,
        "primary_model_name": hurdle_config.model_name,
        "diagnostic_model_name": hurdle_config.hard_gate_model_name,
        "composition_policies": list(hurdle_config.composition_policies),
        "presence_model_name": hurdle_config.presence_model_name,
        "presence_probability_source": hurdle_config.presence_probability_source,
        "presence_threshold_policy": hurdle_config.presence_threshold_policy,
        "presence_threshold": hurdle_config.presence_threshold,
        "presence_target_label": hurdle_config.presence_target_label,
        "presence_target_threshold_fraction": hurdle_config.presence_target_threshold_fraction,
        "conditional_model_name": hurdle_config.conditional_model_name,
        "target": hurdle_config.target_column,
        "target_area_column": hurdle_config.target_area_column,
        "cell_area_m2": hurdle_config.cell_area_m2,
        "selection_split": hurdle_config.selection_split,
        "selection_year": selection_year(hurdle_config),
        "test_split": hurdle_config.test_split,
        "test_year": test_year(hurdle_config),
        "row_counts": {
            "prediction_rows": total_rows,
            "prediction_part_count": part_count,
            "dropped_missing_feature_rows": dropped_missing_feature_rows,
            "metric_rows": len(metric_rows),
            "area_calibration_rows": len(area_rows),
            "residual_bin_rows": len(residual_rows),
            "assumed_background_leakage_rows": len(leakage_rows),
            "model_comparison_rows": len(comparison_rows),
        },
        "mask_status": mask_status(hurdle_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(hurdle_config.reporting_domain_mask),
        "inputs": {
            "inference_table": str(hurdle_config.inference_table_path),
            "binary_full_grid_predictions": str(hurdle_config.binary_full_grid_predictions_path),
            "binary_calibration_model": str(hurdle_config.binary_calibration_model_path),
            "binary_threshold_selection": str(hurdle_config.binary_threshold_selection_path),
            "conditional_model": str(hurdle_config.conditional_model_path),
            "reference_area_calibration": str(hurdle_config.reference_area_calibration_path)
            if hurdle_config.reference_area_calibration_path is not None
            else None,
        },
        "outputs": {
            "predictions": str(hurdle_config.predictions_path),
            "metrics": str(hurdle_config.metrics_path),
            "area_calibration": str(hurdle_config.area_calibration_path),
            "model_comparison": str(hurdle_config.model_comparison_path),
            "residual_by_observed_bin": str(hurdle_config.residual_by_observed_bin_path),
            "assumed_background_leakage": str(hurdle_config.assumed_background_leakage_path),
            "map_figure": str(hurdle_config.map_figure_path)
            if hurdle_config.map_figure_path is not None
            else None,
            "manifest": str(hurdle_config.manifest_path),
        },
        "test_rows_used_for_training_calibration_or_threshold_selection": False,
        "refit_binary_presence_model": False,
        "refit_binary_calibrator": False,
        "refit_conditional_canopy_model": False,
        "qa_notes": [
            (
                "Expected-value hurdle rows are the primary full-grid prediction; "
                "hard-gated rows are diagnostic threshold behavior."
            ),
            (
                "The conditional amount model still inherits the P1-20 high-canopy "
                "underprediction limitation."
            ),
        ],
    }
    hurdle_config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_csv_rows(
    rows: list[dict[str, object]], output_path: Path, fields: tuple[str, ...]
) -> None:
    """Write dictionaries to CSV with a stable header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
