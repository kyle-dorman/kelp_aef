"""Train and evaluate the Phase 1 balanced binary annual-max model."""

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
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    MASK_RETAIN_COLUMN,
    ReportingDomainMask,
    apply_reporting_domain_mask,
    evaluation_scope,
    load_reporting_domain_mask,
    mask_status,
)
from kelp_aef.evaluation.baselines import (
    FULL_GRID_PREDICTION_BATCH_SIZE,
    KELPWATCH_PIXEL_AREA_M2,
    SPLIT_ORDER,
    iter_parquet_batches,
    parse_bands,
    precision_recall_f1,
    reset_output_path,
    safe_ratio,
    write_prediction_part,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize  # noqa: E402

LOGGER = logging.getLogger(__name__)

BINARY_MODEL_NAME = "logistic_annual_max_ge_10pct"
BINARY_SELECTION_SPLIT = "validation"
BINARY_TEST_SPLIT = "test"
ASSUMED_BACKGROUND = "assumed_background"
KELPWATCH_STATION = "kelpwatch_station"
BINARY_THRESHOLD_POLICY = "validation_max_f1_then_precision_then_lower_predicted_positive_rate"
BINARY_CLASSIFICATION_POLICY = "pred_binary_probability_ge_validation_selected_threshold"
DEFAULT_TARGET_LABEL = "annual_max_ge_10pct"
DEFAULT_TARGET_COLUMN = "kelp_fraction_y"
DEFAULT_TARGET_THRESHOLD_FRACTION = 0.10
DEFAULT_TARGET_THRESHOLD_AREA = 90.0
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_MAX_ITER = 1000
DEFAULT_C_GRID = (1.0,)
DEFAULT_THRESHOLD_GRID = tuple(float(round(value, 2)) for value in np.linspace(0.01, 0.99, 99))
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
    "kelp_fraction_y",
    "kelp_max_y",
)
BINARY_PREDICTION_FIELDS = (
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
    "target_label",
    "target_column",
    "target_threshold_fraction",
    "target_threshold_area",
    "binary_observed_y",
    "pred_binary_probability",
    "probability_threshold",
    "pred_binary_class",
    "selection_split",
    "selection_year",
    "classification_policy",
    "regularization_c",
    "class_weight",
)
BINARY_METRIC_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "selection_split",
    "selection_year",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "probability_threshold",
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "true_negative_count",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
)
BINARY_THRESHOLD_SELECTION_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "selection_split",
    "selection_year",
    "selection_policy",
    "selection_status",
    "selected_threshold",
    "probability_threshold",
    "row_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_count",
    "false_positive_rate",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
)
BINARY_FULL_GRID_SUMMARY_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "probability_threshold",
    "row_count",
    "predicted_positive_count",
    "predicted_positive_rate",
    "predicted_positive_cell_count",
    "predicted_positive_area_m2",
    "observed_positive_count",
    "observed_positive_rate",
    "observed_positive_area_m2",
    "assumed_background_count",
    "assumed_background_predicted_positive_count",
    "assumed_background_predicted_positive_rate",
)
BINARY_MODEL_COMPARISON_FIELDS = (
    "model_name",
    "model_family",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "score_column",
    "operating_threshold",
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "true_negative_count",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
)
BINARY_SIDECAR_COMPARISON_FIELDS = (
    "comparison_name",
    "sampling_policy",
    "comparison_scope",
    "split",
    "year",
    "label_source",
    "domain_mask_reason",
    "depth_bin",
    "row_count",
    "positive_count",
    "predicted_positive_count",
    "predicted_positive_rate",
    "auprc",
    "precision",
    "recall",
    "f1",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
    "predicted_positive_area_m2",
    "source_path",
)
CALIBRATION_METHOD_PLATT = "platt"
RAW_LOGISTIC_PROBABILITY_SOURCE = "raw_logistic"
PLATT_PROBABILITY_SOURCE = "platt_calibrated"
RAW_THRESHOLD_POLICY = "p1_18_validation_raw_threshold"
CALIBRATED_MAX_F1_POLICY = "validation_max_f1_calibrated"
CALIBRATED_PREVALENCE_POLICY = "validation_prevalence_match_calibrated"
CALIBRATED_MIN_PRECISION_POLICY = "validation_min_precision_calibrated"
DEFAULT_RELIABILITY_BIN_COUNT = 10
LOGIT_EPSILON = 1e-6
REQUIRED_CALIBRATION_COLUMNS = (
    "split",
    "year",
    "label_source",
    "binary_observed_y",
    "pred_binary_probability",
    "probability_threshold",
    "longitude",
    "latitude",
)
REQUIRED_FULL_GRID_CALIBRATION_COLUMNS = (
    "split",
    "year",
    "label_source",
    "binary_observed_y",
    "pred_binary_probability",
    "probability_threshold",
    "kelp_max_y",
)
CALIBRATED_SAMPLE_PREDICTION_FIELDS = (
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
    "target_label",
    "target_column",
    "target_threshold_fraction",
    "target_threshold_area",
    "binary_observed_y",
    "pred_binary_probability",
    "probability_threshold",
    "pred_binary_class",
    "calibration_method",
    "calibration_status",
    "calibration_split",
    "calibration_year",
    "calibrated_binary_probability",
    "calibrated_probability_threshold",
    "calibrated_pred_binary_class",
    "calibrated_threshold_policy",
)
CALIBRATION_METRIC_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "calibration_method",
    "probability_source",
    "threshold_policy",
    "calibration_split",
    "calibration_year",
    "evaluation_split",
    "evaluation_year",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "probability_threshold",
    "auroc",
    "auprc",
    "brier_score",
    "expected_calibration_error",
    "precision",
    "recall",
    "f1",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "true_negative_count",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
)
CALIBRATION_THRESHOLD_SELECTION_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "calibration_method",
    "probability_source",
    "calibration_split",
    "calibration_year",
    "threshold_policy",
    "selection_status",
    "recommended_policy",
    "selected_threshold",
    "probability_threshold",
    "row_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "target_predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "false_positive_count",
    "false_positive_rate",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
)
CALIBRATED_FULL_GRID_SUMMARY_FIELDS = (
    "model_name",
    "target_label",
    "target_threshold_fraction",
    "target_threshold_area",
    "calibration_method",
    "probability_source",
    "threshold_policy",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "probability_threshold",
    "row_count",
    "predicted_positive_count",
    "predicted_positive_rate",
    "predicted_positive_cell_count",
    "predicted_positive_area_m2",
    "observed_positive_count",
    "observed_positive_rate",
    "observed_positive_area_m2",
    "assumed_background_count",
    "assumed_background_predicted_positive_count",
    "assumed_background_predicted_positive_rate",
)


@dataclass(frozen=True)
class BinaryPresenceConfig:
    """Resolved config values for binary presence model training."""

    config_path: Path
    sample_policy: str
    input_table_path: Path
    split_manifest_path: Path
    inference_table_path: Path
    model_output_path: Path
    sample_predictions_path: Path
    full_grid_predictions_path: Path
    metrics_path: Path
    threshold_selection_path: Path
    full_grid_area_summary_path: Path
    thresholded_model_comparison_path: Path
    prediction_manifest_path: Path
    precision_recall_figure_path: Path
    map_figure_path: Path
    baseline_sample_predictions_path: Path | None
    target_label: str
    target_column: str
    target_threshold_fraction: float
    target_threshold_area: float
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    class_weight: str | None
    c_grid: tuple[float, ...]
    max_iter: int
    drop_missing_features: bool
    allow_missing_split_manifest_rows: bool
    reporting_domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class BinaryPresenceSidecarConfig:
    """Resolved config for an optional binary-presence sidecar run."""

    name: str
    binary_config: BinaryPresenceConfig
    comparison_path: Path | None


@dataclass(frozen=True)
class PreparedBinaryData:
    """Prepared sample rows plus split and target diagnostics."""

    retained_rows: pd.DataFrame
    split_source: str
    dropped_counts_by_split: dict[str, int]


@dataclass(frozen=True)
class BinaryModelSelection:
    """Selected logistic model and validation diagnostics."""

    model: Any
    selected_c: float
    validation_rows: list[dict[str, object]]


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-selected probability operating threshold."""

    threshold: float
    rows: list[dict[str, object]]
    status: str


@dataclass(frozen=True)
class BinaryCalibrationConfig:
    """Resolved config values for binary probability calibration."""

    config_path: Path
    binary_config: BinaryPresenceConfig
    method: str
    calibration_split: str
    calibration_year: int
    evaluation_split: str
    evaluation_year: int
    input_sample_predictions_path: Path
    input_full_grid_predictions_path: Path
    model_output_path: Path
    calibrated_sample_predictions_path: Path
    metrics_path: Path
    threshold_selection_path: Path
    full_grid_area_summary_path: Path
    calibration_curve_figure_path: Path
    threshold_figure_path: Path
    manifest_path: Path
    min_precision: float | None
    include_prevalence_match: bool
    reliability_bin_count: int


@dataclass(frozen=True)
class BinaryCalibrator:
    """Fitted one-dimensional probability calibration model."""

    method: str
    model: Any | None
    status: str
    coefficient: float
    intercept: float


@dataclass(frozen=True)
class CalibrationThresholds:
    """Selected calibrated operating thresholds and diagnostic rows."""

    recommended_threshold: float
    recommended_policy: str
    rows: list[dict[str, object]]
    policy_thresholds: dict[str, tuple[str, float]]


def train_binary_presence(config_path: Path) -> int:
    """Train the balanced binary annual-max model and write configured artifacts."""
    binary_config = load_binary_presence_config(config_path)
    train_binary_presence_config(binary_config)
    for sidecar in load_binary_presence_sidecar_configs(config_path, binary_config):
        LOGGER.info("Training binary-presence sidecar: %s", sidecar.name)
        train_binary_presence_config(sidecar.binary_config)
        if sidecar.comparison_path is not None:
            write_binary_sidecar_comparison(
                base_config=binary_config,
                sidecar=sidecar,
            )
    return 0


def train_binary_presence_config(binary_config: BinaryPresenceConfig) -> None:
    """Train one binary annual-max model from a resolved config."""
    LOGGER.info("Loading binary-presence model input: %s", binary_config.input_table_path)
    sample = pd.read_parquet(binary_config.input_table_path)
    split_manifest = pd.read_parquet(binary_config.split_manifest_path)
    prepared = prepare_binary_model_frame(sample, split_manifest, binary_config)
    train_rows = rows_for_split(prepared.retained_rows, "train")
    validation_rows = rows_for_split(prepared.retained_rows, "validation")
    selection = fit_select_logistic(train_rows, validation_rows, binary_config)
    validation_probabilities = predict_binary_probability(
        selection.model, validation_rows, binary_config
    )
    threshold_selection = select_validation_threshold(
        validation_rows,
        validation_probabilities,
        binary_config,
    )
    sample_predictions = binary_prediction_frame(
        prepared.retained_rows,
        predict_binary_probability(selection.model, prepared.retained_rows, binary_config),
        threshold_selection.threshold,
        binary_config,
        selected_c=selection.selected_c,
    )
    metrics = build_binary_metric_rows(sample_predictions, binary_config)
    write_binary_predictions(sample_predictions, binary_config.sample_predictions_path)
    write_csv_rows(metrics, binary_config.metrics_path, BINARY_METRIC_FIELDS)
    write_csv_rows(
        threshold_selection.rows,
        binary_config.threshold_selection_path,
        BINARY_THRESHOLD_SELECTION_FIELDS,
    )
    write_binary_model(
        selection,
        threshold_selection,
        binary_config,
        prepared=prepared,
    )
    full_grid_rows = predict_binary_full_grid(
        selection.model,
        threshold_selection.threshold,
        selection.selected_c,
        binary_config,
    )
    write_csv_rows(
        full_grid_rows,
        binary_config.full_grid_area_summary_path,
        BINARY_FULL_GRID_SUMMARY_FIELDS,
    )
    model_comparison = build_thresholded_model_comparison(sample_predictions, binary_config)
    write_csv_rows(
        model_comparison,
        binary_config.thresholded_model_comparison_path,
        BINARY_MODEL_COMPARISON_FIELDS,
    )
    write_precision_recall_figure(threshold_selection.rows, binary_config)
    write_binary_full_grid_map(binary_config)
    write_prediction_manifest(
        prepared=prepared,
        selection=selection,
        threshold_selection=threshold_selection,
        sample_predictions=sample_predictions,
        full_grid_summary=full_grid_rows,
        model_comparison=model_comparison,
        binary_config=binary_config,
    )
    LOGGER.info("Wrote binary-presence model: %s", binary_config.model_output_path)
    LOGGER.info("Wrote binary sample predictions: %s", binary_config.sample_predictions_path)
    LOGGER.info("Wrote binary full-grid predictions: %s", binary_config.full_grid_predictions_path)
    LOGGER.info("Wrote binary metrics: %s", binary_config.metrics_path)


def calibrate_binary_presence(config_path: Path) -> int:
    """Calibrate binary annual-max probabilities and write diagnostic artifacts."""
    calibration_config = load_binary_calibration_config(config_path)
    LOGGER.info(
        "Loading binary sample predictions for calibration: %s",
        calibration_config.input_sample_predictions_path,
    )
    sample_predictions = pd.read_parquet(calibration_config.input_sample_predictions_path)
    validate_calibration_prediction_columns(sample_predictions, "binary sample predictions")
    calibration_rows = calibration_fit_rows(sample_predictions, calibration_config)
    calibrator = fit_binary_calibrator(calibration_rows, calibration_config)
    calibrated_sample = calibrated_sample_prediction_frame(
        sample_predictions,
        calibrator,
        calibration_config,
    )
    threshold_selection = select_calibrated_thresholds(calibrated_sample, calibration_config)
    calibrated_sample = apply_recommended_calibrated_threshold(
        calibrated_sample,
        threshold_selection,
    )
    metrics = build_calibration_metric_rows(
        calibrated_sample,
        threshold_selection,
        calibration_config,
    )
    full_grid_summary = summarize_calibrated_full_grid(
        calibrator,
        threshold_selection,
        calibration_config,
    )
    write_binary_predictions(
        calibrated_sample.loc[:, calibrated_sample_output_columns(calibrated_sample)],
        calibration_config.calibrated_sample_predictions_path,
    )
    write_csv_rows(
        metrics,
        calibration_config.metrics_path,
        CALIBRATION_METRIC_FIELDS,
    )
    write_csv_rows(
        threshold_selection.rows,
        calibration_config.threshold_selection_path,
        CALIBRATION_THRESHOLD_SELECTION_FIELDS,
    )
    write_csv_rows(
        full_grid_summary,
        calibration_config.full_grid_area_summary_path,
        CALIBRATED_FULL_GRID_SUMMARY_FIELDS,
    )
    write_binary_calibration_model(calibrator, threshold_selection, calibration_config)
    write_calibration_curve_figure(calibrated_sample, calibration_config)
    write_calibrated_threshold_figure(threshold_selection.rows, calibration_config)
    write_binary_calibration_manifest(
        calibrator=calibrator,
        calibration_rows=calibration_rows,
        calibrated_sample=calibrated_sample,
        threshold_selection=threshold_selection,
        metrics=metrics,
        full_grid_summary=full_grid_summary,
        calibration_config=calibration_config,
    )
    LOGGER.info("Wrote binary calibration model: %s", calibration_config.model_output_path)
    LOGGER.info(
        "Wrote calibrated binary sample predictions: %s",
        calibration_config.calibrated_sample_predictions_path,
    )
    LOGGER.info("Wrote binary calibration metrics: %s", calibration_config.metrics_path)
    return 0


def load_binary_presence_config(config_path: Path) -> BinaryPresenceConfig:
    """Load binary model settings from the workflow config."""
    config = load_yaml_config(config_path)
    alignment = require_mapping(config.get("alignment"), "alignment")
    splits = require_mapping(config.get("splits"), "splits")
    features = require_mapping(config.get("features"), "features")
    models = require_mapping(config.get("models"), "models")
    binary = require_mapping(models.get("binary_presence"), "models.binary_presence")
    reporting_domain_mask = load_reporting_domain_mask(config)
    return BinaryPresenceConfig(
        config_path=config_path,
        sample_policy=str(binary.get("sample_policy", "current_masked_sample")),
        input_table_path=Path(
            require_string(
                binary.get("input_table") or alignment.get("output_table"),
                "models.binary_presence.input_table or alignment.output_table",
            )
        ),
        split_manifest_path=Path(
            require_string(splits.get("output_manifest"), "splits.output_manifest")
        ),
        inference_table_path=Path(
            require_string(
                binary.get("inference_table"),
                "models.binary_presence.inference_table",
            )
        ),
        model_output_path=Path(require_string(binary.get("model"), "models.binary_presence.model")),
        sample_predictions_path=Path(
            require_string(
                binary.get("sample_predictions"),
                "models.binary_presence.sample_predictions",
            )
        ),
        full_grid_predictions_path=Path(
            require_string(
                binary.get("full_grid_predictions"),
                "models.binary_presence.full_grid_predictions",
            )
        ),
        metrics_path=Path(require_string(binary.get("metrics"), "models.binary_presence.metrics")),
        threshold_selection_path=Path(
            require_string(
                binary.get("threshold_selection"),
                "models.binary_presence.threshold_selection",
            )
        ),
        full_grid_area_summary_path=Path(
            require_string(
                binary.get("full_grid_area_summary"),
                "models.binary_presence.full_grid_area_summary",
            )
        ),
        thresholded_model_comparison_path=Path(
            require_string(
                binary.get("thresholded_model_comparison"),
                "models.binary_presence.thresholded_model_comparison",
            )
        ),
        prediction_manifest_path=Path(
            require_string(
                binary.get("prediction_manifest"),
                "models.binary_presence.prediction_manifest",
            )
        ),
        precision_recall_figure_path=Path(
            require_string(
                binary.get("precision_recall_figure"),
                "models.binary_presence.precision_recall_figure",
            )
        ),
        map_figure_path=Path(
            require_string(
                binary.get("map_figure"),
                "models.binary_presence.map_figure",
            )
        ),
        baseline_sample_predictions_path=optional_baseline_sample_predictions_path(models),
        target_label=str(binary.get("target_label", DEFAULT_TARGET_LABEL)),
        target_column=str(binary.get("target_column", DEFAULT_TARGET_COLUMN)),
        target_threshold_fraction=optional_float(
            binary.get("target_threshold_fraction"),
            "models.binary_presence.target_threshold_fraction",
            DEFAULT_TARGET_THRESHOLD_FRACTION,
        ),
        target_threshold_area=optional_float(
            binary.get("target_threshold_area"),
            "models.binary_presence.target_threshold_area",
            DEFAULT_TARGET_THRESHOLD_AREA,
        ),
        feature_columns=parse_bands(binary.get("features") or features.get("bands")),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
        class_weight=read_class_weight(binary.get("class_weight", DEFAULT_CLASS_WEIGHT)),
        c_grid=read_c_grid(binary.get("c_grid")),
        max_iter=optional_positive_int(
            binary.get("max_iter"),
            "models.binary_presence.max_iter",
            DEFAULT_MAX_ITER,
        ),
        drop_missing_features=read_bool(
            binary.get("drop_missing_features"),
            "models.binary_presence.drop_missing_features",
            default=True,
        ),
        allow_missing_split_manifest_rows=read_bool(
            binary.get("allow_missing_split_manifest_rows"),
            "models.binary_presence.allow_missing_split_manifest_rows",
            default=False,
        ),
        reporting_domain_mask=reporting_domain_mask,
    )


def load_binary_presence_sidecar_configs(
    config_path: Path,
    base_config: BinaryPresenceConfig,
) -> tuple[BinaryPresenceSidecarConfig, ...]:
    """Load optional sidecar binary-presence runs from config."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    binary = require_mapping(models.get("binary_presence"), "models.binary_presence")
    sidecars = optional_mapping(binary.get("sidecars"), "models.binary_presence.sidecars")
    output: list[BinaryPresenceSidecarConfig] = []
    for name, value in sidecars.items():
        sidecar_name = str(name)
        sidecar = require_mapping(value, f"models.binary_presence.sidecars.{sidecar_name}")
        if not read_bool(
            sidecar.get("enabled"),
            f"models.binary_presence.sidecars.{sidecar_name}.enabled",
            default=True,
        ):
            continue
        output.append(
            BinaryPresenceSidecarConfig(
                name=sidecar_name,
                binary_config=replace(
                    base_config,
                    sample_policy=str(sidecar.get("sample_policy", sidecar_name)),
                    input_table_path=sidecar_required_path(sidecar, sidecar_name, "input_table"),
                    model_output_path=sidecar_required_path(sidecar, sidecar_name, "model"),
                    sample_predictions_path=sidecar_required_path(
                        sidecar, sidecar_name, "sample_predictions"
                    ),
                    full_grid_predictions_path=sidecar_required_path(
                        sidecar, sidecar_name, "full_grid_predictions"
                    ),
                    metrics_path=sidecar_required_path(sidecar, sidecar_name, "metrics"),
                    threshold_selection_path=sidecar_required_path(
                        sidecar, sidecar_name, "threshold_selection"
                    ),
                    full_grid_area_summary_path=sidecar_required_path(
                        sidecar, sidecar_name, "full_grid_area_summary"
                    ),
                    thresholded_model_comparison_path=sidecar_required_path(
                        sidecar, sidecar_name, "thresholded_model_comparison"
                    ),
                    prediction_manifest_path=sidecar_required_path(
                        sidecar, sidecar_name, "prediction_manifest"
                    ),
                    precision_recall_figure_path=sidecar_required_path(
                        sidecar, sidecar_name, "precision_recall_figure"
                    ),
                    map_figure_path=sidecar_required_path(sidecar, sidecar_name, "map_figure"),
                    allow_missing_split_manifest_rows=read_bool(
                        sidecar.get("allow_missing_split_manifest_rows"),
                        (
                            "models.binary_presence.sidecars."
                            f"{sidecar_name}.allow_missing_split_manifest_rows"
                        ),
                        default=True,
                    ),
                ),
                comparison_path=sidecar_optional_path(sidecar, sidecar_name, "comparison_table"),
            )
        )
    return tuple(output)


def sidecar_required_path(config: dict[str, Any], sidecar_name: str, key: str) -> Path:
    """Read a required binary sidecar output path."""
    return Path(
        require_string(
            config.get(key),
            f"models.binary_presence.sidecars.{sidecar_name}.{key}",
        )
    )


def sidecar_optional_path(config: dict[str, Any], sidecar_name: str, key: str) -> Path | None:
    """Read an optional binary sidecar output path."""
    value = config.get(key)
    if value is None:
        return None
    return Path(
        require_string(
            value,
            f"models.binary_presence.sidecars.{sidecar_name}.{key}",
        )
    )


def load_binary_calibration_config(config_path: Path) -> BinaryCalibrationConfig:
    """Load binary probability calibration settings from the workflow config."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    binary = require_mapping(models.get("binary_presence"), "models.binary_presence")
    calibration = optional_mapping(
        binary.get("calibration"),
        "models.binary_presence.calibration",
    )
    binary_config = load_binary_presence_config(config_path)
    method = str(calibration.get("method", CALIBRATION_METHOD_PLATT))
    if method != CALIBRATION_METHOD_PLATT:
        msg = "models.binary_presence.calibration.method must be 'platt'"
        raise ValueError(msg)
    calibration_split = str(calibration.get("calibration_split", BINARY_SELECTION_SPLIT))
    evaluation_split = str(calibration.get("evaluation_split", BINARY_TEST_SPLIT))
    calibration_year = optional_year(
        calibration.get("calibration_year"),
        "models.binary_presence.calibration.calibration_year",
        primary_selection_year(binary_config),
    )
    evaluation_year = optional_year(
        calibration.get("evaluation_year"),
        "models.binary_presence.calibration.evaluation_year",
        primary_map_year(binary_config),
    )
    return BinaryCalibrationConfig(
        config_path=config_path,
        binary_config=binary_config,
        method=method,
        calibration_split=calibration_split,
        calibration_year=calibration_year,
        evaluation_split=evaluation_split,
        evaluation_year=evaluation_year,
        input_sample_predictions_path=calibration_path(
            calibration,
            "input_sample_predictions",
            binary_config.sample_predictions_path,
        ),
        input_full_grid_predictions_path=calibration_path(
            calibration,
            "input_full_grid_predictions",
            binary_config.full_grid_predictions_path,
        ),
        model_output_path=calibration_path(
            calibration,
            "model",
            binary_config.model_output_path.with_name(
                f"{binary_config.model_output_path.stem}_calibration.joblib"
            ),
        ),
        calibrated_sample_predictions_path=calibration_path(
            calibration,
            "calibrated_sample_predictions",
            binary_config.sample_predictions_path.with_name(
                "binary_presence_calibrated_sample_predictions.parquet"
            ),
        ),
        metrics_path=calibration_path(
            calibration,
            "metrics",
            binary_config.metrics_path.with_name("binary_presence_calibration_metrics.csv"),
        ),
        threshold_selection_path=calibration_path(
            calibration,
            "threshold_selection",
            binary_config.threshold_selection_path.with_name(
                "binary_presence_calibrated_threshold_selection.csv"
            ),
        ),
        full_grid_area_summary_path=calibration_path(
            calibration,
            "full_grid_area_summary",
            binary_config.full_grid_area_summary_path.with_name(
                "binary_presence_calibrated_full_grid_area_summary.csv"
            ),
        ),
        calibration_curve_figure_path=calibration_path(
            calibration,
            "calibration_curve_figure",
            binary_config.precision_recall_figure_path.with_name(
                "binary_presence_calibration_curve.png"
            ),
        ),
        threshold_figure_path=calibration_path(
            calibration,
            "threshold_figure",
            binary_config.precision_recall_figure_path.with_name(
                "binary_presence_calibrated_thresholds.png"
            ),
        ),
        manifest_path=calibration_path(
            calibration,
            "manifest",
            binary_config.prediction_manifest_path.with_name(
                "binary_presence_calibration_manifest.json"
            ),
        ),
        min_precision=optional_probability(
            calibration.get("min_precision"),
            "models.binary_presence.calibration.min_precision",
        ),
        include_prevalence_match=read_bool(
            calibration.get("include_prevalence_match"),
            "models.binary_presence.calibration.include_prevalence_match",
            default=True,
        ),
        reliability_bin_count=optional_positive_int(
            calibration.get("reliability_bin_count"),
            "models.binary_presence.calibration.reliability_bin_count",
            DEFAULT_RELIABILITY_BIN_COUNT,
        ),
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating a missing value as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def calibration_path(config: dict[str, Any], key: str, default: Path) -> Path:
    """Read an optional binary calibration path from config."""
    value = config.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"models.binary_presence.calibration.{key}"))


def optional_year(value: object, name: str, default: int | str) -> int:
    """Read a calibration year, requiring a concrete integer year."""
    if value is None:
        if isinstance(default, str):
            msg = f"config field must be set when multiple years are configured: {name}"
            raise ValueError(msg)
        return default
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        msg = f"field must be an integer year: {name}"
        raise ValueError(msg)
    return operator_index(value)


def optional_probability(value: object, name: str) -> float | None:
    """Read an optional probability value in the closed unit interval."""
    if value is None:
        return None
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    parsed = float(cast(Any, value))
    if not 0.0 <= parsed <= 1.0:
        msg = f"field must be between 0 and 1: {name}"
        raise ValueError(msg)
    return parsed


def optional_float(value: object, name: str, default: float) -> float:
    """Read an optional floating-point config value."""
    if value is None:
        return default
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    return float(cast(Any, value))


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Read an optional positive integer config value."""
    if value is None:
        return default
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    parsed = operator_index(value)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def operator_index(value: object) -> int:
    """Return an index-style integer for mypy-friendly dynamic config parsing."""
    return cast(SupportsIndex, value).__index__()


def read_year_list(config: dict[str, Any], key: str) -> tuple[int, ...]:
    """Read a non-empty split year list from config."""
    values = config.get(key)
    if not isinstance(values, list) or not values:
        msg = f"config field must be a non-empty list of years: splits.{key}"
        raise ValueError(msg)
    years = tuple(operator_index(value) for value in values)
    if any(isinstance(value, bool) for value in values):
        msg = f"split years must be integers, not booleans: splits.{key}"
        raise ValueError(msg)
    return years


def read_bool(value: object, name: str, *, default: bool) -> bool:
    """Read an optional boolean config value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"config field must be a boolean: {name}"
        raise ValueError(msg)
    return value


def read_class_weight(value: object) -> str | None:
    """Read the logistic class-weight setting."""
    if value is None or str(value).lower() in {"none", "null"}:
        return None
    parsed = str(value)
    if parsed != DEFAULT_CLASS_WEIGHT:
        msg = "models.binary_presence.class_weight must be 'balanced' or null"
        raise ValueError(msg)
    return parsed


def read_c_grid(value: object) -> tuple[float, ...]:
    """Read logistic inverse-regularization values."""
    if value is None:
        return DEFAULT_C_GRID
    if not isinstance(value, list) or not value:
        msg = "models.binary_presence.c_grid must be a non-empty list"
        raise ValueError(msg)
    parsed = tuple(float(item) for item in value)
    if any(item <= 0 for item in parsed):
        msg = "models.binary_presence.c_grid values must be positive"
        raise ValueError(msg)
    if len(set(parsed)) != len(parsed):
        msg = "models.binary_presence.c_grid contains duplicate values"
        raise ValueError(msg)
    return parsed


def optional_baseline_sample_predictions_path(models: dict[str, Any]) -> Path | None:
    """Read the optional baseline sample prediction path for thresholded comparison."""
    baselines_value = models.get("baselines")
    if baselines_value is None:
        return None
    baselines = require_mapping(baselines_value, "models.baselines")
    value = baselines.get("sample_predictions")
    if value is None:
        return None
    return Path(require_string(value, "models.baselines.sample_predictions"))


def validate_calibration_prediction_columns(dataframe: pd.DataFrame, name: str) -> None:
    """Validate row-level binary prediction columns needed for calibration."""
    missing = [column for column in REQUIRED_CALIBRATION_COLUMNS if column not in dataframe.columns]
    if missing:
        msg = f"{name} is missing required calibration columns: {missing}"
        raise ValueError(msg)


def validate_full_grid_calibration_columns(dataframe: pd.DataFrame, name: str) -> None:
    """Validate row-level full-grid prediction columns needed for compact summaries."""
    missing = [
        column
        for column in REQUIRED_FULL_GRID_CALIBRATION_COLUMNS
        if column not in dataframe.columns
    ]
    if missing:
        msg = f"{name} is missing required full-grid calibration columns: {missing}"
        raise ValueError(msg)


def calibration_fit_rows(
    predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
) -> pd.DataFrame:
    """Return validation rows used to fit the probability calibrator."""
    rows = predictions.loc[
        (predictions["split"] == calibration_config.calibration_split)
        & (predictions["year"].astype(int) == calibration_config.calibration_year)
    ].copy()
    if rows.empty:
        msg = (
            "no binary prediction rows found for calibration split/year: "
            f"{calibration_config.calibration_split}/{calibration_config.calibration_year}"
        )
        raise ValueError(msg)
    probabilities = rows["pred_binary_probability"].to_numpy(dtype=float)
    observed = rows["binary_observed_y"].to_numpy(dtype=bool)
    valid = np.isfinite(probabilities)
    if not valid.any():
        msg = "calibration rows contain no finite raw probabilities"
        raise ValueError(msg)
    if np.unique(observed[valid]).size < 2:
        msg = "calibration rows must contain both binary classes"
        raise ValueError(msg)
    return rows.loc[valid].copy()


def fit_binary_calibrator(
    calibration_rows: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
) -> BinaryCalibrator:
    """Fit the configured one-dimensional calibration model on validation rows."""
    if calibration_config.method != CALIBRATION_METHOD_PLATT:
        msg = f"unsupported calibration method: {calibration_config.method}"
        raise ValueError(msg)
    probabilities = calibration_rows["pred_binary_probability"].to_numpy(dtype=float)
    observed = calibration_rows["binary_observed_y"].to_numpy(dtype=bool)
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1000)
    model.fit(probability_logit(probabilities), observed)
    coefficient = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    LOGGER.info(
        "Fitted Platt calibrator on %s %s rows from %s with coefficient=%s intercept=%s",
        len(calibration_rows),
        calibration_config.calibration_split,
        calibration_config.calibration_year,
        coefficient,
        intercept,
    )
    return BinaryCalibrator(
        method=calibration_config.method,
        model=model,
        status="fit_on_validation_rows",
        coefficient=coefficient,
        intercept=intercept,
    )


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    """Return clipped probability logits as a two-dimensional feature matrix."""
    clipped = np.clip(probabilities.astype(float), LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return cast(np.ndarray, logits.reshape(-1, 1))


def apply_binary_calibrator(calibrator: BinaryCalibrator, probabilities: np.ndarray) -> np.ndarray:
    """Apply a fitted binary calibrator to raw probabilities."""
    output = np.full(probabilities.shape, np.nan, dtype=float)
    valid = np.isfinite(probabilities)
    if not valid.any():
        return output
    if calibrator.model is None:
        output[valid] = probabilities[valid]
        return output
    calibrated = calibrator.model.predict_proba(probability_logit(probabilities[valid]))[:, 1]
    output[valid] = np.asarray(calibrated, dtype=float)
    return output


def calibrated_sample_prediction_frame(
    sample_predictions: pd.DataFrame,
    calibrator: BinaryCalibrator,
    calibration_config: BinaryCalibrationConfig,
) -> pd.DataFrame:
    """Attach calibrated probabilities to sample prediction rows."""
    frame = sample_predictions.copy()
    raw_probabilities = frame["pred_binary_probability"].to_numpy(dtype=float)
    frame["calibration_method"] = calibration_config.method
    frame["calibration_status"] = calibrator.status
    frame["calibration_split"] = calibration_config.calibration_split
    frame["calibration_year"] = calibration_config.calibration_year
    frame["calibrated_binary_probability"] = apply_binary_calibrator(
        calibrator,
        raw_probabilities,
    )
    frame["calibrated_probability_threshold"] = np.nan
    frame["calibrated_pred_binary_class"] = False
    frame["calibrated_threshold_policy"] = ""
    return frame


def select_calibrated_thresholds(
    sample_predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
) -> CalibrationThresholds:
    """Select calibrated thresholds using validation rows only."""
    validation_rows = calibration_metric_input_rows(
        sample_predictions,
        split=calibration_config.calibration_split,
        year=calibration_config.calibration_year,
    )
    probabilities = validation_rows["calibrated_binary_probability"].to_numpy(dtype=float)
    max_f1_rows = [
        calibration_threshold_row(
            validation_rows,
            probabilities,
            threshold,
            calibration_config,
            threshold_policy=CALIBRATED_MAX_F1_POLICY,
            target_predicted_positive_rate=math.nan,
        )
        for threshold in DEFAULT_THRESHOLD_GRID
    ]
    selected = selected_threshold_row(max_f1_rows)
    recommended_threshold = row_float(selected, "probability_threshold") if selected else 0.5
    status = "selected_from_validation_max_f1" if selected else "no_valid_validation_threshold"
    for row in max_f1_rows:
        mark_calibrated_threshold_row(
            row,
            recommended_threshold,
            status,
            recommended_policy=True,
            selected_threshold=selected is not None,
        )
    rows = list(max_f1_rows)
    policy_thresholds = {
        CALIBRATED_MAX_F1_POLICY: (PLATT_PROBABILITY_SOURCE, recommended_threshold),
        RAW_THRESHOLD_POLICY: (
            RAW_LOGISTIC_PROBABILITY_SOURCE,
            raw_probability_threshold(sample_predictions),
        ),
    }
    if calibration_config.include_prevalence_match:
        prevalence_row = selected_prevalence_threshold_row(
            validation_rows,
            probabilities,
            calibration_config,
        )
        rows.append(prevalence_row)
        policy_thresholds[CALIBRATED_PREVALENCE_POLICY] = (
            PLATT_PROBABILITY_SOURCE,
            row_float(prevalence_row, "probability_threshold"),
        )
    if calibration_config.min_precision is not None:
        precision_row = selected_min_precision_threshold_row(
            max_f1_rows,
            calibration_config.min_precision,
        )
        rows.append(precision_row)
        policy_thresholds[CALIBRATED_MIN_PRECISION_POLICY] = (
            PLATT_PROBABILITY_SOURCE,
            row_float(precision_row, "probability_threshold"),
        )
    return CalibrationThresholds(
        recommended_threshold=recommended_threshold,
        recommended_policy=CALIBRATED_MAX_F1_POLICY,
        rows=rows,
        policy_thresholds=policy_thresholds,
    )


def calibration_metric_input_rows(
    dataframe: pd.DataFrame,
    *,
    split: str,
    year: int,
) -> pd.DataFrame:
    """Return rows for one split and year, requiring at least one finite probability."""
    rows = dataframe.loc[
        (dataframe["split"] == split) & (dataframe["year"].astype(int) == year)
    ].copy()
    if rows.empty:
        msg = f"no rows found for split/year: {split}/{year}"
        raise ValueError(msg)
    return rows


def calibration_threshold_row(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
    probability_threshold: float,
    calibration_config: BinaryCalibrationConfig,
    *,
    threshold_policy: str,
    target_predicted_positive_rate: float,
) -> dict[str, object]:
    """Build one calibrated threshold diagnostic row."""
    valid_mask = np.isfinite(probabilities)
    observed = validation_rows.loc[valid_mask, "binary_observed_y"].to_numpy(dtype=bool)
    predicted = probabilities[valid_mask] >= probability_threshold
    label_sources = label_source_series(validation_rows.loc[valid_mask]).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    false_positive = ~observed & predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": calibration_config.binary_config.target_label,
        "target_threshold_fraction": calibration_config.binary_config.target_threshold_fraction,
        "target_threshold_area": calibration_config.binary_config.target_threshold_area,
        "calibration_method": calibration_config.method,
        "probability_source": PLATT_PROBABILITY_SOURCE,
        "calibration_split": calibration_config.calibration_split,
        "calibration_year": calibration_config.calibration_year,
        "threshold_policy": threshold_policy,
        "selection_status": "",
        "recommended_policy": False,
        "selected_threshold": False,
        "probability_threshold": probability_threshold,
        "row_count": int(observed.size),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, int(predicted.size)),
        "target_predicted_positive_rate": target_predicted_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
    }


def mark_calibrated_threshold_row(
    row: dict[str, object],
    threshold: float,
    status: str,
    *,
    recommended_policy: bool,
    selected_threshold: bool,
) -> None:
    """Mark threshold-selection status fields in place."""
    row["selection_status"] = status
    row["recommended_policy"] = recommended_policy
    row["selected_threshold"] = bool(
        selected_threshold
        and math.isclose(
            row_float(row, "probability_threshold"),
            threshold,
            rel_tol=0,
            abs_tol=1e-12,
        )
    )


def selected_prevalence_threshold_row(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
    calibration_config: BinaryCalibrationConfig,
) -> dict[str, object]:
    """Return the calibrated threshold row closest to validation prevalence."""
    observed = validation_rows["binary_observed_y"].to_numpy(dtype=bool)
    target_rate = safe_ratio(int(np.count_nonzero(observed)), int(observed.size))
    rows = [
        calibration_threshold_row(
            validation_rows,
            probabilities,
            threshold,
            calibration_config,
            threshold_policy=CALIBRATED_PREVALENCE_POLICY,
            target_predicted_positive_rate=target_rate,
        )
        for threshold in DEFAULT_THRESHOLD_GRID
    ]
    selected = min(rows, key=lambda row: prevalence_threshold_sort_key(row, target_rate))
    mark_calibrated_threshold_row(
        selected,
        row_float(selected, "probability_threshold"),
        "selected_from_validation_prevalence_match",
        recommended_policy=False,
        selected_threshold=True,
    )
    return selected


def prevalence_threshold_sort_key(
    row: dict[str, object],
    target_rate: float,
) -> tuple[float, float]:
    """Return the sort key for prevalence-matching threshold selection."""
    rate_gap = abs(row_float(row, "predicted_positive_rate") - target_rate)
    threshold = row_float(row, "probability_threshold")
    return (rate_gap, -threshold)


def selected_min_precision_threshold_row(
    max_f1_rows: list[dict[str, object]],
    min_precision: float,
) -> dict[str, object]:
    """Return the best calibrated threshold satisfying a precision constraint."""
    eligible = [
        row
        for row in max_f1_rows
        if np.isfinite(row_float(row, "precision")) and row_float(row, "precision") >= min_precision
    ]
    selected = max(eligible, key=threshold_sort_key) if eligible else max_f1_rows[-1].copy()
    selected = selected.copy()
    selected["threshold_policy"] = CALIBRATED_MIN_PRECISION_POLICY
    selected["target_predicted_positive_rate"] = math.nan
    status = (
        "selected_from_validation_min_precision" if eligible else "no_threshold_met_min_precision"
    )
    mark_calibrated_threshold_row(
        selected,
        row_float(selected, "probability_threshold"),
        status,
        recommended_policy=False,
        selected_threshold=True,
    )
    return selected


def raw_probability_threshold(predictions: pd.DataFrame) -> float:
    """Return the existing raw P1-18 probability threshold."""
    values = predictions["probability_threshold"].dropna().astype(float).unique()
    return float(values[0]) if len(values) else math.nan


def apply_recommended_calibrated_threshold(
    sample_predictions: pd.DataFrame,
    threshold_selection: CalibrationThresholds,
) -> pd.DataFrame:
    """Attach the recommended calibrated class decision to sample rows."""
    frame = sample_predictions.copy()
    frame["calibrated_probability_threshold"] = threshold_selection.recommended_threshold
    frame["calibrated_pred_binary_class"] = (
        frame["calibrated_binary_probability"].to_numpy(dtype=float)
        >= threshold_selection.recommended_threshold
    )
    frame["calibrated_threshold_policy"] = threshold_selection.recommended_policy
    return frame


def calibrated_sample_output_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return calibrated sample output columns in a stable order."""
    return [column for column in CALIBRATED_SAMPLE_PREDICTION_FIELDS if column in dataframe.columns]


def prepare_binary_model_frame(
    dataframe: pd.DataFrame,
    split_manifest: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> PreparedBinaryData:
    """Attach splits, derive the binary target, and drop unusable rows."""
    validate_input_columns(dataframe, binary_config)
    frame = attach_split_membership(dataframe, split_manifest, binary_config)
    frame["binary_observed_y"] = build_binary_target(
        frame[binary_config.target_column],
        binary_config.target_threshold_fraction,
    )
    feature_complete = frame.loc[:, list(binary_config.feature_columns)].notna().all(axis=1)
    target_complete = frame[binary_config.target_column].notna()
    manifest_used = frame.get("used_for_training_eval")
    manifest_mask = (
        manifest_used.fillna(False).astype(bool)
        if isinstance(manifest_used, pd.Series)
        else pd.Series(True, index=frame.index)
    )
    frame["has_complete_features"] = feature_complete.to_numpy(dtype=bool)
    frame["has_binary_target"] = target_complete.to_numpy(dtype=bool)
    frame["used_for_binary_model"] = (
        manifest_mask & frame["has_complete_features"] & frame["has_binary_target"]
    )
    retained = frame.loc[frame["used_for_binary_model"]].copy()
    if not binary_config.drop_missing_features and len(retained) != len(frame):
        msg = "configured to keep missing features, but logistic model cannot fit missing rows"
        raise ValueError(msg)
    ensure_required_splits_present(retained)
    ensure_train_has_two_classes(retained)
    return PreparedBinaryData(
        retained_rows=retained,
        split_source="split_manifest",
        dropped_counts_by_split=dropped_counts_by_split(frame),
    )


def validate_input_columns(dataframe: pd.DataFrame, binary_config: BinaryPresenceConfig) -> None:
    """Validate sample input columns needed for the binary model."""
    required = [*REQUIRED_INPUT_COLUMNS, *binary_config.feature_columns]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        msg = f"binary model input is missing required columns: {missing}"
        raise ValueError(msg)


def attach_split_membership(
    dataframe: pd.DataFrame,
    split_manifest: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> pd.DataFrame:
    """Attach split labels from the configured split manifest."""
    key_columns = split_join_columns(dataframe, split_manifest)
    if not key_columns:
        frame = dataframe.copy()
        frame["split"] = assign_splits_by_year(frame["year"], binary_config)
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
        msg = f"split manifest has duplicate binary-model keys: {key_columns}"
        raise ValueError(msg)
    frame = dataframe.merge(manifest, on=key_columns, how="left", validate="many_to_one")
    if frame["split"].isna().any():
        missing_count = int(frame["split"].isna().sum())
        if not binary_config.allow_missing_split_manifest_rows:
            msg = f"split manifest is missing {missing_count} binary model rows"
            raise ValueError(msg)
        missing = frame["split"].isna()
        frame.loc[missing, "split"] = assign_splits_by_year(
            frame.loc[missing, "year"],
            binary_config,
        )
        if "used_for_training_eval" in frame.columns:
            frame.loc[missing, "used_for_training_eval"] = True
        if "drop_reason" in frame.columns:
            frame.loc[missing, "drop_reason"] = "year_split_fallback"
        LOGGER.info(
            "Assigned %s binary rows to splits by year because they were absent from %s",
            missing_count,
            binary_config.split_manifest_path,
        )
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


def assign_splits_by_year(years: pd.Series, binary_config: BinaryPresenceConfig) -> pd.Series:
    """Assign split labels from configured year lists."""
    train_years = set(binary_config.train_years)
    validation_years = set(binary_config.validation_years)
    test_years = set(binary_config.test_years)
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
        msg = f"binary model rows contain years not assigned to a split: {missing}"
        raise ValueError(msg)
    return split


def build_binary_target(values: pd.Series, threshold_fraction: float) -> pd.Series:
    """Return the annual-max binary target from Kelpwatch-style fraction labels."""
    return values.astype(float) >= threshold_fraction


def dropped_counts_by_split(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count dropped binary-model rows by split."""
    counts = {split: 0 for split in SPLIT_ORDER}
    dropped = dataframe.loc[~dataframe["used_for_binary_model"]]
    for split, group in dropped.groupby("split", sort=False, dropna=False):
        counts[str(split)] = int(len(group))
    return counts


def ensure_required_splits_present(dataframe: pd.DataFrame) -> None:
    """Validate that retained rows cover train, validation, and test splits."""
    missing = [split for split in SPLIT_ORDER if split not in set(dataframe["split"])]
    if missing:
        msg = f"retained binary model rows are missing splits: {missing}"
        raise ValueError(msg)


def ensure_train_has_two_classes(dataframe: pd.DataFrame) -> None:
    """Validate that the training target has both binary classes."""
    train = dataframe.loc[dataframe["split"] == "train", "binary_observed_y"].astype(bool)
    if train.nunique(dropna=True) < 2:
        msg = "binary logistic training rows must contain both target classes"
        raise ValueError(msg)


def rows_for_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return retained rows for one split."""
    rows = dataframe.loc[dataframe["split"] == split].copy()
    if rows.empty:
        msg = f"no retained binary rows for split: {split}"
        raise ValueError(msg)
    return rows


def fit_select_logistic(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> BinaryModelSelection:
    """Fit logistic candidates and select inverse regularization on validation AUPRC."""
    x_train = feature_matrix(train_rows, binary_config.feature_columns)
    y_train = binary_target_vector(train_rows)
    validation_rows_out: list[dict[str, object]] = []
    candidates: list[tuple[float, float, float, Any]] = []
    for c_value in binary_config.c_grid:
        model = make_logistic_pipeline(c_value, binary_config)
        model.fit(x_train, y_train)
        probabilities = predict_binary_probability(model, validation_rows, binary_config)
        auprc = binary_auprc(binary_target_vector(validation_rows), probabilities)
        auroc = binary_auroc(binary_target_vector(validation_rows), probabilities)
        validation_rows_out.append(
            {
                "regularization_c": c_value,
                "validation_auprc": auprc,
                "validation_auroc": auroc,
            }
        )
        candidates.append(
            (finite_or_negative_inf(auprc), finite_or_negative_inf(auroc), c_value, model)
        )
        LOGGER.info("Binary logistic C=%s validation AUPRC=%s AUROC=%s", c_value, auprc, auroc)
    selected_auprc, selected_auroc, selected_c, selected_model = max(candidates)
    LOGGER.info(
        "Selected binary logistic C=%s with validation AUPRC=%s AUROC=%s",
        selected_c,
        selected_auprc,
        selected_auroc,
    )
    return BinaryModelSelection(
        model=selected_model,
        selected_c=selected_c,
        validation_rows=validation_rows_out,
    )


def make_logistic_pipeline(c_value: float, binary_config: BinaryPresenceConfig) -> Any:
    """Build the standard scaler plus class-weighted logistic model."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    class_weight=binary_config.class_weight,
                    max_iter=binary_config.max_iter,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def feature_matrix(dataframe: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Return configured feature columns as a floating-point matrix."""
    return cast(np.ndarray, dataframe.loc[:, list(feature_columns)].to_numpy(dtype=float))


def binary_target_vector(dataframe: pd.DataFrame) -> np.ndarray:
    """Return binary target values as a boolean vector."""
    return cast(np.ndarray, dataframe["binary_observed_y"].to_numpy(dtype=bool))


def predict_binary_probability(
    model: Any,
    dataframe: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> np.ndarray:
    """Predict positive-class probabilities, preserving NaN for incomplete features."""
    feature_complete = dataframe.loc[:, list(binary_config.feature_columns)].notna().all(axis=1)
    probabilities = np.full(len(dataframe), np.nan, dtype=float)
    if feature_complete.any():
        complete_rows = dataframe.loc[feature_complete]
        predicted = model.predict_proba(
            feature_matrix(complete_rows, binary_config.feature_columns)
        )
        probabilities[feature_complete.to_numpy(dtype=bool)] = np.asarray(
            predicted[:, 1], dtype=float
        )
    return probabilities


def select_validation_threshold(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
    binary_config: BinaryPresenceConfig,
) -> ThresholdSelection:
    """Select a diagnostic probability threshold using validation rows only."""
    selection_year = primary_selection_year(binary_config)
    rows = [
        threshold_selection_row(validation_rows, probabilities, threshold, binary_config)
        for threshold in DEFAULT_THRESHOLD_GRID
    ]
    selected = selected_threshold_row(rows)
    threshold = row_float(selected, "probability_threshold") if selected else 0.5
    status = "selected_from_validation_max_f1" if selected else "no_valid_validation_threshold"
    for row in rows:
        row["selection_status"] = status
        row["selected_threshold"] = bool(
            selected is not None
            and math.isclose(
                row_float(row, "probability_threshold"),
                threshold,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
        row["selection_year"] = selection_year
    return ThresholdSelection(threshold=threshold, rows=rows, status=status)


def primary_selection_year(binary_config: BinaryPresenceConfig) -> int | str:
    """Return the validation year label used for threshold selection."""
    if len(binary_config.validation_years) == 1:
        return binary_config.validation_years[0]
    return "all_validation_years"


def threshold_selection_row(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
    probability_threshold: float,
    binary_config: BinaryPresenceConfig,
) -> dict[str, object]:
    """Build one validation-threshold diagnostic row."""
    valid_mask = np.isfinite(probabilities)
    observed = validation_rows.loc[valid_mask, "binary_observed_y"].to_numpy(dtype=bool)
    predicted = probabilities[valid_mask] >= probability_threshold
    label_sources = label_source_series(validation_rows.loc[valid_mask]).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    false_positive = ~observed & predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": binary_config.target_label,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "selection_split": BINARY_SELECTION_SPLIT,
        "selection_year": primary_selection_year(binary_config),
        "selection_policy": BINARY_THRESHOLD_POLICY,
        "selection_status": "",
        "selected_threshold": False,
        "probability_threshold": probability_threshold,
        "row_count": int(observed.size),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, int(predicted.size)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
    }


def selected_threshold_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the best validation threshold row by the configured policy."""
    valid_rows = [row for row in rows if row_int(row, "positive_count") > 0]
    if not valid_rows:
        return None
    return max(valid_rows, key=threshold_sort_key)


def threshold_sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    """Return the validation threshold selection key."""
    f1 = finite_or_negative_inf(row_float(row, "f1"))
    precision = finite_or_negative_inf(row_float(row, "precision"))
    predicted_rate = finite_or_positive_inf(row_float(row, "predicted_positive_rate"))
    threshold = row_float(row, "probability_threshold")
    return (f1, precision, -predicted_rate, threshold)


def binary_prediction_frame(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
    probability_threshold: float,
    binary_config: BinaryPresenceConfig,
    *,
    selected_c: float,
) -> pd.DataFrame:
    """Build row-level binary prediction output."""
    columns = prediction_identity_columns(dataframe)
    frame = dataframe[columns].copy()
    if "split" not in frame.columns:
        frame["split"] = dataframe["split"].to_numpy(dtype=object)
    if "binary_observed_y" not in frame.columns:
        frame["binary_observed_y"] = dataframe["binary_observed_y"].to_numpy(dtype=bool)
    frame["model_name"] = BINARY_MODEL_NAME
    frame["target_label"] = binary_config.target_label
    frame["target_column"] = binary_config.target_column
    frame["target_threshold_fraction"] = binary_config.target_threshold_fraction
    frame["target_threshold_area"] = binary_config.target_threshold_area
    frame["pred_binary_probability"] = probabilities
    frame["probability_threshold"] = probability_threshold
    frame["pred_binary_class"] = probabilities >= probability_threshold
    frame["selection_split"] = BINARY_SELECTION_SPLIT
    frame["selection_year"] = primary_selection_year(binary_config)
    frame["classification_policy"] = BINARY_CLASSIFICATION_POLICY
    frame["regularization_c"] = selected_c
    frame["class_weight"] = binary_config.class_weight or "none"
    return frame


def prediction_identity_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return identity and provenance columns present on a prediction input frame."""
    columns = [
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        "kelp_fraction_y",
        "kelp_max_y",
    ]
    for column in OPTIONAL_ID_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    if "binary_observed_y" in dataframe.columns:
        columns.append("binary_observed_y")
    return columns


def build_binary_metric_rows(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> list[dict[str, object]]:
    """Build binary sample metrics grouped by split, year, and label source."""
    rows: list[dict[str, object]] = []
    rows.extend(grouped_binary_metric_rows(predictions, binary_config, ["split", "year"]))
    rows.extend(
        grouped_binary_metric_rows(predictions, binary_config, ["split", "year", "label_source"])
    )
    return rows


def grouped_binary_metric_rows(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate binary metrics for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(binary_metric_row(group, group_values, binary_config))
    return rows


def binary_metric_row(
    group: pd.DataFrame,
    group_values: dict[str, object],
    binary_config: BinaryPresenceConfig,
) -> dict[str, object]:
    """Build one grouped binary classification metric row."""
    probabilities = group["pred_binary_probability"].to_numpy(dtype=float)
    valid_mask = np.isfinite(probabilities)
    valid_group = group.loc[valid_mask]
    observed = valid_group["binary_observed_y"].to_numpy(dtype=bool)
    predicted = valid_group["pred_binary_class"].to_numpy(dtype=bool)
    label_sources = label_source_series(valid_group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    true_positive = observed & predicted
    false_positive = ~observed & predicted
    false_negative = observed & ~predicted
    true_negative = ~observed & ~predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": binary_config.target_label,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "selection_split": BINARY_SELECTION_SPLIT,
        "selection_year": primary_selection_year(binary_config),
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, binary_config),
        "evaluation_scope": "model_input_sample",
        "row_count": int(len(valid_group)),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, int(predicted.size)),
        "probability_threshold": first_probability_threshold(group),
        "auroc": binary_auroc(observed, probabilities[valid_mask]),
        "auprc": binary_auprc(observed, probabilities[valid_mask]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(np.count_nonzero(true_positive)),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_rate": safe_ratio(int(np.count_nonzero(false_negative)), positive_count),
        "true_negative_count": int(np.count_nonzero(true_negative)),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
    }


def input_mask_status(group: pd.DataFrame, binary_config: BinaryPresenceConfig) -> str:
    """Infer sample mask status for metric rows."""
    if binary_config.reporting_domain_mask is None or MASK_RETAIN_COLUMN not in group.columns:
        return "unmasked"
    retained = group[MASK_RETAIN_COLUMN].dropna().astype(bool)
    if retained.empty or not bool(retained.all()):
        return "unmasked"
    return mask_status(binary_config.reporting_domain_mask)


def first_probability_threshold(group: pd.DataFrame) -> float:
    """Return the configured probability threshold from a prediction group."""
    values = group["probability_threshold"].dropna().unique()
    return float(values[0]) if len(values) else math.nan


def predict_binary_full_grid(
    model: Any,
    probability_threshold: float,
    selected_c: float,
    binary_config: BinaryPresenceConfig,
) -> list[dict[str, object]]:
    """Stream masked full-grid probability predictions and return area summaries."""
    reset_output_path(binary_config.full_grid_predictions_path)
    row_count = 0
    part_count = 0
    label_source_counts: dict[str, int] = {}
    summary_rows: list[dict[str, object]] = []
    columns = full_grid_input_columns(binary_config)
    LOGGER.info("Streaming binary full-grid inference from %s", binary_config.inference_table_path)
    for batch in iter_parquet_batches(
        binary_config.inference_table_path,
        columns,
        FULL_GRID_PREDICTION_BATCH_SIZE,
    ):
        batch["split"] = assign_splits_by_year(batch["year"], binary_config)
        batch["binary_observed_y"] = build_binary_target(
            batch[binary_config.target_column],
            binary_config.target_threshold_fraction,
        )
        masked = apply_reporting_domain_mask(batch, binary_config.reporting_domain_mask)
        probabilities = predict_binary_probability(model, masked, binary_config)
        prediction_rows = binary_prediction_frame(
            masked,
            probabilities,
            probability_threshold,
            binary_config,
            selected_c=selected_c,
        )
        write_prediction_part(prediction_rows, binary_config.full_grid_predictions_path, part_count)
        summary_rows.extend(full_grid_summary_rows_for_frame(prediction_rows, binary_config))
        row_count += len(prediction_rows)
        part_count += 1
        update_label_source_counts(label_source_counts, prediction_rows)
        LOGGER.info(
            "Wrote binary full-grid prediction part %s with %s retained rows",
            part_count,
            len(prediction_rows),
        )
    summary = aggregate_full_grid_summary_rows(summary_rows)
    binary_config.prediction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    binary_config.prediction_manifest_path.write_text(
        json.dumps(
            {
                "full_grid_row_count": row_count,
                "full_grid_part_count": part_count,
                "full_grid_label_source_counts": label_source_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def full_grid_input_columns(binary_config: BinaryPresenceConfig) -> list[str]:
    """Return columns required for full-grid binary prediction."""
    columns = [
        *REQUIRED_INPUT_COLUMNS,
        *binary_config.feature_columns,
        *OPTIONAL_ID_COLUMNS,
    ]
    deduped: list[str] = []
    for column in columns:
        if column not in deduped:
            deduped.append(column)
    return deduped


def update_label_source_counts(counts: dict[str, int], predictions: pd.DataFrame) -> None:
    """Update streamed label-source counts from one prediction batch."""
    for label_source, count in label_source_series(predictions).value_counts().items():
        key = str(label_source)
        counts[key] = counts.get(key, 0) + int(count)


def full_grid_summary_rows_for_frame(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> list[dict[str, object]]:
    """Build full-grid area summary rows for one prediction batch."""
    rows: list[dict[str, object]] = []
    rows.extend(full_grid_summary_group_rows(predictions, binary_config, ["split", "year"]))
    rows.extend(
        full_grid_summary_group_rows(predictions, binary_config, ["split", "year", "label_source"])
    )
    return rows


def full_grid_summary_group_rows(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate full-grid behavior for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(full_grid_summary_row(group, group_values, binary_config))
    return rows


def full_grid_summary_row(
    group: pd.DataFrame,
    group_values: dict[str, object],
    binary_config: BinaryPresenceConfig,
) -> dict[str, object]:
    """Build one full-grid predicted-positive area summary row."""
    predicted = group["pred_binary_class"].fillna(False).to_numpy(dtype=bool)
    observed = group["binary_observed_y"].fillna(False).to_numpy(dtype=bool)
    label_sources = label_source_series(group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    predicted_positive_count = int(np.count_nonzero(predicted))
    observed_positive_count = int(np.count_nonzero(observed))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_predicted = assumed_background & predicted
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": binary_config.target_label,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": mask_status(binary_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(binary_config.reporting_domain_mask),
        "probability_threshold": first_probability_threshold(group),
        "row_count": int(len(group)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, len(group)),
        "predicted_positive_cell_count": predicted_positive_count,
        "predicted_positive_area_m2": predicted_positive_count * KELPWATCH_PIXEL_AREA_M2,
        "observed_positive_count": observed_positive_count,
        "observed_positive_rate": safe_ratio(observed_positive_count, len(group)),
        "observed_positive_area_m2": float(np.nansum(group.loc[observed, "kelp_max_y"])),
        "assumed_background_count": assumed_background_count,
        "assumed_background_predicted_positive_count": int(
            np.count_nonzero(assumed_background_predicted)
        ),
        "assumed_background_predicted_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_predicted)),
            assumed_background_count,
        ),
    }


def aggregate_full_grid_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-batch full-grid area summary rows."""
    key_fields = tuple(
        field
        for field in BINARY_FULL_GRID_SUMMARY_FIELDS
        if field
        not in {
            "row_count",
            "predicted_positive_count",
            "predicted_positive_rate",
            "predicted_positive_cell_count",
            "predicted_positive_area_m2",
            "observed_positive_count",
            "observed_positive_rate",
            "observed_positive_area_m2",
            "assumed_background_count",
            "assumed_background_predicted_positive_count",
            "assumed_background_predicted_positive_rate",
        }
    )
    totals: dict[tuple[object, ...], dict[str, float]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        current = totals.setdefault(
            key,
            {
                "row_count": 0.0,
                "predicted_positive_count": 0.0,
                "predicted_positive_cell_count": 0.0,
                "predicted_positive_area_m2": 0.0,
                "observed_positive_count": 0.0,
                "observed_positive_area_m2": 0.0,
                "assumed_background_count": 0.0,
                "assumed_background_predicted_positive_count": 0.0,
            },
        )
        for field in current:
            current[field] += row_float(row, field, default=0.0)
    output: list[dict[str, object]] = []
    for key, total in sorted(
        totals.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        row = {field: value for field, value in zip(key_fields, key, strict=True)}
        row_count = int(total["row_count"])
        predicted_positive_count = int(total["predicted_positive_count"])
        observed_positive_count = int(total["observed_positive_count"])
        assumed_background_count = int(total["assumed_background_count"])
        assumed_background_predicted = int(total["assumed_background_predicted_positive_count"])
        row.update(
            {
                "row_count": row_count,
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": safe_ratio(predicted_positive_count, row_count),
                "predicted_positive_cell_count": int(total["predicted_positive_cell_count"]),
                "predicted_positive_area_m2": total["predicted_positive_area_m2"],
                "observed_positive_count": observed_positive_count,
                "observed_positive_rate": safe_ratio(observed_positive_count, row_count),
                "observed_positive_area_m2": total["observed_positive_area_m2"],
                "assumed_background_count": assumed_background_count,
                "assumed_background_predicted_positive_count": assumed_background_predicted,
                "assumed_background_predicted_positive_rate": safe_ratio(
                    assumed_background_predicted,
                    assumed_background_count,
                ),
            }
        )
        output.append(row)
    return output


def build_thresholded_model_comparison(
    binary_predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
) -> list[dict[str, object]]:
    """Compare the binary model against continuous baselines thresholded at 10%."""
    rows: list[dict[str, object]] = []
    rows.extend(
        thresholded_prediction_comparison_rows(
            binary_predictions,
            binary_config,
            model_family="balanced_binary",
            score_column="pred_binary_probability",
            prediction_column="pred_binary_class",
            operating_threshold=first_probability_threshold(binary_predictions),
        )
    )
    baseline_predictions = read_baseline_sample_predictions(binary_config)
    if baseline_predictions.empty:
        return rows
    rows.extend(
        thresholded_prediction_comparison_rows(
            baseline_predictions,
            binary_config,
            model_family="thresholded_continuous_baseline",
            score_column="pred_kelp_fraction_y_clipped",
            prediction_column=None,
            operating_threshold=binary_config.target_threshold_fraction,
        )
    )
    return rows


def read_baseline_sample_predictions(binary_config: BinaryPresenceConfig) -> pd.DataFrame:
    """Read baseline sample predictions when available for model comparison."""
    path = binary_config.baseline_sample_predictions_path
    if path is None or not path.exists():
        LOGGER.info("Skipping thresholded-baseline comparison; baseline sample predictions missing")
        return pd.DataFrame()
    columns = [
        "model_name",
        "split",
        "year",
        "label_source",
        "is_kelpwatch_observed",
        "kelp_fraction_y",
        "kelp_max_y",
        "pred_kelp_fraction_y_clipped",
        "is_plausible_kelp_domain",
    ]
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    selected_columns = [column for column in columns if column in set(dataset.schema.names)]
    return cast(pd.DataFrame, pd.read_parquet(path, columns=selected_columns))


def thresholded_prediction_comparison_rows(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
    *,
    model_family: str,
    score_column: str,
    prediction_column: str | None,
    operating_threshold: float,
) -> list[dict[str, object]]:
    """Build thresholded binary comparison rows for all and label-source groups."""
    if predictions.empty or score_column not in predictions.columns:
        return []
    rows: list[dict[str, object]] = []
    rows.extend(
        grouped_thresholded_comparison_rows(
            predictions,
            binary_config,
            model_family=model_family,
            score_column=score_column,
            prediction_column=prediction_column,
            operating_threshold=operating_threshold,
            group_columns=["model_name", "split", "year"],
        )
    )
    rows.extend(
        grouped_thresholded_comparison_rows(
            predictions,
            binary_config,
            model_family=model_family,
            score_column=score_column,
            prediction_column=prediction_column,
            operating_threshold=operating_threshold,
            group_columns=["model_name", "split", "year", "label_source"],
        )
    )
    return rows


def grouped_thresholded_comparison_rows(
    predictions: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
    *,
    model_family: str,
    score_column: str,
    prediction_column: str | None,
    operating_threshold: float,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate thresholded binary comparison rows for one grouping layout."""
    rows: list[dict[str, object]] = []
    frame = predictions.copy()
    if "label_source" not in frame.columns:
        frame["label_source"] = label_source_series(frame)
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {
            "model_name": "",
            "split": "all",
            "year": "all",
            "label_source": "all",
        }
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(
            thresholded_comparison_row(
                group,
                binary_config,
                group_values=group_values,
                model_family=model_family,
                score_column=score_column,
                prediction_column=prediction_column,
                operating_threshold=operating_threshold,
            )
        )
    return rows


def thresholded_comparison_row(
    group: pd.DataFrame,
    binary_config: BinaryPresenceConfig,
    *,
    group_values: dict[str, object],
    model_family: str,
    score_column: str,
    prediction_column: str | None,
    operating_threshold: float,
) -> dict[str, object]:
    """Build one thresholded model comparison row."""
    score = group[score_column].to_numpy(dtype=float)
    observed_fraction = group[binary_config.target_column].to_numpy(dtype=float)
    valid_mask = np.isfinite(score) & np.isfinite(observed_fraction)
    valid_group = group.loc[valid_mask]
    observed = observed_fraction[valid_mask] >= binary_config.target_threshold_fraction
    if prediction_column is not None and prediction_column in valid_group.columns:
        predicted = valid_group[prediction_column].to_numpy(dtype=bool)
    else:
        predicted = score[valid_mask] >= operating_threshold
    valid_score = score[valid_mask]
    label_sources = label_source_series(valid_group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    true_positive = observed & predicted
    false_positive = ~observed & predicted
    false_negative = observed & ~predicted
    true_negative = ~observed & ~predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "model_name": group_values["model_name"],
        "model_family": model_family,
        "target_label": binary_config.target_label,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, binary_config),
        "evaluation_scope": "model_input_sample",
        "row_count": int(observed.size),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, int(predicted.size)),
        "score_column": score_column,
        "operating_threshold": operating_threshold,
        "auroc": binary_auroc(observed, valid_score),
        "auprc": binary_auprc(observed, valid_score),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(np.count_nonzero(true_positive)),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_rate": safe_ratio(int(np.count_nonzero(false_negative)), positive_count),
        "true_negative_count": int(np.count_nonzero(true_negative)),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
    }


def write_binary_sidecar_comparison(
    *,
    base_config: BinaryPresenceConfig,
    sidecar: BinaryPresenceSidecarConfig,
) -> None:
    """Write compact current-vs-sidecar binary comparison rows."""
    if sidecar.comparison_path is None:
        return
    rows: list[dict[str, object]] = []
    rows.extend(binary_metric_comparison_rows(base_config, "current_masked_sample"))
    rows.extend(binary_metric_comparison_rows(sidecar.binary_config, sidecar.name))
    rows.extend(binary_full_grid_area_comparison_rows(base_config, "current_masked_sample"))
    rows.extend(binary_full_grid_area_comparison_rows(sidecar.binary_config, sidecar.name))
    rows.extend(binary_full_grid_stratum_comparison_rows(base_config, "current_masked_sample"))
    rows.extend(binary_full_grid_stratum_comparison_rows(sidecar.binary_config, sidecar.name))
    write_csv_rows(rows, sidecar.comparison_path, BINARY_SIDECAR_COMPARISON_FIELDS)
    LOGGER.info("Wrote binary sidecar comparison: %s", sidecar.comparison_path)


def binary_metric_comparison_rows(
    binary_config: BinaryPresenceConfig,
    sampling_policy: str,
) -> list[dict[str, object]]:
    """Return validation/test sample metric rows for a sampling policy."""
    if not binary_config.metrics_path.exists():
        return []
    metrics = pd.read_csv(binary_config.metrics_path)
    rows = []
    for row in metrics.to_dict("records"):
        if str(row.get("split")) not in {BINARY_SELECTION_SPLIT, BINARY_TEST_SPLIT}:
            continue
        if str(row.get("label_source")) not in {"all", ASSUMED_BACKGROUND, KELPWATCH_STATION}:
            continue
        rows.append(
            binary_sidecar_comparison_row(
                comparison_name="crm_stratified_background_sampling",
                sampling_policy=sampling_policy,
                comparison_scope="sample_metric",
                split=row.get("split", ""),
                year=row.get("year", ""),
                label_source=row.get("label_source", ""),
                row_count=row.get("row_count", 0),
                positive_count=row.get("positive_count", 0),
                predicted_positive_count=row.get("predicted_positive_count", 0),
                predicted_positive_rate=row.get("predicted_positive_rate", math.nan),
                auprc=row.get("auprc", math.nan),
                precision=row.get("precision", math.nan),
                recall=row.get("recall", math.nan),
                f1=row.get("f1", math.nan),
                assumed_background_count=row.get("assumed_background_count", 0),
                assumed_background_false_positive_count=row.get(
                    "assumed_background_false_positive_count", 0
                ),
                assumed_background_false_positive_rate=row.get(
                    "assumed_background_false_positive_rate", math.nan
                ),
                source_path=binary_config.metrics_path,
            )
        )
    return rows


def binary_full_grid_area_comparison_rows(
    binary_config: BinaryPresenceConfig,
    sampling_policy: str,
) -> list[dict[str, object]]:
    """Return full-grid area/leakage summary rows for a sampling policy."""
    if not binary_config.full_grid_area_summary_path.exists():
        return []
    summary = pd.read_csv(binary_config.full_grid_area_summary_path)
    rows = []
    for row in summary.to_dict("records"):
        if str(row.get("split")) not in {BINARY_SELECTION_SPLIT, BINARY_TEST_SPLIT}:
            continue
        if str(row.get("label_source")) not in {"all", ASSUMED_BACKGROUND, KELPWATCH_STATION}:
            continue
        rows.append(
            binary_sidecar_comparison_row(
                comparison_name="crm_stratified_background_sampling",
                sampling_policy=sampling_policy,
                comparison_scope="full_grid_area_summary",
                split=row.get("split", ""),
                year=row.get("year", ""),
                label_source=row.get("label_source", ""),
                row_count=row.get("row_count", 0),
                positive_count=row.get("observed_positive_count", 0),
                predicted_positive_count=row.get("predicted_positive_count", 0),
                predicted_positive_rate=row.get("predicted_positive_rate", math.nan),
                assumed_background_count=row.get("assumed_background_count", 0),
                assumed_background_false_positive_count=row.get(
                    "assumed_background_predicted_positive_count", 0
                ),
                assumed_background_false_positive_rate=row.get(
                    "assumed_background_predicted_positive_rate", math.nan
                ),
                predicted_positive_area_m2=row.get("predicted_positive_area_m2", math.nan),
                source_path=binary_config.full_grid_area_summary_path,
            )
        )
    return rows


def binary_full_grid_stratum_comparison_rows(
    binary_config: BinaryPresenceConfig,
    sampling_policy: str,
) -> list[dict[str, object]]:
    """Summarize assumed-background full-grid predictions by CRM stratum."""
    if not binary_config.full_grid_predictions_path.exists():
        return []
    required = [
        "split",
        "year",
        "label_source",
        "domain_mask_reason",
        "depth_bin",
        "pred_binary_class",
    ]
    dataset = ds.dataset(binary_config.full_grid_predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    columns = [column for column in required if column in set(dataset.schema.names)]
    if len(columns) != len(required):
        return []
    totals: dict[tuple[str, int, str, str], dict[str, int]] = {}
    for batch in dataset.to_batches(columns=columns, batch_size=FULL_GRID_PREDICTION_BATCH_SIZE):
        frame = batch.to_pandas()
        frame = frame.loc[
            (frame["label_source"] == ASSUMED_BACKGROUND)
            & frame["split"].isin({BINARY_SELECTION_SPLIT, BINARY_TEST_SPLIT})
        ]
        if frame.empty:
            continue
        for keys, group in frame.groupby(["split", "year", "domain_mask_reason", "depth_bin"]):
            split, year, reason, depth_bin = cast(tuple[str, int, str, str], keys)
            key = (str(split), int(year), str(reason), str(depth_bin))
            current = totals.setdefault(key, {"row_count": 0, "predicted_positive_count": 0})
            current["row_count"] += int(len(group))
            current["predicted_positive_count"] += int(
                group["pred_binary_class"].fillna(False).sum()
            )
    rows = []
    for (split, year, reason, depth_bin), values in sorted(totals.items()):
        row_count = values["row_count"]
        predicted_count = values["predicted_positive_count"]
        rows.append(
            binary_sidecar_comparison_row(
                comparison_name="crm_stratified_background_sampling",
                sampling_policy=sampling_policy,
                comparison_scope="full_grid_assumed_background_stratum",
                split=split,
                year=year,
                label_source=ASSUMED_BACKGROUND,
                domain_mask_reason=reason,
                depth_bin=depth_bin,
                row_count=row_count,
                predicted_positive_count=predicted_count,
                predicted_positive_rate=safe_ratio(predicted_count, row_count),
                assumed_background_count=row_count,
                assumed_background_false_positive_count=predicted_count,
                assumed_background_false_positive_rate=safe_ratio(predicted_count, row_count),
                predicted_positive_area_m2=predicted_count * KELPWATCH_PIXEL_AREA_M2,
                source_path=binary_config.full_grid_predictions_path,
            )
        )
    return rows


def binary_sidecar_comparison_row(
    *,
    comparison_name: str,
    sampling_policy: str,
    comparison_scope: str,
    split: object,
    year: object,
    label_source: object,
    domain_mask_reason: object = "",
    depth_bin: object = "",
    row_count: object = 0,
    positive_count: object = math.nan,
    predicted_positive_count: object = math.nan,
    predicted_positive_rate: object = math.nan,
    auprc: object = math.nan,
    precision: object = math.nan,
    recall: object = math.nan,
    f1: object = math.nan,
    assumed_background_count: object = math.nan,
    assumed_background_false_positive_count: object = math.nan,
    assumed_background_false_positive_rate: object = math.nan,
    predicted_positive_area_m2: object = math.nan,
    source_path: object = "",
) -> dict[str, object]:
    """Build one row in the current-vs-sidecar comparison table."""
    return {
        "comparison_name": comparison_name,
        "sampling_policy": sampling_policy,
        "comparison_scope": comparison_scope,
        "split": split,
        "year": year,
        "label_source": label_source,
        "domain_mask_reason": domain_mask_reason,
        "depth_bin": depth_bin,
        "row_count": row_count,
        "positive_count": positive_count,
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": predicted_positive_rate,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": assumed_background_false_positive_count,
        "assumed_background_false_positive_rate": assumed_background_false_positive_rate,
        "predicted_positive_area_m2": predicted_positive_area_m2,
        "source_path": str(source_path),
    }


def build_calibration_metric_rows(
    sample_predictions: pd.DataFrame,
    threshold_selection: CalibrationThresholds,
    calibration_config: BinaryCalibrationConfig,
) -> list[dict[str, object]]:
    """Build raw and calibrated probability metrics for sample predictions."""
    rows: list[dict[str, object]] = []
    raw_threshold = raw_probability_threshold(sample_predictions)
    rows.extend(
        grouped_calibration_metric_rows(
            sample_predictions,
            calibration_config,
            probability_source=RAW_LOGISTIC_PROBABILITY_SOURCE,
            probability_column="pred_binary_probability",
            threshold_policy=RAW_THRESHOLD_POLICY,
            probability_threshold=raw_threshold,
            group_columns=["split", "year"],
        )
    )
    rows.extend(
        grouped_calibration_metric_rows(
            sample_predictions,
            calibration_config,
            probability_source=RAW_LOGISTIC_PROBABILITY_SOURCE,
            probability_column="pred_binary_probability",
            threshold_policy=RAW_THRESHOLD_POLICY,
            probability_threshold=raw_threshold,
            group_columns=["split", "year", "label_source"],
        )
    )
    rows.extend(
        grouped_calibration_metric_rows(
            sample_predictions,
            calibration_config,
            probability_source=PLATT_PROBABILITY_SOURCE,
            probability_column="calibrated_binary_probability",
            threshold_policy=threshold_selection.recommended_policy,
            probability_threshold=threshold_selection.recommended_threshold,
            group_columns=["split", "year"],
        )
    )
    rows.extend(
        grouped_calibration_metric_rows(
            sample_predictions,
            calibration_config,
            probability_source=PLATT_PROBABILITY_SOURCE,
            probability_column="calibrated_binary_probability",
            threshold_policy=threshold_selection.recommended_policy,
            probability_threshold=threshold_selection.recommended_threshold,
            group_columns=["split", "year", "label_source"],
        )
    )
    return rows


def grouped_calibration_metric_rows(
    predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
    *,
    probability_source: str,
    probability_column: str,
    threshold_policy: str,
    probability_threshold: float,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate calibration metrics for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(
            calibration_metric_row(
                group,
                calibration_config,
                group_values=group_values,
                probability_source=probability_source,
                probability_column=probability_column,
                threshold_policy=threshold_policy,
                probability_threshold=probability_threshold,
            )
        )
    return rows


def calibration_metric_row(
    group: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
    *,
    group_values: dict[str, object],
    probability_source: str,
    probability_column: str,
    threshold_policy: str,
    probability_threshold: float,
) -> dict[str, object]:
    """Build one grouped probability calibration and threshold metric row."""
    probabilities = group[probability_column].to_numpy(dtype=float)
    valid_mask = np.isfinite(probabilities)
    valid_group = group.loc[valid_mask]
    observed = valid_group["binary_observed_y"].to_numpy(dtype=bool)
    valid_probabilities = probabilities[valid_mask]
    predicted = valid_probabilities >= probability_threshold
    label_sources = label_source_series(valid_group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    true_positive = observed & predicted
    false_positive = ~observed & predicted
    false_negative = observed & ~predicted
    true_negative = ~observed & ~predicted
    precision, recall, f1 = precision_recall_f1(observed, predicted)
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": calibration_config.binary_config.target_label,
        "target_threshold_fraction": calibration_config.binary_config.target_threshold_fraction,
        "target_threshold_area": calibration_config.binary_config.target_threshold_area,
        "calibration_method": calibration_config.method
        if probability_source != RAW_LOGISTIC_PROBABILITY_SOURCE
        else "none",
        "probability_source": probability_source,
        "threshold_policy": threshold_policy,
        "calibration_split": calibration_config.calibration_split,
        "calibration_year": calibration_config.calibration_year,
        "evaluation_split": calibration_config.evaluation_split,
        "evaluation_year": calibration_config.evaluation_year,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": input_mask_status(group, calibration_config.binary_config),
        "evaluation_scope": "model_input_sample",
        "row_count": int(observed.size),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, int(predicted.size)),
        "probability_threshold": probability_threshold,
        "auroc": binary_auroc(observed, valid_probabilities),
        "auprc": binary_auprc(observed, valid_probabilities),
        "brier_score": binary_brier_score(observed, valid_probabilities),
        "expected_calibration_error": expected_calibration_error(
            observed,
            valid_probabilities,
            calibration_config.reliability_bin_count,
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(np.count_nonzero(true_positive)),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_rate": safe_ratio(int(np.count_nonzero(false_negative)), positive_count),
        "true_negative_count": int(np.count_nonzero(true_negative)),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
    }


def summarize_calibrated_full_grid(
    calibrator: BinaryCalibrator,
    threshold_selection: CalibrationThresholds,
    calibration_config: BinaryCalibrationConfig,
) -> list[dict[str, object]]:
    """Read full-grid predictions and write compact calibrated area summaries."""
    validate_full_grid_calibration_path(calibration_config.input_full_grid_predictions_path)
    summary_rows: list[dict[str, object]] = []
    columns = full_grid_calibration_columns(calibration_config.input_full_grid_predictions_path)
    LOGGER.info(
        "Streaming calibrated full-grid summaries from %s",
        calibration_config.input_full_grid_predictions_path,
    )
    for batch in iter_parquet_batches(
        calibration_config.input_full_grid_predictions_path,
        columns,
        FULL_GRID_PREDICTION_BATCH_SIZE,
    ):
        validate_full_grid_calibration_columns(batch, "binary full-grid predictions")
        raw_probabilities = batch["pred_binary_probability"].to_numpy(dtype=float)
        calibrated_probabilities = apply_binary_calibrator(calibrator, raw_probabilities)
        for threshold_policy, (
            probability_source,
            probability_threshold,
        ) in threshold_selection.policy_thresholds.items():
            probabilities = (
                raw_probabilities
                if probability_source == RAW_LOGISTIC_PROBABILITY_SOURCE
                else calibrated_probabilities
            )
            prediction_rows = batch.copy()
            prediction_rows["calibration_method"] = (
                "none"
                if probability_source == RAW_LOGISTIC_PROBABILITY_SOURCE
                else calibration_config.method
            )
            prediction_rows["probability_source"] = probability_source
            prediction_rows["threshold_policy"] = threshold_policy
            prediction_rows["probability_threshold"] = probability_threshold
            prediction_rows["pred_binary_class"] = probabilities >= probability_threshold
            summary_rows.extend(
                calibrated_full_grid_summary_rows_for_frame(
                    prediction_rows,
                    calibration_config,
                )
            )
    return aggregate_calibrated_full_grid_summary_rows(summary_rows)


def validate_full_grid_calibration_path(path: Path) -> None:
    """Validate that full-grid binary predictions are available for calibration summaries."""
    if not path.exists():
        msg = f"binary full-grid prediction path does not exist: {path}"
        raise FileNotFoundError(msg)


def full_grid_calibration_columns(path: Path) -> list[str]:
    """Return columns needed to summarize calibrated full-grid predictions."""
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    available = set(dataset.schema.names)
    candidates = [
        "split",
        "year",
        "label_source",
        "is_kelpwatch_observed",
        "binary_observed_y",
        "pred_binary_probability",
        "probability_threshold",
        "kelp_max_y",
        "is_plausible_kelp_domain",
        "domain_mask_reason",
        "domain_mask_detail",
        "domain_mask_version",
    ]
    return [column for column in candidates if column in available]


def calibrated_full_grid_summary_rows_for_frame(
    predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
) -> list[dict[str, object]]:
    """Build calibrated full-grid area summary rows for one prediction batch."""
    rows: list[dict[str, object]] = []
    rows.extend(
        calibrated_full_grid_summary_group_rows(
            predictions,
            calibration_config,
            ["split", "year"],
        )
    )
    rows.extend(
        calibrated_full_grid_summary_group_rows(
            predictions,
            calibration_config,
            ["split", "year", "label_source"],
        )
    )
    return rows


def calibrated_full_grid_summary_group_rows(
    predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate calibrated full-grid behavior for one grouping layout."""
    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(calibrated_full_grid_summary_row(group, group_values, calibration_config))
    return rows


def calibrated_full_grid_summary_row(
    group: pd.DataFrame,
    group_values: dict[str, object],
    calibration_config: BinaryCalibrationConfig,
) -> dict[str, object]:
    """Build one calibrated full-grid predicted-positive area summary row."""
    predicted = group["pred_binary_class"].fillna(False).to_numpy(dtype=bool)
    observed = group["binary_observed_y"].fillna(False).to_numpy(dtype=bool)
    label_sources = label_source_series(group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    predicted_positive_count = int(np.count_nonzero(predicted))
    observed_positive_count = int(np.count_nonzero(observed))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_predicted = assumed_background & predicted
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": calibration_config.binary_config.target_label,
        "target_threshold_fraction": calibration_config.binary_config.target_threshold_fraction,
        "target_threshold_area": calibration_config.binary_config.target_threshold_area,
        "calibration_method": str(group["calibration_method"].iloc[0]),
        "probability_source": str(group["probability_source"].iloc[0]),
        "threshold_policy": str(group["threshold_policy"].iloc[0]),
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": mask_status(calibration_config.binary_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(
            calibration_config.binary_config.reporting_domain_mask
        ),
        "probability_threshold": first_probability_threshold(group),
        "row_count": int(len(group)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(predicted_positive_count, len(group)),
        "predicted_positive_cell_count": predicted_positive_count,
        "predicted_positive_area_m2": predicted_positive_count * KELPWATCH_PIXEL_AREA_M2,
        "observed_positive_count": observed_positive_count,
        "observed_positive_rate": safe_ratio(observed_positive_count, len(group)),
        "observed_positive_area_m2": float(np.nansum(group.loc[observed, "kelp_max_y"])),
        "assumed_background_count": assumed_background_count,
        "assumed_background_predicted_positive_count": int(
            np.count_nonzero(assumed_background_predicted)
        ),
        "assumed_background_predicted_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_predicted)),
            assumed_background_count,
        ),
    }


def aggregate_calibrated_full_grid_summary_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-batch calibrated full-grid area summary rows."""
    key_fields = tuple(
        field
        for field in CALIBRATED_FULL_GRID_SUMMARY_FIELDS
        if field
        not in {
            "row_count",
            "predicted_positive_count",
            "predicted_positive_rate",
            "predicted_positive_cell_count",
            "predicted_positive_area_m2",
            "observed_positive_count",
            "observed_positive_rate",
            "observed_positive_area_m2",
            "assumed_background_count",
            "assumed_background_predicted_positive_count",
            "assumed_background_predicted_positive_rate",
        }
    )
    totals: dict[tuple[object, ...], dict[str, float]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        current = totals.setdefault(
            key,
            {
                "row_count": 0.0,
                "predicted_positive_count": 0.0,
                "predicted_positive_cell_count": 0.0,
                "predicted_positive_area_m2": 0.0,
                "observed_positive_count": 0.0,
                "observed_positive_area_m2": 0.0,
                "assumed_background_count": 0.0,
                "assumed_background_predicted_positive_count": 0.0,
            },
        )
        for field in current:
            current[field] += row_float(row, field, default=0.0)
    output: list[dict[str, object]] = []
    for key, total in sorted(
        totals.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        row = {field: value for field, value in zip(key_fields, key, strict=True)}
        row_count = int(total["row_count"])
        predicted_positive_count = int(total["predicted_positive_count"])
        observed_positive_count = int(total["observed_positive_count"])
        assumed_background_count = int(total["assumed_background_count"])
        assumed_background_predicted = int(total["assumed_background_predicted_positive_count"])
        row.update(
            {
                "row_count": row_count,
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": safe_ratio(predicted_positive_count, row_count),
                "predicted_positive_cell_count": int(total["predicted_positive_cell_count"]),
                "predicted_positive_area_m2": total["predicted_positive_area_m2"],
                "observed_positive_count": observed_positive_count,
                "observed_positive_rate": safe_ratio(observed_positive_count, row_count),
                "observed_positive_area_m2": total["observed_positive_area_m2"],
                "assumed_background_count": assumed_background_count,
                "assumed_background_predicted_positive_count": assumed_background_predicted,
                "assumed_background_predicted_positive_rate": safe_ratio(
                    assumed_background_predicted,
                    assumed_background_count,
                ),
            }
        )
        output.append(row)
    return output


def write_binary_full_grid_map(binary_config: BinaryPresenceConfig) -> None:
    """Write a map of binary model predictions for the primary test year."""
    map_rows = read_binary_map_rows(binary_config)
    if map_rows.empty:
        LOGGER.info("Skipping binary map; no selected rows were available")
        return
    binary_config.map_figure_path.parent.mkdir(parents=True, exist_ok=True)
    observed = map_rows["binary_observed_y"].to_numpy(dtype=bool)
    probability = map_rows["pred_binary_probability"].to_numpy(dtype=float)
    predicted = map_rows["pred_binary_class"].to_numpy(dtype=bool)
    outcome = binary_outcome_codes(observed, predicted)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    plot_binary_scatter_panel(
        axes[0],
        map_rows,
        observed.astype(float),
        title=f"Observed >=10% {primary_map_year(binary_config)}",
        cmap=ListedColormap(["#f1f5f9", "#1b9e77"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
        colorbar_ticks=[0, 1],
        colorbar_labels=["no", "yes"],
    )
    plot_binary_scatter_panel(
        axes[1],
        map_rows,
        probability,
        title="Predicted probability",
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    plot_binary_scatter_panel(
        axes[2],
        map_rows,
        predicted.astype(float),
        title="Selected class",
        cmap=ListedColormap(["#f1f5f9", "#1b9e77"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
        colorbar_ticks=[0, 1],
        colorbar_labels=["negative", "positive"],
    )
    plot_binary_scatter_panel(
        axes[3],
        map_rows,
        outcome,
        title="Classification outcome",
        cmap=ListedColormap(["#f1f5f9", "#1b9e77", "#e76f51", "#457b9d"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
        colorbar_ticks=[0, 1, 2, 3],
        colorbar_labels=["TN", "TP", "FP", "FN"],
    )
    fig.suptitle(
        f"{BINARY_MODEL_NAME} | {BINARY_TEST_SPLIT} | {primary_map_year(binary_config)} | "
        f"threshold {first_probability_threshold(map_rows):.2f}"
    )
    fig.savefig(binary_config.map_figure_path, dpi=180)
    plt.close(fig)
    LOGGER.info("Wrote binary full-grid map: %s", binary_config.map_figure_path)


def read_binary_map_rows(binary_config: BinaryPresenceConfig) -> pd.DataFrame:
    """Read binary full-grid prediction rows for the primary map split/year."""
    dataset = ds.dataset(binary_config.full_grid_predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    columns = [
        "split",
        "year",
        "longitude",
        "latitude",
        "binary_observed_y",
        "pred_binary_probability",
        "pred_binary_class",
        "probability_threshold",
        "label_source",
    ]
    selected_columns = [column for column in columns if column in set(dataset.schema.names)]
    expression = (dataset_field("split") == BINARY_TEST_SPLIT) & (
        dataset_field("year") == primary_map_year(binary_config)
    )
    return cast(
        pd.DataFrame, dataset.to_table(columns=selected_columns, filter=expression).to_pandas()
    )


def primary_map_year(binary_config: BinaryPresenceConfig) -> int:
    """Return the configured year used for the binary full-grid map."""
    if len(binary_config.test_years) == 1:
        return binary_config.test_years[0]
    return max(binary_config.test_years)


def dataset_field(name: str) -> Any:
    """Return a PyArrow dataset field expression with a typed wrapper."""
    return cast(Any, ds).field(name)


def binary_outcome_codes(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Return compact outcome codes: 0 TN, 1 TP, 2 FP, 3 FN."""
    codes = np.zeros(observed.shape, dtype=float)
    codes[observed & predicted] = 1
    codes[~observed & predicted] = 2
    codes[observed & ~predicted] = 3
    return codes


def plot_binary_scatter_panel(
    axis: Any,
    dataframe: pd.DataFrame,
    values: np.ndarray,
    *,
    title: str,
    cmap: str | ListedColormap,
    norm: Normalize | BoundaryNorm,
    colorbar_ticks: list[int] | None = None,
    colorbar_labels: list[str] | None = None,
) -> None:
    """Draw one binary model map panel."""
    artist = axis.scatter(
        dataframe["longitude"],
        dataframe["latitude"],
        c=values,
        s=0.35,
        marker="s",
        linewidths=0,
        alpha=0.88,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    set_map_bounds(axis, dataframe)
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_aspect("equal", adjustable="box")
    colorbar = plt.colorbar(artist, ax=axis, shrink=0.78, ticks=colorbar_ticks)
    if colorbar_ticks is not None and colorbar_labels is not None:
        colorbar.ax.set_yticklabels(colorbar_labels)


def set_map_bounds(axis: Any, dataframe: pd.DataFrame) -> None:
    """Set map bounds from selected prediction rows with a small pad."""
    longitudes = dataframe["longitude"].to_numpy(dtype=float)
    latitudes = dataframe["latitude"].to_numpy(dtype=float)
    longitude_min = float(np.nanmin(longitudes))
    longitude_max = float(np.nanmax(longitudes))
    latitude_min = float(np.nanmin(latitudes))
    latitude_max = float(np.nanmax(latitudes))
    longitude_pad = (longitude_max - longitude_min) * 0.04 or 0.01
    latitude_pad = (latitude_max - latitude_min) * 0.04 or 0.01
    axis.set_xlim(longitude_min - longitude_pad, longitude_max + longitude_pad)
    axis.set_ylim(latitude_min - latitude_pad, latitude_max + latitude_pad)


def label_source_series(dataframe: pd.DataFrame) -> pd.Series:
    """Return label-source values with a stable fallback."""
    if "label_source" in dataframe.columns:
        return dataframe["label_source"].fillna("unknown").astype(str)
    if "is_kelpwatch_observed" in dataframe.columns:
        observed = dataframe["is_kelpwatch_observed"].fillna(False).astype(bool)
        return pd.Series(
            np.where(observed, "kelpwatch_station", "assumed_background"),
            index=dataframe.index,
            dtype="object",
        )
    return pd.Series("all", index=dataframe.index, dtype="object")


def normalized_group_value(value: object) -> object:
    """Normalize pandas group keys for stable CSV output."""
    if pd.isna(value):
        return "unknown"
    return value


def binary_auroc(observed: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute AUROC, returning NaN when a group has one class."""
    valid = np.isfinite(probabilities)
    observed = observed[valid]
    probabilities = probabilities[valid]
    if observed.size == 0 or np.unique(observed).size < 2:
        return math.nan
    return float(roc_auc_score(observed, probabilities))


def binary_auprc(observed: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute AUPRC, returning NaN when a group has no positive rows."""
    valid = np.isfinite(probabilities)
    observed = observed[valid]
    probabilities = probabilities[valid]
    if observed.size == 0 or int(np.count_nonzero(observed)) == 0:
        return math.nan
    return float(average_precision_score(observed, probabilities))


def binary_brier_score(observed: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute Brier score, returning NaN when no finite probabilities exist."""
    valid = np.isfinite(probabilities)
    observed = observed[valid]
    probabilities = probabilities[valid]
    if observed.size == 0:
        return math.nan
    return float(brier_score_loss(observed, probabilities))


def expected_calibration_error(
    observed: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> float:
    """Compute fixed-bin expected calibration error for binary probabilities."""
    valid = np.isfinite(probabilities)
    observed = observed[valid].astype(float)
    probabilities = probabilities[valid].astype(float)
    if observed.size == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not mask.any():
            continue
        weight = float(np.count_nonzero(mask)) / float(observed.size)
        ece += weight * abs(float(np.mean(probabilities[mask])) - float(np.mean(observed[mask])))
    return ece


def finite_or_negative_inf(value: float) -> float:
    """Return a finite value or negative infinity for sorting."""
    return value if np.isfinite(value) else -math.inf


def finite_or_positive_inf(value: float) -> float:
    """Return a finite value or positive infinity for sorting."""
    return value if np.isfinite(value) else math.inf


def row_float(row: dict[str, object], field: str, *, default: float = math.nan) -> float:
    """Read a row value as float with a fallback for blanks."""
    value = row.get(field, default)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def row_int(row: dict[str, object], field: str) -> int:
    """Read a row value as int with a zero fallback."""
    value = row.get(field, 0)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return 0


def write_binary_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write sample binary predictions to a Parquet file."""
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
        writer.writerows(rows)


def write_binary_model(
    selection: BinaryModelSelection,
    threshold_selection: ThresholdSelection,
    binary_config: BinaryPresenceConfig,
    *,
    prepared: PreparedBinaryData,
) -> None:
    """Serialize the selected binary model and metadata."""
    binary_config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": selection.model,
        "model_name": BINARY_MODEL_NAME,
        "sample_policy": binary_config.sample_policy,
        "target_label": binary_config.target_label,
        "target_column": binary_config.target_column,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "feature_columns": list(binary_config.feature_columns),
        "class_weight": binary_config.class_weight,
        "selected_c": selection.selected_c,
        "probability_threshold": threshold_selection.threshold,
        "selection_split": BINARY_SELECTION_SPLIT,
        "selection_year": primary_selection_year(binary_config),
        "selection_policy": BINARY_THRESHOLD_POLICY,
        "split_source": prepared.split_source,
    }
    joblib.dump(payload, binary_config.model_output_path)


def write_binary_calibration_model(
    calibrator: BinaryCalibrator,
    threshold_selection: CalibrationThresholds,
    calibration_config: BinaryCalibrationConfig,
) -> None:
    """Serialize the fitted binary calibrator and compact metadata."""
    calibration_config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrator": calibrator.model,
        "model_name": BINARY_MODEL_NAME,
        "calibration_method": calibration_config.method,
        "calibration_status": calibrator.status,
        "calibration_split": calibration_config.calibration_split,
        "calibration_year": calibration_config.calibration_year,
        "evaluation_split": calibration_config.evaluation_split,
        "evaluation_year": calibration_config.evaluation_year,
        "probability_source": PLATT_PROBABILITY_SOURCE,
        "recommended_threshold": threshold_selection.recommended_threshold,
        "recommended_policy": threshold_selection.recommended_policy,
        "policy_thresholds": threshold_selection.policy_thresholds,
        "coefficient": calibrator.coefficient,
        "intercept": calibrator.intercept,
        "input_sample_predictions": str(calibration_config.input_sample_predictions_path),
    }
    joblib.dump(payload, calibration_config.model_output_path)


def write_prediction_manifest(
    *,
    prepared: PreparedBinaryData,
    selection: BinaryModelSelection,
    threshold_selection: ThresholdSelection,
    sample_predictions: pd.DataFrame,
    full_grid_summary: list[dict[str, object]],
    model_comparison: list[dict[str, object]],
    binary_config: BinaryPresenceConfig,
) -> None:
    """Write a compact manifest for binary model outputs."""
    existing: dict[str, object] = {}
    if binary_config.prediction_manifest_path.exists():
        existing = json.loads(binary_config.prediction_manifest_path.read_text())
    manifest = {
        **existing,
        "command": "train-binary-presence",
        "config": str(binary_config.config_path),
        "model_name": BINARY_MODEL_NAME,
        "sample_policy": binary_config.sample_policy,
        "target_label": binary_config.target_label,
        "target_column": binary_config.target_column,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "feature_columns": list(binary_config.feature_columns),
        "class_weight": binary_config.class_weight,
        "selected_c": selection.selected_c,
        "c_grid": list(binary_config.c_grid),
        "validation_model_selection": selection.validation_rows,
        "probability_threshold": threshold_selection.threshold,
        "threshold_selection_status": threshold_selection.status,
        "selection_split": BINARY_SELECTION_SPLIT,
        "selection_year": primary_selection_year(binary_config),
        "selection_policy": BINARY_THRESHOLD_POLICY,
        "split_source": prepared.split_source,
        "allow_missing_split_manifest_rows": binary_config.allow_missing_split_manifest_rows,
        "dropped_counts_by_split": prepared.dropped_counts_by_split,
        "sample_prediction_row_count": int(len(sample_predictions)),
        "sample_label_source_counts": label_source_series(sample_predictions)
        .value_counts()
        .to_dict(),
        "full_grid_summary_row_count": len(full_grid_summary),
        "thresholded_model_comparison_row_count": len(model_comparison),
        "mask_status": mask_status(binary_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(binary_config.reporting_domain_mask),
        "inputs": {
            "sample": str(binary_config.input_table_path),
            "split_manifest": str(binary_config.split_manifest_path),
            "inference_table": str(binary_config.inference_table_path),
        },
        "outputs": {
            "model": str(binary_config.model_output_path),
            "sample_predictions": str(binary_config.sample_predictions_path),
            "full_grid_predictions": str(binary_config.full_grid_predictions_path),
            "metrics": str(binary_config.metrics_path),
            "threshold_selection": str(binary_config.threshold_selection_path),
            "full_grid_area_summary": str(binary_config.full_grid_area_summary_path),
            "thresholded_model_comparison": str(binary_config.thresholded_model_comparison_path),
            "precision_recall_figure": str(binary_config.precision_recall_figure_path),
            "map_figure": str(binary_config.map_figure_path),
        },
    }
    binary_config.prediction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    binary_config.prediction_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def write_binary_calibration_manifest(
    *,
    calibrator: BinaryCalibrator,
    calibration_rows: pd.DataFrame,
    calibrated_sample: pd.DataFrame,
    threshold_selection: CalibrationThresholds,
    metrics: list[dict[str, object]],
    full_grid_summary: list[dict[str, object]],
    calibration_config: BinaryCalibrationConfig,
) -> None:
    """Write a compact manifest for binary calibration outputs."""
    label_source_counts = label_source_series(calibration_rows).value_counts().to_dict()
    assumed_background_count = int(
        np.count_nonzero(label_source_series(calibration_rows) == "assumed_background")
    )
    manifest = {
        "command": "calibrate-binary-presence",
        "config": str(calibration_config.config_path),
        "model_name": BINARY_MODEL_NAME,
        "target_label": calibration_config.binary_config.target_label,
        "target_threshold_fraction": calibration_config.binary_config.target_threshold_fraction,
        "target_threshold_area": calibration_config.binary_config.target_threshold_area,
        "calibration_method": calibration_config.method,
        "calibration_status": calibrator.status,
        "calibration_split": calibration_config.calibration_split,
        "calibration_year": calibration_config.calibration_year,
        "calibration_row_count": int(len(calibration_rows)),
        "calibration_label_source_counts": label_source_counts,
        "calibration_includes_assumed_background_negatives": assumed_background_count > 0,
        "assumed_background_calibration_row_count": assumed_background_count,
        "evaluation_split": calibration_config.evaluation_split,
        "evaluation_year": calibration_config.evaluation_year,
        "recommended_threshold_policy": threshold_selection.recommended_policy,
        "recommended_threshold": threshold_selection.recommended_threshold,
        "policy_thresholds": {
            policy: {"probability_source": source, "probability_threshold": threshold}
            for policy, (source, threshold) in threshold_selection.policy_thresholds.items()
        },
        "platt_coefficient": calibrator.coefficient,
        "platt_intercept": calibrator.intercept,
        "platt_monotonic_increasing": calibrator.coefficient >= 0,
        "sample_prediction_row_count": int(len(calibrated_sample)),
        "metrics_row_count": len(metrics),
        "full_grid_summary_row_count": len(full_grid_summary),
        "mask_status": mask_status(calibration_config.binary_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(
            calibration_config.binary_config.reporting_domain_mask
        ),
        "qa_notes": [
            (
                "The P1-18 2022 binary map has a visible false-positive cluster near or in "
                "the river mouth; this calibration task reports threshold effects but does "
                "not change the domain mask."
            )
        ],
        "inputs": {
            "sample_predictions": str(calibration_config.input_sample_predictions_path),
            "full_grid_predictions": str(calibration_config.input_full_grid_predictions_path),
        },
        "outputs": {
            "model": str(calibration_config.model_output_path),
            "calibrated_sample_predictions": str(
                calibration_config.calibrated_sample_predictions_path
            ),
            "metrics": str(calibration_config.metrics_path),
            "threshold_selection": str(calibration_config.threshold_selection_path),
            "full_grid_area_summary": str(calibration_config.full_grid_area_summary_path),
            "calibration_curve_figure": str(calibration_config.calibration_curve_figure_path),
            "threshold_figure": str(calibration_config.threshold_figure_path),
        },
    }
    calibration_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_precision_recall_figure(
    threshold_rows: list[dict[str, object]],
    binary_config: BinaryPresenceConfig,
) -> None:
    """Write a compact validation precision/recall/F1 threshold diagnostic figure."""
    binary_config.precision_recall_figure_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(threshold_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not frame.empty:
        ax.plot(
            frame["probability_threshold"], frame["precision"], label="precision", linewidth=1.6
        )
        ax.plot(frame["probability_threshold"], frame["recall"], label="recall", linewidth=1.6)
        ax.plot(frame["probability_threshold"], frame["f1"], label="F1", linewidth=1.8)
        selected = frame.loc[frame["selected_threshold"]]
        if not selected.empty:
            ax.axvline(
                float(selected.iloc[0]["probability_threshold"]),
                color="black",
                linestyle="--",
                linewidth=1.1,
                label="selected threshold",
            )
    ax.set_title("Validation Binary Annual-Max Threshold")
    ax.set_xlabel("Predicted probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(binary_config.precision_recall_figure_path, dpi=180)
    plt.close(fig)


def write_calibration_curve_figure(
    sample_predictions: pd.DataFrame,
    calibration_config: BinaryCalibrationConfig,
) -> None:
    """Write validation/test reliability curves for raw and calibrated probabilities."""
    calibration_config.calibration_curve_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot([0, 1], [0, 1], color="#334155", linestyle="--", linewidth=1.0, label="ideal")
    for split, year in (
        (calibration_config.calibration_split, calibration_config.calibration_year),
        (calibration_config.evaluation_split, calibration_config.evaluation_year),
    ):
        rows = calibration_metric_input_rows(sample_predictions, split=split, year=year)
        for label, column, style in (
            ("raw", "pred_binary_probability", ":"),
            ("platt", "calibrated_binary_probability", "-"),
        ):
            points = reliability_points(
                rows["binary_observed_y"].to_numpy(dtype=bool),
                rows[column].to_numpy(dtype=float),
                calibration_config.reliability_bin_count,
            )
            if points.empty:
                continue
            ax.plot(
                points["mean_probability"],
                points["observed_rate"],
                linestyle=style,
                marker="o",
                markersize=3.0,
                linewidth=1.4,
                label=f"{split} {year} {label}",
            )
    ax.set_title("Binary Annual-Max Probability Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed annual-max >=10% rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(calibration_config.calibration_curve_figure_path, dpi=180)
    plt.close(fig)


def reliability_points(
    observed: np.ndarray,
    probabilities: np.ndarray,
    bin_count: int,
) -> pd.DataFrame:
    """Return non-empty fixed-bin reliability points."""
    valid = np.isfinite(probabilities)
    observed_float = observed[valid].astype(float)
    probability_values = probabilities[valid].astype(float)
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    rows: list[dict[str, float]] = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if upper == 1.0:
            mask = (probability_values >= lower) & (probability_values <= upper)
        else:
            mask = (probability_values >= lower) & (probability_values < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "mean_probability": float(np.mean(probability_values[mask])),
                "observed_rate": float(np.mean(observed_float[mask])),
                "row_count": float(np.count_nonzero(mask)),
            }
        )
    return pd.DataFrame(rows)


def write_calibrated_threshold_figure(
    threshold_rows: list[dict[str, object]],
    calibration_config: BinaryCalibrationConfig,
) -> None:
    """Write a calibrated validation precision/recall/F1 threshold figure."""
    calibration_config.threshold_figure_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [row for row in threshold_rows if row.get("threshold_policy") == CALIBRATED_MAX_F1_POLICY]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if not frame.empty:
        ax.plot(
            frame["probability_threshold"],
            frame["precision"],
            label="precision",
            linewidth=1.6,
        )
        ax.plot(frame["probability_threshold"], frame["recall"], label="recall", linewidth=1.6)
        ax.plot(frame["probability_threshold"], frame["f1"], label="F1", linewidth=1.8)
        selected = frame.loc[frame["selected_threshold"].astype(bool)]
        if not selected.empty:
            ax.axvline(
                float(selected.iloc[0]["probability_threshold"]),
                color="black",
                linestyle="--",
                linewidth=1.1,
                label="selected threshold",
            )
    ax.set_title("Calibrated Validation Binary Annual-Max Threshold")
    ax.set_xlabel("Calibrated probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(calibration_config.threshold_figure_path, dpi=180)
    plt.close(fig)
