"""Analyze smoke-test model behavior and write a Phase 1 report."""
# ruff: noqa: E501

from __future__ import annotations

import base64
import csv
import html
import json
import logging
import math
import operator
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
import xarray as xr
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.evaluation.baselines import (
    KELPWATCH_PIXEL_AREA_M2,
    load_baseline_config,
    parse_bands,
    percent_bias,
    precision_recall_f1,
    root_mean_squared_error,
    safe_ratio,
    write_reference_area_calibration,
)
from kelp_aef.labels.kelpwatch import NETCDF_ENGINE

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ridge_regression"
DEFAULT_ANALYSIS_SPLIT = "test"
DEFAULT_ANALYSIS_YEAR = 2022
DEFAULT_LATITUDE_BAND_COUNT = 12
DEFAULT_MAX_PROJECTION_ROWS = 50_000
DEFAULT_FALL_QUARTER = 4
DEFAULT_WINTER_QUARTER = 1
DEFAULT_OBSERVED_AREA_BINS = (0.0, 1.0, 90.0, 225.0, 450.0, 810.0, 900.0)
DEFAULT_THRESHOLD_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.50, 0.90)
PERSISTENCE_CLASS_ORDER = (
    "no_quarter_present",
    "transient_one_quarter",
    "intermittent_two_or_three_quarters",
    "persistent_all_valid_quarters",
    "missing_quarters",
)
PDF_PAGE_WIDTH_IN = 8.5
PDF_PAGE_HEIGHT_IN = 11.0
PDF_MARGIN_X = 0.055
PDF_TOP_Y = 0.965
PDF_BODY_FONT_SIZE = 8.3
PDF_MONO_FONT_SIZE = 7.2
PDF_LINE_HEIGHT = 0.0185
QUARTER_COLUMNS = ("area_q1", "area_q2", "area_q3", "area_q4")
REQUIRED_LABEL_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_max_y",
    "kelp_fraction_y",
    *QUARTER_COLUMNS,
    "valid_quarter_count",
    "nonzero_quarter_count",
)
REQUIRED_PREDICTION_COLUMNS = (
    "year",
    "split",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_max_y",
    "kelp_fraction_y",
    "model_name",
    "pred_kelp_fraction_y_clipped",
    "pred_kelp_max_y",
    "residual_kelp_max_y",
)
OPTIONAL_PREDICTION_COLUMNS = (
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "label_source",
    "is_kelpwatch_observed",
    "kelpwatch_station_count",
)
STAGE_DISTRIBUTION_FIELDS = (
    "stage",
    "split",
    "year",
    "row_count",
    "station_count",
    "valid_count",
    "missing_count",
    "zero_count",
    "zero_fraction",
    "positive_count",
    "positive_fraction",
    "saturated_count",
    "saturated_fraction",
    "min",
    "median",
    "p90",
    "p95",
    "p99",
    "max",
    "aggregate_canopy_area",
)
PREDICTION_DISTRIBUTION_FIELDS = (
    "model_name",
    "split",
    "year",
    "row_count",
    "observed_zero_count",
    "observed_positive_count",
    "observed_saturated_count",
    "observed_p95",
    "observed_p99",
    "observed_max",
    "predicted_p50",
    "predicted_p95",
    "predicted_p99",
    "predicted_p999",
    "predicted_max",
    "predicted_ge_810_count",
    "observed_900_prediction_mean",
    "observed_900_prediction_p95",
    "observed_900_prediction_max",
)
RESIDUAL_BIN_FIELDS = (
    "model_name",
    "split",
    "year",
    "observed_bin",
    "row_count",
    "observed_mean",
    "predicted_mean",
    "mean_residual",
    "median_residual",
    "mae",
    "rmse",
    "underprediction_count",
    "overprediction_count",
    "saturated_count",
)
PERSISTENCE_FIELDS = (
    "model_name",
    "split",
    "year",
    "persistence_class",
    "row_count",
    "observed_mean",
    "predicted_mean",
    "mean_residual",
    "mae",
    "saturated_count",
)
THRESHOLD_FIELDS = (
    "model_name",
    "split",
    "year",
    "threshold_fraction",
    "threshold_area",
    "positive_count",
    "positive_fraction",
    "predicted_positive_count",
    "predicted_positive_fraction",
    "observed_area_positive_rows",
    "predicted_area_positive_rows",
    "precision",
    "recall",
    "f1",
)
TARGET_FRAMING_FIELDS = (
    "target_name",
    "target_kind",
    "row_count",
    "valid_count",
    "mean",
    "median",
    "p95",
    "positive_fraction",
    "pearson_with_prediction",
    "spearman_with_prediction",
    "mae_vs_prediction_area",
    "area_bias_vs_prediction",
    "area_pct_bias_vs_prediction",
)
SPATIAL_READINESS_FIELDS = (
    "latitude_band",
    "latitude_min",
    "latitude_max",
    "row_count",
    "station_count",
    "zero_count",
    "positive_count",
    "high_canopy_count",
    "saturated_count",
    "observed_area",
    "predicted_area",
    "mean_residual",
    "top_abs_residual_count",
    "enough_for_holdout",
)
FEATURE_SEPARABILITY_FIELDS = (
    "label_group",
    "row_count",
    "pc1_mean",
    "pc2_mean",
    "pc1_std",
    "pc2_std",
    "distance_from_zero_group",
)
QUARTER_MAPPING_FIELDS = (
    "source_time",
    "source_year",
    "source_quarter",
    "derived_year",
    "derived_quarter",
    "season_label",
)
PHASE1_DECISION_FIELDS = (
    "branch",
    "evidence_status",
    "triggering_evidence",
    "proposed_next_tasks",
    "expected_artifacts",
    "decision_unlocked",
)
PHASE1_MODEL_COMPARISON_FIELDS = (
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
REFERENCE_AREA_CALIBRATION_FIELDS = (
    "model_name",
    "split",
    "year",
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
DATA_HEALTH_FIELDS = (
    "check_name",
    "split",
    "year",
    "label_source",
    "row_count",
    "reference_row_count",
    "rate",
    "detail",
)


@dataclass(frozen=True)
class ModelAnalysisConfig:
    """Resolved config values for model-analysis report generation."""

    config_path: Path
    data_root: Path
    label_path: Path
    label_manifest_path: Path
    source_manifest_path: Path | None
    aligned_table_path: Path
    split_manifest_path: Path
    predictions_path: Path
    metrics_path: Path
    model_name: str
    analysis_split: str
    analysis_year: int
    feature_columns: tuple[str, ...]
    observed_area_bins: tuple[float, ...]
    threshold_fractions: tuple[float, ...]
    latitude_band_count: int
    max_projection_rows: int
    fall_quarter: int
    winter_quarter: int
    report_path: Path
    html_report_path: Path
    pdf_report_path: Path
    manifest_path: Path
    label_distribution_path: Path
    target_framing_path: Path
    residual_by_bin_path: Path
    residual_by_persistence_path: Path
    prediction_distribution_path: Path
    threshold_sensitivity_path: Path
    spatial_readiness_path: Path
    feature_separability_path: Path
    phase1_decision_path: Path
    phase1_model_comparison_path: Path
    data_health_path: Path
    reference_area_calibration_path: Path
    fallback_summary_path: Path
    quarter_mapping_path: Path
    label_distribution_figure: Path
    observed_predicted_figure: Path
    residual_by_bin_figure: Path
    observed_900_figure: Path
    residual_by_persistence_figure: Path
    alternative_targets_figure: Path
    feature_projection_figure: Path
    spatial_readiness_figure: Path
    observed_predicted_residual_map_figure: Path
    residual_interactive_html: Path


@dataclass(frozen=True)
class AnalysisTables:
    """All tabular outputs needed to write the report and manifest."""

    stage_distribution: list[dict[str, object]]
    prediction_distribution: list[dict[str, object]]
    residual_by_bin: list[dict[str, object]]
    residual_by_persistence: list[dict[str, object]]
    target_framing: list[dict[str, object]]
    threshold_sensitivity: list[dict[str, object]]
    spatial_readiness: list[dict[str, object]]
    feature_separability: list[dict[str, object]]
    phase1_decision: list[dict[str, object]]
    phase1_model_comparison: list[dict[str, object]]
    data_health: list[dict[str, object]]
    quarter_mapping: list[dict[str, object]]
    reference_area_calibration: list[dict[str, object]]


@dataclass(frozen=True)
class AnalysisData:
    """Loaded model-analysis inputs after validation and model filtering."""

    labels: pd.DataFrame
    aligned: pd.DataFrame
    split_manifest: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    model_predictions: pd.DataFrame
    labels_with_targets: pd.DataFrame
    prediction_targets: pd.DataFrame


def analyze_model(config_path: Path) -> int:
    """Analyze the smoke-test model and write report artifacts."""
    analysis_config = load_model_analysis_config(config_path)
    LOGGER.info("Loading model-analysis inputs")
    data = load_analysis_data(analysis_config)
    reference_area_calibration = build_or_load_reference_area_calibration(analysis_config)
    tables = build_analysis_tables(data, analysis_config, reference_area_calibration)
    write_analysis_tables(tables, analysis_config)
    write_analysis_figures(data, tables, analysis_config)
    write_report(data, tables, analysis_config)
    write_manifest(data, tables, analysis_config)
    LOGGER.info("Wrote model-analysis report: %s", analysis_config.report_path)
    LOGGER.info("Wrote model-analysis HTML report: %s", analysis_config.html_report_path)
    LOGGER.info("Wrote model-analysis PDF report: %s", analysis_config.pdf_report_path)
    LOGGER.info("Wrote model-analysis manifest: %s", analysis_config.manifest_path)
    return 0


def load_model_analysis_config(config_path: Path) -> ModelAnalysisConfig:
    """Load model-analysis settings from the workflow config."""
    config = load_yaml_config(config_path)
    data_root = Path(require_string(config.get("data_root"), "data_root"))
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    features = require_mapping(config.get("features"), "features")
    alignment = require_mapping(config.get("alignment"), "alignment")
    splits = require_mapping(config.get("splits"), "splits")
    models = require_mapping(config.get("models"), "models")
    baselines = require_mapping(models.get("baselines"), "models.baselines")
    reports = require_mapping(config.get("reports"), "reports")
    outputs = require_mapping(reports.get("outputs"), "reports.outputs")
    settings = optional_mapping(reports.get("model_analysis"), "reports.model_analysis")
    figures_dir = Path(require_string(reports.get("figures_dir"), "reports.figures_dir"))
    tables_dir = Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))
    report_dir = (
        data_root / "reports/model_analysis"
        if settings.get("report_dir") is None
        else Path(require_string(settings.get("report_dir"), "reports.model_analysis.report_dir"))
    )
    return ModelAnalysisConfig(
        config_path=config_path,
        data_root=data_root,
        label_path=Path(
            require_string(label_paths.get("annual_labels"), "labels.paths.annual_labels")
        ),
        label_manifest_path=Path(
            require_string(
                label_paths.get("annual_label_manifest"),
                "labels.paths.annual_label_manifest",
            )
        ),
        source_manifest_path=optional_path(label_paths.get("source_manifest")),
        aligned_table_path=Path(
            require_string(
                baselines.get("input_table") or alignment.get("output_table"),
                "models.baselines.input_table or alignment.output_table",
            )
        ),
        split_manifest_path=Path(
            require_string(splits.get("output_manifest"), "splits.output_manifest")
        ),
        predictions_path=Path(
            require_string(baselines.get("predictions"), "models.baselines.predictions")
        ),
        metrics_path=Path(require_string(baselines.get("metrics"), "models.baselines.metrics")),
        model_name=str(settings.get("model_name", DEFAULT_MODEL_NAME)),
        analysis_split=str(settings.get("split", DEFAULT_ANALYSIS_SPLIT)),
        analysis_year=optional_int(
            settings.get("year"), "reports.model_analysis.year", DEFAULT_ANALYSIS_YEAR
        ),
        feature_columns=parse_bands(baselines.get("features") or features.get("bands")),
        observed_area_bins=read_float_tuple(
            settings.get("observed_area_bins"),
            "reports.model_analysis.observed_area_bins",
            DEFAULT_OBSERVED_AREA_BINS,
        ),
        threshold_fractions=read_float_tuple(
            settings.get("threshold_fractions"),
            "reports.model_analysis.threshold_fractions",
            DEFAULT_THRESHOLD_FRACTIONS,
        ),
        latitude_band_count=optional_positive_int(
            settings.get("latitude_band_count"),
            "reports.model_analysis.latitude_band_count",
            DEFAULT_LATITUDE_BAND_COUNT,
        ),
        max_projection_rows=optional_positive_int(
            settings.get("max_projection_rows"),
            "reports.model_analysis.max_projection_rows",
            DEFAULT_MAX_PROJECTION_ROWS,
        ),
        fall_quarter=optional_quarter(
            settings.get("fall_quarter"),
            "reports.model_analysis.fall_quarter",
            DEFAULT_FALL_QUARTER,
        ),
        winter_quarter=optional_quarter(
            settings.get("winter_quarter"),
            "reports.model_analysis.winter_quarter",
            DEFAULT_WINTER_QUARTER,
        ),
        report_path=output_path(
            outputs,
            "model_analysis_report",
            report_dir / "monterey_phase1_model_analysis.md",
        ),
        html_report_path=output_path(
            outputs,
            "model_analysis_html_report",
            report_dir / "monterey_phase1_model_analysis.html",
        ),
        pdf_report_path=output_path(
            outputs,
            "model_analysis_pdf_report",
            report_dir / "monterey_phase1_model_analysis.pdf",
        ),
        manifest_path=output_path(
            outputs,
            "model_analysis_manifest",
            data_root / "interim/model_analysis_manifest.json",
        ),
        label_distribution_path=output_path(
            outputs,
            "model_analysis_label_distribution_by_stage",
            tables_dir / "model_analysis_label_distribution_by_stage.csv",
        ),
        target_framing_path=output_path(
            outputs,
            "model_analysis_target_framing_summary",
            tables_dir / "model_analysis_target_framing_summary.csv",
        ),
        residual_by_bin_path=output_path(
            outputs,
            "model_analysis_residual_by_observed_bin",
            tables_dir / "model_analysis_residual_by_observed_bin.csv",
        ),
        residual_by_persistence_path=output_path(
            outputs,
            "model_analysis_residual_by_persistence",
            tables_dir / "model_analysis_residual_by_persistence.csv",
        ),
        prediction_distribution_path=output_path(
            outputs,
            "model_analysis_prediction_distribution",
            tables_dir / "model_analysis_prediction_distribution.csv",
        ),
        threshold_sensitivity_path=output_path(
            outputs,
            "model_analysis_threshold_sensitivity",
            tables_dir / "model_analysis_threshold_sensitivity.csv",
        ),
        spatial_readiness_path=output_path(
            outputs,
            "model_analysis_spatial_holdout_readiness",
            tables_dir / "model_analysis_spatial_holdout_readiness.csv",
        ),
        feature_separability_path=output_path(
            outputs,
            "model_analysis_feature_separability",
            tables_dir / "model_analysis_feature_separability.csv",
        ),
        phase1_decision_path=output_path(
            outputs,
            "model_analysis_phase1_decision_matrix",
            tables_dir / "model_analysis_phase1_decision_matrix.csv",
        ),
        phase1_model_comparison_path=output_path(
            outputs,
            "model_analysis_phase1_model_comparison",
            tables_dir / "model_analysis_phase1_model_comparison.csv",
        ),
        data_health_path=output_path(
            outputs,
            "model_analysis_data_health",
            tables_dir / "model_analysis_data_health.csv",
        ),
        reference_area_calibration_path=output_path(
            outputs,
            "reference_baseline_area_calibration",
            tables_dir / "reference_baseline_area_calibration.csv",
        ),
        fallback_summary_path=output_path(
            outputs,
            "reference_baseline_fallback_summary",
            tables_dir / "reference_baseline_fallback_summary.csv",
        ),
        quarter_mapping_path=output_path(
            outputs,
            "model_analysis_quarter_mapping",
            tables_dir / "model_analysis_quarter_mapping.csv",
        ),
        label_distribution_figure=output_path(
            outputs,
            "model_analysis_label_distribution_figure",
            figures_dir / "model_analysis_label_distribution_by_stage.png",
        ),
        observed_predicted_figure=output_path(
            outputs,
            "model_analysis_observed_predicted_figure",
            figures_dir / "model_analysis_observed_vs_predicted_distribution.png",
        ),
        residual_by_bin_figure=output_path(
            outputs,
            "model_analysis_residual_by_bin_figure",
            figures_dir / "model_analysis_residual_by_observed_bin.png",
        ),
        observed_900_figure=output_path(
            outputs,
            "model_analysis_observed_900_figure",
            figures_dir / "model_analysis_observed_900_predictions.png",
        ),
        residual_by_persistence_figure=output_path(
            outputs,
            "model_analysis_residual_by_persistence_figure",
            figures_dir / "model_analysis_residual_by_persistence.png",
        ),
        alternative_targets_figure=output_path(
            outputs,
            "model_analysis_alternative_targets_figure",
            figures_dir / "model_analysis_alternative_target_framings.png",
        ),
        feature_projection_figure=output_path(
            outputs,
            "model_analysis_feature_projection_figure",
            figures_dir / "model_analysis_feature_projection.png",
        ),
        spatial_readiness_figure=output_path(
            outputs,
            "model_analysis_spatial_readiness_figure",
            figures_dir / "model_analysis_spatial_holdout_readiness.png",
        ),
        observed_predicted_residual_map_figure=output_path(
            outputs,
            "ridge_observed_predicted_residual_map",
            figures_dir / "ridge_2022_observed_predicted_residual.png",
        ),
        residual_interactive_html=output_path(
            outputs,
            "ridge_residual_interactive",
            figures_dir / "ridge_2022_residual_interactive.html",
        ),
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating missing values as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def optional_path(value: object) -> Path | None:
    """Return an optional path from a dynamic config value."""
    if value is None:
        return None
    return Path(str(value))


def output_path(outputs: dict[str, Any], key: str, default: Path) -> Path:
    """Read an optional report output path from config."""
    value = outputs.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"reports.outputs.{key}"))


def read_float_tuple(value: object, name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Read an optional list of numeric config values."""
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        msg = f"config field must be a non-empty list of numbers: {name}"
        raise ValueError(msg)
    return tuple(float(item) for item in value)


def optional_int(value: object, name: str, default: int) -> int:
    """Read an optional integer config value."""
    if value is None:
        return default
    return require_int_value(value, name)


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Read an optional positive integer config value."""
    parsed = optional_int(value, name, default)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def optional_quarter(value: object, name: str, default: int) -> int:
    """Read an optional quarter value from one to four."""
    parsed = optional_int(value, name, default)
    if parsed not in {1, 2, 3, 4}:
        msg = f"quarter must be one of 1, 2, 3, or 4: {name}"
        raise ValueError(msg)
    return parsed


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like dynamic value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))


def load_analysis_data(analysis_config: ModelAnalysisConfig) -> AnalysisData:
    """Load and validate all tabular inputs for model analysis."""
    labels = pd.read_parquet(analysis_config.label_path)
    aligned = pd.read_parquet(analysis_config.aligned_table_path)
    split_manifest = pd.read_parquet(analysis_config.split_manifest_path)
    predictions = read_model_prediction_rows(analysis_config)
    metrics = pd.read_csv(analysis_config.metrics_path)
    validate_columns(labels, REQUIRED_LABEL_COLUMNS, "annual labels")
    validate_columns(aligned, REQUIRED_LABEL_COLUMNS, "aligned table")
    validate_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "baseline predictions")
    model_predictions = predictions.loc[
        predictions["model_name"] == analysis_config.model_name
    ].copy()
    if model_predictions.empty:
        msg = f"no prediction rows found for model: {analysis_config.model_name}"
        raise ValueError(msg)
    labels_with_targets = add_target_framings(labels, analysis_config)
    prediction_targets = attach_target_framings(
        kelpwatch_supported_prediction_rows(model_predictions),
        labels_with_targets,
    )
    LOGGER.info(
        "Loaded %s labels, %s aligned rows, %s split rows, and %s %s prediction rows",
        len(labels),
        len(aligned),
        len(split_manifest),
        len(model_predictions),
        analysis_config.model_name,
    )
    return AnalysisData(
        labels=labels,
        aligned=aligned,
        split_manifest=split_manifest,
        predictions=predictions,
        metrics=metrics,
        model_predictions=model_predictions,
        labels_with_targets=labels_with_targets,
        prediction_targets=prediction_targets,
    )


def build_or_load_reference_area_calibration(
    analysis_config: ModelAnalysisConfig,
) -> list[dict[str, object]]:
    """Build cached full-grid reference area rows, falling back to an existing table."""
    cached_rows = fresh_reference_area_calibration_rows(analysis_config)
    if cached_rows is not None:
        return cached_rows
    try:
        return write_reference_area_calibration(analysis_config.config_path)
    except FileNotFoundError as error:
        LOGGER.info("Skipping reference area calibration; input artifact is missing: %s", error)
    except ValueError as error:
        if not is_skippable_reference_calibration_error(error):
            raise
        LOGGER.info("Skipping reference area calibration; config is incomplete: %s", error)
    return read_reference_area_calibration(analysis_config.reference_area_calibration_path)


def fresh_reference_area_calibration_rows(
    analysis_config: ModelAnalysisConfig,
) -> list[dict[str, object]] | None:
    """Return cached reference area rows when the output is newer than its inputs."""
    output_path = analysis_config.reference_area_calibration_path
    if not output_path.exists():
        return None
    try:
        baseline_config = load_baseline_config(analysis_config.config_path)
    except ValueError as error:
        if not is_skippable_reference_calibration_error(error):
            raise
        LOGGER.info("Using existing reference area calibration: %s", output_path)
        return read_reference_area_calibration(output_path)
    input_paths = [
        analysis_config.config_path,
        baseline_config.model_output_path,
        baseline_config.geographic_model_output_path,
    ]
    if baseline_config.inference_table_path is not None:
        input_paths.append(baseline_config.inference_table_path)
    output_mtime = output_path.stat().st_mtime
    for input_path in input_paths:
        if input_path.exists() and input_path.stat().st_mtime > output_mtime:
            return None
    LOGGER.info("Using cached reference area calibration: %s", output_path)
    return read_reference_area_calibration(output_path)


def is_skippable_reference_calibration_error(error: ValueError) -> bool:
    """Return whether a config error means the optional calibration stage is unavailable."""
    message = str(error)
    return any(
        token in message
        for token in (
            "models.baselines.target",
            "models.baselines.alpha_grid",
            "models.baselines.ridge_model",
            "models.baselines.input_table",
            "models.baselines.metrics",
            "splits.train_years",
            "splits.validation_years",
            "splits.test_years",
        )
    )


def read_reference_area_calibration(path: Path) -> list[dict[str, object]]:
    """Read an existing reference area-calibration table when available."""
    if not path.exists():
        return []
    return cast(list[dict[str, object]], pd.read_csv(path).to_dict("records"))


def read_model_prediction_rows(analysis_config: ModelAnalysisConfig) -> pd.DataFrame:
    """Read prediction rows needed for model analysis without loading all partitions."""
    columns = available_prediction_columns(analysis_config.predictions_path)
    dataset = ds.dataset(analysis_config.predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    expression = dataset_field("model_name") == analysis_config.model_name
    if analysis_config.predictions_path.is_dir():
        LOGGER.info(
            "Prediction path is a dataset directory; loading primary split/year only: split=%s year=%s",
            analysis_config.analysis_split,
            analysis_config.analysis_year,
        )
        expression = (
            expression
            & (dataset_field("split") == analysis_config.analysis_split)
            & (dataset_field("year") == analysis_config.analysis_year)
        )
    table = dataset.to_table(columns=columns, filter=expression)
    frame = table.to_pandas()
    LOGGER.info("Loaded %s prediction rows from %s", len(frame), analysis_config.predictions_path)
    return cast(pd.DataFrame, frame)


def available_prediction_columns(path: Path) -> list[str]:
    """Return required and optional prediction columns available in a Parquet path."""
    columns = [*REQUIRED_PREDICTION_COLUMNS, *OPTIONAL_PREDICTION_COLUMNS]
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    available_columns = set(dataset.schema.names)
    return [column for column in columns if column in available_columns]


def dataset_field(name: str) -> Any:
    """Return a PyArrow dataset field expression with a typed wrapper."""
    return cast(Any, ds).field(name)


def kelpwatch_supported_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select prediction rows that have true Kelpwatch station-label support."""
    if "label_source" in predictions.columns:
        return predictions.loc[predictions["label_source"] == "kelpwatch_station"].copy()
    if "is_kelpwatch_observed" in predictions.columns:
        observed = predictions["is_kelpwatch_observed"].fillna(False).astype(bool)
        return predictions.loc[observed].copy()
    return predictions.loc[predictions["kelpwatch_station_id"].notna()].copy()


def validate_columns(dataframe: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    """Validate that a dataframe has expected columns."""
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        msg = f"{name} is missing required columns: {missing}"
        raise ValueError(msg)


def add_target_framings(labels: pd.DataFrame, analysis_config: ModelAnalysisConfig) -> pd.DataFrame:
    """Add alternative target framing columns to annual labels."""
    frame = labels.copy()
    quarter_values = frame[list(QUARTER_COLUMNS)].to_numpy(dtype=float)
    valid_counts = frame["valid_quarter_count"].to_numpy(dtype=float)
    nonzero_counts = frame["nonzero_quarter_count"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        frame["kelp_mean_area_y"] = np.nanmean(quarter_values, axis=1)
        frame["kelp_min_area_y"] = np.nanmin(quarter_values, axis=1)
    frame["mean_presence_fraction_y"] = np.divide(
        nonzero_counts,
        valid_counts,
        out=np.full(nonzero_counts.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    frame["persistent_presence_y"] = (valid_counts > 0) & (nonzero_counts == valid_counts)
    frame["any_presence_y"] = nonzero_counts > 0
    fall_column = quarter_column(analysis_config.fall_quarter)
    winter_column = quarter_column(analysis_config.winter_quarter)
    frame["fall_area_y"] = frame[fall_column].astype(float)
    frame["winter_area_y"] = frame[winter_column].astype(float)
    frame["fall_presence_y"] = frame["fall_area_y"] > 0
    frame["winter_presence_y"] = frame["winter_area_y"] > 0
    frame["fall_minus_winter_area_y"] = frame["fall_area_y"] - frame["winter_area_y"]
    for threshold in analysis_config.threshold_fractions:
        suffix = threshold_suffix(threshold)
        if threshold == 0:
            frame[f"high_presence_ge_{suffix}_y"] = frame["kelp_fraction_y"] > 0
        else:
            frame[f"high_presence_ge_{suffix}_y"] = frame["kelp_fraction_y"] >= threshold
    return frame


def quarter_column(quarter: int) -> str:
    """Return the annual label column name for a configured quarter."""
    return f"area_q{quarter}"


def threshold_suffix(threshold: float) -> str:
    """Return a stable threshold suffix for column and table labels."""
    if threshold == 0:
        return "gt0"
    return f"{int(round(threshold * 100)):02d}pct"


def attach_target_framings(
    predictions: pd.DataFrame, labels_with_targets: pd.DataFrame
) -> pd.DataFrame:
    """Join alternative target framing columns onto prediction rows."""
    target_columns = [
        "year",
        "kelpwatch_station_id",
        "kelp_mean_area_y",
        "kelp_min_area_y",
        "mean_presence_fraction_y",
        "persistent_presence_y",
        "any_presence_y",
        "fall_area_y",
        "winter_area_y",
        "fall_presence_y",
        "winter_presence_y",
        "fall_minus_winter_area_y",
        "valid_quarter_count",
        "nonzero_quarter_count",
    ]
    high_presence_columns = [
        column for column in labels_with_targets.columns if column.startswith("high_presence_ge_")
    ]
    return predictions.merge(
        labels_with_targets[target_columns + high_presence_columns],
        on=["year", "kelpwatch_station_id"],
        how="left",
        validate="many_to_one",
    )


def build_analysis_tables(
    data: AnalysisData,
    analysis_config: ModelAnalysisConfig,
    reference_area_calibration: list[dict[str, object]],
) -> AnalysisTables:
    """Build all analysis tables from loaded inputs."""
    prediction_distribution = build_prediction_distribution(data.model_predictions)
    residual_by_bin = build_residual_by_observed_bin(
        data.model_predictions, analysis_config.observed_area_bins
    )
    residual_by_persistence = build_residual_by_persistence(data.prediction_targets)
    threshold_sensitivity = build_threshold_sensitivity(
        data.model_predictions, analysis_config.threshold_fractions
    )
    spatial_readiness = build_spatial_readiness(
        data.model_predictions, analysis_config.latitude_band_count
    )
    feature_separability, projection_frame = build_feature_separability(
        data.aligned, analysis_config
    )
    target_framing = build_target_framing_summary(data.prediction_targets, analysis_config)
    phase1_decision = build_phase1_decision_matrix(
        data=data,
        prediction_distribution=prediction_distribution,
        residual_by_bin=residual_by_bin,
        spatial_readiness=spatial_readiness,
        feature_separability=feature_separability,
    )
    phase1_model_comparison = build_phase1_model_comparison(
        data, analysis_config, reference_area_calibration
    )
    data_health = build_data_health_rows(data, analysis_config)
    quarter_mapping = build_quarter_mapping(analysis_config)
    data.aligned.attrs["model_analysis_projection_frame"] = projection_frame
    return AnalysisTables(
        stage_distribution=build_stage_distribution(data),
        prediction_distribution=prediction_distribution,
        residual_by_bin=residual_by_bin,
        residual_by_persistence=residual_by_persistence,
        target_framing=target_framing,
        threshold_sensitivity=threshold_sensitivity,
        spatial_readiness=spatial_readiness,
        feature_separability=feature_separability,
        phase1_decision=phase1_decision,
        phase1_model_comparison=phase1_model_comparison,
        data_health=data_health,
        quarter_mapping=quarter_mapping,
        reference_area_calibration=reference_area_calibration,
    )


def build_stage_distribution(data: AnalysisData) -> list[dict[str, object]]:
    """Build label distribution rows for each pipeline stage."""
    rows: list[dict[str, object]] = []
    rows.extend(distribution_rows("annual_labels", data.labels, split_column=None))
    rows.extend(distribution_rows("model_input_sample", data.aligned, split_column=None))
    split_rows = split_manifest_with_area(data.split_manifest)
    rows.extend(distribution_rows("split_manifest_all", split_rows, split_column="split"))
    retained = split_rows.loc[split_rows["used_for_training_eval"]].copy()
    rows.extend(distribution_rows("retained_model_rows", retained, split_column="split"))
    rows.extend(
        distribution_rows("ridge_prediction_rows", data.model_predictions, split_column="split")
    )
    return rows


def split_manifest_with_area(split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Add area target columns to the split manifest for distribution summaries."""
    frame = split_manifest.copy()
    frame["kelp_max_y"] = frame["kelp_fraction_y"].astype(float) * KELPWATCH_PIXEL_AREA_M2
    return frame


def distribution_rows(
    stage: str, dataframe: pd.DataFrame, *, split_column: str | None
) -> list[dict[str, object]]:
    """Build stage distribution rows grouped by available year and split columns."""
    group_columns = ["year"] if split_column is None else [split_column, "year"]
    rows: list[dict[str, object]] = []
    for keys, group in dataframe.groupby(group_columns, sort=True, dropna=False):
        if split_column is None:
            split = "all"
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            year = int(cast(Any, key_tuple[0]))
        else:
            split, year = cast(tuple[str, int], keys)
        rows.append(distribution_row(stage, str(split), int(year), group))
    return rows


def distribution_row(
    stage: str, split: str, year: int, dataframe: pd.DataFrame
) -> dict[str, object]:
    """Build one distribution row for a target area series."""
    values = dataframe["kelp_max_y"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    zero_count = int(np.count_nonzero(finite == 0))
    positive_count = int(np.count_nonzero(finite > 0))
    saturated_count = int(np.count_nonzero(finite >= KELPWATCH_PIXEL_AREA_M2))
    row_count = int(len(dataframe))
    valid_count = int(finite.size)
    return {
        "stage": stage,
        "split": split,
        "year": year,
        "row_count": row_count,
        "station_count": int(dataframe["kelpwatch_station_id"].nunique()),
        "valid_count": valid_count,
        "missing_count": row_count - valid_count,
        "zero_count": zero_count,
        "zero_fraction": safe_ratio(zero_count, valid_count),
        "positive_count": positive_count,
        "positive_fraction": safe_ratio(positive_count, valid_count),
        "saturated_count": saturated_count,
        "saturated_fraction": safe_ratio(saturated_count, valid_count),
        "min": safe_percentile(finite, 0),
        "median": safe_percentile(finite, 50),
        "p90": safe_percentile(finite, 90),
        "p95": safe_percentile(finite, 95),
        "p99": safe_percentile(finite, 99),
        "max": safe_percentile(finite, 100),
        "aggregate_canopy_area": float(np.nansum(finite)) if finite.size else math.nan,
    }


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a percentile or NaN for empty arrays."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanpercentile(finite, percentile))


def build_prediction_distribution(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Build observed and predicted distribution rows by split and year."""
    rows: list[dict[str, object]] = []
    for keys, group in dataframe.groupby(["model_name", "split", "year"], sort=True):
        model_name, split, year = cast(tuple[str, str, int], keys)
        observed = group["kelp_max_y"].to_numpy(dtype=float)
        predicted = group["pred_kelp_max_y"].to_numpy(dtype=float)
        observed_900_predictions = group.loc[
            group["kelp_max_y"] >= KELPWATCH_PIXEL_AREA_M2, "pred_kelp_max_y"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": str(model_name),
                "split": str(split),
                "year": int(year),
                "row_count": int(len(group)),
                "observed_zero_count": int(np.count_nonzero(observed == 0)),
                "observed_positive_count": int(np.count_nonzero(observed > 0)),
                "observed_saturated_count": int(
                    np.count_nonzero(observed >= KELPWATCH_PIXEL_AREA_M2)
                ),
                "observed_p95": safe_percentile(observed, 95),
                "observed_p99": safe_percentile(observed, 99),
                "observed_max": safe_percentile(observed, 100),
                "predicted_p50": safe_percentile(predicted, 50),
                "predicted_p95": safe_percentile(predicted, 95),
                "predicted_p99": safe_percentile(predicted, 99),
                "predicted_p999": safe_percentile(predicted, 99.9),
                "predicted_max": safe_percentile(predicted, 100),
                "predicted_ge_810_count": int(np.count_nonzero(predicted >= 810)),
                "observed_900_prediction_mean": safe_mean(observed_900_predictions),
                "observed_900_prediction_p95": safe_percentile(observed_900_predictions, 95),
                "observed_900_prediction_max": safe_percentile(observed_900_predictions, 100),
            }
        )
    return rows


def safe_mean(values: np.ndarray) -> float:
    """Return a NaN-aware mean or NaN for empty arrays."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(finite))


def object_to_float(value: object, default: float = math.nan) -> float:
    """Convert a generic row value to float with a fallback for missing values."""
    if value is None:
        return default
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def object_to_int(value: object, default: int = 0) -> int:
    """Convert a generic row value to int with a fallback for missing values."""
    if value is None:
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


def row_float(row: dict[str, object], key: str, default: float = math.nan) -> float:
    """Read one generic row value as a float."""
    return object_to_float(row.get(key), default)


def row_int(row: dict[str, object], key: str, default: int = 0) -> int:
    """Read one generic row value as an integer."""
    return object_to_int(row.get(key), default)


def mean_absolute_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute a mean absolute error for paired arrays."""
    return float(np.nanmean(np.abs(observed - predicted)))


def build_residual_by_observed_bin(
    dataframe: pd.DataFrame, observed_bins: tuple[float, ...]
) -> list[dict[str, object]]:
    """Build residual summary rows grouped by observed canopy bins."""
    frame = dataframe.copy()
    frame["observed_bin"] = observed_area_bin_labels(frame["kelp_max_y"], observed_bins)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["model_name", "split", "year", "observed_bin"], sort=True, observed=True
    ):
        model_name, split, year, observed_bin = cast(tuple[str, str, int, str], keys)
        rows.append(
            residual_bin_row(str(model_name), str(split), int(year), str(observed_bin), group)
        )
    return rows


def observed_area_bin_labels(values: pd.Series, observed_bins: tuple[float, ...]) -> pd.Series:
    """Assign readable observed-area bin labels with exact zero isolated."""
    bins = sorted(set(float(value) for value in observed_bins if value > 0))
    labels: list[str] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            labels.append("missing")
        elif value == 0:
            labels.append("000_zero")
        else:
            lower = 0.0
            selected = f">{bins[-1]:g}" if bins else ">0"
            for upper in bins:
                if value <= upper:
                    selected = f"({lower:g}, {upper:g}]"
                    break
                lower = upper
            labels.append(selected)
    return pd.Series(
        pd.Categorical(labels, categories=observed_area_bin_order(observed_bins), ordered=True),
        index=values.index,
    )


def observed_area_bin_order(observed_bins: tuple[float, ...]) -> list[str]:
    """Return the numeric display order for observed-area bins."""
    bins = sorted(set(float(value) for value in observed_bins if value > 0))
    labels = ["000_zero"]
    lower = 0.0
    for upper in bins:
        labels.append(f"({lower:g}, {upper:g}]")
        lower = upper
    labels.append(f">{bins[-1]:g}" if bins else ">0")
    labels.append("missing")
    return labels


def residual_bin_row(
    model_name: str, split: str, year: int, observed_bin: str, group: pd.DataFrame
) -> dict[str, object]:
    """Build one residual-by-observed-bin row."""
    observed = group["kelp_max_y"].to_numpy(dtype=float)
    predicted = group["pred_kelp_max_y"].to_numpy(dtype=float)
    residual = group["residual_kelp_max_y"].to_numpy(dtype=float)
    return {
        "model_name": model_name,
        "split": split,
        "year": year,
        "observed_bin": observed_bin,
        "row_count": int(len(group)),
        "observed_mean": safe_mean(observed),
        "predicted_mean": safe_mean(predicted),
        "mean_residual": safe_mean(residual),
        "median_residual": safe_percentile(residual, 50),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "underprediction_count": int(np.count_nonzero(residual > 0)),
        "overprediction_count": int(np.count_nonzero(residual < 0)),
        "saturated_count": int(np.count_nonzero(observed >= KELPWATCH_PIXEL_AREA_M2)),
    }


def build_residual_by_persistence(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Build residual rows grouped by quarterly persistence class."""
    frame = dataframe.copy()
    frame["persistence_class"] = persistence_classes(frame)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["model_name", "split", "year", "persistence_class"], sort=True
    ):
        model_name, split, year, persistence_class = cast(tuple[str, str, int, str], keys)
        observed = group["kelp_max_y"].to_numpy(dtype=float)
        predicted = group["pred_kelp_max_y"].to_numpy(dtype=float)
        residual = group["residual_kelp_max_y"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": str(model_name),
                "split": str(split),
                "year": int(year),
                "persistence_class": str(persistence_class),
                "row_count": int(len(group)),
                "observed_mean": safe_mean(observed),
                "predicted_mean": safe_mean(predicted),
                "mean_residual": safe_mean(residual),
                "mae": mean_absolute_error(observed, predicted),
                "saturated_count": int(np.count_nonzero(observed >= KELPWATCH_PIXEL_AREA_M2)),
            }
        )
    return rows


def persistence_classes(dataframe: pd.DataFrame) -> pd.Series:
    """Classify rows by number of quarters with nonzero kelp."""
    classes: list[str] = []
    for valid_count, nonzero_count in zip(
        dataframe["valid_quarter_count"].to_numpy(dtype=float),
        dataframe["nonzero_quarter_count"].to_numpy(dtype=float),
        strict=True,
    ):
        if not np.isfinite(valid_count) or valid_count == 0:
            classes.append("missing_quarters")
        elif nonzero_count == 0:
            classes.append("no_quarter_present")
        elif nonzero_count == valid_count:
            classes.append("persistent_all_valid_quarters")
        elif nonzero_count == 1:
            classes.append("transient_one_quarter")
        else:
            classes.append("intermittent_two_or_three_quarters")
    return pd.Series(classes, index=dataframe.index, dtype="object")


def build_threshold_sensitivity(
    dataframe: pd.DataFrame, thresholds: tuple[float, ...]
) -> list[dict[str, object]]:
    """Build binary-threshold sensitivity summaries."""
    rows: list[dict[str, object]] = []
    for keys, group in dataframe.groupby(["model_name", "split", "year"], sort=True):
        model_name, split, year = cast(tuple[str, str, int], keys)
        observed_fraction = group["kelp_fraction_y"].to_numpy(dtype=float)
        predicted_fraction = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
        observed_area = group["kelp_max_y"].to_numpy(dtype=float)
        predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
        for threshold in thresholds:
            if threshold == 0:
                observed_positive = observed_fraction > 0
                predicted_positive = predicted_fraction > 0
            else:
                observed_positive = observed_fraction >= threshold
                predicted_positive = predicted_fraction >= threshold
            precision, recall, f1 = precision_recall_f1(observed_positive, predicted_positive)
            rows.append(
                {
                    "model_name": str(model_name),
                    "split": str(split),
                    "year": int(year),
                    "threshold_fraction": threshold,
                    "threshold_area": threshold * KELPWATCH_PIXEL_AREA_M2,
                    "positive_count": int(np.count_nonzero(observed_positive)),
                    "positive_fraction": safe_ratio(
                        int(np.count_nonzero(observed_positive)), len(group)
                    ),
                    "predicted_positive_count": int(np.count_nonzero(predicted_positive)),
                    "predicted_positive_fraction": safe_ratio(
                        int(np.count_nonzero(predicted_positive)), len(group)
                    ),
                    "observed_area_positive_rows": float(
                        np.nansum(observed_area[observed_positive])
                    ),
                    "predicted_area_positive_rows": float(
                        np.nansum(predicted_area[predicted_positive])
                    ),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def build_spatial_readiness(dataframe: pd.DataFrame, band_count: int) -> list[dict[str, object]]:
    """Build latitude-band summaries for spatial holdout readiness."""
    frame = dataframe.copy()
    frame["latitude_band"] = latitude_band_indices(
        frame["latitude"].to_numpy(dtype=float), band_count
    )
    abs_residual = frame["residual_kelp_max_y"].abs()
    top_cutoff = float(abs_residual.quantile(0.95)) if len(abs_residual) else math.nan
    rows: list[dict[str, object]] = []
    for band, group in frame.groupby("latitude_band", sort=True):
        observed = group["kelp_max_y"].to_numpy(dtype=float)
        predicted = group["pred_kelp_max_y"].to_numpy(dtype=float)
        residual = group["residual_kelp_max_y"].to_numpy(dtype=float)
        positive_count = int(np.count_nonzero(observed > 0))
        saturated_count = int(np.count_nonzero(observed >= KELPWATCH_PIXEL_AREA_M2))
        row_count = int(len(group))
        enough = row_count >= 1_000 and positive_count >= 100 and saturated_count >= 10
        rows.append(
            {
                "latitude_band": f"{int(band):02d}",
                "latitude_min": float(group["latitude"].min()),
                "latitude_max": float(group["latitude"].max()),
                "row_count": row_count,
                "station_count": int(group["kelpwatch_station_id"].nunique()),
                "zero_count": int(np.count_nonzero(observed == 0)),
                "positive_count": positive_count,
                "high_canopy_count": int(np.count_nonzero(observed >= 450)),
                "saturated_count": saturated_count,
                "observed_area": float(np.nansum(observed)),
                "predicted_area": float(np.nansum(predicted)),
                "mean_residual": safe_mean(residual),
                "top_abs_residual_count": int(np.count_nonzero(np.abs(residual) >= top_cutoff)),
                "enough_for_holdout": enough,
            }
        )
    return rows


def latitude_band_indices(latitudes: np.ndarray, band_count: int) -> np.ndarray:
    """Assign latitudes to deterministic equal-width band indices."""
    minimum = float(np.nanmin(latitudes))
    maximum = float(np.nanmax(latitudes))
    if maximum == minimum:
        return np.zeros(latitudes.shape, dtype=int)
    scaled = (latitudes - minimum) / (maximum - minimum)
    indices = np.floor(scaled * band_count).astype(int)
    return cast(np.ndarray, np.clip(indices, 0, band_count - 1))


def build_feature_separability(
    aligned: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Build PCA-based feature separability summaries and projection rows."""
    required_columns = list(analysis_config.feature_columns)
    complete = aligned.dropna(subset=required_columns).copy()
    if complete.empty:
        return [], pd.DataFrame()
    if len(complete) > analysis_config.max_projection_rows:
        sample = complete.sample(n=analysis_config.max_projection_rows, random_state=0)
    else:
        sample = complete
    label_group = feature_label_groups(sample)
    scaled = StandardScaler().fit_transform(sample[required_columns].to_numpy(dtype=float))
    projection = PCA(n_components=2, random_state=0).fit_transform(scaled)
    projection_frame = pd.DataFrame(
        {
            "pc1": projection[:, 0],
            "pc2": projection[:, 1],
            "label_group": label_group.to_numpy(),
            "year": sample["year"].to_numpy(),
            "kelp_max_y": sample["kelp_max_y"].to_numpy(dtype=float),
        }
    )
    zero_center = group_center(projection_frame, "zero")
    rows: list[dict[str, object]] = []
    for group_name, group in projection_frame.groupby("label_group", sort=True):
        pc1 = group["pc1"].to_numpy(dtype=float)
        pc2 = group["pc2"].to_numpy(dtype=float)
        center = np.array([safe_mean(pc1), safe_mean(pc2)])
        rows.append(
            {
                "label_group": str(group_name),
                "row_count": int(len(group)),
                "pc1_mean": center[0],
                "pc2_mean": center[1],
                "pc1_std": float(np.nanstd(pc1)),
                "pc2_std": float(np.nanstd(pc2)),
                "distance_from_zero_group": float(np.linalg.norm(center - zero_center)),
            }
        )
    return rows, projection_frame


def feature_label_groups(dataframe: pd.DataFrame) -> pd.Series:
    """Assign rows to broad label groups for feature-space diagnostics."""
    values = dataframe["kelp_max_y"].to_numpy(dtype=float)
    groups = np.full(values.shape, "positive_low_mid", dtype=object)
    groups[values == 0] = "zero"
    groups[values >= 450] = "high_canopy"
    groups[values >= KELPWATCH_PIXEL_AREA_M2] = "saturated_900"
    return pd.Series(groups, index=dataframe.index, dtype="object")


def group_center(projection_frame: pd.DataFrame, label_group: str) -> np.ndarray:
    """Return a PCA group center, defaulting to the global center if absent."""
    group = projection_frame.loc[projection_frame["label_group"] == label_group]
    if group.empty:
        group = projection_frame
    return np.array([float(group["pc1"].mean()), float(group["pc2"].mean())])


def build_target_framing_summary(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build summaries comparing ridge predictions with alternative target framings."""
    rows: list[dict[str, object]] = []
    continuous_targets = {
        "annual_max_area": dataframe["kelp_max_y"],
        "annual_mean_area": dataframe["kelp_mean_area_y"],
        "annual_min_area": dataframe["kelp_min_area_y"],
        "fall_area": dataframe["fall_area_y"],
        "winter_area": dataframe["winter_area_y"],
        "fall_minus_winter_area": dataframe["fall_minus_winter_area_y"],
    }
    predicted_area = dataframe["pred_kelp_max_y"].to_numpy(dtype=float)
    predicted_fraction = dataframe["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    for name, series in continuous_targets.items():
        target = series.to_numpy(dtype=float)
        rows.append(target_framing_row(name, "continuous_area", target, predicted_area))
    binary_targets = {
        "any_presence": dataframe["any_presence_y"],
        "persistent_presence": dataframe["persistent_presence_y"],
        "fall_presence": dataframe["fall_presence_y"],
        "winter_presence": dataframe["winter_presence_y"],
    }
    for threshold in analysis_config.threshold_fractions:
        suffix = threshold_suffix(threshold)
        column = f"high_presence_ge_{suffix}_y"
        if column in dataframe.columns:
            binary_targets[f"presence_ge_{suffix}"] = dataframe[column]
    for name, series in binary_targets.items():
        target = series.astype(float).to_numpy(dtype=float)
        rows.append(target_framing_row(name, "binary_or_fraction", target, predicted_fraction))
    mean_presence = dataframe["mean_presence_fraction_y"].to_numpy(dtype=float)
    rows.append(
        target_framing_row(
            "mean_presence_fraction",
            "binary_or_fraction",
            mean_presence,
            predicted_fraction,
        )
    )
    return rows


def target_framing_row(
    target_name: str, target_kind: str, target: np.ndarray, predicted: np.ndarray
) -> dict[str, object]:
    """Build one target framing summary row."""
    valid_mask = np.isfinite(target) & np.isfinite(predicted)
    valid_target = target[valid_mask]
    valid_predicted = predicted[valid_mask]
    return {
        "target_name": target_name,
        "target_kind": target_kind,
        "row_count": int(target.size),
        "valid_count": int(valid_target.size),
        "mean": safe_mean(valid_target),
        "median": safe_percentile(valid_target, 50),
        "p95": safe_percentile(valid_target, 95),
        "positive_fraction": safe_ratio(int(np.count_nonzero(valid_target > 0)), valid_target.size),
        "pearson_with_prediction": correlation(valid_target, valid_predicted, method="pearson"),
        "spearman_with_prediction": correlation(valid_target, valid_predicted, method="spearman"),
        "mae_vs_prediction_area": (
            mean_absolute_error(valid_target, valid_predicted)
            if target_kind == "continuous_area"
            else math.nan
        ),
        "area_bias_vs_prediction": (
            float(np.nansum(valid_predicted) - np.nansum(valid_target))
            if target_kind == "continuous_area"
            else math.nan
        ),
        "area_pct_bias_vs_prediction": (
            percent_bias(float(np.nansum(valid_predicted)), float(np.nansum(valid_target)))
            if target_kind == "continuous_area"
            else math.nan
        ),
    }


def correlation(observed: np.ndarray, predicted: np.ndarray, *, method: str) -> float:
    """Compute a Pearson or Spearman correlation for finite paired values."""
    if observed.size < 2:
        return math.nan
    if np.nanstd(observed) == 0 or np.nanstd(predicted) == 0:
        return math.nan
    if method == "spearman":
        observed = pd.Series(observed).rank(method="average").to_numpy(dtype=float)
        predicted = pd.Series(predicted).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(observed, predicted)[0, 1])


def build_phase1_decision_matrix(
    *,
    data: AnalysisData,
    prediction_distribution: list[dict[str, object]],
    residual_by_bin: list[dict[str, object]],
    spatial_readiness: list[dict[str, object]],
    feature_separability: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build evidence-linked Phase 1 branch recommendations."""
    test_distribution = first_matching_row(
        prediction_distribution,
        split="test",
        year=int(data.model_predictions["year"].max()),
    )
    saturated_mean_prediction = row_float(test_distribution, "observed_900_prediction_mean")
    saturated_underprediction = (
        np.isfinite(saturated_mean_prediction)
        and saturated_mean_prediction < KELPWATCH_PIXEL_AREA_M2 * 0.75
    )
    station_ridge = station_metric_row(data.metrics, DEFAULT_MODEL_NAME, "test")
    sample_ridge = background_sample_overall_metric_row(data.metrics, DEFAULT_MODEL_NAME, "test")
    missing_drop_fraction = missing_feature_drop_fraction(data.split_manifest)
    enough_spatial_bands = sum(bool(row["enough_for_holdout"]) for row in spatial_readiness)
    feature_distance = max_feature_distance(feature_separability)
    rows: list[dict[str, object]] = [
        {
            "branch": "Sampling/objective calibration Phase 1",
            "evidence_status": "strong",
            "triggering_evidence": (
                f"Kelpwatch-station ridge area bias is {format_percent(row_float(station_ridge, 'area_pct_bias'), 1)}, "
                f"while background-inclusive sample area bias is {format_percent(row_float(sample_ridge, 'area_pct_bias'), 1)}; "
                "the same model can underpredict observed canopy support while leaking positives across background."
            ),
            "proposed_next_tasks": "Separate training objective weights from population-calibration metrics; test capped background weights, stratified losses, and post-fit calibration.",
            "expected_artifacts": "Sampling policy comparison table, calibrated predictions, station-vs-background metric summary.",
            "decision_unlocked": "Choose a defensible background sampling and calibration policy before scaling.",
        },
        {
            "branch": "Derived-label Phase 1",
            "evidence_status": "strong" if saturated_underprediction else "moderate",
            "triggering_evidence": (
                f"Observed-900 rows have mean ridge prediction {saturated_mean_prediction:.1f}; "
                "annual max, annual mean, persistence, and threshold diagnostics need retrained target variants before changing models."
            ),
            "proposed_next_tasks": "Implement evaluated label variants: mean canopy, persistent presence, fall/winter labels, and threshold diagnostics.",
            "expected_artifacts": "Derived label parquet variants, label QA tables, target-comparison report.",
            "decision_unlocked": "Choose the first production target framing for Phase 1 models.",
        },
        {
            "branch": "Baseline-hardening Phase 1",
            "evidence_status": "strong",
            "triggering_evidence": "Reference-baseline rows now include train-mean no-skill, previous-year persistence, grid-cell climatology, lat/lon/year geography, and AEF ridge once the Phase 1 baseline loop is rerun.",
            "proposed_next_tasks": "Interpret whether AEF ridge beats persistence, site memory, and geography before moving to domain masks or imbalance-aware models.",
            "expected_artifacts": "Ranked baseline predictions, metrics tables, fallback summaries, and calibration rows.",
            "decision_unlocked": "Interpret whether AEF embeddings beat meaningful non-embedding references.",
        },
        {
            "branch": "Stronger-tabular-model Phase 1",
            "evidence_status": "moderate" if feature_distance > 0.25 else "weak",
            "triggering_evidence": f"Feature projection maximum distance from zero group is {feature_distance:.2f}; use this only as separability evidence, not model skill.",
            "proposed_next_tasks": "Train a tree-based or histogram-gradient boosting baseline after target, baseline, and calibration checks.",
            "expected_artifacts": "Nonlinear tabular model, predictions, metrics, and residual maps.",
            "decision_unlocked": "Test whether ridge underfits a separable nonlinear feature signal.",
        },
        {
            "branch": "Spatial-evaluation Phase 1",
            "evidence_status": "moderate" if enough_spatial_bands >= 3 else "weak",
            "triggering_evidence": f"{enough_spatial_bands} latitude bands meet minimum holdout-readiness counts in Monterey.",
            "proposed_next_tasks": "Design latitude/site holdout manifests and decide whether to add a second smoke region.",
            "expected_artifacts": "Spatial split manifests and split-balance diagnostics.",
            "decision_unlocked": "Evaluate spatial generalization instead of only year holdout.",
        },
        {
            "branch": "Ingestion/alignment-hardening Phase 1",
            "evidence_status": "weak" if missing_drop_fraction < 0.01 else "moderate",
            "triggering_evidence": f"Missing-feature drops are {missing_drop_fraction:.3%} of split-manifest rows; label row counts should remain monitored.",
            "proposed_next_tasks": "Harden manifests, missing-feature diagnostics, and alignment comparison checks before scale-up if drop rates grow.",
            "expected_artifacts": "Alignment QA manifest and stage-distribution regression tests.",
            "decision_unlocked": "Know whether failures are data plumbing or model behavior.",
        },
        {
            "branch": "Scale-up Phase 1",
            "evidence_status": "moderate",
            "triggering_evidence": "The smoke pipeline is runnable end to end, but target framing, baseline, and calibration gaps should be resolved before full West Coast processing.",
            "proposed_next_tasks": "Expand to a second region or broader California slice after label, baseline, and calibration choices are settled.",
            "expected_artifacts": "Expanded config, manifests, aligned table, and smoke report for more geography.",
            "decision_unlocked": "Decide whether observed Monterey behavior generalizes.",
        },
    ]
    return rows


def build_phase1_model_comparison(
    data: AnalysisData,
    analysis_config: ModelAnalysisConfig,
    reference_area_calibration: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build the Phase 1 compact model comparison rows."""
    rows: list[dict[str, object]] = []
    split_years = split_year_labels(data.split_manifest)
    for metric in data.metrics.to_dict("records"):
        comparison_row = metric_comparison_row(
            cast(dict[str, object], metric),
            split_years,
        )
        if comparison_row:
            rows.append(comparison_row)
    rows.extend(reference_area_calibration_comparison_rows(reference_area_calibration))
    if not any(row.get("evaluation_scope") == "full_grid_prediction" for row in rows):
        rows.extend(full_grid_comparison_rows(data.model_predictions, analysis_config))
    return rows


def reference_area_calibration_comparison_rows(
    reference_area_calibration: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Convert compact reference area-calibration rows into comparison rows."""
    required = {
        "model_name",
        "split",
        "year",
        "label_source",
        "row_count",
        "mae",
        "rmse",
        "r2",
        "f1_ge_10pct",
        "observed_canopy_area",
        "predicted_canopy_area",
        "area_pct_bias",
    }
    rows: list[dict[str, object]] = []
    for row in reference_area_calibration:
        row_dict = row
        if not required.issubset(row_dict):
            continue
        rows.append(
            {
                "model_name": row_dict.get("model_name", ""),
                "split": row_dict.get("split", ""),
                "year": row_dict.get("year", ""),
                "mask_status": "unmasked",
                "evaluation_scope": "full_grid_prediction",
                "label_source": row_dict.get("label_source", "all"),
                "row_count": row_dict.get("row_count", ""),
                "mae": row_dict.get("mae", math.nan),
                "rmse": row_dict.get("rmse", math.nan),
                "r2": row_dict.get("r2", math.nan),
                "spearman": math.nan,
                "f1_ge_10pct": row_dict.get("f1_ge_10pct", math.nan),
                "observed_canopy_area": row_dict.get("observed_canopy_area", math.nan),
                "predicted_canopy_area": row_dict.get("predicted_canopy_area", math.nan),
                "area_pct_bias": row_dict.get("area_pct_bias", math.nan),
            }
        )
    return rows


def split_year_labels(split_manifest: pd.DataFrame) -> dict[str, str]:
    """Return compact year labels for each split in the split manifest."""
    if "split" not in split_manifest.columns or "year" not in split_manifest.columns:
        return {}
    labels: dict[str, str] = {}
    for split, group in split_manifest.groupby("split", sort=True):
        years = sorted(int(year) for year in group["year"].dropna().unique())
        if not years:
            labels[str(split)] = ""
        elif len(years) == 1:
            labels[str(split)] = str(years[0])
        elif years == list(range(years[0], years[-1] + 1)):
            labels[str(split)] = f"{years[0]}-{years[-1]}"
        else:
            labels[str(split)] = ",".join(str(year) for year in years)
    return labels


def metric_comparison_row(
    metric: dict[str, object], split_years: dict[str, str]
) -> dict[str, object]:
    """Convert one metrics CSV row into the Phase 1 comparison schema."""
    if bool_from_metric(metric.get("weighted")):
        return {}
    metric_group = str(metric.get("metric_group", "overall"))
    metric_group_value = str(metric.get("metric_group_value", "all"))
    if metric_group == "overall":
        evaluation_scope = "background_inclusive_sample"
        label_source = "all"
    elif metric_group == "label_source" and metric_group_value == "kelpwatch_station":
        evaluation_scope = "kelpwatch_station_sample"
        label_source = metric_group_value
    elif "metric_group" not in metric:
        evaluation_scope = "kelpwatch_station_sample"
        label_source = "kelpwatch_station"
    else:
        return {}
    split = str(metric.get("split", ""))
    return {
        "model_name": metric.get("model_name", ""),
        "split": split,
        "year": split_years.get(split, ""),
        "mask_status": "unmasked",
        "evaluation_scope": evaluation_scope,
        "label_source": label_source,
        "row_count": metric.get("row_count", ""),
        "mae": metric.get("mae", math.nan),
        "rmse": metric.get("rmse", math.nan),
        "r2": metric.get("r2", math.nan),
        "spearman": metric.get("spearman", math.nan),
        "f1_ge_10pct": metric.get("f1_ge_10pct", math.nan),
        "observed_canopy_area": metric.get("observed_canopy_area", math.nan),
        "predicted_canopy_area": metric.get("predicted_canopy_area", math.nan),
        "area_pct_bias": metric.get("area_pct_bias", math.nan),
    }


def bool_from_metric(value: object) -> bool:
    """Return whether a CSV/object value should be treated as true."""
    return str(value).lower() in {"true", "1"}


def full_grid_comparison_rows(
    predictions: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build full-grid calibration rows from prediction artifacts."""
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return rows
    for keys, group in predictions.groupby(["model_name", "split", "year"], sort=True):
        model_name, split, year = cast(tuple[str, str, int], keys)
        rows.append(
            full_grid_comparison_row(
                str(model_name),
                str(split),
                int(year),
                group,
                analysis_config,
            )
        )
    return rows


def full_grid_comparison_row(
    model_name: str,
    split: str,
    year: int,
    group: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
) -> dict[str, object]:
    """Build one full-grid prediction comparison row."""
    observed = group["kelp_fraction_y"].to_numpy(dtype=float)
    predicted = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = group["kelp_max_y"].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    observed_positive = observed >= 0.10
    predicted_positive = predicted >= 0.10
    _precision, _recall, f1 = precision_recall_f1(observed_positive, predicted_positive)
    return {
        "model_name": model_name,
        "split": split,
        "year": year,
        "mask_status": "unmasked",
        "evaluation_scope": "full_grid_prediction",
        "label_source": "all",
        "row_count": int(len(group)),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "r2": unweighted_r2(observed, predicted),
        "spearman": correlation(observed, predicted, method="spearman"),
        "f1_ge_10pct": f1,
        "observed_canopy_area": float(np.nansum(observed_area)),
        "predicted_canopy_area": float(np.nansum(predicted_area)),
        "area_pct_bias": percent_bias(
            float(np.nansum(predicted_area)),
            float(np.nansum(observed_area)),
        ),
    }


def unweighted_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute unweighted R2 or NaN when the observed target is constant."""
    finite_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite_mask]
    predicted = predicted[finite_mask]
    if observed.size == 0:
        return math.nan
    total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum_squares == 0:
        return math.nan
    residual_sum_squares = float(np.sum((observed - predicted) ** 2))
    return 1.0 - residual_sum_squares / total_sum_squares


def build_data_health_rows(
    data: AnalysisData, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build Phase 1 data-health rows for the report harness."""
    rows: list[dict[str, object]] = []
    rows.extend(year_count_health_rows("annual_label_rows", data.labels))
    rows.extend(label_source_health_rows("model_input_rows", data.aligned, split="all"))
    rows.extend(split_manifest_health_rows(data.split_manifest))
    rows.extend(label_source_health_rows("prediction_rows", data.model_predictions))
    primary = primary_rows(data.model_predictions, analysis_config)
    rows.append(
        data_health_row(
            check_name="primary_report_prediction_rows",
            split=analysis_config.analysis_split,
            year=analysis_config.analysis_year,
            label_source="all",
            row_count=len(primary),
            reference_row_count=len(data.model_predictions),
            detail="rows selected for the primary model-analysis split/year",
        )
    )
    return rows


def year_count_health_rows(check_name: str, dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Build data-health count rows by year."""
    rows: list[dict[str, object]] = []
    if "year" not in dataframe.columns:
        return [
            data_health_row(
                check_name=check_name,
                split="all",
                year="all",
                label_source="all",
                row_count=len(dataframe),
                reference_row_count=len(dataframe),
                detail=check_name,
            )
        ]
    for year, group in dataframe.groupby("year", sort=True):
        rows.append(
            data_health_row(
                check_name=check_name,
                split="all",
                year=int(cast(int, year)),
                label_source="all",
                row_count=len(group),
                reference_row_count=len(group),
                detail=check_name,
            )
        )
    return rows


def label_source_health_rows(
    check_name: str, dataframe: pd.DataFrame, *, split: str | None = None
) -> list[dict[str, object]]:
    """Build data-health count rows by split, year, and label source."""
    if dataframe.empty:
        return []
    frame = dataframe.copy()
    frame["_health_split"] = split if split is not None else frame.get("split", "all")
    frame["_health_label_source"] = label_source_series(frame)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["_health_split", "year", "_health_label_source"], sort=True, dropna=False
    ):
        group_split, year, label_source = cast(tuple[str, int, str], keys)
        year_total = (
            len(frame.loc[frame["year"] == year]) if "year" in frame.columns else len(frame)
        )
        rows.append(
            data_health_row(
                check_name=check_name,
                split=str(group_split),
                year=int(year),
                label_source=str(label_source),
                row_count=len(group),
                reference_row_count=year_total,
                detail=check_name,
            )
        )
    return rows


def split_manifest_health_rows(split_manifest: pd.DataFrame) -> list[dict[str, object]]:
    """Build data-health rows for split-manifest retained and dropped counts."""
    if (
        split_manifest.empty
        or "split" not in split_manifest.columns
        or "year" not in split_manifest.columns
    ):
        return []
    rows: list[dict[str, object]] = []
    for keys, group in split_manifest.groupby(["split", "year"], sort=True):
        split, year = cast(tuple[str, int], keys)
        total = len(group)
        if "used_for_training_eval" in group.columns:
            retained_mask = group["used_for_training_eval"].fillna(False).astype(bool)
        else:
            retained_mask = pd.Series(True, index=group.index)
        dropped_mask = ~retained_mask
        missing_mask = missing_feature_drop_mask(group, dropped_mask)
        rows.append(
            data_health_row(
                check_name="split_manifest_rows",
                split=str(split),
                year=int(year),
                label_source="all",
                row_count=total,
                reference_row_count=total,
                detail="all split-manifest rows",
            )
        )
        rows.append(
            data_health_row(
                check_name="retained_model_rows",
                split=str(split),
                year=int(year),
                label_source="all",
                row_count=int(retained_mask.sum()),
                reference_row_count=total,
                detail="rows retained for model training/evaluation",
            )
        )
        rows.append(
            data_health_row(
                check_name="dropped_model_rows",
                split=str(split),
                year=int(year),
                label_source="all",
                row_count=int(dropped_mask.sum()),
                reference_row_count=total,
                detail="rows dropped before model training/evaluation",
            )
        )
        rows.append(
            data_health_row(
                check_name="missing_feature_drop_rate",
                split=str(split),
                year=int(year),
                label_source="all",
                row_count=int(missing_mask.sum()),
                reference_row_count=total,
                detail="dropped rows attributed to missing features",
            )
        )
    return rows


def missing_feature_drop_mask(group: pd.DataFrame, dropped_mask: pd.Series) -> pd.Series:
    """Return rows dropped because features are missing."""
    if "drop_reason" in group.columns:
        reason_mask = group["drop_reason"].fillna("").astype(str) == "missing_features"
        return dropped_mask & reason_mask
    if "has_complete_features" in group.columns:
        return dropped_mask & ~group["has_complete_features"].fillna(False).astype(bool)
    return dropped_mask


def label_source_series(dataframe: pd.DataFrame) -> pd.Series:
    """Return label-source provenance with fallbacks for older artifacts."""
    if "label_source" in dataframe.columns:
        return dataframe["label_source"].fillna("unknown").astype(str)
    if "is_kelpwatch_observed" in dataframe.columns:
        observed = dataframe["is_kelpwatch_observed"].fillna(False).astype(bool)
        return pd.Series(
            np.where(observed, "kelpwatch_station", "assumed_background"),
            index=dataframe.index,
            dtype="object",
        )
    if "kelpwatch_station_id" in dataframe.columns:
        observed = dataframe["kelpwatch_station_id"].notna()
        return pd.Series(
            np.where(observed, "kelpwatch_station", "assumed_background"),
            index=dataframe.index,
            dtype="object",
        )
    return pd.Series("all", index=dataframe.index, dtype="object")


def data_health_row(
    *,
    check_name: str,
    split: object,
    year: object,
    label_source: object,
    row_count: int,
    reference_row_count: int,
    detail: str,
) -> dict[str, object]:
    """Build one data-health row."""
    return {
        "check_name": check_name,
        "split": split,
        "year": year,
        "label_source": label_source,
        "row_count": int(row_count),
        "reference_row_count": int(reference_row_count),
        "rate": safe_ratio(int(row_count), int(reference_row_count)),
        "detail": detail,
    }


def first_matching_row(
    rows: list[dict[str, object]], *, split: str, year: int
) -> dict[str, object]:
    """Return the first distribution row matching a split and year."""
    for row in rows:
        if row.get("split") == split and int(cast(int, row.get("year", -1))) == year:
            return row
    return {}


def missing_feature_drop_fraction(split_manifest: pd.DataFrame) -> float:
    """Compute the fraction of split-manifest rows dropped for missing features."""
    if "used_for_training_eval" not in split_manifest.columns or split_manifest.empty:
        return math.nan
    dropped = int((~split_manifest["used_for_training_eval"]).sum())
    return safe_ratio(dropped, len(split_manifest))


def max_feature_distance(rows: list[dict[str, object]]) -> float:
    """Return the largest feature-projection distance from the zero group."""
    distances = [row_float(row, "distance_from_zero_group") for row in rows]
    return max(distances) if distances else math.nan


def build_quarter_mapping(analysis_config: ModelAnalysisConfig) -> list[dict[str, object]]:
    """Build a Kelpwatch source quarter mapping table."""
    if (
        analysis_config.source_manifest_path is None
        or not analysis_config.source_manifest_path.exists()
    ):
        return fallback_quarter_mapping(analysis_config)
    manifest = load_json_object(analysis_config.source_manifest_path)
    transfer = require_mapping(manifest.get("transfer"), "source_manifest.transfer")
    netcdf_path = Path(
        require_string(transfer.get("local_path"), "source_manifest.transfer.local_path")
    )
    if not netcdf_path.exists():
        return fallback_quarter_mapping(analysis_config)
    rows: list[dict[str, object]] = []
    with xr.open_dataset(netcdf_path, engine=NETCDF_ENGINE, decode_cf=True) as dataset:
        time_values = dataset["time"].values if "time" in dataset else np.array([])
        years = dataset["year"].values if "year" in dataset else np.array([])
        quarters = dataset["quarter"].values if "quarter" in dataset else np.array([])
        for index in range(min(len(time_values), len(years), len(quarters))):
            quarter = int(quarters[index])
            rows.append(
                {
                    "source_time": str(pd.Timestamp(time_values[index]).date()),
                    "source_year": int(years[index]),
                    "source_quarter": quarter,
                    "derived_year": int(years[index]),
                    "derived_quarter": quarter,
                    "season_label": season_label(
                        quarter,
                        analysis_config.fall_quarter,
                        analysis_config.winter_quarter,
                    ),
                }
            )
    return rows


def fallback_quarter_mapping(analysis_config: ModelAnalysisConfig) -> list[dict[str, object]]:
    """Build a fallback quarter mapping when source time metadata is unavailable."""
    return [
        {
            "source_time": "",
            "source_year": "",
            "source_quarter": quarter,
            "derived_year": "",
            "derived_quarter": quarter,
            "season_label": season_label(
                quarter,
                analysis_config.fall_quarter,
                analysis_config.winter_quarter,
            ),
        }
        for quarter in range(1, 5)
    ]


def season_label(quarter: int, fall_quarter: int, winter_quarter: int) -> str:
    """Label a source quarter with configured seasonal interpretation."""
    if quarter == fall_quarter:
        return "configured_fall"
    if quarter == winter_quarter:
        return "configured_winter"
    return "other"


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"expected JSON object at {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def write_analysis_tables(tables: AnalysisTables, analysis_config: ModelAnalysisConfig) -> None:
    """Write all model-analysis CSV outputs."""
    write_csv(
        tables.stage_distribution,
        analysis_config.label_distribution_path,
        STAGE_DISTRIBUTION_FIELDS,
    )
    write_csv(tables.target_framing, analysis_config.target_framing_path, TARGET_FRAMING_FIELDS)
    write_csv(tables.residual_by_bin, analysis_config.residual_by_bin_path, RESIDUAL_BIN_FIELDS)
    write_csv(
        tables.residual_by_persistence,
        analysis_config.residual_by_persistence_path,
        PERSISTENCE_FIELDS,
    )
    write_csv(
        tables.prediction_distribution,
        analysis_config.prediction_distribution_path,
        PREDICTION_DISTRIBUTION_FIELDS,
    )
    write_csv(
        tables.threshold_sensitivity, analysis_config.threshold_sensitivity_path, THRESHOLD_FIELDS
    )
    write_csv(
        tables.spatial_readiness, analysis_config.spatial_readiness_path, SPATIAL_READINESS_FIELDS
    )
    write_csv(
        tables.feature_separability,
        analysis_config.feature_separability_path,
        FEATURE_SEPARABILITY_FIELDS,
    )
    write_csv(tables.phase1_decision, analysis_config.phase1_decision_path, PHASE1_DECISION_FIELDS)
    write_csv(
        tables.phase1_model_comparison,
        analysis_config.phase1_model_comparison_path,
        PHASE1_MODEL_COMPARISON_FIELDS,
    )
    write_csv(
        tables.reference_area_calibration,
        analysis_config.reference_area_calibration_path,
        REFERENCE_AREA_CALIBRATION_FIELDS,
    )
    write_csv(tables.data_health, analysis_config.data_health_path, DATA_HEALTH_FIELDS)
    write_csv(tables.quarter_mapping, analysis_config.quarter_mapping_path, QUARTER_MAPPING_FIELDS)


def write_csv(rows: list[dict[str, object]], output_path: Path, fields: tuple[str, ...]) -> None:
    """Write rows to CSV with a stable field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_figures(
    data: AnalysisData, tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write all model-analysis figures."""
    write_label_distribution_figure(data, analysis_config)
    write_observed_predicted_distribution_figure(data, analysis_config)
    write_residual_by_bin_figure(tables, analysis_config)
    write_observed_900_figure(data, analysis_config)
    write_residual_by_persistence_figure(tables, analysis_config)
    write_alternative_targets_figure(tables, analysis_config)
    projection_frame = cast(
        pd.DataFrame, data.aligned.attrs.get("model_analysis_projection_frame", pd.DataFrame())
    )
    write_feature_projection_figure(projection_frame, analysis_config)
    write_spatial_readiness_figure(tables, analysis_config)


def write_label_distribution_figure(
    data: AnalysisData, analysis_config: ModelAnalysisConfig
) -> None:
    """Write annual label distribution histograms for labels and aligned rows."""
    output_path = analysis_config.label_distribution_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, dataframe, title in [
        (axes[0], data.labels, "Annual labels"),
        (axes[1], data.aligned, "Aligned labels"),
    ]:
        axis.hist(
            dataframe["kelp_max_y"], bins=np.linspace(0, 900, 31), color="#3b7ea1", alpha=0.85
        )
        axis.set_title(title)
        axis.set_xlabel("Kelpwatch annual max area")
        axis.set_ylabel("Rows")
        axis.set_yscale("log")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_observed_predicted_distribution_figure(
    data: AnalysisData, analysis_config: ModelAnalysisConfig
) -> None:
    """Write observed-vs-predicted distribution histograms for the primary split/year."""
    rows = primary_rows(data.model_predictions, analysis_config)
    output_path = analysis_config.observed_predicted_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bins = np.linspace(0, 900, 31).tolist()
    axis.hist(rows["kelp_max_y"], bins=bins, alpha=0.55, label="Observed", color="#2c7fb8")
    axis.hist(rows["pred_kelp_max_y"], bins=bins, alpha=0.55, label="Predicted", color="#f03b20")
    axis.set_title(
        f"Observed vs predicted distribution | {analysis_config.analysis_split} {analysis_config.analysis_year}"
    )
    axis.set_xlabel("Canopy area")
    axis.set_ylabel("Rows")
    axis.set_yscale("log")
    axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def primary_rows(dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig) -> pd.DataFrame:
    """Select the configured primary split and year."""
    rows = dataframe.loc[
        (dataframe["split"] == analysis_config.analysis_split)
        & (dataframe["year"] == analysis_config.analysis_year)
    ].copy()
    if rows.empty:
        LOGGER.warning(
            "Primary analysis rows are empty for split=%s year=%s; using all model rows",
            analysis_config.analysis_split,
            analysis_config.analysis_year,
        )
        return dataframe
    return rows


def write_residual_by_bin_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write mean residual by observed-bin figure for the primary split/year."""
    rows = filter_table_rows(
        tables.residual_by_bin,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    bin_order = {
        label: index
        for index, label in enumerate(observed_area_bin_order(analysis_config.observed_area_bins))
    }
    rows = sorted(
        rows,
        key=lambda row: bin_order.get(str(row["observed_bin"]), len(bin_order)),
    )
    output_path = analysis_config.residual_by_bin_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = [str(row["observed_bin"]) for row in rows]
    values = [row_float(row, "mean_residual") for row in rows]
    axis.bar(labels, values, color="#756bb1")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Mean residual by observed canopy bin")
    axis.set_xlabel("Observed canopy bin")
    axis.set_ylabel("Residual observed - predicted")
    axis.tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def filter_table_rows(
    rows: list[dict[str, object]], *, split: str, year: int
) -> list[dict[str, object]]:
    """Filter generic table rows by split and year."""
    return [
        row
        for row in rows
        if row.get("split") == split and int(cast(int, row.get("year", -1))) == year
    ]


def write_observed_900_figure(data: AnalysisData, analysis_config: ModelAnalysisConfig) -> None:
    """Write prediction distribution for observed saturated pixels."""
    rows = primary_rows(data.model_predictions, analysis_config)
    saturated = rows.loc[rows["kelp_max_y"] >= KELPWATCH_PIXEL_AREA_M2]
    output_path = analysis_config.observed_900_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.hist(
        saturated["pred_kelp_max_y"],
        bins=np.linspace(0, 900, 31).tolist(),
        color="#b2182b",
        alpha=0.85,
    )
    axis.axvline(900, color="black", linestyle="--", linewidth=1.0)
    axis.set_title("Ridge predictions where observed area is 900")
    axis.set_xlabel("Predicted canopy area")
    axis.set_ylabel("Rows")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_residual_by_persistence_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write mean residual by persistence class."""
    rows = filter_table_rows(
        tables.residual_by_persistence,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    class_order = {label: index for index, label in enumerate(PERSISTENCE_CLASS_ORDER)}
    rows = sorted(
        rows,
        key=lambda row: class_order.get(str(row["persistence_class"]), len(class_order)),
    )
    output_path = analysis_config.residual_by_persistence_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = [str(row["persistence_class"]) for row in rows]
    values = [row_float(row, "mean_residual") for row in rows]
    axis.bar(labels, values, color="#238b45")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Mean residual by quarterly persistence")
    axis.set_xlabel("Persistence class")
    axis.set_ylabel("Residual observed - predicted")
    axis.tick_params(axis="x", rotation=30)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_alternative_targets_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write target framing correlation figure."""
    rows = sorted(
        tables.target_framing,
        key=lambda row: (
            row_float(row, "spearman_with_prediction")
            if np.isfinite(row_float(row, "spearman_with_prediction"))
            else -math.inf
        ),
        reverse=True,
    )
    output_path = analysis_config.alternative_targets_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    labels = [str(row["target_name"]) for row in rows]
    values = [row_float(row, "spearman_with_prediction") for row in rows]
    axis.bar(labels, values, color="#6a51a3")
    axis.set_title("Spearman correlation with ridge predictions")
    axis.set_ylabel("Spearman rho")
    axis.set_ylim(-1, 1)
    axis.tick_params(axis="x", rotation=45)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_feature_projection_figure(
    projection_frame: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> None:
    """Write a PCA feature-projection scatter figure."""
    output_path = analysis_config.feature_projection_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    if projection_frame.empty:
        axis.text(0.5, 0.5, "No complete feature rows", ha="center", va="center")
    else:
        for label_group, group in projection_frame.groupby("label_group", sort=True):
            axis.scatter(group["pc1"], group["pc2"], s=4, alpha=0.35, label=str(label_group))
        axis.legend(markerscale=3)
    axis.set_title("AEF feature projection by label group")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_spatial_readiness_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write latitude-band positive and saturated count figure."""
    rows = tables.spatial_readiness
    output_path = analysis_config.spatial_readiness_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    labels = [str(row["latitude_band"]) for row in rows]
    positive = [row_int(row, "positive_count") for row in rows]
    saturated = [row_int(row, "saturated_count") for row in rows]
    x_values = np.arange(len(labels))
    axis.bar(x_values - 0.2, positive, width=0.4, label="Positive")
    axis.bar(x_values + 0.2, saturated, width=0.4, label="Saturated")
    axis.set_xticks(x_values, labels)
    axis.set_title("Spatial holdout readiness by latitude band")
    axis.set_xlabel("Latitude band")
    axis.set_ylabel("Rows")
    axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    data: AnalysisData, tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write the Markdown Phase 1 model-analysis report."""
    output_path = analysis_config.report_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_distribution = first_matching_row(
        tables.prediction_distribution,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    stage_rows = [
        row
        for row in tables.stage_distribution
        if row["year"] == analysis_config.analysis_year
        and row["stage"] in {"annual_labels", "model_input_sample", "retained_model_rows"}
    ]
    report = [
        "# Monterey Phase 1 Model Analysis",
        "",
        "## Executive Summary",
        "",
        "This is the active hardening report for the Monterey annual-max pipeline. It keeps the Kelpwatch annual-max target fixed and treats the current ridge output as a reference model to improve, not as a production result. Results should be interpreted as learning Kelpwatch-style labels, not field-verified kelp truth.",
        "",
        shrinkage_summary_markdown(data.metrics, tables.residual_by_bin, analysis_config),
        "",
        "Phase 1 focus: **Monterey annual-max model and domain hardening**. The harness is set up to compare reference baselines, domain-mask variants, imbalance-aware models, pixel skill, and area calibration in the main reports after each pipeline rerun.",
        "",
        "## Current Annual-Max Scope And Artifacts",
        "",
        f"- Config: `{analysis_config.config_path}`",
        f"- Annual labels: `{analysis_config.label_path}`",
        f"- Model input sample: `{analysis_config.aligned_table_path}`",
        f"- Predictions: `{analysis_config.predictions_path}`",
        f"- Metrics: `{analysis_config.metrics_path}`",
        "",
        "## Phase 1 Harness Status",
        "",
        phase1_harness_status_markdown(tables, analysis_config),
        "",
        "## Model Comparison",
        "",
        model_comparison_markdown(tables.phase1_model_comparison, analysis_config),
        "",
        "## Reference Baseline Ranking",
        "",
        reference_baseline_ranking_markdown(tables.phase1_model_comparison, analysis_config),
        "",
        baseline_comparison_markdown(data.metrics, analysis_config),
        "",
        baseline_calibration_markdown(data.metrics, analysis_config),
        "",
        "## Data Health And Label Distribution",
        "",
        data_health_markdown(tables.data_health, analysis_config),
        "",
        stage_distribution_markdown(stage_rows),
        "",
        image_markdown(
            "Label distribution", analysis_config.label_distribution_figure, output_path
        ),
        "",
        "## Quarter And Season Grounding",
        "",
        "Kelpwatch source quarter metadata is written to the quarter mapping table. In the current source, quarter 1 is Jan-Mar, quarter 2 is Apr-Jun, quarter 3 is Jul-Sep, and quarter 4 is Oct-Dec. This report treats configured winter as quarter 1 and configured fall as quarter 4.",
        "",
        f"- Quarter mapping table: `{analysis_config.quarter_mapping_path}`",
        "",
        "## Pixel Skill And Area Calibration",
        "",
        metric_summary_markdown(data.metrics, analysis_config),
        "",
        image_markdown(
            "Observed vs predicted distribution",
            analysis_config.observed_predicted_figure,
            output_path,
        ),
        "",
        "## Observed, Predicted, And Error Map",
        "",
        map_section_markdown(analysis_config),
        "",
        image_markdown(
            "Observed, predicted, and residual map",
            analysis_config.observed_predicted_residual_map_figure,
            output_path,
        ),
        "",
        "## Residual And Saturation Findings",
        "",
        prediction_distribution_markdown(test_distribution),
        "",
        residual_bin_interpretation_markdown(tables.residual_by_bin, analysis_config),
        "",
        image_markdown(
            "Observed 900 predictions", analysis_config.observed_900_figure, output_path
        ),
        "",
        image_markdown(
            "Residual by observed bin", analysis_config.residual_by_bin_figure, output_path
        ),
        "",
        image_markdown(
            "Residual by persistence",
            analysis_config.residual_by_persistence_figure,
            output_path,
        ),
        "",
        "## Binary Threshold Sensitivity",
        "",
        threshold_sensitivity_markdown(tables.threshold_sensitivity, analysis_config),
        "",
        f"Detailed threshold sensitivity is written to `{analysis_config.threshold_sensitivity_path}`. These rows are diagnostics only; this task does not choose a production binary threshold.",
        "",
        "## Phase 1 Coverage Gaps",
        "",
        "Implemented rows currently cover train-mean no-skill, previous-year persistence, grid-cell climatology, lat/lon/year geography, and AEF ridge regression after the reference-baseline task has been rerun. Missing rows that should appear in this same report as Phase 1 progresses: bathymetry/DEM mask variants and imbalance-aware model variants.",
        "",
        "## Interpretation",
        "",
        interpretation_markdown(data.metrics, tables, analysis_config),
        "",
        "## Appendix",
        "",
        f"- Stage distribution table: `{analysis_config.label_distribution_path}`",
        f"- Prediction distribution table: `{analysis_config.prediction_distribution_path}`",
        f"- Residual by observed bin table: `{analysis_config.residual_by_bin_path}`",
        f"- Residual by persistence table: `{analysis_config.residual_by_persistence_path}`",
        f"- Phase 1 model comparison table: `{analysis_config.phase1_model_comparison_path}`",
        f"- Phase 1 data-health table: `{analysis_config.data_health_path}`",
        f"- Reference fallback summary table: `{analysis_config.fallback_summary_path}`",
        f"- Reference area calibration table: `{analysis_config.reference_area_calibration_path}`",
        f"- Phase 1 PDF report: `{analysis_config.pdf_report_path}`",
        "",
        "Validation command:",
        "",
        "```bash",
        "make check",
        "uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml",
        "```",
        "",
    ]
    report_text = "\n".join(report)
    output_path.write_text(report_text)
    write_html_report(
        report_text.splitlines(), analysis_config.html_report_path, output_path.parent
    )
    write_pdf_report(report_text.splitlines(), analysis_config.pdf_report_path, output_path.parent)


def image_markdown(alt_text: str, image_path: Path, report_path: Path) -> str:
    """Build a Markdown image link that is relative to the report file."""
    return f"![{alt_text}]({relative_markdown_path(image_path, report_path.parent)})"


def relative_markdown_path(target_path: Path, base_dir: Path) -> str:
    """Return a POSIX relative path suitable for Markdown links."""
    return Path(os.path.relpath(target_path, start=base_dir)).as_posix()


def shrinkage_summary_markdown(
    metrics: pd.DataFrame,
    residual_rows: list[dict[str, object]],
    analysis_config: ModelAnalysisConfig,
) -> str:
    """Build executive-summary text for the primary residual pattern."""
    zero_row = first_residual_bin_row(residual_rows, analysis_config, "000_zero")
    highest_row = highest_observed_bin_row(residual_rows, analysis_config)
    zero_prediction = -row_float(zero_row, "mean_residual")
    high_residual = row_float(highest_row, "mean_residual")
    station_row = station_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    sample_row = background_sample_overall_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    return (
        "The strongest current finding is that the corrected background-inclusive setup exposes "
        "a failed ridge objective, not a successful baseline. On Kelpwatch-station rows in the "
        f"configured {analysis_config.analysis_split} {analysis_config.analysis_year} split, ridge "
        f"RMSE is `{format_decimal(row_float(station_row, 'rmse'), 4)}` and station-area bias is "
        f"`{format_percent(row_float(station_row, 'area_pct_bias'), 1)}`. On the background-inclusive "
        f"sample, area bias is `{format_percent(row_float(sample_row, 'area_pct_bias'), 1)}`, "
        "which is tracked separately from full-grid map calibration. "
        f"Zero-canopy rows receive mean predicted area `{format_decimal(zero_prediction, 1)} m2`, "
        f"while the highest observed canopy bin has mean residual `{format_decimal(high_residual, 1)} m2`."
    )


def first_residual_bin_row(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig, observed_bin: str
) -> dict[str, object]:
    """Return one primary residual-bin row by observed-bin label."""
    for row in filter_table_rows(
        rows, split=analysis_config.analysis_split, year=analysis_config.analysis_year
    ):
        if row.get("observed_bin") == observed_bin:
            return row
    return {}


def highest_observed_bin_row(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> dict[str, object]:
    """Return the primary residual-bin row with the highest observed mean."""
    primary = filter_table_rows(
        rows, split=analysis_config.analysis_split, year=analysis_config.analysis_year
    )
    if not primary:
        return {}
    return max(primary, key=lambda row: row_float(row, "observed_mean", default=-math.inf))


def metric_row(
    metrics: pd.DataFrame,
    model_name: str,
    split: str,
    *,
    metric_group: str = "overall",
    metric_group_value: str = "all",
    weighted: bool | None = None,
) -> dict[str, object]:
    """Return one metric row as a generic dictionary."""
    rows = metrics.loc[(metrics["model_name"] == model_name) & (metrics["split"] == split)].copy()
    if "metric_group" in rows.columns:
        rows = rows.loc[rows["metric_group"] == metric_group]
    if "metric_group_value" in rows.columns:
        rows = rows.loc[rows["metric_group_value"] == metric_group_value]
    if weighted is not None and "weighted" in rows.columns:
        rows = rows.loc[metric_weight_mask(rows) == weighted]
    if rows.empty:
        return {}
    return cast(dict[str, object], rows.iloc[0].to_dict())


def station_metric_row(metrics: pd.DataFrame, model_name: str, split: str) -> dict[str, object]:
    """Return the Kelpwatch-station metric row used for label-learning skill."""
    row = metric_row(
        metrics,
        model_name,
        split,
        metric_group="label_source",
        metric_group_value="kelpwatch_station",
        weighted=False,
    )
    if row:
        return row
    return metric_row(metrics, model_name, split, weighted=False)


def background_sample_overall_metric_row(
    metrics: pd.DataFrame, model_name: str, split: str
) -> dict[str, object]:
    """Return the overall row used for background-inclusive sample metrics."""
    row = metric_row(metrics, model_name, split, weighted=True)
    if row:
        return row
    return metric_row(metrics, model_name, split, weighted=False)


def report_metric_rows(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return Kelpwatch-station rows for the primary report comparison table."""
    rows = metrics.loc[metrics["split"] == split].copy()
    if "metric_group" in rows.columns:
        station_rows = rows.loc[
            (rows["metric_group"] == "label_source")
            & (rows["metric_group_value"] == "kelpwatch_station")
        ]
        if not station_rows.empty:
            rows = station_rows
        else:
            rows = rows.loc[rows["metric_group"] == "overall"]
    if "weighted" in rows.columns:
        rows = rows.loc[~metric_weight_mask(rows)]
    return rows


def metric_weight_mask(metrics: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for metric rows marked as weighted."""
    return metrics["weighted"].astype(str).str.lower().isin({"true", "1"})


def format_decimal(value: float, digits: int) -> str:
    """Format a finite decimal value or return `nan`."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    """Format a finite fraction as a percentage or return `nan`."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}%}"


IMAGE_MARKDOWN_PATTERN = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")


def write_html_report(
    markdown_lines: list[str], output_path: Path, markdown_base_dir: Path
) -> None:
    """Write a standalone HTML report with image files embedded as data URIs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body_lines = markdown_lines_to_html(markdown_lines, markdown_base_dir)
    html_lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Monterey Phase 1 Model Analysis</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.5; margin: 0; background: #f6f7f8; color: #1f2933; }",
        "main { max-width: 1040px; margin: 0 auto; padding: 32px 24px 56px; background: #ffffff; }",
        "h1, h2 { line-height: 1.2; color: #0f1720; }",
        "h1 { margin-top: 0; font-size: 2rem; }",
        "h2 { margin-top: 2rem; border-top: 1px solid #d8dee4; padding-top: 1.25rem; }",
        "table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }",
        "th, td { border: 1px solid #d8dee4; padding: 0.45rem 0.55rem; text-align: left; vertical-align: top; }",
        "th { background: #eef2f5; }",
        "code { background: #eef2f5; border-radius: 4px; padding: 0.1rem 0.25rem; }",
        "pre { background: #111827; color: #f9fafb; border-radius: 6px; padding: 1rem; overflow-x: auto; }",
        "pre code { background: transparent; padding: 0; }",
        "figure { margin: 1.4rem 0; }",
        "figcaption { color: #52616b; font-size: 0.86rem; margin-top: 0.4rem; }",
        "img { display: block; width: 100%; height: auto; border: 1px solid #d8dee4; border-radius: 6px; background: #ffffff; }",
        "ul { padding-left: 1.4rem; }",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        *body_lines,
        "</main>",
        "</body>",
        "</html>",
        "",
    ]
    output_path.write_text("\n".join(html_lines))


PdfLine = tuple[str, str]


def write_pdf_report(markdown_lines: list[str], output_path: Path, markdown_base_dir: Path) -> None:
    """Write a paginated PDF text report from the generated Markdown subset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = pdf_report_lines(markdown_lines, markdown_base_dir)
    with PdfPages(output_path) as pdf:
        fig = new_pdf_figure()
        y_position = PDF_TOP_Y
        for text, style in report_lines:
            line_height = pdf_line_height(style)
            if y_position - line_height < 0.04:
                pdf.savefig(fig)
                plt.close(fig)
                fig = new_pdf_figure()
                y_position = PDF_TOP_Y
            draw_pdf_line(fig, text, style, y_position)
            y_position -= line_height
        pdf.savefig(fig)
        plt.close(fig)


def pdf_report_lines(markdown_lines: list[str], markdown_base_dir: Path) -> list[PdfLine]:
    """Convert generated Markdown lines into plain styled PDF text lines."""
    output: list[PdfLine] = []
    in_code_block = False
    for line in markdown_lines:
        stripped = line.rstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not stripped:
            output.append(("", "body"))
            continue
        image_match = IMAGE_MARKDOWN_PATTERN.fullmatch(stripped)
        if image_match is not None:
            alt_text, source = image_match.groups()
            figure_path = resolve_markdown_pdf_path(source, markdown_base_dir)
            output.extend(wrapped_pdf_lines(f"[Figure] {alt_text}: {figure_path}", "body", 106))
        elif stripped.startswith("# "):
            output.extend(wrapped_pdf_lines(stripped[2:], "h1", 84))
        elif stripped.startswith("## "):
            output.extend(wrapped_pdf_lines(stripped[3:], "h2", 92))
        elif stripped.startswith("|") or in_code_block:
            output.extend(wrapped_pdf_lines(stripped, "mono", 116))
        else:
            output.extend(wrapped_pdf_lines(stripped, "body", 106))
    return output or [("No report content.", "body")]


def resolve_markdown_pdf_path(source: str, markdown_base_dir: Path) -> Path:
    """Resolve a Markdown asset source for PDF text references."""
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path
    return (markdown_base_dir / source_path).resolve()


def wrapped_pdf_lines(text: str, style: str, width: int) -> list[PdfLine]:
    """Wrap one text line into styled PDF line tuples."""
    wrapped = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
        subsequent_indent="  " if text.startswith("- ") else "",
    )
    return [(line, style) for line in wrapped] or [("", style)]


def new_pdf_figure() -> Any:
    """Create one blank report PDF page figure."""
    fig = plt.figure(figsize=(PDF_PAGE_WIDTH_IN, PDF_PAGE_HEIGHT_IN))
    axis = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.axis("off")
    return fig


def draw_pdf_line(fig: Any, text: str, style: str, y_position: float) -> None:
    """Draw one styled text line onto a PDF page."""
    font_size = PDF_MONO_FONT_SIZE if style == "mono" else PDF_BODY_FONT_SIZE
    if style == "h1":
        font_size = 14.0
    elif style == "h2":
        font_size = 11.0
    fig.text(
        PDF_MARGIN_X,
        y_position,
        text,
        ha="left",
        va="top",
        fontsize=font_size,
        fontweight="bold" if style in {"h1", "h2"} else "normal",
        family="monospace" if style == "mono" else "sans-serif",
        color="#111827",
    )


def pdf_line_height(style: str) -> float:
    """Return the vertical line spacing for one PDF style."""
    if style == "h1":
        return PDF_LINE_HEIGHT * 1.8
    if style == "h2":
        return PDF_LINE_HEIGHT * 1.55
    if style == "mono":
        return PDF_LINE_HEIGHT * 0.92
    return PDF_LINE_HEIGHT


def markdown_lines_to_html(markdown_lines: list[str], markdown_base_dir: Path) -> list[str]:
    """Convert the generated report Markdown subset into HTML."""
    output: list[str] = []
    index = 0
    while index < len(markdown_lines):
        line = markdown_lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
        elif stripped.startswith("```"):
            code_html, index = fenced_code_block_to_html(markdown_lines, index)
            output.append(code_html)
        elif stripped.startswith("|"):
            table_lines, index = collect_prefixed_lines(markdown_lines, index, "|")
            output.append(markdown_table_to_html(table_lines))
        elif stripped.startswith("- "):
            list_lines, index = collect_prefixed_lines(markdown_lines, index, "- ")
            output.append(markdown_list_to_html(list_lines))
        elif stripped.startswith("# "):
            output.append(f"<h1>{inline_markdown_to_html(stripped[2:])}</h1>")
            index += 1
        elif stripped.startswith("## "):
            output.append(f"<h2>{inline_markdown_to_html(stripped[3:])}</h2>")
            index += 1
        elif IMAGE_MARKDOWN_PATTERN.fullmatch(stripped):
            output.append(image_line_to_html(stripped, markdown_base_dir))
            index += 1
        else:
            paragraph_lines, index = collect_paragraph_lines(markdown_lines, index)
            paragraph = " ".join(item.strip() for item in paragraph_lines)
            output.append(f"<p>{inline_markdown_to_html(paragraph)}</p>")
    return output


def fenced_code_block_to_html(markdown_lines: list[str], start_index: int) -> tuple[str, int]:
    """Convert one fenced code block to HTML and return the next line index."""
    code_lines: list[str] = []
    index = start_index + 1
    while index < len(markdown_lines) and not markdown_lines[index].strip().startswith("```"):
        code_lines.append(markdown_lines[index])
        index += 1
    next_index = index + 1 if index < len(markdown_lines) else index
    code = html.escape("\n".join(code_lines))
    return f"<pre><code>{code}</code></pre>", next_index


def collect_prefixed_lines(
    markdown_lines: list[str], start_index: int, prefix: str
) -> tuple[list[str], int]:
    """Collect consecutive nonblank Markdown lines with a shared prefix."""
    collected: list[str] = []
    index = start_index
    while index < len(markdown_lines) and markdown_lines[index].strip().startswith(prefix):
        collected.append(markdown_lines[index].strip())
        index += 1
    return collected, index


def collect_paragraph_lines(markdown_lines: list[str], start_index: int) -> tuple[list[str], int]:
    """Collect lines that form one plain paragraph in the generated Markdown."""
    collected: list[str] = []
    index = start_index
    while index < len(markdown_lines):
        stripped = markdown_lines[index].strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("- ")
            or stripped.startswith("```")
            or IMAGE_MARKDOWN_PATTERN.fullmatch(stripped)
        ):
            break
        collected.append(markdown_lines[index])
        index += 1
    return collected, index


def markdown_table_to_html(table_lines: list[str]) -> str:
    """Convert a simple Markdown pipe table into HTML."""
    if len(table_lines) < 2:
        return ""
    headers = split_markdown_table_row(table_lines[0])
    rows = [split_markdown_table_row(line) for line in table_lines[2:]]
    header_html = "".join(f"<th>{inline_markdown_to_html(value)}</th>" for value in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{inline_markdown_to_html(value)}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return "\n".join(
        [
            "<table>",
            "<thead>",
            f"<tr>{header_html}</tr>",
            "</thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
        ]
    )


def split_markdown_table_row(line: str) -> list[str]:
    """Split a Markdown pipe-table row into trimmed cell values."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_list_to_html(list_lines: list[str]) -> str:
    """Convert consecutive Markdown bullet lines into an HTML unordered list."""
    items = [
        f"<li>{inline_markdown_to_html(line.strip()[2:])}</li>"
        for line in list_lines
        if line.strip().startswith("- ")
    ]
    return "\n".join(["<ul>", *items, "</ul>"])


def image_line_to_html(line: str, markdown_base_dir: Path) -> str:
    """Convert one Markdown image line into an embedded HTML image figure."""
    match = IMAGE_MARKDOWN_PATTERN.fullmatch(line)
    if match is None:
        return f"<p>{inline_markdown_to_html(line)}</p>"
    alt_text, image_reference = match.groups()
    image_path = Path(image_reference)
    if not image_path.is_absolute():
        image_path = markdown_base_dir / image_path
    source = image_data_uri(image_path) if image_path.is_file() else image_reference
    escaped_source = html.escape(source, quote=True)
    escaped_alt = html.escape(alt_text, quote=True)
    caption = inline_markdown_to_html(alt_text)
    return (
        "<figure>"
        f'<img src="{escaped_source}" alt="{escaped_alt}">'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def image_data_uri(image_path: Path) -> str:
    """Read an image file and encode it as a browser-ready data URI."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type_for_image(image_path)};base64,{encoded}"


def mime_type_for_image(image_path: Path) -> str:
    """Return a MIME type for supported report image suffixes."""
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".svg":
        return "image/svg+xml"
    return "image/png"


def inline_markdown_to_html(text: str) -> str:
    """Convert the small inline Markdown subset used by the report to HTML."""
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


def stage_distribution_markdown(rows: list[dict[str, object]]) -> str:
    """Build Markdown summary text for key stage distribution rows."""
    lines = [
        "| Stage | Split | Rows | Zero | Positive | 900 m2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['split']} | {row['row_count']} | {row['zero_count']} | {row['positive_count']} | {row['saturated_count']} |"
        )
    return "\n".join(lines)


def background_correction_markdown(data: AnalysisData) -> str:
    """Build report text describing the corrected full-grid/background contract."""
    aligned_counts = label_source_counts(data.aligned)
    prediction_counts = label_source_counts(data.model_predictions)
    return "\n".join(
        [
            "The first Phase 0 alignment only sampled AEF values at Kelpwatch station centers. Task 11 corrects that data contract: non-station cells are retained as weak-label `assumed_background` rows, and station-supported rows are flagged as `kelpwatch_station`.",
            "",
            "| Artifact | Loaded rows | Kelpwatch station rows | Assumed background rows |",
            "|---|---:|---:|---:|",
            (
                f"| Model input sample | {len(data.aligned)} | "
                f"{aligned_counts.get('kelpwatch_station', 0)} | "
                f"{aligned_counts.get('assumed_background', 0)} |"
            ),
            (
                f"| Prediction rows in this report | {len(data.model_predictions)} | "
                f"{prediction_counts.get('kelpwatch_station', 0)} | "
                f"{prediction_counts.get('assumed_background', 0)} |"
            ),
            "",
            "Overall full-grid metrics are useful for calibration and map QA, but quarterly target-framing and persistence diagnostics are computed only on Kelpwatch-station rows because background cells do not have quarterly Kelpwatch support.",
        ]
    )


def label_source_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count label-source provenance values with a fallback for older fixtures."""
    if "label_source" in dataframe.columns:
        return {
            str(label_source): int(count)
            for label_source, count in dataframe["label_source"].value_counts().items()
        }
    if "is_kelpwatch_observed" in dataframe.columns:
        observed = dataframe["is_kelpwatch_observed"].fillna(False).astype(bool)
        return {
            "kelpwatch_station": int(observed.sum()),
            "assumed_background": int((~observed).sum()),
        }
    return {"kelpwatch_station": int(dataframe["kelpwatch_station_id"].notna().sum())}


def metric_summary_markdown(metrics: pd.DataFrame, analysis_config: ModelAnalysisConfig) -> str:
    """Build Markdown text for primary ridge metrics."""
    station_row = station_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    sample_row = background_sample_overall_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    if not station_row:
        return "Primary metric row was not found."
    return (
        f"For the primary `{analysis_config.analysis_split}` split, the Kelpwatch-station "
        f"ridge row has RMSE `{format_decimal(row_float(station_row, 'rmse'), 4)}`, R2 "
        f"`{format_decimal(row_float(station_row, 'r2'), 4)}`, and area percent bias "
        f"`{format_percent(row_float(station_row, 'area_pct_bias'), 2)}`. The background-inclusive "
        f"sample area percent bias is `{format_percent(row_float(sample_row, 'area_pct_bias'), 2)}`."
    )


def baseline_comparison_markdown(
    metrics: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> str:
    """Build a compact implemented-baseline comparison table."""
    rows = report_metric_rows(metrics, analysis_config.analysis_split)
    if rows.empty:
        return "No baseline metric rows were available for the primary split."
    lines = [
        "| Model | MAE | RMSE | R2 | Spearman | Area pct bias | F1 at 10% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.to_dict("records"):
        row_dict = cast(dict[str, object], row)
        lines.append(
            f"| {row_dict['model_name']} | "
            f"{format_decimal(row_float(row_dict, 'mae'), 4)} | "
            f"{format_decimal(row_float(row_dict, 'rmse'), 4)} | "
            f"{format_decimal(row_float(row_dict, 'r2'), 3)} | "
            f"{format_decimal(row_float(row_dict, 'spearman'), 3)} | "
            f"{format_percent(row_float(row_dict, 'area_pct_bias'), 2)} | "
            f"{format_decimal(row_float(row_dict, 'f1_ge_10pct'), 3)} |"
        )
    return "\n".join(lines)


def baseline_calibration_markdown(
    metrics: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> str:
    """Build prose comparing ridge skill against no-skill area calibration."""
    ridge_station = station_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    no_skill_station = station_metric_row(
        metrics, "no_skill_train_mean", analysis_config.analysis_split
    )
    ridge_sample = background_sample_overall_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    no_skill_sample = background_sample_overall_metric_row(
        metrics, "no_skill_train_mean", analysis_config.analysis_split
    )
    if not ridge_station or not no_skill_station:
        return (
            "Baseline comparison is incomplete because at least one implemented reference "
            "metric row is missing for the primary split."
        )
    no_skill_rmse = row_float(no_skill_station, "rmse")
    ridge_rmse = row_float(ridge_station, "rmse")
    rmse_reduction = safe_ratio(no_skill_rmse - ridge_rmse, no_skill_rmse)
    return (
        "On Kelpwatch-station rows, ridge only modestly improves over the train-mean baseline "
        f"(RMSE reduction `{format_percent(rmse_reduction, 1)}`) and still misses most canopy "
        f"area: station area bias is `{format_percent(row_float(ridge_station, 'area_pct_bias'), 2)}`. "
        "The background-inclusive sample shows the calibration tradeoff: no-skill area "
        f"bias is `{format_percent(row_float(no_skill_sample, 'area_pct_bias'), 2)}` and ridge "
        f"area bias is `{format_percent(row_float(ridge_sample, 'area_pct_bias'), 2)}`. "
        "This is still a sampling/objective calibration problem to revisit, but it is a more useful reference baseline than the population-expanded weighted ridge."
    )


def phase1_harness_status_markdown(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> str:
    """Build prose describing the Phase 1 report harness outputs."""
    comparison_count = len(tables.phase1_model_comparison)
    data_health_count = len(tables.data_health)
    return (
        "The Phase 1 harness is active for the current annual-max workflow. It adds stable "
        "table contracts for future reference baselines, domain-mask rows, and imbalance-aware "
        "models without changing the current ridge predictions. "
        f"The model comparison table currently has `{comparison_count}` rows at "
        f"`{analysis_config.phase1_model_comparison_path}`. The data-health table has "
        f"`{data_health_count}` rows at `{analysis_config.data_health_path}`. Future tasks "
        "should append comparable rows instead of changing these schemas."
    )


def model_comparison_markdown(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> str:
    """Build a compact Phase 1 model-comparison preview table."""
    primary_rows = [
        row
        for row in rows
        if row.get("split") == analysis_config.analysis_split
        and str(row.get("year")) == str(analysis_config.analysis_year)
    ]
    if not primary_rows:
        primary_rows = rows[:6]
    if not primary_rows:
        return "No Phase 1 model-comparison rows were available."
    lines = [
        "| Model | Scope | Mask | Rows | RMSE | R2 | Area pct bias |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in primary_rows[:8]:
        lines.append(
            f"| {row.get('model_name', '')} | "
            f"{row.get('evaluation_scope', '')} | "
            f"{row.get('mask_status', '')} | "
            f"{row.get('row_count', '')} | "
            f"{format_decimal(row_float(row, 'rmse'), 4)} | "
            f"{format_decimal(row_float(row, 'r2'), 3)} | "
            f"{format_percent(row_float(row, 'area_pct_bias'), 2)} |"
        )
    return "\n".join(lines)


def reference_baseline_ranking_markdown(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> str:
    """Build Markdown ranking reference baselines against the AEF ridge model."""
    station_rows = sorted(
        [
            row
            for row in rows
            if row.get("split") == analysis_config.analysis_split
            and row.get("evaluation_scope") == "kelpwatch_station_sample"
        ],
        key=lambda row: row_float(row, "rmse", default=math.inf),
    )
    full_grid_rows = sorted(
        [
            row
            for row in rows
            if row.get("split") == analysis_config.analysis_split
            and row.get("evaluation_scope") == "full_grid_prediction"
            and row.get("label_source") in {"all", "kelpwatch_station", "assumed_background"}
        ],
        key=lambda row: (
            str(row.get("label_source", "")),
            abs(row_float(row, "area_pct_bias", default=math.inf)),
        ),
    )
    if not station_rows and not full_grid_rows:
        return "Reference-baseline ranking rows are not available yet."
    lines = [
        "Lower RMSE is better for Kelpwatch-station pixel skill; lower absolute area percent bias is better for full-grid calibration. These comparisons evaluate Kelpwatch-style label reproduction, not independent field truth.",
        "",
    ]
    if station_rows:
        lines.extend(
            [
                "| Station rank | Model | Rows | RMSE | MAE | F1 at 10% |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(station_rows[:8], start=1):
            lines.append(
                f"| {rank} | {row.get('model_name', '')} | "
                f"{row.get('row_count', '')} | "
                f"{format_decimal(row_float(row, 'rmse'), 4)} | "
                f"{format_decimal(row_float(row, 'mae'), 4)} | "
                f"{format_decimal(row_float(row, 'f1_ge_10pct'), 3)} |"
            )
        lines.append("")
    if full_grid_rows:
        lines.extend(
            [
                "| Area rank | Label source | Model | Rows | Area pct bias | Predicted area |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(full_grid_rows[:12], start=1):
            lines.append(
                f"| {rank} | {row.get('label_source', '')} | "
                f"{row.get('model_name', '')} | "
                f"{row.get('row_count', '')} | "
                f"{format_percent(row_float(row, 'area_pct_bias'), 2)} | "
                f"{format_decimal(row_float(row, 'predicted_canopy_area'), 1)} |"
            )
    return "\n".join(lines)


def data_health_markdown(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> str:
    """Build a compact Phase 1 data-health preview table."""
    primary_rows = [
        row
        for row in rows
        if row.get("split") in {"all", analysis_config.analysis_split}
        and str(row.get("year")) in {str(analysis_config.analysis_year), "all"}
    ]
    if not primary_rows:
        primary_rows = rows[:8]
    if not primary_rows:
        return "No Phase 1 data-health rows were available."
    lines = [
        "| Check | Split | Year | Label source | Rows | Rate |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in primary_rows[:10]:
        lines.append(
            f"| {row.get('check_name', '')} | "
            f"{row.get('split', '')} | "
            f"{row.get('year', '')} | "
            f"{row.get('label_source', '')} | "
            f"{row.get('row_count', '')} | "
            f"{format_decimal(row_float(row, 'rate'), 3)} |"
        )
    return "\n".join(lines)


def map_section_markdown(analysis_config: ModelAnalysisConfig) -> str:
    """Build prose for the observed, predicted, and residual map section."""
    return (
        "The three-panel model review map uses the latest static map for the primary "
        f"`{analysis_config.analysis_split}` `{analysis_config.analysis_year}` split. The "
        "observed and predicted panels use the same canopy-area scale, and the error panel "
        "uses `observed - predicted`, so positive residuals are underprediction. The linked "
        f"interactive map is `{analysis_config.residual_interactive_html}`."
    )


def prediction_distribution_markdown(row: dict[str, object]) -> str:
    """Build Markdown text for primary prediction distribution."""
    return (
        f"Observed saturated count: `{row.get('observed_saturated_count')}`. "
        f"Predicted p99: `{row_float(row, 'predicted_p99'):.1f}`. "
        f"Predicted max: `{row_float(row, 'predicted_max'):.1f}`. "
        f"Mean prediction on observed `900 m2` rows: "
        f"`{row_float(row, 'observed_900_prediction_mean'):.1f}`."
    )


def residual_bin_interpretation_markdown(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> str:
    """Build prose interpreting residuals by ordered observed canopy bin."""
    primary_rows = filter_table_rows(
        rows, split=analysis_config.analysis_split, year=analysis_config.analysis_year
    )
    if not primary_rows:
        return "Primary residual-bin rows were not available."
    zero_row = first_residual_bin_row(rows, analysis_config, "000_zero")
    highest_row = highest_observed_bin_row(rows, analysis_config)
    return (
        "Residual bins are sorted by numeric observed canopy area. Positive residuals mean "
        "`observed - predicted`, so positive values are underprediction. The pattern is "
        f"shrinkage: zero rows have mean predicted area "
        f"`{format_decimal(-row_float(zero_row, 'mean_residual'), 1)} m2`, while the "
        f"highest observed bin `{highest_row.get('observed_bin', 'missing')}` has mean "
        f"observed area `{format_decimal(row_float(highest_row, 'observed_mean'), 1)} m2`, "
        f"mean predicted area `{format_decimal(row_float(highest_row, 'predicted_mean'), 1)} m2`, "
        f"and mean residual `{format_decimal(row_float(highest_row, 'mean_residual'), 1)} m2`."
    )


def threshold_sensitivity_markdown(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> str:
    """Build Markdown threshold diagnostics for the primary split and year."""
    primary_rows = filter_table_rows(
        rows, split=analysis_config.analysis_split, year=analysis_config.analysis_year
    )
    if not primary_rows:
        return "Primary threshold-sensitivity rows were not available."
    best_row = max(primary_rows, key=lambda row: row_float(row, "f1", default=-math.inf))
    high_row = max(primary_rows, key=lambda row: row_float(row, "threshold_fraction"))
    lines = [
        (
            f"Among the configured diagnostic thresholds, `{format_percent(row_float(best_row, 'threshold_fraction'), 0)}` "
            f"has the highest F1 (`{format_decimal(row_float(best_row, 'f1'), 3)}`) on the primary split. "
            f"The highest threshold, `{format_percent(row_float(high_row, 'threshold_fraction'), 0)}`, has recall "
            f"`{format_decimal(row_float(high_row, 'recall'), 3)}`, which confirms poor high-canopy recall."
        ),
        "",
        "| Threshold | Observed positive | Predicted positive | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {format_percent(row_float(row, 'threshold_fraction'), 0)} | "
            f"{format_percent(row_float(row, 'positive_fraction'), 1)} | "
            f"{format_percent(row_float(row, 'predicted_positive_fraction'), 1)} | "
            f"{format_decimal(row_float(row, 'precision'), 3)} | "
            f"{format_decimal(row_float(row, 'recall'), 3)} | "
            f"{format_decimal(row_float(row, 'f1'), 3)} |"
        )
    return "\n".join(lines)


def top_target_framings_markdown(rows: list[dict[str, object]]) -> str:
    """Build Markdown table for target framings with highest rank correlation."""
    ranked = sorted(
        rows,
        key=lambda row: (
            abs(row_float(row, "spearman_with_prediction"))
            if np.isfinite(row_float(row, "spearman_with_prediction"))
            else -1
        ),
        reverse=True,
    )[:6]
    lines = ["| Target | Kind | Spearman | Mean | Positive fraction |", "|---|---|---:|---:|---:|"]
    for row in ranked:
        lines.append(
            f"| {row['target_name']} | {row['target_kind']} | {row_float(row, 'spearman_with_prediction'):.3f} | {row_float(row, 'mean'):.3f} | {row_float(row, 'positive_fraction'):.3f} |"
        )
    return "\n".join(lines)


def target_framing_interpretation_markdown(rows: list[dict[str, object]]) -> str:
    """Build prose explaining how to interpret target-framing Spearman values."""
    annual_max = first_named_row(rows, "annual_max_area")
    annual_mean = first_named_row(rows, "annual_mean_area")
    mean_presence = first_named_row(rows, "mean_presence_fraction")
    return (
        "The Spearman target-framing plot is a rank-agreement diagnostic for the existing "
        "annual-max ridge predictions. The model was not retrained for each target, so this "
        "does not prove that any alternative target is better. Alternative temporal target "
        "inputs are out of active Phase 1 scope; these rows are retained as Phase 0 evidence "
        "while annual max remains fixed. Current predictions rank "
        f"`annual_mean_area` (`{format_decimal(row_float(annual_mean, 'spearman_with_prediction'), 3)}`), "
        f"`annual_max_area` (`{format_decimal(row_float(annual_max, 'spearman_with_prediction'), 3)}`), "
        f"and `mean_presence_fraction` (`{format_decimal(row_float(mean_presence, 'spearman_with_prediction'), 3)}`) "
        "similarly, suggesting the ridge output behaves like a general annual kelp-intensity "
        "or seasonal-regularity score rather than a clean saturated-peak predictor."
    )


def first_named_row(rows: list[dict[str, object]], target_name: str) -> dict[str, object]:
    """Return the first target-framing row with a matching target name."""
    for row in rows:
        if row.get("target_name") == target_name:
            return row
    return {}


def feature_separability_markdown(rows: list[dict[str, object]]) -> str:
    """Build Markdown text for feature separability summaries."""
    if not rows:
        return "No complete feature rows were available for projection diagnostics."
    max_distance = max_feature_distance(rows)
    return (
        f"The largest PCA group-center distance from the zero group is `{max_distance:.2f}`. "
        "This is a diagnostic for separability only; it is not model skill."
    )


def spatial_readiness_markdown(rows: list[dict[str, object]]) -> str:
    """Build Markdown text for spatial holdout readiness."""
    ready_count = sum(bool(row["enough_for_holdout"]) for row in rows)
    return (
        f"`{ready_count}` latitude bands meet the current minimum counts for a crude "
        "within-Monterey spatial holdout. This is useful for smoke-test diagnostics, but "
        "it is not enough to claim robust spatial generalization. Phase 1 should either "
        "add a second smoke region or broaden the California slice before making larger "
        "spatial claims."
    )


def interpretation_markdown(
    metrics: pd.DataFrame, tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> str:
    """Build the report-level interpretation for the active Phase 1 workflow."""
    ridge_station = station_metric_row(
        metrics, analysis_config.model_name, analysis_config.analysis_split
    )
    return (
        "The current background-inclusive ridge result should be treated as a weak starting "
        "baseline. The "
        f"Kelpwatch-station test R2 is `{format_decimal(row_float(ridge_station, 'r2'), 3)}` "
        "and the model still severely underpredicts canopy-support rows while leaking small "
        "positive predictions over a large assumed-background population. The next work should "
        "therefore focus on reference baselines, bathymetry/DEM domain filtering, calibration, "
        "and imbalance-aware modeling before interpreting stronger nonlinear models as the main "
        "answer."
    )


def decision_matrix_markdown(rows: list[dict[str, object]]) -> str:
    """Build Markdown table for Phase 1 decision branches."""
    lines = [
        "| Branch | Evidence | Proposed next tasks |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['branch']} | {row['evidence_status']}: {row['triggering_evidence']} | {row['proposed_next_tasks']} |"
        )
    return "\n".join(lines)


def recommended_phase1_branch(rows: list[dict[str, object]]) -> str:
    """Pick the first strong non-scale-up branch as the report recommendation."""
    for row in rows:
        if row["evidence_status"] == "strong" and row["branch"] != "Scale-up Phase 1":
            return str(row["branch"])
    return str(rows[0]["branch"]) if rows else "Undetermined"


def write_manifest(
    data: AnalysisData, tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write the model-analysis JSON manifest."""
    payload = {
        "command": "analyze-model",
        "config_path": str(analysis_config.config_path),
        "model_name": analysis_config.model_name,
        "split": analysis_config.analysis_split,
        "year": analysis_config.analysis_year,
        "inputs": {
            "labels": str(analysis_config.label_path),
            "aligned_table": str(analysis_config.aligned_table_path),
            "split_manifest": str(analysis_config.split_manifest_path),
            "predictions": str(analysis_config.predictions_path),
            "metrics": str(analysis_config.metrics_path),
        },
        "outputs": {
            "report": str(analysis_config.report_path),
            "html_report": str(analysis_config.html_report_path),
            "pdf_report": str(analysis_config.pdf_report_path),
            "observed_predicted_residual_map": str(
                analysis_config.observed_predicted_residual_map_figure
            ),
            "residual_interactive_html": str(analysis_config.residual_interactive_html),
            "label_distribution": str(analysis_config.label_distribution_path),
            "target_framing": str(analysis_config.target_framing_path),
            "prediction_distribution": str(analysis_config.prediction_distribution_path),
            "phase1_decision": str(analysis_config.phase1_decision_path),
            "phase1_model_comparison": str(analysis_config.phase1_model_comparison_path),
            "reference_area_calibration": str(analysis_config.reference_area_calibration_path),
            "data_health": str(analysis_config.data_health_path),
            "manifest": str(analysis_config.manifest_path),
        },
        "row_counts": {
            "labels": int(len(data.labels)),
            "aligned": int(len(data.aligned)),
            "split_manifest": int(len(data.split_manifest)),
            "model_predictions": int(len(data.model_predictions)),
            "stage_distribution": len(tables.stage_distribution),
            "target_framing": len(tables.target_framing),
            "phase1_decision": len(tables.phase1_decision),
            "phase1_model_comparison": len(tables.phase1_model_comparison),
            "reference_area_calibration": len(tables.reference_area_calibration),
            "data_health": len(tables.data_health),
        },
    }
    analysis_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_config.manifest_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
