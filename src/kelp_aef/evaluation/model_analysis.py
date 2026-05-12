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
from kelp_aef.domain.reporting_mask import (
    MASK_RETAIN_COLUMN,
    ReportingDomainMask,
    apply_reporting_domain_mask,
    evaluation_scope,
    load_reporting_domain_mask,
    mask_status,
    masked_output_path,
)
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
DEFAULT_THRESHOLD_SELECTION_SPLIT = "validation"
DEFAULT_THRESHOLD_TEST_SPLIT = "test"
DEFAULT_LATITUDE_BAND_COUNT = 12
DEFAULT_MAX_PROJECTION_ROWS = 50_000
DEFAULT_TOP_RESIDUAL_COUNT = 50
DEFAULT_FALL_QUARTER = 4
DEFAULT_WINTER_QUARTER = 1
DEFAULT_OBSERVED_AREA_BINS = (0.0, 1.0, 90.0, 225.0, 450.0, 810.0, 900.0)
DEFAULT_THRESHOLD_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.50, 0.90)
REQUIRED_BINARY_THRESHOLD_FRACTIONS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_MIN_BINARY_SELECTION_POSITIVES = 100
DEFAULT_MIN_BINARY_SELECTION_POSITIVE_RATE = 0.001
BINARY_THRESHOLD_SELECTION_POLICY = (
    "validation_only_prefer_highest_threshold_with_positive_support_then_f1"
)
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
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "crm_elevation_m",
    "crm_depth_m",
    "depth_bin",
    "elevation_bin",
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
BINARY_THRESHOLD_PREVALENCE_FIELDS = (
    "data_scope",
    "mask_status",
    "evaluation_scope",
    "split",
    "year",
    "label_source",
    "threshold_fraction",
    "threshold_area",
    "threshold_label",
    "threshold_operator",
    "threshold_role",
    "row_count",
    "target_count",
    "station_count",
    "positive_count",
    "positive_rate",
    "assumed_background_count",
    "assumed_background_positive_count",
    "assumed_background_positive_rate",
    "observed_positive_area",
    "mean_positive_area",
)
BINARY_THRESHOLD_COMPARISON_FIELDS = (
    "data_scope",
    "mask_status",
    "evaluation_scope",
    "split",
    "year",
    "label_source",
    "model_name",
    "threshold_fraction",
    "threshold_area",
    "threshold_label",
    "threshold_operator",
    "threshold_role",
    "row_count",
    "target_count",
    "station_count",
    "positive_count",
    "positive_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
    "precision",
    "recall",
    "f1",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_positive_area",
    "false_negative_count",
    "false_negative_rate",
    "false_negative_area",
    "observed_positive_area",
    "predicted_positive_area",
    "assumed_background_count",
    "assumed_background_false_positive_count",
    "assumed_background_false_positive_rate",
    "assumed_background_false_positive_area",
)
BINARY_THRESHOLD_RECOMMENDATION_FIELDS = (
    "selection_split",
    "selection_year",
    "test_split",
    "selection_policy",
    "recommendation_status",
    "selected_candidate",
    "threshold_rank",
    "model_name",
    "threshold_fraction",
    "threshold_area",
    "threshold_label",
    "threshold_operator",
    "threshold_role",
    "positive_support_ok",
    "validation_row_count",
    "validation_positive_count",
    "validation_positive_rate",
    "validation_predicted_positive_rate",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_false_positive_rate",
    "validation_false_positive_area",
    "validation_false_negative_area",
    "validation_assumed_background_false_positive_rate",
    "validation_assumed_background_false_positive_area",
    "recommended_threshold_fraction",
    "recommended_threshold_label",
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
CLASS_BALANCE_FIELDS = (
    "data_scope",
    "mask_status",
    "evaluation_scope",
    "split",
    "year",
    "label_source",
    "row_count",
    "station_count",
    "zero_count",
    "zero_rate",
    "positive_count",
    "positive_rate",
    "positive_ge_1pct_count",
    "positive_ge_1pct_rate",
    "positive_ge_5pct_count",
    "positive_ge_5pct_rate",
    "positive_ge_10pct_count",
    "positive_ge_10pct_rate",
    "high_canopy_count",
    "high_canopy_rate",
    "very_high_canopy_count",
    "very_high_canopy_rate",
    "saturated_count",
    "saturated_rate",
    "assumed_background_count",
    "assumed_background_rate",
    "observed_canopy_area",
    "mean_observed_canopy_area",
)
TARGET_BALANCE_FIELDS = CLASS_BALANCE_FIELDS
BACKGROUND_RATE_FIELDS = CLASS_BALANCE_FIELDS
RESIDUAL_DOMAIN_CONTEXT_FIELDS = (
    "model_name",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "domain_mask_reason",
    "depth_bin",
    "elevation_bin",
    "label_source",
    "observed_bin",
    "residual_class",
    "row_count",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mae",
    "rmse",
    "mean_residual",
    "median_residual",
    "underprediction_count",
    "overprediction_count",
    "high_error_count",
    "high_error_threshold",
    "zero_observed_count",
    "positive_observed_count",
    "high_canopy_count",
    "saturated_count",
    "mean_crm_depth_m",
    "mean_crm_elevation_m",
)
TOP_RESIDUAL_CONTEXT_FIELDS = (
    "residual_type",
    "rank",
    "model_name",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "kelpwatch_station_id",
    "label_source",
    "is_kelpwatch_observed",
    "longitude",
    "latitude",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "crm_elevation_m",
    "crm_depth_m",
    "depth_bin",
    "elevation_bin",
    "kelp_max_y",
    "pred_kelp_max_y",
    "observed_canopy_area",
    "predicted_canopy_area",
    "residual_kelp_max_y",
    "abs_residual_kelp_max_y",
    "residual_class",
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
    sample_predictions_path: Path | None
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
    binary_threshold_comparison_path: Path
    binary_threshold_recommendation_path: Path
    binary_threshold_prevalence_path: Path
    spatial_readiness_path: Path
    feature_separability_path: Path
    phase1_decision_path: Path
    phase1_model_comparison_path: Path
    data_health_path: Path
    class_balance_by_split_path: Path
    target_balance_by_label_source_path: Path
    background_rate_summary_path: Path
    residual_domain_context_path: Path
    residual_by_mask_reason_path: Path
    residual_by_depth_bin_path: Path
    top_residual_context_path: Path
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
    class_balance_figure: Path
    binary_threshold_comparison_figure: Path
    residual_domain_context_figure: Path
    observed_predicted_residual_map_figure: Path
    residual_interactive_html: Path
    top_residual_count: int
    domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class BalanceSource:
    """Prepared target-balance source frame plus report metadata."""

    data_scope: str
    mask_status: str
    evaluation_scope: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class BinaryThresholdDefinition:
    """Configured annual-max binary threshold metadata."""

    fraction: float
    label: str
    operator: str
    role: str


@dataclass(frozen=True)
class AnalysisTables:
    """All tabular outputs needed to write the report and manifest."""

    stage_distribution: list[dict[str, object]]
    prediction_distribution: list[dict[str, object]]
    residual_by_bin: list[dict[str, object]]
    residual_by_persistence: list[dict[str, object]]
    target_framing: list[dict[str, object]]
    threshold_sensitivity: list[dict[str, object]]
    binary_threshold_prevalence: list[dict[str, object]]
    binary_threshold_comparison: list[dict[str, object]]
    binary_threshold_recommendation: list[dict[str, object]]
    spatial_readiness: list[dict[str, object]]
    feature_separability: list[dict[str, object]]
    phase1_decision: list[dict[str, object]]
    phase1_model_comparison: list[dict[str, object]]
    data_health: list[dict[str, object]]
    class_balance_by_split: list[dict[str, object]]
    target_balance_by_label_source: list[dict[str, object]]
    background_rate_summary: list[dict[str, object]]
    residual_domain_context: list[dict[str, object]]
    residual_by_mask_reason: list[dict[str, object]]
    residual_by_depth_bin: list[dict[str, object]]
    top_residual_context: list[dict[str, object]]
    quarter_mapping: list[dict[str, object]]
    reference_area_calibration: list[dict[str, object]]


@dataclass(frozen=True)
class AnalysisData:
    """Loaded model-analysis inputs after validation and model filtering."""

    labels: pd.DataFrame
    aligned: pd.DataFrame
    split_manifest: pd.DataFrame
    predictions: pd.DataFrame
    sample_predictions: pd.DataFrame
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
    map_settings = optional_mapping(reports.get("map_residuals"), "reports.map_residuals")
    domain_mask = load_reporting_domain_mask(config)
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
        sample_predictions_path=optional_path(baselines.get("sample_predictions")),
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
        binary_threshold_comparison_path=output_path(
            outputs,
            "model_analysis_binary_threshold_comparison",
            tables_dir / "model_analysis_binary_threshold_comparison.csv",
        ),
        binary_threshold_recommendation_path=output_path(
            outputs,
            "model_analysis_binary_threshold_recommendation",
            tables_dir / "model_analysis_binary_threshold_recommendation.csv",
        ),
        binary_threshold_prevalence_path=output_path(
            outputs,
            "model_analysis_binary_threshold_prevalence",
            tables_dir / "model_analysis_binary_threshold_prevalence.csv",
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
        class_balance_by_split_path=output_path(
            outputs,
            "model_analysis_class_balance_by_split",
            tables_dir / "model_analysis_class_balance_by_split.csv",
        ),
        target_balance_by_label_source_path=output_path(
            outputs,
            "model_analysis_target_balance_by_label_source",
            tables_dir / "model_analysis_target_balance_by_label_source.csv",
        ),
        background_rate_summary_path=output_path(
            outputs,
            "model_analysis_background_rate_summary",
            tables_dir / "model_analysis_background_rate_summary.csv",
        ),
        residual_domain_context_path=output_path(
            outputs,
            "model_analysis_residual_by_domain_context",
            tables_dir / "model_analysis_residual_by_domain_context.csv",
        ),
        residual_by_mask_reason_path=output_path(
            outputs,
            "model_analysis_residual_by_mask_reason",
            tables_dir / "model_analysis_residual_by_mask_reason.csv",
        ),
        residual_by_depth_bin_path=output_path(
            outputs,
            "model_analysis_residual_by_depth_bin",
            tables_dir / "model_analysis_residual_by_depth_bin.csv",
        ),
        top_residual_context_path=output_path(
            outputs,
            "top_residual_stations_domain_context",
            tables_dir / "top_residual_stations.domain_context.csv",
        ),
        reference_area_calibration_path=masked_output_path(
            outputs,
            unmasked_key="reference_baseline_area_calibration",
            masked_key="reference_baseline_area_calibration_masked",
            default=tables_dir / "reference_baseline_area_calibration.csv",
            mask_config=domain_mask,
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
        class_balance_figure=output_path(
            outputs,
            "model_analysis_class_balance_figure",
            figures_dir / "model_analysis_class_balance.png",
        ),
        binary_threshold_comparison_figure=output_path(
            outputs,
            "model_analysis_binary_threshold_comparison_figure",
            figures_dir / "model_analysis_binary_threshold_comparison.png",
        ),
        residual_domain_context_figure=output_path(
            outputs,
            "model_analysis_residual_by_domain_context_figure",
            figures_dir / "model_analysis_residual_by_domain_context.png",
        ),
        observed_predicted_residual_map_figure=masked_output_path(
            outputs,
            unmasked_key="ridge_observed_predicted_residual_map",
            masked_key="ridge_observed_predicted_residual_map_masked",
            default=figures_dir / "ridge_2022_observed_predicted_residual.png",
            mask_config=domain_mask,
        ),
        residual_interactive_html=masked_output_path(
            outputs,
            unmasked_key="ridge_residual_interactive",
            masked_key="ridge_residual_interactive_masked",
            default=figures_dir / "ridge_2022_residual_interactive.html",
            mask_config=domain_mask,
        ),
        top_residual_count=optional_positive_int(
            settings.get("top_residual_count", map_settings.get("top_residual_count")),
            "reports.model_analysis.top_residual_count",
            DEFAULT_TOP_RESIDUAL_COUNT,
        ),
        domain_mask=domain_mask,
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
    sample_predictions = read_sample_prediction_rows(analysis_config)
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
        sample_predictions=sample_predictions,
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
    if analysis_config.domain_mask is not None:
        input_paths.append(analysis_config.domain_mask.table_path)
        if analysis_config.domain_mask.manifest_path is not None:
            input_paths.append(analysis_config.domain_mask.manifest_path)
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
    masked = apply_reporting_domain_mask(cast(pd.DataFrame, frame), analysis_config.domain_mask)
    LOGGER.info(
        "Loaded %s prediction rows from %s after reporting-domain filtering",
        len(masked),
        analysis_config.predictions_path,
    )
    return masked


def read_sample_prediction_rows(analysis_config: ModelAnalysisConfig) -> pd.DataFrame:
    """Read configured sample-prediction rows for balance diagnostics."""
    if analysis_config.sample_predictions_path is None:
        return pd.DataFrame()
    if not analysis_config.sample_predictions_path.exists():
        LOGGER.info(
            "Skipping sample-prediction balance diagnostics; path is missing: %s",
            analysis_config.sample_predictions_path,
        )
        return pd.DataFrame()
    columns = available_prediction_columns(analysis_config.sample_predictions_path)
    dataset = ds.dataset(analysis_config.sample_predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    table = dataset.to_table(
        columns=columns,
        filter=dataset_field("model_name") == analysis_config.model_name,
    )
    frame = table.to_pandas()
    LOGGER.info(
        "Loaded %s %s sample-prediction rows for balance diagnostics",
        len(frame),
        analysis_config.model_name,
    )
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
    threshold_definitions = binary_threshold_definitions(analysis_config.threshold_fractions)
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
    balance_sources = build_balance_sources(data, analysis_config)
    class_balance_by_split = build_class_balance_by_split(balance_sources)
    target_balance_by_label_source = build_target_balance_by_label_source(balance_sources)
    background_rate_summary = build_background_rate_summary(balance_sources)
    binary_threshold_prevalence = build_binary_threshold_prevalence(
        balance_sources, threshold_definitions
    )
    binary_threshold_comparison = build_binary_threshold_comparison(
        data.sample_predictions,
        analysis_config,
        threshold_definitions,
        split_by_year_mapping(data.split_manifest),
    )
    binary_threshold_recommendation = build_binary_threshold_recommendation(
        binary_threshold_comparison,
        primary_validation_year(data.split_manifest),
    )
    residual_domain_context = build_residual_domain_context(data.model_predictions, analysis_config)
    residual_by_mask_reason = build_residual_by_mask_reason(data.model_predictions, analysis_config)
    residual_by_depth_bin = build_residual_by_depth_bin(data.model_predictions, analysis_config)
    top_residual_context = build_top_residual_context(data.model_predictions, analysis_config)
    quarter_mapping = build_quarter_mapping(analysis_config)
    data.aligned.attrs["model_analysis_projection_frame"] = projection_frame
    return AnalysisTables(
        stage_distribution=build_stage_distribution(data),
        prediction_distribution=prediction_distribution,
        residual_by_bin=residual_by_bin,
        residual_by_persistence=residual_by_persistence,
        target_framing=target_framing,
        threshold_sensitivity=threshold_sensitivity,
        binary_threshold_prevalence=binary_threshold_prevalence,
        binary_threshold_comparison=binary_threshold_comparison,
        binary_threshold_recommendation=binary_threshold_recommendation,
        spatial_readiness=spatial_readiness,
        feature_separability=feature_separability,
        phase1_decision=phase1_decision,
        phase1_model_comparison=phase1_model_comparison,
        data_health=data_health,
        class_balance_by_split=class_balance_by_split,
        target_balance_by_label_source=target_balance_by_label_source,
        background_rate_summary=background_rate_summary,
        residual_domain_context=residual_domain_context,
        residual_by_mask_reason=residual_by_mask_reason,
        residual_by_depth_bin=residual_by_depth_bin,
        top_residual_context=top_residual_context,
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
    difference = observed - predicted
    finite = difference[np.isfinite(difference)]
    if finite.size == 0:
        return math.nan
    return float(np.nanmean(np.abs(finite)))


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


def build_residual_domain_context(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build retained-domain residual taxonomy rows by joined mask context."""
    return residual_context_group_rows(
        dataframe,
        analysis_config,
        group_columns=[
            "domain_mask_reason",
            "depth_bin",
            "elevation_bin",
            "label_source",
            "observed_bin",
            "residual_class",
        ],
    )


def build_residual_by_mask_reason(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build retained-domain residual summaries grouped by mask reason."""
    return residual_context_group_rows(
        dataframe,
        analysis_config,
        group_columns=["domain_mask_reason"],
    )


def build_residual_by_depth_bin(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build retained-domain residual summaries grouped by depth/elevation bin."""
    return residual_context_group_rows(
        dataframe,
        analysis_config,
        group_columns=["depth_bin", "elevation_bin"],
    )


def residual_context_group_rows(
    dataframe: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
    *,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate residual diagnostics over a requested retained-domain grouping."""
    frame = residual_diagnostic_frame(dataframe, analysis_config)
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        group_columns,
        sort=True,
        dropna=False,
        observed=True,
    ):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values = {
            column: normalized_group_value(value)
            for column, value in zip(group_columns, key_tuple, strict=True)
        }
        rows.append(residual_context_row(group, analysis_config, group_values))
    return rows


def residual_diagnostic_frame(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> pd.DataFrame:
    """Return prediction rows annotated for retained-domain residual diagnostics."""
    frame = dataframe.copy()
    if MASK_RETAIN_COLUMN in frame.columns:
        frame = frame.loc[frame[MASK_RETAIN_COLUMN].fillna(False).astype(bool)].copy()
    frame = ensure_residual_context_columns(frame, analysis_config)
    if frame.empty:
        return frame
    frame["observed_bin"] = observed_area_bin_labels(
        frame["kelp_max_y"],
        analysis_config.observed_area_bins,
    )
    frame["residual_class"] = residual_taxonomy_classes(frame)
    abs_residual = frame["residual_kelp_max_y"].abs()
    threshold = float(abs_residual.quantile(0.95)) if len(abs_residual) else math.nan
    high_error = abs_residual >= threshold if np.isfinite(threshold) and threshold > 0 else False
    frame["is_high_error"] = high_error
    frame["high_error_threshold"] = threshold
    frame.attrs["high_error_threshold"] = threshold
    return frame


def ensure_residual_context_columns(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> pd.DataFrame:
    """Fill optional mask-context columns used by residual diagnostics."""
    frame = dataframe.copy()
    defaults: dict[str, object] = {
        "domain_mask_reason": "retained_domain"
        if analysis_config.domain_mask is not None
        else "unmasked",
        "depth_bin": "missing_depth_bin",
        "elevation_bin": "missing_elevation_bin",
        "label_source": "unknown",
        "crm_depth_m": math.nan,
        "crm_elevation_m": math.nan,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    if MASK_RETAIN_COLUMN not in frame.columns:
        frame[MASK_RETAIN_COLUMN] = analysis_config.domain_mask is not None
    return frame


def residual_taxonomy_classes(dataframe: pd.DataFrame) -> pd.Series:
    """Classify residual rows into stable interpretation buckets."""
    classes: list[str] = []
    for observed, predicted, residual in zip(
        dataframe["kelp_max_y"].to_numpy(dtype=float),
        dataframe["pred_kelp_max_y"].to_numpy(dtype=float),
        dataframe["residual_kelp_max_y"].to_numpy(dtype=float),
        strict=True,
    ):
        if not np.isfinite(observed) or not np.isfinite(predicted) or not np.isfinite(residual):
            classes.append("missing_or_uncalculable")
        elif abs(residual) <= 1.0:
            classes.append("near_correct")
        elif observed == 0 and predicted > 0:
            classes.append("observed_zero_false_positive")
        elif observed >= 450.0 and residual > 0:
            classes.append("high_canopy_underprediction")
        elif residual > 0:
            classes.append("positive_underprediction")
        elif residual < 0:
            classes.append("low_canopy_overprediction")
        else:
            classes.append("near_correct")
    return pd.Series(classes, index=dataframe.index, dtype="object")


def residual_context_row(
    group: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
    group_values: dict[str, str],
) -> dict[str, object]:
    """Build one retained-domain residual diagnostic summary row."""
    observed = group["kelp_max_y"].to_numpy(dtype=float)
    predicted = group["pred_kelp_max_y"].to_numpy(dtype=float)
    residual = group["residual_kelp_max_y"].to_numpy(dtype=float)
    observed_area = float(np.nansum(observed))
    predicted_area = float(np.nansum(predicted))
    return {
        "model_name": analysis_config.model_name,
        "split": str(group["split"].iloc[0])
        if "split" in group
        else analysis_config.analysis_split,
        "year": int(group["year"].iloc[0]) if "year" in group else analysis_config.analysis_year,
        "mask_status": mask_status(analysis_config.domain_mask),
        "evaluation_scope": evaluation_scope(analysis_config.domain_mask),
        "domain_mask_reason": group_values.get("domain_mask_reason", "all"),
        "depth_bin": group_values.get("depth_bin", "all"),
        "elevation_bin": group_values.get("elevation_bin", "all"),
        "label_source": group_values.get("label_source", "all"),
        "observed_bin": group_values.get("observed_bin", "all"),
        "residual_class": group_values.get("residual_class", "all"),
        "row_count": int(len(group)),
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "area_bias": predicted_area - observed_area,
        "area_pct_bias": percent_bias(predicted_area, observed_area),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "mean_residual": safe_mean(residual),
        "median_residual": safe_percentile(residual, 50),
        "underprediction_count": int(np.count_nonzero(residual > 0)),
        "overprediction_count": int(np.count_nonzero(residual < 0)),
        "high_error_count": int(group["is_high_error"].sum()),
        "high_error_threshold": float(group["high_error_threshold"].iloc[0]),
        "zero_observed_count": int(np.count_nonzero(observed == 0)),
        "positive_observed_count": int(np.count_nonzero(observed > 0)),
        "high_canopy_count": int(np.count_nonzero(observed >= 450.0)),
        "saturated_count": int(np.count_nonzero(observed >= KELPWATCH_PIXEL_AREA_M2)),
        "mean_crm_depth_m": safe_mean(group["crm_depth_m"].to_numpy(dtype=float)),
        "mean_crm_elevation_m": safe_mean(group["crm_elevation_m"].to_numpy(dtype=float)),
    }


def normalized_group_value(value: object) -> str:
    """Normalize a pandas group key to a stable string label."""
    if pd.isna(value):
        return "missing"
    return str(value)


def build_top_residual_context(
    dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig
) -> list[dict[str, object]]:
    """Build top underprediction and overprediction rows with domain context."""
    frame = residual_diagnostic_frame(dataframe, analysis_config)
    underpredicted = top_residual_context_rows(
        frame.loc[frame["residual_kelp_max_y"] > 0],
        analysis_config,
        residual_type="underprediction",
        ascending=False,
    )
    overpredicted = top_residual_context_rows(
        frame.loc[frame["residual_kelp_max_y"] < 0],
        analysis_config,
        residual_type="overprediction",
        ascending=True,
    )
    return underpredicted + overpredicted


def top_residual_context_rows(
    dataframe: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
    *,
    residual_type: str,
    ascending: bool,
) -> list[dict[str, object]]:
    """Build ranked top-residual rows for one residual sign."""
    sorted_rows = dataframe.sort_values("residual_kelp_max_y", ascending=ascending).head(
        analysis_config.top_residual_count
    )
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(sorted_rows.to_dict("records"), start=1):
        rows.append(top_residual_context_row(row, analysis_config, residual_type, rank))
    return rows


def top_residual_context_row(
    row: dict[str, object],
    analysis_config: ModelAnalysisConfig,
    residual_type: str,
    rank: int,
) -> dict[str, object]:
    """Build one top-residual row with joined domain-mask context."""
    observed_area = object_to_float(row.get("kelp_max_y"))
    predicted_area = object_to_float(row.get("pred_kelp_max_y"))
    residual = object_to_float(row.get("residual_kelp_max_y"))
    return {
        "residual_type": residual_type,
        "rank": rank,
        "model_name": str(row.get("model_name", analysis_config.model_name)),
        "split": str(row.get("split", analysis_config.analysis_split)),
        "year": object_to_int(row.get("year"), analysis_config.analysis_year),
        "mask_status": mask_status(analysis_config.domain_mask),
        "evaluation_scope": evaluation_scope(analysis_config.domain_mask),
        "aef_grid_cell_id": nullable_int(row.get("aef_grid_cell_id")),
        "aef_grid_row": nullable_int(row.get("aef_grid_row")),
        "aef_grid_col": nullable_int(row.get("aef_grid_col")),
        "kelpwatch_station_id": nullable_int(row.get("kelpwatch_station_id")),
        "label_source": nullable_string(row.get("label_source")),
        "is_kelpwatch_observed": nullable_bool(row.get("is_kelpwatch_observed")),
        "longitude": object_to_float(row.get("longitude")),
        "latitude": object_to_float(row.get("latitude")),
        "is_plausible_kelp_domain": nullable_bool(row.get(MASK_RETAIN_COLUMN)),
        "domain_mask_reason": nullable_string(row.get("domain_mask_reason")),
        "domain_mask_detail": nullable_string(row.get("domain_mask_detail")),
        "domain_mask_version": nullable_string(row.get("domain_mask_version")),
        "crm_elevation_m": nullable_float(row.get("crm_elevation_m")),
        "crm_depth_m": nullable_float(row.get("crm_depth_m")),
        "depth_bin": nullable_string(row.get("depth_bin")),
        "elevation_bin": nullable_string(row.get("elevation_bin")),
        "kelp_max_y": observed_area,
        "pred_kelp_max_y": predicted_area,
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "residual_kelp_max_y": residual,
        "abs_residual_kelp_max_y": abs(residual),
        "residual_class": str(row.get("residual_class", "missing_or_uncalculable")),
    }


def nullable_int(value: object) -> int | None:
    """Convert a nullable numeric value to int or None."""
    if pd.isna(value):
        return None
    if not isinstance(value, int | float | np.integer | np.floating):
        return None
    return int(value)


def nullable_float(value: object) -> float | None:
    """Convert a nullable numeric value to float or None."""
    if pd.isna(value):
        return None
    if not isinstance(value, int | float | np.integer | np.floating):
        return None
    return float(value)


def nullable_bool(value: object) -> bool | None:
    """Convert a nullable value to bool or None."""
    if pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return None


def nullable_string(value: object) -> str | None:
    """Convert a nullable value to string or None."""
    if pd.isna(value):
        return None
    return str(value)


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


def binary_threshold_definitions(
    configured_thresholds: tuple[float, ...],
) -> tuple[BinaryThresholdDefinition, ...]:
    """Return required and configured annual-max binary threshold definitions."""
    threshold_values = {
        round(float(threshold), 10)
        for threshold in (*REQUIRED_BINARY_THRESHOLD_FRACTIONS, *configured_thresholds)
    }
    definitions: list[BinaryThresholdDefinition] = []
    required_values = {round(value, 10) for value in REQUIRED_BINARY_THRESHOLD_FRACTIONS}
    for threshold in sorted(threshold_values):
        role = "selection_candidate" if threshold in required_values else "diagnostic"
        definitions.append(
            BinaryThresholdDefinition(
                fraction=threshold,
                label=binary_threshold_label(threshold),
                operator=">" if threshold_is_zero(threshold) else ">=",
                role=role,
            )
        )
    return tuple(definitions)


def binary_threshold_label(threshold: float) -> str:
    """Return a report-safe label that encodes the annual-max threshold rule."""
    if threshold_is_zero(threshold):
        return "annual_max_gt0"
    return f"annual_max_ge_{compact_percent_label(threshold)}"


def compact_percent_label(threshold: float) -> str:
    """Return a compact percentage label for a fraction threshold."""
    percent = threshold * 100.0
    if math.isclose(percent, round(percent), abs_tol=1e-9):
        return f"{int(round(percent))}pct"
    return f"{percent:.3g}".replace(".", "p").replace("-", "neg") + "pct"


def threshold_is_zero(threshold: float) -> bool:
    """Return whether a threshold should use the strict greater-than-zero rule."""
    return math.isclose(threshold, 0.0, abs_tol=1e-12)


def annual_max_positive_mask(
    values: np.ndarray, threshold: BinaryThresholdDefinition
) -> np.ndarray:
    """Return annual-max binary positives for one threshold definition."""
    finite = np.isfinite(values)
    if threshold_is_zero(threshold.fraction):
        return cast(np.ndarray, finite & (values > 0))
    return cast(np.ndarray, finite & (values >= threshold.fraction))


def build_binary_threshold_prevalence(
    sources: list[BalanceSource],
    thresholds: tuple[BinaryThresholdDefinition, ...],
) -> list[dict[str, object]]:
    """Build threshold prevalence rows from label source-of-truth tables."""
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.extend(binary_prevalence_group_rows(source, thresholds, ["split", "year"]))
        rows.extend(
            binary_prevalence_group_rows(source, thresholds, ["split", "year", "label_source"])
        )
    return rows


def binary_prevalence_group_rows(
    source: BalanceSource,
    thresholds: tuple[BinaryThresholdDefinition, ...],
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate threshold prevalence for one source and grouping."""
    rows: list[dict[str, object]] = []
    for keys, group in source.frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {
            "split": "all",
            "year": "all",
            "label_source": "all",
        }
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        for threshold in thresholds:
            rows.append(binary_prevalence_summary_row(source, group, group_values, threshold))
    return rows


def binary_prevalence_summary_row(
    source: BalanceSource,
    group: pd.DataFrame,
    group_values: dict[str, object],
    threshold: BinaryThresholdDefinition,
) -> dict[str, object]:
    """Build one grouped annual-max binary threshold prevalence row."""
    observed_fraction = group["kelp_fraction_y"].to_numpy(dtype=float)
    observed_area = group["kelp_max_y"].to_numpy(dtype=float)
    valid_mask = np.isfinite(observed_fraction)
    positive = annual_max_positive_mask(observed_fraction, threshold)
    label_sources = label_source_series(group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    positive_count = int(np.count_nonzero(positive))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_positive = assumed_background & positive
    positive_area = observed_area[positive]
    return {
        "data_scope": source.data_scope,
        "mask_status": source.mask_status,
        "evaluation_scope": source.evaluation_scope,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "threshold_fraction": threshold.fraction,
        "threshold_area": threshold.fraction * KELPWATCH_PIXEL_AREA_M2,
        "threshold_label": threshold.label,
        "threshold_operator": threshold.operator,
        "threshold_role": threshold.role,
        "row_count": int(len(group)),
        "target_count": int(np.count_nonzero(valid_mask)),
        "station_count": balance_station_count(group),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(np.count_nonzero(valid_mask))),
        "assumed_background_count": assumed_background_count,
        "assumed_background_positive_count": int(np.count_nonzero(assumed_background_positive)),
        "assumed_background_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_positive)),
            assumed_background_count,
        ),
        "observed_positive_area": float(np.nansum(positive_area)) if positive_area.size else 0.0,
        "mean_positive_area": safe_mean(positive_area),
    }


def build_binary_threshold_comparison(
    sample_predictions: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
    thresholds: tuple[BinaryThresholdDefinition, ...],
    split_by_year: dict[int, str],
) -> list[dict[str, object]]:
    """Build validation-ready binary threshold metrics from sample predictions."""
    if sample_predictions.empty:
        return []
    frame = prepare_binary_prediction_frame(sample_predictions, analysis_config, split_by_year)
    if frame.empty:
        return []
    rows: list[dict[str, object]] = []
    source_mask_status = frame_mask_status(frame, analysis_config)
    source_scope = "sample_predictions"
    for keys, group in frame.groupby(["model_name", "split", "year"], sort=True, dropna=False):
        model_name, split, year = cast(tuple[str, str, int], keys)
        group_values = {
            "model_name": str(model_name),
            "split": str(split),
            "year": int(year),
            "label_source": "all",
        }
        for threshold in thresholds:
            rows.append(
                binary_threshold_comparison_row(
                    group,
                    threshold,
                    data_scope=source_scope,
                    mask_status_value=source_mask_status,
                    evaluation_scope_value=source_scope,
                    group_values=group_values,
                )
            )
        for label_source, label_group in group.groupby("label_source", sort=True, dropna=False):
            label_values = {
                **group_values,
                "label_source": normalized_group_value(label_source),
            }
            for threshold in thresholds:
                rows.append(
                    binary_threshold_comparison_row(
                        label_group,
                        threshold,
                        data_scope=source_scope,
                        mask_status_value=source_mask_status,
                        evaluation_scope_value=source_scope,
                        group_values=label_values,
                    )
                )
    return rows


def prepare_binary_prediction_frame(
    sample_predictions: pd.DataFrame,
    analysis_config: ModelAnalysisConfig,
    split_by_year: dict[int, str],
) -> pd.DataFrame:
    """Normalize sample predictions for binary threshold comparison."""
    frame = prepare_balance_frame(sample_predictions, split_by_year)
    if "model_name" not in frame.columns:
        frame["model_name"] = analysis_config.model_name
    if "pred_kelp_fraction_y_clipped" not in frame.columns and "pred_kelp_max_y" in frame.columns:
        frame["pred_kelp_fraction_y_clipped"] = (
            frame["pred_kelp_max_y"].astype(float) / KELPWATCH_PIXEL_AREA_M2
        )
    if "pred_kelp_max_y" not in frame.columns and "pred_kelp_fraction_y_clipped" in frame.columns:
        frame["pred_kelp_max_y"] = (
            frame["pred_kelp_fraction_y_clipped"].astype(float) * KELPWATCH_PIXEL_AREA_M2
        )
    required = ["kelp_fraction_y", "kelp_max_y", "pred_kelp_fraction_y_clipped", "pred_kelp_max_y"]
    if any(column not in frame.columns for column in required):
        LOGGER.info(
            "Skipping binary threshold comparison; sample predictions lack required columns"
        )
        return pd.DataFrame()
    return frame


def binary_threshold_comparison_row(
    group: pd.DataFrame,
    threshold: BinaryThresholdDefinition,
    *,
    data_scope: str,
    mask_status_value: str,
    evaluation_scope_value: str,
    group_values: dict[str, object],
) -> dict[str, object]:
    """Build one binary threshold prediction-metric row."""
    observed_fraction = group["kelp_fraction_y"].to_numpy(dtype=float)
    predicted_fraction = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = group["kelp_max_y"].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    valid_mask = np.isfinite(observed_fraction) & np.isfinite(predicted_fraction)
    observed_fraction = observed_fraction[valid_mask]
    predicted_fraction = predicted_fraction[valid_mask]
    observed_area = observed_area[valid_mask]
    predicted_area = predicted_area[valid_mask]
    valid_group = group.loc[valid_mask]
    observed_positive = annual_max_positive_mask(observed_fraction, threshold)
    predicted_positive = annual_max_positive_mask(predicted_fraction, threshold)
    true_positive = observed_positive & predicted_positive
    false_positive = ~observed_positive & predicted_positive
    false_negative = observed_positive & ~predicted_positive
    precision, recall, f1 = precision_recall_f1(observed_positive, predicted_positive)
    positive_count = int(np.count_nonzero(observed_positive))
    negative_count = int(observed_positive.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted_positive))
    label_sources = label_source_series(valid_group).to_numpy(dtype=object)
    assumed_background = label_sources == "assumed_background"
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    return {
        "data_scope": data_scope,
        "mask_status": mask_status_value,
        "evaluation_scope": evaluation_scope_value,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "model_name": group_values["model_name"],
        "threshold_fraction": threshold.fraction,
        "threshold_area": threshold.fraction * KELPWATCH_PIXEL_AREA_M2,
        "threshold_label": threshold.label,
        "threshold_operator": threshold.operator,
        "threshold_role": threshold.role,
        "row_count": int(len(group)),
        "target_count": int(observed_positive.size),
        "station_count": balance_station_count(valid_group),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, int(observed_positive.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(
            predicted_positive_count,
            int(predicted_positive.size),
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(np.count_nonzero(true_positive)),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(int(np.count_nonzero(false_positive)), negative_count),
        "false_positive_area": float(np.nansum(predicted_area[false_positive])),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_rate": safe_ratio(int(np.count_nonzero(false_negative)), positive_count),
        "false_negative_area": float(np.nansum(observed_area[false_negative])),
        "observed_positive_area": float(np.nansum(observed_area[observed_positive])),
        "predicted_positive_area": float(np.nansum(predicted_area[predicted_positive])),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            int(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
        "assumed_background_false_positive_area": float(
            np.nansum(predicted_area[assumed_background_false_positive])
        ),
    }


def build_binary_threshold_recommendation(
    comparison_rows: list[dict[str, object]], selection_year: int | None
) -> list[dict[str, object]]:
    """Rank candidate thresholds using validation rows only."""
    selection_rows = validation_candidate_threshold_rows(comparison_rows, selection_year)
    if not selection_rows:
        return []
    ranked_rows = sorted(selection_rows, key=recommendation_sort_key, reverse=True)
    selected_row = selected_threshold_row(ranked_rows)
    selected_fraction = (
        row_float(selected_row, "threshold_fraction") if selected_row is not None else math.nan
    )
    selected_label = str(selected_row.get("threshold_label", "")) if selected_row else ""
    status = recommendation_status(selection_rows, selected_row)
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(ranked_rows, start=1):
        threshold_fraction = row_float(row, "threshold_fraction")
        rows.append(
            {
                "selection_split": DEFAULT_THRESHOLD_SELECTION_SPLIT,
                "selection_year": selection_year
                if selection_year is not None
                else "all_validation_years",
                "test_split": DEFAULT_THRESHOLD_TEST_SPLIT,
                "selection_policy": BINARY_THRESHOLD_SELECTION_POLICY,
                "recommendation_status": status,
                "selected_candidate": bool(
                    selected_row is not None and math.isclose(threshold_fraction, selected_fraction)
                ),
                "threshold_rank": rank,
                "model_name": row.get("model_name", ""),
                "threshold_fraction": threshold_fraction,
                "threshold_area": row_float(row, "threshold_area"),
                "threshold_label": row.get("threshold_label", ""),
                "threshold_operator": row.get("threshold_operator", ""),
                "threshold_role": row.get("threshold_role", ""),
                "positive_support_ok": threshold_positive_support_ok(row),
                "validation_row_count": row_int(row, "target_count"),
                "validation_positive_count": row_int(row, "positive_count"),
                "validation_positive_rate": row_float(row, "positive_rate"),
                "validation_predicted_positive_rate": row_float(row, "predicted_positive_rate"),
                "validation_precision": row_float(row, "precision"),
                "validation_recall": row_float(row, "recall"),
                "validation_f1": row_float(row, "f1"),
                "validation_false_positive_rate": row_float(row, "false_positive_rate"),
                "validation_false_positive_area": row_float(row, "false_positive_area"),
                "validation_false_negative_area": row_float(row, "false_negative_area"),
                "validation_assumed_background_false_positive_rate": row_float(
                    row, "assumed_background_false_positive_rate"
                ),
                "validation_assumed_background_false_positive_area": row_float(
                    row, "assumed_background_false_positive_area"
                ),
                "recommended_threshold_fraction": selected_fraction,
                "recommended_threshold_label": selected_label,
            }
        )
    return rows


def validation_candidate_threshold_rows(
    comparison_rows: list[dict[str, object]], selection_year: int | None
) -> list[dict[str, object]]:
    """Filter comparison rows to validation all-label candidate thresholds."""
    rows = [
        row
        for row in comparison_rows
        if row.get("split") == DEFAULT_THRESHOLD_SELECTION_SPLIT
        and row.get("label_source") == "all"
        and row.get("threshold_role") == "selection_candidate"
    ]
    if selection_year is None:
        return rows
    return [row for row in rows if str(row.get("year")) == str(selection_year)]


def recommendation_sort_key(row: dict[str, object]) -> tuple[int, int, float, float, float]:
    """Return the validation-only ranking key for threshold candidates."""
    support_ok = int(threshold_positive_support_ok(row))
    has_positive = int(row_int(row, "positive_count") > 0)
    f1 = row_float(row, "f1", default=-math.inf)
    assumed_background_rate = finite_or_default(
        row_float(row, "assumed_background_false_positive_rate"),
        math.inf,
    )
    return (
        support_ok,
        has_positive,
        row_float(row, "threshold_fraction", default=-math.inf),
        f1 if np.isfinite(f1) else -math.inf,
        -assumed_background_rate,
    )


def threshold_positive_support_ok(row: dict[str, object]) -> bool:
    """Return whether a candidate threshold keeps enough validation positives."""
    return (
        row_int(row, "positive_count") >= DEFAULT_MIN_BINARY_SELECTION_POSITIVES
        and row_float(row, "positive_rate") >= DEFAULT_MIN_BINARY_SELECTION_POSITIVE_RATE
    )


def finite_or_default(value: float, default: float) -> float:
    """Return a finite value or a fallback used for sorting."""
    return value if np.isfinite(value) else default


def selected_threshold_row(
    ranked_rows: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return the selected validation threshold row, or none when no positives exist."""
    if not ranked_rows:
        return None
    if all(row_int(row, "positive_count") == 0 for row in ranked_rows):
        return None
    return ranked_rows[0]


def recommendation_status(
    selection_rows: list[dict[str, object]], selected_row: dict[str, object] | None
) -> str:
    """Return a compact recommendation status for the output table."""
    if not selection_rows:
        return "no_validation_rows"
    if selected_row is None:
        return "no_positive_validation_rows"
    if threshold_positive_support_ok(selected_row):
        return "selected_from_validation_support_floor"
    return "selected_from_validation_low_support_fallback"


def primary_validation_year(split_manifest: pd.DataFrame) -> int | None:
    """Return the configured validation year when exactly inferable from the split manifest."""
    if "split" not in split_manifest.columns or "year" not in split_manifest.columns:
        return None
    years = sorted(
        int(year)
        for year in split_manifest.loc[
            split_manifest["split"] == DEFAULT_THRESHOLD_SELECTION_SPLIT, "year"
        ]
        .dropna()
        .unique()
    )
    if len(years) == 1:
        return years[0]
    return None


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
    input_mask_status = model_input_mask_status(data.aligned, analysis_config)
    for metric in data.metrics.to_dict("records"):
        comparison_row = metric_comparison_row(
            cast(dict[str, object], metric),
            split_years,
            input_mask_status,
        )
        if comparison_row:
            rows.append(comparison_row)
    rows.extend(reference_area_calibration_comparison_rows(reference_area_calibration))
    if not has_primary_full_grid_comparison(rows, analysis_config):
        rows.extend(full_grid_comparison_rows(data.model_predictions, analysis_config))
    return rows


def model_input_mask_status(dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig) -> str:
    """Infer whether sample metric rows came from the configured retained mask domain."""
    if analysis_config.domain_mask is None or MASK_RETAIN_COLUMN not in dataframe.columns:
        return "unmasked"
    retained = dataframe[MASK_RETAIN_COLUMN].dropna().astype(bool)
    if retained.empty or not bool(retained.all()):
        return "unmasked"
    return mask_status(analysis_config.domain_mask)


def has_primary_full_grid_comparison(
    rows: list[dict[str, object]], analysis_config: ModelAnalysisConfig
) -> bool:
    """Return whether compact full-grid calibration rows are already present."""
    primary_scope = evaluation_scope(analysis_config.domain_mask)
    return any(row.get("evaluation_scope") == primary_scope for row in rows)


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
                "mask_status": row_dict.get("mask_status", "unmasked"),
                "evaluation_scope": row_dict.get("evaluation_scope", "full_grid_prediction"),
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
    metric: dict[str, object], split_years: dict[str, str], input_mask_status: str
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
        "mask_status": input_mask_status,
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
        "mask_status": mask_status(analysis_config.domain_mask),
        "evaluation_scope": evaluation_scope(analysis_config.domain_mask),
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


def build_balance_sources(
    data: AnalysisData, analysis_config: ModelAnalysisConfig
) -> list[BalanceSource]:
    """Prepare the model-analysis sources used for target-balance diagnostics."""
    split_by_year = split_by_year_mapping(data.split_manifest)
    sources = [
        BalanceSource(
            data_scope="model_input_sample",
            mask_status=frame_mask_status(data.aligned, analysis_config),
            evaluation_scope="model_input_sample",
            frame=prepare_balance_frame(data.aligned, split_by_year),
        )
    ]
    retained_split_manifest = retained_split_manifest_rows(data.split_manifest)
    sources.append(
        BalanceSource(
            data_scope="split_manifest_retained",
            mask_status=frame_mask_status(retained_split_manifest, analysis_config),
            evaluation_scope="split_manifest_retained",
            frame=prepare_balance_frame(retained_split_manifest, split_by_year),
        )
    )
    if not data.sample_predictions.empty:
        sources.append(
            BalanceSource(
                data_scope="sample_predictions",
                mask_status=frame_mask_status(data.sample_predictions, analysis_config),
                evaluation_scope="sample_predictions",
                frame=prepare_balance_frame(data.sample_predictions, split_by_year),
            )
        )
    sources.append(
        BalanceSource(
            data_scope=evaluation_scope(analysis_config.domain_mask),
            mask_status=mask_status(analysis_config.domain_mask),
            evaluation_scope=evaluation_scope(analysis_config.domain_mask),
            frame=prepare_balance_frame(data.model_predictions, split_by_year),
        )
    )
    return [source for source in sources if not source.frame.empty]


def split_by_year_mapping(split_manifest: pd.DataFrame) -> dict[int, str]:
    """Infer split labels from split-manifest years for tables lacking split columns."""
    if "year" not in split_manifest.columns or "split" not in split_manifest.columns:
        return {}
    mapping: dict[int, str] = {}
    for year, group in split_manifest.groupby("year", sort=True):
        splits = sorted(str(value) for value in group["split"].dropna().unique())
        mapping[int(cast(int, year))] = splits[0] if len(splits) == 1 else ",".join(splits)
    return mapping


def retained_split_manifest_rows(split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Return split-manifest rows retained for model training and evaluation."""
    if "used_for_training_eval" not in split_manifest.columns:
        return split_manifest.copy()
    retained = split_manifest["used_for_training_eval"].fillna(False).astype(bool)
    return cast(pd.DataFrame, split_manifest.loc[retained].copy())


def frame_mask_status(dataframe: pd.DataFrame, analysis_config: ModelAnalysisConfig) -> str:
    """Infer the mask-status label for a sample-like balance source."""
    if analysis_config.domain_mask is None:
        return "unmasked"
    if MASK_RETAIN_COLUMN not in dataframe.columns:
        return "unmasked"
    retained = dataframe[MASK_RETAIN_COLUMN].dropna().astype(bool)
    if retained.empty or not bool(retained.all()):
        return "unmasked"
    return mask_status(analysis_config.domain_mask)


def prepare_balance_frame(dataframe: pd.DataFrame, split_by_year: dict[int, str]) -> pd.DataFrame:
    """Normalize target, split, and label-source columns for balance summaries."""
    frame = dataframe.copy()
    if "kelp_max_y" not in frame.columns and "kelp_fraction_y" in frame.columns:
        frame["kelp_max_y"] = frame["kelp_fraction_y"].astype(float) * KELPWATCH_PIXEL_AREA_M2
    if "kelp_fraction_y" not in frame.columns and "kelp_max_y" in frame.columns:
        frame["kelp_fraction_y"] = frame["kelp_max_y"].astype(float) / KELPWATCH_PIXEL_AREA_M2
    if "split" not in frame.columns:
        frame["split"] = frame["year"].map(lambda year: split_by_year.get(int(year), "all"))
    frame["label_source"] = label_source_series(frame)
    return frame


def build_class_balance_by_split(sources: list[BalanceSource]) -> list[dict[str, object]]:
    """Build class-balance diagnostics grouped by split, year, and label source."""
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.extend(balance_group_rows(source, ["split", "year"]))
        rows.extend(balance_group_rows(source, ["split", "year", "label_source"]))
    return rows


def build_target_balance_by_label_source(
    sources: list[BalanceSource],
) -> list[dict[str, object]]:
    """Build target-balance diagnostics grouped by source and label provenance."""
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.extend(balance_group_rows(source, []))
        rows.extend(balance_group_rows(source, ["label_source"]))
    return rows


def build_background_rate_summary(sources: list[BalanceSource]) -> list[dict[str, object]]:
    """Build compact assumed-background rate rows by split and source scope."""
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.extend(balance_group_rows(source, ["split", "year"]))
    return rows


def balance_group_rows(source: BalanceSource, group_columns: list[str]) -> list[dict[str, object]]:
    """Aggregate one balance source over requested grouping columns."""
    if source.frame.empty:
        return []
    if not group_columns:
        return [
            balance_summary_row(
                source,
                source.frame,
                {"split": "all", "year": "all", "label_source": "all"},
            )
        ]
    rows: list[dict[str, object]] = []
    for keys, group in source.frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {
            "split": "all",
            "year": "all",
            "label_source": "all",
        }
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(balance_summary_row(source, group, group_values))
    return rows


def balance_summary_row(
    source: BalanceSource, group: pd.DataFrame, group_values: dict[str, object]
) -> dict[str, object]:
    """Build one class and target-balance row."""
    observed_area = group["kelp_max_y"].to_numpy(dtype=float)
    observed_fraction = group["kelp_fraction_y"].to_numpy(dtype=float)
    finite_area = observed_area[np.isfinite(observed_area)]
    target_count = int(finite_area.size)
    label_sources = label_source_series(group)
    assumed_background_count = int(np.count_nonzero(label_sources == "assumed_background"))
    row_count = int(len(group))
    zero_count = int(np.count_nonzero(finite_area == 0))
    positive_count = int(np.count_nonzero(finite_area > 0))
    positive_ge_1pct_count = int(np.count_nonzero(observed_fraction >= 0.01))
    positive_ge_5pct_count = int(np.count_nonzero(observed_fraction >= 0.05))
    positive_ge_10pct_count = int(np.count_nonzero(observed_fraction >= 0.10))
    high_canopy_count = int(np.count_nonzero(finite_area >= 450.0))
    very_high_canopy_count = int(np.count_nonzero(finite_area >= 810.0))
    saturated_count = int(np.count_nonzero(finite_area >= KELPWATCH_PIXEL_AREA_M2))
    observed_canopy_area = float(np.nansum(finite_area)) if target_count else math.nan
    return {
        "data_scope": source.data_scope,
        "mask_status": source.mask_status,
        "evaluation_scope": source.evaluation_scope,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "row_count": row_count,
        "station_count": balance_station_count(group),
        "zero_count": zero_count,
        "zero_rate": safe_ratio(zero_count, target_count),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(positive_count, target_count),
        "positive_ge_1pct_count": positive_ge_1pct_count,
        "positive_ge_1pct_rate": safe_ratio(positive_ge_1pct_count, target_count),
        "positive_ge_5pct_count": positive_ge_5pct_count,
        "positive_ge_5pct_rate": safe_ratio(positive_ge_5pct_count, target_count),
        "positive_ge_10pct_count": positive_ge_10pct_count,
        "positive_ge_10pct_rate": safe_ratio(positive_ge_10pct_count, target_count),
        "high_canopy_count": high_canopy_count,
        "high_canopy_rate": safe_ratio(high_canopy_count, target_count),
        "very_high_canopy_count": very_high_canopy_count,
        "very_high_canopy_rate": safe_ratio(very_high_canopy_count, target_count),
        "saturated_count": saturated_count,
        "saturated_rate": safe_ratio(saturated_count, target_count),
        "assumed_background_count": assumed_background_count,
        "assumed_background_rate": safe_ratio(assumed_background_count, row_count),
        "observed_canopy_area": observed_canopy_area,
        "mean_observed_canopy_area": safe_mean(finite_area),
    }


def balance_station_count(group: pd.DataFrame) -> int:
    """Count unique Kelpwatch station ids when a grouped frame has station support."""
    if "kelpwatch_station_id" not in group.columns:
        return 0
    return int(group["kelpwatch_station_id"].dropna().nunique())


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
        tables.binary_threshold_prevalence,
        analysis_config.binary_threshold_prevalence_path,
        BINARY_THRESHOLD_PREVALENCE_FIELDS,
    )
    write_csv(
        tables.binary_threshold_comparison,
        analysis_config.binary_threshold_comparison_path,
        BINARY_THRESHOLD_COMPARISON_FIELDS,
    )
    write_csv(
        tables.binary_threshold_recommendation,
        analysis_config.binary_threshold_recommendation_path,
        BINARY_THRESHOLD_RECOMMENDATION_FIELDS,
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
        tables.class_balance_by_split,
        analysis_config.class_balance_by_split_path,
        CLASS_BALANCE_FIELDS,
    )
    write_csv(
        tables.target_balance_by_label_source,
        analysis_config.target_balance_by_label_source_path,
        TARGET_BALANCE_FIELDS,
    )
    write_csv(
        tables.background_rate_summary,
        analysis_config.background_rate_summary_path,
        BACKGROUND_RATE_FIELDS,
    )
    write_csv(
        tables.residual_domain_context,
        analysis_config.residual_domain_context_path,
        RESIDUAL_DOMAIN_CONTEXT_FIELDS,
    )
    write_csv(
        tables.residual_by_mask_reason,
        analysis_config.residual_by_mask_reason_path,
        RESIDUAL_DOMAIN_CONTEXT_FIELDS,
    )
    write_csv(
        tables.residual_by_depth_bin,
        analysis_config.residual_by_depth_bin_path,
        RESIDUAL_DOMAIN_CONTEXT_FIELDS,
    )
    write_csv(
        tables.top_residual_context,
        analysis_config.top_residual_context_path,
        TOP_RESIDUAL_CONTEXT_FIELDS,
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
    write_class_balance_figure(tables, analysis_config)
    write_binary_threshold_comparison_figure(tables, analysis_config)
    write_residual_domain_context_figure(tables, analysis_config)


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


def write_class_balance_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write a compact class-balance figure for the primary analysis split."""
    output_path = analysis_config.class_balance_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in tables.class_balance_by_split
        if row.get("label_source") == "all"
        and row.get("split") == analysis_config.analysis_split
        and str(row.get("year")) == str(analysis_config.analysis_year)
    ]
    if not rows:
        rows = [
            row for row in tables.target_balance_by_label_source if row.get("label_source") == "all"
        ]
    fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    if not rows:
        axis.text(0.5, 0.5, "No class-balance rows", ha="center", va="center")
        axis.set_axis_off()
    else:
        labels = [str(row["data_scope"]) for row in rows]
        x_values = np.arange(len(labels))
        width = 0.2
        series = [
            ("zero", "zero_rate"),
            ("positive", "positive_rate"),
            ("high", "high_canopy_rate"),
            ("saturated", "saturated_rate"),
        ]
        for index, (label, column) in enumerate(series):
            offset = (index - 1.5) * width
            axis.bar(
                x_values + offset,
                [row_float(row, column, default=0.0) for row in rows],
                width=width,
                label=label,
            )
        axis.set_xticks(x_values, labels, rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title("Annual-max class balance")
        axis.set_ylabel("Share of rows")
        axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_binary_threshold_comparison_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write a compact validation threshold comparison figure."""
    output_path = analysis_config.binary_threshold_comparison_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation = selected_or_first_recommendation_row(tables.binary_threshold_recommendation)
    selection_year = recommendation.get("selection_year", "all_validation_years")
    rows = [
        row
        for row in tables.binary_threshold_comparison
        if row.get("split") == DEFAULT_THRESHOLD_SELECTION_SPLIT
        and str(row.get("year")) == str(selection_year)
        and row.get("label_source") == "all"
        and row.get("threshold_role") == "selection_candidate"
    ]
    rows = sorted(rows, key=lambda row: row_float(row, "threshold_fraction"))
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if not rows:
        axis.text(0.5, 0.5, "No validation threshold rows", ha="center", va="center")
        axis.set_axis_off()
    else:
        labels = [str(row["threshold_label"]).replace("annual_max_", "") for row in rows]
        x_values = np.arange(len(labels))
        axis.bar(
            x_values - 0.18,
            [row_float(row, "positive_rate", default=0.0) for row in rows],
            width=0.36,
            label="observed positive rate",
            color="#2c7fb8",
        )
        axis.bar(
            x_values + 0.18,
            [row_float(row, "predicted_positive_rate", default=0.0) for row in rows],
            width=0.36,
            label="predicted positive rate",
            color="#f03b20",
        )
        selected_label = str(recommendation.get("recommended_threshold_label", ""))
        for index, row in enumerate(rows):
            if row.get("threshold_label") == selected_label:
                axis.axvline(index, color="black", linestyle="--", linewidth=1.0)
                break
        axis.set_xticks(x_values, labels, rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(f"Validation annual-max binary thresholds | {selection_year}")
        axis.set_ylabel("Share of validation rows")
        axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_residual_domain_context_figure(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> None:
    """Write a compact retained-domain residual figure by depth/elevation bin."""
    output_path = analysis_config.residual_domain_context_figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = filter_table_rows(
        tables.residual_by_depth_bin,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    if not rows:
        axis.text(0.5, 0.5, "No retained-domain residual rows", ha="center", va="center")
        axis.set_axis_off()
    else:
        labels = [f"{row['depth_bin']} | {row['elevation_bin']}" for row in rows]
        values = [row_float(row, "mean_residual") for row in rows]
        colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]
        x_values = np.arange(len(labels))
        axis.bar(x_values, values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x_values, labels, rotation=35, ha="right")
        axis.set_title("Mean residual by retained depth/elevation bin")
        axis.set_xlabel("Depth and elevation bin")
        axis.set_ylabel("Mean residual area, observed - predicted")
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
        "## Mask-Aware Residual Diagnostics",
        "",
        mask_aware_residual_markdown(tables, analysis_config),
        "",
        image_markdown(
            "Residual by retained domain context",
            analysis_config.residual_domain_context_figure,
            output_path,
        ),
        "",
        "## Class And Target Balance",
        "",
        class_target_balance_markdown(tables, analysis_config),
        "",
        image_markdown(
            "Annual-max class balance",
            analysis_config.class_balance_figure,
            output_path,
        ),
        "",
        "## Annual-Max Binary Threshold Comparison",
        "",
        binary_threshold_comparison_markdown(tables, analysis_config),
        "",
        image_markdown(
            "Annual-max binary threshold comparison",
            analysis_config.binary_threshold_comparison_figure,
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
        f"- Residual by domain context table: `{analysis_config.residual_domain_context_path}`",
        f"- Residual by mask reason table: `{analysis_config.residual_by_mask_reason_path}`",
        f"- Residual by depth/elevation bin table: `{analysis_config.residual_by_depth_bin_path}`",
        f"- Top residual domain-context table: `{analysis_config.top_residual_context_path}`",
        f"- Class balance by split table: `{analysis_config.class_balance_by_split_path}`",
        f"- Target balance by label source table: `{analysis_config.target_balance_by_label_source_path}`",
        f"- Background rate summary table: `{analysis_config.background_rate_summary_path}`",
        f"- Binary threshold prevalence table: `{analysis_config.binary_threshold_prevalence_path}`",
        f"- Binary threshold comparison table: `{analysis_config.binary_threshold_comparison_path}`",
        f"- Binary threshold recommendation table: `{analysis_config.binary_threshold_recommendation_path}`",
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
        "models without changing the current ridge predictions. Full-grid reporting now treats "
        f"`{mask_status(analysis_config.domain_mask)}` as the primary area-calibration domain. "
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
            and row.get("evaluation_scope") in {"full_grid_masked", "full_grid_prediction"}
            and row.get("label_source") in {"all", "kelpwatch_station", "assumed_background"}
        ],
        key=lambda row: (
            0 if row.get("evaluation_scope") == "full_grid_masked" else 1,
            str(row.get("label_source", "")),
            abs(row_float(row, "area_pct_bias", default=math.inf)),
        ),
    )
    if not station_rows and not full_grid_rows:
        return "Reference-baseline ranking rows are not available yet."
    lines = [
        "Lower RMSE is better for Kelpwatch-station pixel skill; lower absolute area percent bias is better for masked full-grid calibration. These comparisons evaluate Kelpwatch-style label reproduction, not independent field truth.",
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
        f"rows are filtered to `{mask_status(analysis_config.domain_mask)}` before plotting. "
        "The observed and predicted panels use the same canopy-area scale, and the error "
        "panel uses `observed - predicted`, so positive residuals are underprediction. The linked "
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


def mask_aware_residual_markdown(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> str:
    """Build Markdown for retained-domain residual diagnostics."""
    reason_rows = filter_table_rows(
        tables.residual_by_mask_reason,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    context_rows = filter_table_rows(
        tables.residual_domain_context,
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    if not reason_rows:
        return "No retained-domain mask-aware residual rows were available."
    largest_reason = max(reason_rows, key=lambda row: row_int(row, "row_count"))
    highest_abs_reason = max(
        reason_rows,
        key=lambda row: abs(row_float(row, "mean_residual", default=0.0)),
    )
    false_positive_rows = [
        row for row in context_rows if row.get("residual_class") == "observed_zero_false_positive"
    ]
    underprediction_rows = [
        row
        for row in context_rows
        if row.get("residual_class") in {"positive_underprediction", "high_canopy_underprediction"}
    ]
    underprediction_focus = (
        max(
            underprediction_rows,
            key=lambda row: (
                row_int(row, "high_error_count"),
                row_int(row, "row_count"),
                row_float(row, "mean_residual", default=-math.inf),
            ),
        )
        if underprediction_rows
        else {}
    )
    false_positive_count = sum(row_int(row, "row_count") for row in false_positive_rows)
    false_positive_area = sum(
        row_float(row, "predicted_canopy_area") for row in false_positive_rows
    )
    underprediction_count = sum(row_int(row, "row_count") for row in underprediction_rows)
    top_false_positive_source = dominant_label_source(false_positive_rows)
    interpretation = residual_diagnostic_interpretation(
        false_positive_count,
        underprediction_count,
        highest_abs_reason,
        underprediction_focus,
    )
    lines = [
        (
            f"Primary mask-aware diagnostics use `mask_status = {mask_status(analysis_config.domain_mask)}` "
            f"and `evaluation_scope = {evaluation_scope(analysis_config.domain_mask)}`. "
            "Rows are restricted to retained plausible-kelp cells; off-domain leakage stays in the separate audit table."
        ),
        (
            f"The largest retained group by row count is `{largest_reason['domain_mask_reason']}` "
            f"(`{row_int(largest_reason, 'row_count')}` rows). The retained mask reason with the largest "
            f"absolute mean residual is `{highest_abs_reason['domain_mask_reason']}` "
            f"(`{format_decimal(row_float(highest_abs_reason, 'mean_residual'), 1)} m2`)."
        ),
    ]
    if underprediction_focus:
        lines.append(
            f"The largest underprediction concentration is `{underprediction_focus['depth_bin']}` / "
            f"`{underprediction_focus['elevation_bin']}` in "
            f"`{underprediction_focus['residual_class']}` rows, with "
            f"`{row_int(underprediction_focus, 'row_count')}` rows and "
            f"`{row_int(underprediction_focus, 'high_error_count')}` high-error rows."
        )
    lines.extend(
        [
            (
                "Observed-zero false positives inside retained habitat account for "
                f"`{false_positive_count}` rows and `{format_decimal(false_positive_area, 1)} m2` "
                f"of predicted canopy area; the largest label-source group is `{top_false_positive_source}`."
            ),
            interpretation,
            "",
            "| Mask reason | Rows | Mean residual | Area bias | High-error rows |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(reason_rows, key=lambda item: row_int(item, "row_count"), reverse=True)[:6]:
        lines.append(
            f"| {row['domain_mask_reason']} | {row_int(row, 'row_count')} | "
            f"{format_decimal(row_float(row, 'mean_residual'), 1)} | "
            f"{format_decimal(row_float(row, 'area_bias'), 1)} | "
            f"{row_int(row, 'high_error_count')} |"
        )
    return "\n".join(lines)


def dominant_label_source(rows: list[dict[str, object]]) -> str:
    """Return the label-source value with the largest row count."""
    if not rows:
        return "none"
    by_source: dict[str, int] = {}
    for row in rows:
        source = str(row.get("label_source", "unknown"))
        by_source[source] = by_source.get(source, 0) + row_int(row, "row_count")
    return max(by_source.items(), key=lambda item: item[1])[0]


def residual_diagnostic_interpretation(
    false_positive_count: int,
    underprediction_count: int,
    highest_abs_reason: dict[str, object],
    under_depth: dict[str, object],
) -> str:
    """Build a concise interpretation of the residual diagnostic pattern."""
    reason = str(highest_abs_reason.get("domain_mask_reason", "unknown"))
    depth_bin = str(under_depth.get("depth_bin", "unknown"))
    if false_positive_count > underprediction_count:
        dominant_error = "false positives"
    elif underprediction_count > false_positive_count:
        dominant_error = "underprediction"
    else:
        dominant_error = "mixed false positives and underprediction"
    return (
        f"The retained-domain residuals point first to `{dominant_error}` inside the plausible habitat "
        f"rather than off-domain leakage. Concentration in `{reason}` and `{depth_bin}` should be treated "
        "as triage evidence for the next imbalance/objective task, not as a reason to tune mask thresholds here."
    )


def class_target_balance_markdown(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> str:
    """Build Markdown explaining annual-max class and target imbalance."""
    primary_rows = [
        row
        for row in tables.class_balance_by_split
        if row.get("label_source") == "all"
        and row.get("split") == analysis_config.analysis_split
        and str(row.get("year")) == str(analysis_config.analysis_year)
    ]
    if not primary_rows:
        primary_rows = [
            row for row in tables.target_balance_by_label_source if row.get("label_source") == "all"
        ]
    if not primary_rows:
        return "Class and target-balance rows were not available."
    model_sample = first_balance_scope_row(
        tables.target_balance_by_label_source, "model_input_sample"
    )
    full_grid = first_balance_scope_row(
        tables.class_balance_by_split,
        evaluation_scope(analysis_config.domain_mask),
        split=analysis_config.analysis_split,
        year=analysis_config.analysis_year,
    )
    if not full_grid:
        full_grid = first_balance_scope_row(
            tables.target_balance_by_label_source,
            evaluation_scope(analysis_config.domain_mask),
        )
    sample_zero_rate = row_float(model_sample, "zero_rate")
    sample_background_rate = row_float(model_sample, "assumed_background_rate")
    full_grid_zero_rate = row_float(full_grid, "zero_rate")
    full_grid_background_rate = row_float(full_grid, "assumed_background_rate")
    full_grid_positive_rate = row_float(full_grid, "positive_rate")
    full_grid_high_rate = row_float(full_grid, "high_canopy_rate")
    full_grid_saturated_rate = row_float(full_grid, "saturated_rate")
    lines = [
        (
            "These diagnostics quantify Kelpwatch-style annual-max imbalance before any binary, "
            "balanced, hurdle, or conditional model is introduced. They do not choose a production "
            "threshold and do not tune on the primary test split."
        ),
        (
            f"In the current model-input sample, zero rows are `{format_percent(sample_zero_rate, 1)}` "
            f"and assumed-background rows are `{format_percent(sample_background_rate, 1)}` of rows. "
            f"In the primary `{evaluation_scope(analysis_config.domain_mask)}` report scope, zero rows "
            f"are `{format_percent(full_grid_zero_rate, 1)}` and assumed-background rows are "
            f"`{format_percent(full_grid_background_rate, 1)}`."
        ),
        (
            f"Primary full-grid positives are `{format_percent(full_grid_positive_rate, 2)}` of rows; "
            f"high canopy rows at `>=450 m2` are `{format_percent(full_grid_high_rate, 2)}`, and "
            f"saturated or near-saturated rows are `{format_percent(full_grid_saturated_rate, 2)}`. "
            "The mask removes off-domain background from the report scope, but the retained plausible "
            "domain still has strong within-domain target imbalance."
        ),
        (
            "Later threshold, binary, balanced, hurdle, and conditional models should be evaluated "
            "against these rates so improvements are measured against the actual annual-max class mix, "
            "not introduced blindly."
        ),
        "",
        "| Scope | Split | Year | Rows | Positive | High canopy | Saturated | Assumed background |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows[:8]:
        lines.append(
            f"| {row.get('data_scope', '')} | "
            f"{row.get('split', '')} | "
            f"{row.get('year', '')} | "
            f"{row.get('row_count', '')} | "
            f"{format_percent(row_float(row, 'positive_rate'), 2)} | "
            f"{format_percent(row_float(row, 'high_canopy_rate'), 2)} | "
            f"{format_percent(row_float(row, 'saturated_rate'), 2)} | "
            f"{format_percent(row_float(row, 'assumed_background_rate'), 1)} |"
        )
    return "\n".join(lines)


def first_balance_scope_row(
    rows: list[dict[str, object]],
    data_scope: str,
    *,
    split: str | None = None,
    year: int | None = None,
) -> dict[str, object]:
    """Return the first balance row matching a source scope and optional split/year."""
    for row in rows:
        if row.get("data_scope") != data_scope:
            continue
        if row.get("label_source") != "all":
            continue
        if split is not None and row.get("split") != split:
            continue
        if year is not None and str(row.get("year")) != str(year):
            continue
        return row
    return {}


def binary_threshold_comparison_markdown(
    tables: AnalysisTables, analysis_config: ModelAnalysisConfig
) -> str:
    """Build Markdown for validation-only annual-max binary threshold selection."""
    recommendation = selected_or_first_recommendation_row(tables.binary_threshold_recommendation)
    if not recommendation:
        return "Annual-max binary threshold comparison rows were not available."
    selection_year = recommendation.get("selection_year", "all_validation_years")
    selected_label = str(recommendation.get("recommended_threshold_label", ""))
    selected_fraction = row_float(recommendation, "recommended_threshold_fraction")
    status = str(recommendation.get("recommendation_status", "missing"))
    selected_text = (
        f"The recommended next P1-18 candidate is `{selected_label}` "
        f"({format_percent(selected_fraction, 0)} annual max)."
        if selected_label
        else "No threshold is recommended because validation rows contain no positive annual-max target."
    )
    rows = [
        row
        for row in tables.binary_threshold_comparison
        if row.get("split") == DEFAULT_THRESHOLD_SELECTION_SPLIT
        and str(row.get("year")) == str(selection_year)
        and row.get("label_source") == "all"
        and row.get("threshold_role") == "selection_candidate"
    ]
    if not rows:
        rows = [
            row
            for row in tables.binary_threshold_comparison
            if row.get("split") == DEFAULT_THRESHOLD_SELECTION_SPLIT
            and row.get("label_source") == "all"
            and row.get("threshold_role") == "selection_candidate"
        ]
    rows = sorted(rows, key=lambda row: row_float(row, "threshold_fraction"))
    lines = [
        (
            f"Threshold selection uses only `{DEFAULT_THRESHOLD_SELECTION_SPLIT}` rows "
            f"from `{selection_year}` in the sample-prediction table; `{DEFAULT_THRESHOLD_TEST_SPLIT}` "
            "rows remain locked audit/context rows and are not used to choose the candidate. "
            "All thresholds are derived from the existing Kelpwatch annual-max fraction, so this "
            "does not change the label input or claim ecological truth."
        ),
        (
            f"{selected_text} Selection status is `{status}` under policy "
            f"`{BINARY_THRESHOLD_SELECTION_POLICY}`."
        ),
        "",
        "| Threshold | Positive | Predicted positive | Precision | Recall | F1 | Assumed-background FP area |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('threshold_label', '')} | "
            f"{format_percent(row_float(row, 'positive_rate'), 2)} | "
            f"{format_percent(row_float(row, 'predicted_positive_rate'), 2)} | "
            f"{format_decimal(row_float(row, 'precision'), 3)} | "
            f"{format_decimal(row_float(row, 'recall'), 3)} | "
            f"{format_decimal(row_float(row, 'f1'), 3)} | "
            f"{format_decimal(row_float(row, 'assumed_background_false_positive_area'), 1)} |"
        )
    lines.extend(
        [
            "",
            (
                f"Detailed prevalence, comparison, and recommendation rows are written to "
                f"`{analysis_config.binary_threshold_prevalence_path}`, "
                f"`{analysis_config.binary_threshold_comparison_path}`, and "
                f"`{analysis_config.binary_threshold_recommendation_path}`."
            ),
        ]
    )
    return "\n".join(lines)


def selected_or_first_recommendation_row(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return the selected recommendation row, falling back to the first row."""
    for row in rows:
        if bool(row.get("selected_candidate")):
            return row
    return rows[0] if rows else {}


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
            "sample_predictions": str(analysis_config.sample_predictions_path)
            if analysis_config.sample_predictions_path is not None
            else None,
            "predictions": str(analysis_config.predictions_path),
            "metrics": str(analysis_config.metrics_path),
            "domain_mask": str(analysis_config.domain_mask.table_path)
            if analysis_config.domain_mask is not None
            else None,
            "domain_mask_manifest": str(analysis_config.domain_mask.manifest_path)
            if analysis_config.domain_mask is not None
            and analysis_config.domain_mask.manifest_path is not None
            else None,
        },
        "mask_status": mask_status(analysis_config.domain_mask),
        "evaluation_scope": evaluation_scope(analysis_config.domain_mask),
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
            "class_balance_by_split": str(analysis_config.class_balance_by_split_path),
            "target_balance_by_label_source": str(
                analysis_config.target_balance_by_label_source_path
            ),
            "background_rate_summary": str(analysis_config.background_rate_summary_path),
            "binary_threshold_prevalence": str(analysis_config.binary_threshold_prevalence_path),
            "binary_threshold_comparison": str(analysis_config.binary_threshold_comparison_path),
            "binary_threshold_recommendation": str(
                analysis_config.binary_threshold_recommendation_path
            ),
            "residual_domain_context": str(analysis_config.residual_domain_context_path),
            "residual_by_mask_reason": str(analysis_config.residual_by_mask_reason_path),
            "residual_by_depth_bin": str(analysis_config.residual_by_depth_bin_path),
            "top_residual_context": str(analysis_config.top_residual_context_path),
            "class_balance_figure": str(analysis_config.class_balance_figure),
            "binary_threshold_comparison_figure": str(
                analysis_config.binary_threshold_comparison_figure
            ),
            "residual_domain_context_figure": str(analysis_config.residual_domain_context_figure),
            "reference_area_calibration": str(analysis_config.reference_area_calibration_path),
            "data_health": str(analysis_config.data_health_path),
            "manifest": str(analysis_config.manifest_path),
        },
        "row_counts": {
            "labels": int(len(data.labels)),
            "aligned": int(len(data.aligned)),
            "split_manifest": int(len(data.split_manifest)),
            "sample_predictions": int(len(data.sample_predictions)),
            "model_predictions": int(len(data.model_predictions)),
            "stage_distribution": len(tables.stage_distribution),
            "target_framing": len(tables.target_framing),
            "phase1_decision": len(tables.phase1_decision),
            "phase1_model_comparison": len(tables.phase1_model_comparison),
            "class_balance_by_split": len(tables.class_balance_by_split),
            "target_balance_by_label_source": len(tables.target_balance_by_label_source),
            "background_rate_summary": len(tables.background_rate_summary),
            "binary_threshold_prevalence": len(tables.binary_threshold_prevalence),
            "binary_threshold_comparison": len(tables.binary_threshold_comparison),
            "binary_threshold_recommendation": len(tables.binary_threshold_recommendation),
            "residual_domain_context": len(tables.residual_domain_context),
            "residual_by_mask_reason": len(tables.residual_by_mask_reason),
            "residual_by_depth_bin": len(tables.residual_by_depth_bin),
            "top_residual_context": len(tables.top_residual_context),
            "reference_area_calibration": len(tables.reference_area_calibration),
            "data_health": len(tables.data_health),
        },
    }
    analysis_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_config.manifest_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
