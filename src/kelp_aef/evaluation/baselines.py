"""Train and evaluate first tabular baseline models."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    ReportingDomainMask,
    evaluation_scope,
    filter_polars_to_reporting_domain,
    load_reporting_domain_mask,
    mask_status,
    masked_output_path,
)

LOGGER = logging.getLogger(__name__)

KELPWATCH_PIXEL_AREA_M2 = 900.0
NO_SKILL_MODEL_NAME = "no_skill_train_mean"
RIDGE_MODEL_NAME = "ridge_regression"
PREVIOUS_YEAR_MODEL_NAME = "previous_year_annual_max"
CLIMATOLOGY_MODEL_NAME = "grid_cell_climatology"
GEOGRAPHIC_MODEL_NAME = "geographic_ridge_lon_lat_year"
REFERENCE_MODEL_ORDER = (
    NO_SKILL_MODEL_NAME,
    RIDGE_MODEL_NAME,
    PREVIOUS_YEAR_MODEL_NAME,
    CLIMATOLOGY_MODEL_NAME,
    GEOGRAPHIC_MODEL_NAME,
)
SPLIT_ORDER = ("train", "validation", "test")
THRESHOLDS = ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"))
GEOGRAPHIC_FEATURE_COLUMNS = ("longitude", "latitude", "year")
REQUIRED_ID_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_fraction_y",
    "kelp_max_y",
)
METRIC_FIELDS = (
    "model_name",
    "split",
    "metric_group",
    "metric_group_value",
    "weighted",
    "target",
    "alpha",
    "row_count",
    "mae",
    "rmse",
    "r2",
    "pearson",
    "spearman",
    "observed_mean_fraction",
    "predicted_mean_fraction",
    "predicted_mean_fraction_clipped",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "observed_positive_rate_ge_1pct",
    "predicted_positive_rate_ge_1pct",
    "precision_ge_1pct",
    "recall_ge_1pct",
    "f1_ge_1pct",
    "observed_positive_rate_ge_5pct",
    "predicted_positive_rate_ge_5pct",
    "precision_ge_5pct",
    "recall_ge_5pct",
    "f1_ge_5pct",
    "observed_positive_rate_ge_10pct",
    "predicted_positive_rate_ge_10pct",
    "precision_ge_10pct",
    "recall_ge_10pct",
    "f1_ge_10pct",
)
FALLBACK_SUMMARY_FIELDS = (
    "evaluation_scope",
    "model_name",
    "split",
    "year",
    "label_source",
    "fallback_reason",
    "row_count",
)
AREA_CALIBRATION_FIELDS = (
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
OPTIONAL_PROVENANCE_COLUMNS = (
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
)
FULL_GRID_PREDICTION_BATCH_SIZE = 100_000


@dataclass(frozen=True)
class BaselineConfig:
    """Resolved config values for baseline training and evaluation."""

    config_path: Path
    sample_policy: str
    aligned_table_path: Path
    split_manifest_path: Path
    model_output_path: Path
    sample_predictions_path: Path
    predictions_path: Path
    prediction_manifest_path: Path
    inference_table_path: Path | None
    metrics_path: Path
    eval_manifest_path: Path
    geographic_model_output_path: Path
    fallback_summary_path: Path
    area_calibration_path: Path
    target_column: str
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    drop_missing_features: bool
    use_sample_weight: bool
    sample_weight_column: str
    reporting_domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class BaselineSidecarConfig:
    """Resolved config values for an optional baseline sidecar run."""

    name: str
    baseline_config: BaselineConfig


@dataclass(frozen=True)
class PreparedData:
    """Aligned rows with split assignments and missing-feature diagnostics."""

    split_manifest: pd.DataFrame
    retained_rows: pd.DataFrame
    dropped_counts_by_split: dict[str, int]


@dataclass(frozen=True)
class RidgeSelection:
    """Selected ridge model and validation diagnostics."""

    model: Any
    selected_alpha: float
    validation_rows: list[dict[str, object]]


@dataclass(frozen=True)
class ReferencePredictionResult:
    """Prediction rows and fallback diagnostics for reference baselines."""

    predictions: pd.DataFrame
    fallback_rows: list[dict[str, object]]


@dataclass(frozen=True)
class ReferenceLookupState:
    """Reusable lookup tables for persistence and climatology baselines."""

    key_columns: tuple[str, ...]
    previous_year: pd.DataFrame
    climatology_totals: pd.DataFrame
    climatology_year_totals: pd.DataFrame
    label_source_means: dict[str, float]
    global_mean: float


def train_baselines(config_path: Path) -> int:
    """Train and evaluate the first no-skill and ridge baselines."""
    baseline_config = load_baseline_config(config_path)
    train_baselines_config(baseline_config)
    for sidecar in load_baseline_sidecar_configs(config_path, baseline_config):
        LOGGER.info("Training baseline sidecar: %s", sidecar.name)
        train_baselines_config(sidecar.baseline_config)
    return 0


def train_baselines_config(baseline_config: BaselineConfig) -> None:
    """Train and evaluate one resolved no-skill, ridge, and reference config."""
    LOGGER.info("Loading aligned table: %s", baseline_config.aligned_table_path)
    aligned = pd.read_parquet(baseline_config.aligned_table_path)
    validate_aligned_table(aligned, baseline_config)
    prepared = prepare_model_frame(aligned, baseline_config)
    write_split_manifest(prepared.split_manifest, baseline_config.split_manifest_path)
    train_rows = rows_for_split(prepared.retained_rows, "train")
    validation_rows = rows_for_split(prepared.retained_rows, "validation")

    train_weights = model_weights(train_rows, baseline_config)
    train_mean = weighted_mean(
        train_rows[baseline_config.target_column].to_numpy(dtype=float),
        train_weights,
    )
    ridge_selection = fit_select_ridge(train_rows, validation_rows, baseline_config)
    geographic_selection = fit_select_geographic(train_rows, validation_rows, baseline_config)
    predictions = build_prediction_rows(
        prepared.retained_rows,
        baseline_config,
        train_mean=train_mean,
        ridge_model=ridge_selection.model,
        selected_alpha=ridge_selection.selected_alpha,
        geographic_model=geographic_selection.model,
        geographic_alpha=geographic_selection.selected_alpha,
    )
    reference_predictions = build_reference_predictions(
        prepared.retained_rows,
        prepared.retained_rows,
        baseline_config,
        evaluation_scope="background_inclusive_sample",
    )
    predictions = pd.concat(
        [predictions, reference_predictions.predictions],
        ignore_index=True,
    )
    metrics = build_metric_rows(predictions, baseline_config)
    write_predictions(predictions, baseline_config.sample_predictions_path)
    write_metrics(metrics, baseline_config.metrics_path)
    write_fallback_summary(
        reference_predictions.fallback_rows,
        baseline_config.fallback_summary_path,
    )
    write_ridge_model(
        ridge_selection,
        baseline_config,
        train_mean=train_mean,
    )
    write_geographic_model(geographic_selection, baseline_config)
    area_calibration_rows = build_reference_area_calibration_rows(baseline_config)
    if area_calibration_rows:
        write_csv(
            area_calibration_rows,
            baseline_config.area_calibration_path,
            AREA_CALIBRATION_FIELDS,
        )
    write_eval_manifest(
        baseline_config=baseline_config,
        prepared=prepared,
        ridge_selection=ridge_selection,
        geographic_selection=geographic_selection,
        train_mean=train_mean,
        predictions=predictions,
        fallback_rows=reference_predictions.fallback_rows,
    )
    LOGGER.info("Wrote split manifest: %s", baseline_config.split_manifest_path)
    LOGGER.info("Wrote baseline sample predictions: %s", baseline_config.sample_predictions_path)
    LOGGER.info("Wrote baseline metrics: %s", baseline_config.metrics_path)
    LOGGER.info("Wrote reference fallback summary: %s", baseline_config.fallback_summary_path)
    LOGGER.info("Wrote ridge model: %s", baseline_config.model_output_path)
    LOGGER.info("Wrote geographic model: %s", baseline_config.geographic_model_output_path)
    if area_calibration_rows:
        LOGGER.info("Wrote reference area calibration: %s", baseline_config.area_calibration_path)
    LOGGER.info("Wrote baseline evaluation manifest: %s", baseline_config.eval_manifest_path)


def predict_full_grid(config_path: Path, *, fast: bool = False) -> int:
    """Apply the trained ridge model to the full-grid inference table in chunks."""
    baseline_config = load_baseline_config(config_path)
    predict_full_grid_config(baseline_config, fast=fast)
    for sidecar in load_baseline_sidecar_configs(config_path, baseline_config):
        LOGGER.info("Streaming baseline full-grid sidecar predictions: %s", sidecar.name)
        predict_full_grid_config(sidecar.baseline_config, fast=fast)
    return 0


def predict_full_grid_config(baseline_config: BaselineConfig, *, fast: bool = False) -> None:
    """Apply one resolved baseline ridge model to the full-grid table in chunks."""
    if baseline_config.inference_table_path is None:
        msg = "models.baselines.inference_table is required for predict-full-grid"
        raise ValueError(msg)
    inference_path = (
        suffix_path(baseline_config.inference_table_path, ".fast")
        if fast
        else baseline_config.inference_table_path
    )
    model_payload = load_model_payload(baseline_config.model_output_path)
    model = model_payload["model"]
    output_path = (
        suffix_path(baseline_config.predictions_path, ".fast")
        if fast
        else baseline_config.predictions_path
    )
    manifest_path = (
        suffix_path(baseline_config.prediction_manifest_path, ".fast")
        if fast
        else baseline_config.prediction_manifest_path
    )
    reset_output_path(output_path)
    row_count = 0
    part_count = 0
    label_source_counts: dict[str, int] = {}
    columns = prediction_input_columns(baseline_config)
    LOGGER.info("Streaming full-grid inference from %s", inference_path)
    for batch in iter_parquet_batches(
        inference_path,
        columns,
        FULL_GRID_PREDICTION_BATCH_SIZE,
    ):
        batch["split"] = assign_splits(batch["year"], baseline_config)
        predictions = predict_with_missing_features(batch, baseline_config, model)
        prediction_rows = prediction_frame(
            batch,
            baseline_config,
            model_name=RIDGE_MODEL_NAME,
            alpha=float(model_payload.get("selected_alpha", math.nan)),
            predictions=predictions,
        )
        write_prediction_part(prediction_rows, output_path, part_count)
        row_count += len(prediction_rows)
        part_count += 1
        if "label_source" in prediction_rows.columns:
            for label_source, count in prediction_rows["label_source"].value_counts().items():
                label_source_counts[str(label_source)] = label_source_counts.get(
                    str(label_source), 0
                ) + int(count)
        LOGGER.info("Wrote full-grid prediction part %s with %s rows", part_count, len(batch))
    write_prediction_manifest(
        baseline_config=baseline_config,
        inference_path=inference_path,
        output_path=output_path,
        manifest_path=manifest_path,
        row_count=row_count,
        part_count=part_count,
        label_source_counts=label_source_counts,
        fast=fast,
    )
    LOGGER.info("Wrote full-grid predictions: %s", output_path)
    LOGGER.info("Wrote full-grid prediction manifest: %s", manifest_path)


def load_baseline_config(config_path: Path) -> BaselineConfig:
    """Load baseline training settings from the workflow config."""
    config = load_yaml_config(config_path)
    alignment = require_mapping(config.get("alignment"), "alignment")
    splits = require_mapping(config.get("splits"), "splits")
    features = require_mapping(config.get("features"), "features")
    models = require_mapping(config.get("models"), "models")
    baseline = require_mapping(models.get("baselines"), "models.baselines")
    reports = config.get("reports")
    outputs: dict[str, Any] = {}
    if reports is not None:
        reports_mapping = require_mapping(reports, "reports")
        outputs_value = reports_mapping.get("outputs")
        if outputs_value is not None:
            outputs = require_mapping(outputs_value, "reports.outputs")
    predictions_path = Path(
        require_string(baseline.get("predictions"), "models.baselines.predictions")
    )
    sample_predictions_value = baseline.get("sample_predictions")
    inference_table_value = baseline.get("inference_table")
    prediction_manifest_value = baseline.get("prediction_manifest")
    ridge_model_path = Path(
        require_string(baseline.get("ridge_model"), "models.baselines.ridge_model")
    )
    geographic_model_value = baseline.get("geographic_model")
    fallback_summary_value = outputs.get("reference_baseline_fallback_summary")
    reporting_domain_mask = load_reporting_domain_mask(config)
    area_calibration_path = masked_output_path(
        outputs,
        unmasked_key="reference_baseline_area_calibration",
        masked_key="reference_baseline_area_calibration_masked",
        default=Path(require_string(baseline.get("metrics"), "models.baselines.metrics")).parent
        / "reference_baseline_area_calibration.csv",
        mask_config=reporting_domain_mask,
    )
    return BaselineConfig(
        config_path=config_path,
        sample_policy=str(baseline.get("sample_policy", "current_masked_sample")),
        aligned_table_path=Path(
            require_string(
                baseline.get("input_table") or alignment.get("output_table"),
                "models.baselines.input_table or alignment.output_table",
            )
        ),
        split_manifest_path=Path(
            require_string(splits.get("output_manifest"), "splits.output_manifest")
        ),
        model_output_path=ridge_model_path,
        sample_predictions_path=(
            Path(
                require_string(
                    sample_predictions_value,
                    "models.baselines.sample_predictions",
                )
            )
            if sample_predictions_value is not None
            else predictions_path
        ),
        predictions_path=predictions_path,
        prediction_manifest_path=(
            Path(
                require_string(
                    prediction_manifest_value,
                    "models.baselines.prediction_manifest",
                )
            )
            if prediction_manifest_value is not None
            else predictions_path.parent / "baseline_full_grid_prediction_manifest.json"
        ),
        inference_table_path=(
            Path(require_string(inference_table_value, "models.baselines.inference_table"))
            if inference_table_value is not None
            else None
        ),
        metrics_path=Path(require_string(baseline.get("metrics"), "models.baselines.metrics")),
        eval_manifest_path=Path(
            require_string(baseline.get("manifest"), "models.baselines.manifest")
        ),
        geographic_model_output_path=(
            Path(
                require_string(
                    geographic_model_value,
                    "models.baselines.geographic_model",
                )
            )
            if geographic_model_value is not None
            else ridge_model_path.parent / "geographic_ridge_lon_lat_year.joblib"
        ),
        fallback_summary_path=(
            Path(
                require_string(
                    fallback_summary_value,
                    "reports.outputs.reference_baseline_fallback_summary",
                )
            )
            if fallback_summary_value is not None
            else Path(require_string(baseline.get("metrics"), "models.baselines.metrics")).parent
            / "reference_baseline_fallback_summary.csv"
        ),
        area_calibration_path=area_calibration_path,
        target_column=require_string(baseline.get("target"), "models.baselines.target"),
        feature_columns=parse_bands(baseline.get("features") or features.get("bands")),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
        alpha_grid=read_alpha_grid(baseline.get("alpha_grid")),
        drop_missing_features=read_bool(
            baseline.get("drop_missing_features"),
            "models.baselines.drop_missing_features",
            default=True,
        ),
        use_sample_weight=read_bool(
            baseline.get("use_sample_weight"),
            "models.baselines.use_sample_weight",
            default=False,
        ),
        sample_weight_column=str(baseline.get("sample_weight_column", "sample_weight")),
        reporting_domain_mask=reporting_domain_mask,
    )


def load_baseline_sidecar_configs(
    config_path: Path,
    base_config: BaselineConfig,
) -> tuple[BaselineSidecarConfig, ...]:
    """Load optional baseline sidecar configs from the workflow config."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    baseline = require_mapping(models.get("baselines"), "models.baselines")
    sidecars = optional_mapping(baseline.get("sidecars"), "models.baselines.sidecars")
    output: list[BaselineSidecarConfig] = []
    for name, value in sidecars.items():
        sidecar_name = str(name)
        sidecar = require_mapping(value, f"models.baselines.sidecars.{sidecar_name}")
        if not read_bool(
            sidecar.get("enabled"),
            f"models.baselines.sidecars.{sidecar_name}.enabled",
            default=True,
        ):
            continue
        output.append(
            BaselineSidecarConfig(
                name=sidecar_name,
                baseline_config=replace(
                    base_config,
                    sample_policy=str(sidecar.get("sample_policy", sidecar_name)),
                    aligned_table_path=baseline_sidecar_path(sidecar, sidecar_name, "input_table"),
                    split_manifest_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "split_manifest"
                    ),
                    model_output_path=baseline_sidecar_path(sidecar, sidecar_name, "ridge_model"),
                    geographic_model_output_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "geographic_model"
                    ),
                    sample_predictions_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "sample_predictions"
                    ),
                    predictions_path=baseline_sidecar_path(sidecar, sidecar_name, "predictions"),
                    prediction_manifest_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "prediction_manifest"
                    ),
                    metrics_path=baseline_sidecar_path(sidecar, sidecar_name, "metrics"),
                    eval_manifest_path=baseline_sidecar_path(sidecar, sidecar_name, "manifest"),
                    fallback_summary_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "fallback_summary"
                    ),
                    area_calibration_path=baseline_sidecar_path(
                        sidecar, sidecar_name, "area_calibration"
                    ),
                ),
            )
        )
    return tuple(output)


def baseline_sidecar_path(config: dict[str, Any], sidecar_name: str, key: str) -> Path:
    """Read a required baseline sidecar path from config."""
    return Path(
        require_string(
            config.get(key),
            f"models.baselines.sidecars.{sidecar_name}.{key}",
        )
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating a missing value as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def read_year_list(config: dict[str, Any], key: str) -> tuple[int, ...]:
    """Read a non-empty year list from a config mapping."""
    values = config.get(key)
    if not isinstance(values, list) or not values:
        msg = f"config field must be a non-empty list of years: splits.{key}"
        raise ValueError(msg)
    return tuple(require_int_value(value, f"splits.{key}[]") for value in values)


def read_alpha_grid(value: object) -> tuple[float, ...]:
    """Read a non-empty ridge alpha grid from config."""
    if not isinstance(value, list) or not value:
        msg = "config field must be a non-empty list: models.baselines.alpha_grid"
        raise ValueError(msg)
    alphas = tuple(require_positive_float(item, "models.baselines.alpha_grid[]") for item in value)
    if len(set(alphas)) != len(alphas):
        msg = "ridge alpha grid contains duplicate values"
        raise ValueError(msg)
    return alphas


def read_bool(value: object, name: str, *, default: bool) -> bool:
    """Read an optional boolean config value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"config field must be a boolean: {name}"
        raise ValueError(msg)
    return value


def parse_bands(value: object) -> tuple[str, ...]:
    """Parse configured AEF band names from a range string or list."""
    if isinstance(value, list):
        bands = tuple(str(item) for item in value)
    elif isinstance(value, str):
        bands = parse_band_string(value)
    else:
        msg = "config field must be a band range string or list"
        raise ValueError(msg)
    if not bands:
        msg = "at least one feature band must be configured"
        raise ValueError(msg)
    return bands


def parse_band_string(value: str) -> tuple[str, ...]:
    """Parse a feature band string such as A00-A63 or A00,A01."""
    compact = value.replace(" ", "")
    match = re.fullmatch(r"A(\d+)-A(\d+)", compact)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if end < start:
            msg = f"band range must be ascending: {value}"
            raise ValueError(msg)
        width = max(len(match.group(1)), len(match.group(2)))
        return tuple(f"A{index:0{width}d}" for index in range(start, end + 1))
    return tuple(part for part in compact.split(",") if part)


def validate_aligned_table(dataframe: pd.DataFrame, baseline_config: BaselineConfig) -> None:
    """Validate that the aligned table has the columns needed for modeling."""
    required_columns = [
        *REQUIRED_ID_COLUMNS,
        baseline_config.target_column,
        *baseline_config.feature_columns,
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        msg = f"aligned table is missing required columns: {missing}"
        raise ValueError(msg)


def prepare_model_frame(dataframe: pd.DataFrame, baseline_config: BaselineConfig) -> PreparedData:
    """Assign splits, mark dropped rows, and return retained modeling rows."""
    split_columns = split_manifest_columns(dataframe, baseline_config)
    split_manifest = dataframe[split_columns].copy()
    split_manifest["split"] = assign_splits(split_manifest["year"], baseline_config)
    feature_complete = dataframe.loc[:, list(baseline_config.feature_columns)].notna().all(axis=1)
    target_complete = dataframe[baseline_config.target_column].notna()
    split_manifest["has_complete_features"] = feature_complete.to_numpy(dtype=bool)
    split_manifest["has_target"] = target_complete.to_numpy(dtype=bool)
    split_manifest["used_for_training_eval"] = (
        split_manifest["has_complete_features"] & split_manifest["has_target"]
    )
    split_manifest["drop_reason"] = drop_reasons(split_manifest)
    retained_mask = split_manifest["used_for_training_eval"].to_numpy(dtype=bool)
    if not baseline_config.drop_missing_features and not bool(retained_mask.all()):
        msg = "configured to keep missing features, but ridge baseline cannot fit NaN feature rows"
        raise ValueError(msg)
    retained = dataframe.loc[retained_mask].copy()
    retained["split"] = split_manifest.loc[retained_mask, "split"].to_numpy()
    dropped_counts = dropped_counts_by_split(split_manifest)
    LOGGER.info(
        "Retained %s rows for modeling; dropped rows by split: %s", len(retained), dropped_counts
    )
    ensure_required_splits_present(retained)
    return PreparedData(
        split_manifest=split_manifest,
        retained_rows=retained,
        dropped_counts_by_split=dropped_counts,
    )


def split_manifest_columns(dataframe: pd.DataFrame, baseline_config: BaselineConfig) -> list[str]:
    """Return stable split-manifest identity and target columns."""
    columns = [
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        baseline_config.target_column,
    ]
    for column in OPTIONAL_PROVENANCE_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    return columns


def assign_splits(years: pd.Series, baseline_config: BaselineConfig) -> pd.Series:
    """Assign each configured year to train, validation, or test."""
    train_years = set(baseline_config.train_years)
    validation_years = set(baseline_config.validation_years)
    test_years = set(baseline_config.test_years)
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
    unassigned_years = sorted(set(int(year) for year in years[split == "unassigned"].unique()))
    if unassigned_years:
        msg = f"aligned table contains years not assigned to a split: {unassigned_years}"
        raise ValueError(msg)
    return split


def drop_reasons(split_manifest: pd.DataFrame) -> pd.Series:
    """Build human-readable drop reasons for the split manifest."""
    reasons = pd.Series(index=split_manifest.index, data="", dtype="object")
    reasons.loc[~split_manifest["has_complete_features"]] = "missing_features"
    missing_target_mask = ~split_manifest["has_target"]
    reasons.loc[missing_target_mask & (reasons != "")] += ";missing_target"
    reasons.loc[missing_target_mask & (reasons == "")] = "missing_target"
    return reasons


def dropped_counts_by_split(split_manifest: pd.DataFrame) -> dict[str, int]:
    """Count dropped rows in each split."""
    dropped = split_manifest.loc[~split_manifest["used_for_training_eval"]]
    counts = {split: 0 for split in SPLIT_ORDER}
    for split, group in dropped.groupby("split", sort=False):
        counts[str(split)] = int(len(group))
    return counts


def ensure_required_splits_present(dataframe: pd.DataFrame) -> None:
    """Validate that retained rows include all expected split labels."""
    missing = [split for split in SPLIT_ORDER if split not in set(dataframe["split"])]
    if missing:
        msg = f"retained modeling rows are missing splits: {missing}"
        raise ValueError(msg)


def rows_for_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return rows for one split, raising when the split is empty."""
    rows = dataframe.loc[dataframe["split"] == split].copy()
    if rows.empty:
        msg = f"no retained rows for split: {split}"
        raise ValueError(msg)
    return rows


def fit_select_ridge(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    baseline_config: BaselineConfig,
) -> RidgeSelection:
    """Fit ridge candidates on train rows and select alpha by validation RMSE."""
    x_train = feature_matrix(train_rows, baseline_config.feature_columns)
    y_train = target_vector(train_rows, baseline_config.target_column)
    train_weights = model_weights(train_rows, baseline_config)
    x_validation = feature_matrix(validation_rows, baseline_config.feature_columns)
    y_validation = target_vector(validation_rows, baseline_config.target_column)
    validation_rows_for_grid: list[dict[str, object]] = []
    candidates: list[tuple[float, float, Any]] = []
    for alpha in baseline_config.alpha_grid:
        model = make_ridge_pipeline(alpha)
        fit_ridge_model(model, x_train, y_train, train_weights)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        rmse = root_mean_squared_error(y_validation, predictions)
        validation_rows_for_grid.append(
            {
                "alpha": alpha,
                "validation_rmse": rmse,
                "validation_mae": mean_absolute_error(y_validation, predictions),
            }
        )
        candidates.append((rmse, alpha, model))
        LOGGER.info("Ridge alpha=%s validation RMSE=%s", alpha, rmse)
    selected_rmse, selected_alpha, selected_model = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    LOGGER.info("Selected ridge alpha=%s with validation RMSE=%s", selected_alpha, selected_rmse)
    return RidgeSelection(
        model=selected_model,
        selected_alpha=selected_alpha,
        validation_rows=validation_rows_for_grid,
    )


def fit_select_geographic(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    baseline_config: BaselineConfig,
) -> RidgeSelection:
    """Fit a lat/lon/year-only ridge model and select alpha on validation RMSE."""
    x_train = feature_matrix(train_rows, GEOGRAPHIC_FEATURE_COLUMNS)
    y_train = target_vector(train_rows, baseline_config.target_column)
    train_weights = model_weights(train_rows, baseline_config)
    x_validation = feature_matrix(validation_rows, GEOGRAPHIC_FEATURE_COLUMNS)
    y_validation = target_vector(validation_rows, baseline_config.target_column)
    validation_rows_for_grid: list[dict[str, object]] = []
    candidates: list[tuple[float, float, Any]] = []
    for alpha in baseline_config.alpha_grid:
        model = make_ridge_pipeline(alpha)
        fit_ridge_model(model, x_train, y_train, train_weights)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        rmse = root_mean_squared_error(y_validation, predictions)
        validation_rows_for_grid.append(
            {
                "alpha": alpha,
                "validation_rmse": rmse,
                "validation_mae": mean_absolute_error(y_validation, predictions),
            }
        )
        candidates.append((rmse, alpha, model))
        LOGGER.info("Geographic ridge alpha=%s validation RMSE=%s", alpha, rmse)
    selected_rmse, selected_alpha, selected_model = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    LOGGER.info(
        "Selected geographic ridge alpha=%s with validation RMSE=%s",
        selected_alpha,
        selected_rmse,
    )
    return RidgeSelection(
        model=selected_model,
        selected_alpha=selected_alpha,
        validation_rows=validation_rows_for_grid,
    )


def fit_ridge_model(
    model: Any, features: np.ndarray, target: np.ndarray, weights: np.ndarray | None
) -> None:
    """Fit the ridge pipeline with optional sample weights."""
    if weights is None:
        model.fit(features, target)
        return
    model.fit(features, target, ridge__sample_weight=weights)


def make_ridge_pipeline(alpha: float) -> Any:
    """Build the standard scaler plus ridge pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def feature_matrix(dataframe: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Return model features as a floating-point matrix."""
    return cast(np.ndarray, dataframe.loc[:, list(feature_columns)].to_numpy(dtype=float))


def target_vector(dataframe: pd.DataFrame, target_column: str) -> np.ndarray:
    """Return the target column as a floating-point vector."""
    return cast(np.ndarray, dataframe[target_column].to_numpy(dtype=float))


def build_prediction_rows(
    retained_rows: pd.DataFrame,
    baseline_config: BaselineConfig,
    *,
    train_mean: float,
    ridge_model: Any,
    selected_alpha: float,
    geographic_model: Any,
    geographic_alpha: float,
) -> pd.DataFrame:
    """Build prediction rows for no-skill, AEF ridge, and geographic ridge models."""
    x_all = feature_matrix(retained_rows, baseline_config.feature_columns)
    x_geographic = feature_matrix(retained_rows, GEOGRAPHIC_FEATURE_COLUMNS)
    no_skill_predictions = np.full(len(retained_rows), train_mean, dtype=float)
    ridge_predictions = np.asarray(ridge_model.predict(x_all), dtype=float)
    geographic_predictions = np.asarray(geographic_model.predict(x_geographic), dtype=float)
    frames = [
        prediction_frame(
            retained_rows,
            baseline_config,
            model_name=NO_SKILL_MODEL_NAME,
            alpha=math.nan,
            predictions=no_skill_predictions,
        ),
        prediction_frame(
            retained_rows,
            baseline_config,
            model_name=RIDGE_MODEL_NAME,
            alpha=selected_alpha,
            predictions=ridge_predictions,
        ),
        prediction_frame(
            retained_rows,
            baseline_config,
            model_name=GEOGRAPHIC_MODEL_NAME,
            alpha=geographic_alpha,
            predictions=geographic_predictions,
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def prediction_frame(
    retained_rows: pd.DataFrame,
    baseline_config: BaselineConfig,
    *,
    model_name: str,
    alpha: float,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build the standard prediction schema for one model."""
    frame = retained_rows[prediction_identity_columns(retained_rows, baseline_config)].copy()
    clipped = np.clip(predictions, 0.0, 1.0)
    frame["model_name"] = model_name
    frame["alpha"] = alpha
    frame["pred_kelp_fraction_y"] = predictions
    frame["pred_kelp_fraction_y_clipped"] = clipped
    frame["pred_kelp_max_y"] = clipped * KELPWATCH_PIXEL_AREA_M2
    frame["residual_kelp_fraction_y"] = frame[baseline_config.target_column] - predictions
    frame["residual_kelp_fraction_y_clipped"] = frame[baseline_config.target_column] - clipped
    frame["residual_kelp_max_y"] = frame["kelp_max_y"] - frame["pred_kelp_max_y"]
    return frame


def prediction_identity_columns(
    dataframe: pd.DataFrame, baseline_config: BaselineConfig
) -> list[str]:
    """Return identity, provenance, and target columns for prediction outputs."""
    columns = [
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        baseline_config.target_column,
        "kelp_max_y",
    ]
    for column in OPTIONAL_PROVENANCE_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    return columns


def build_reference_predictions(
    dataframe: pd.DataFrame,
    lookup_source: pd.DataFrame,
    baseline_config: BaselineConfig,
    *,
    evaluation_scope: str,
) -> ReferencePredictionResult:
    """Build previous-year and grid-cell climatology prediction rows."""
    key_columns = cell_key_columns(lookup_source)
    lookup_state = build_reference_lookup_state(lookup_source, baseline_config, key_columns)
    return build_reference_predictions_from_state(
        dataframe,
        baseline_config,
        lookup_state,
        evaluation_scope=evaluation_scope,
    )


def build_reference_predictions_from_state(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    lookup_state: ReferenceLookupState,
    *,
    evaluation_scope: str,
) -> ReferencePredictionResult:
    """Build reference prediction rows from precomputed lookup state."""
    previous_predictions, previous_reasons = previous_year_predictions(
        dataframe, baseline_config, lookup_state
    )
    climatology_values, climatology_reasons = climatology_predictions(
        dataframe, baseline_config, lookup_state
    )
    previous_frame = reference_prediction_frame(
        dataframe,
        previous_predictions,
        baseline_config,
        model_name=PREVIOUS_YEAR_MODEL_NAME,
    )
    climatology_frame = prediction_frame(
        dataframe,
        baseline_config,
        model_name=CLIMATOLOGY_MODEL_NAME,
        alpha=math.nan,
        predictions=climatology_values.to_numpy(dtype=float),
    )
    fallback_rows = fallback_summary_rows(
        dataframe,
        previous_reasons,
        evaluation_scope=evaluation_scope,
        model_name=PREVIOUS_YEAR_MODEL_NAME,
    )
    fallback_rows.extend(
        fallback_summary_rows(
            dataframe,
            climatology_reasons,
            evaluation_scope=evaluation_scope,
            model_name=CLIMATOLOGY_MODEL_NAME,
        )
    )
    return ReferencePredictionResult(
        predictions=pd.concat([previous_frame, climatology_frame], ignore_index=True),
        fallback_rows=fallback_rows,
    )


def cell_key_columns(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Return stable target-grid cell key columns from a dataframe."""
    return cell_key_columns_from_names(tuple(dataframe.columns))


def cell_key_columns_from_names(column_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable target-grid cell key columns from available column names."""
    columns = set(column_names)
    if "aef_grid_cell_id" in columns:
        return ("aef_grid_cell_id",)
    if {"aef_grid_row", "aef_grid_col"}.issubset(columns):
        return ("aef_grid_row", "aef_grid_col")
    msg = "reference baselines require a stable AEF grid cell key"
    raise ValueError(msg)


def build_reference_lookup_state(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
) -> ReferenceLookupState:
    """Build lookup tables used by persistence and climatology baselines."""
    source_columns = [*key_columns, "year", baseline_config.target_column]
    if "label_source" in dataframe.columns:
        source_columns.append("label_source")
    source = dataframe[source_columns].copy()
    source[baseline_config.target_column] = source[baseline_config.target_column].astype(float)
    previous_year = previous_year_lookup(source, baseline_config, key_columns)
    training = source.loc[source["year"].isin(baseline_config.train_years)].copy()
    if training.empty:
        msg = "grid-cell climatology requires at least one training-year row"
        raise ValueError(msg)
    climatology_totals = climatology_total_lookup(training, baseline_config, key_columns)
    climatology_year_totals = climatology_year_lookup(training, baseline_config, key_columns)
    label_source_means = training_label_source_means(training, baseline_config)
    global_mean = safe_mean(training[baseline_config.target_column].to_numpy(dtype=float))
    return ReferenceLookupState(
        key_columns=key_columns,
        previous_year=previous_year,
        climatology_totals=climatology_totals,
        climatology_year_totals=climatology_year_totals,
        label_source_means=label_source_means,
        global_mean=global_mean,
    )


def previous_year_lookup(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build a lookup where row year points to the previous year's target value."""
    grouped = (
        dataframe.groupby([*key_columns, "year"], dropna=False)[baseline_config.target_column]
        .mean()
        .reset_index()
    )
    grouped["year"] = grouped["year"].astype(int) + 1
    return grouped.rename(columns={baseline_config.target_column: "reference_prediction"})


def climatology_total_lookup(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build cell-level training-year target sums and counts."""
    grouped = (
        dataframe.groupby(list(key_columns), dropna=False)[baseline_config.target_column]
        .agg(["sum", "count"])
        .reset_index()
    )
    return grouped.rename(columns={"sum": "training_sum", "count": "training_count"})


def climatology_year_lookup(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build cell-year training target sums and counts for train leave-one-year-out."""
    grouped = (
        dataframe.groupby([*key_columns, "year"], dropna=False)[baseline_config.target_column]
        .agg(["sum", "count"])
        .reset_index()
    )
    return grouped.rename(columns={"sum": "year_sum", "count": "year_count"})


def training_label_source_means(
    dataframe: pd.DataFrame, baseline_config: BaselineConfig
) -> dict[str, float]:
    """Return training means by label source for climatology fallback."""
    if "label_source" not in dataframe.columns:
        return {}
    grouped = dataframe.groupby("label_source", dropna=False)[baseline_config.target_column].mean()
    return {str(label_source): float(value) for label_source, value in grouped.items()}


def previous_year_predictions(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    lookup_state: ReferenceLookupState,
) -> tuple[pd.Series, pd.Series]:
    """Return previous-year predictions and availability reasons for rows."""
    merge_columns = [*lookup_state.key_columns, "year"]
    merged = dataframe[merge_columns].merge(
        lookup_state.previous_year,
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )
    predictions = pd.Series(
        merged["reference_prediction"].to_numpy(dtype=float),
        index=dataframe.index,
        dtype="float64",
    )
    reasons = pd.Series("missing_previous_year", index=dataframe.index, dtype="object")
    reasons.loc[predictions.notna()] = "previous_year"
    return predictions, reasons


def climatology_predictions(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    lookup_state: ReferenceLookupState,
) -> tuple[pd.Series, pd.Series]:
    """Return grid-cell climatology predictions and fallback reasons."""
    base = dataframe[[*lookup_state.key_columns, "year"]].copy()
    merged = base.merge(
        lookup_state.climatology_totals,
        on=list(lookup_state.key_columns),
        how="left",
        validate="many_to_one",
    )
    predictions = pd.Series(
        merged["training_sum"].to_numpy(dtype=float)
        / merged["training_count"].to_numpy(dtype=float),
        index=dataframe.index,
        dtype="float64",
    )
    reasons = pd.Series("cell_training_mean", index=dataframe.index, dtype="object")
    train_mask = dataframe["year"].isin(baseline_config.train_years)
    if train_mask.any():
        train_predictions, train_reasons = train_leave_one_year_out_climatology(
            dataframe.loc[train_mask],
            lookup_state,
        )
        predictions.loc[train_mask] = train_predictions
        reasons.loc[train_mask] = train_reasons
    predictions, reasons = apply_climatology_fallbacks(
        dataframe,
        predictions,
        reasons,
        lookup_state,
    )
    return predictions, reasons


def train_leave_one_year_out_climatology(
    dataframe: pd.DataFrame,
    lookup_state: ReferenceLookupState,
) -> tuple[pd.Series, pd.Series]:
    """Compute training-row climatology without each row's own year."""
    merge_columns = [*lookup_state.key_columns, "year"]
    merged = (
        dataframe[merge_columns]
        .merge(
            lookup_state.climatology_totals,
            on=list(lookup_state.key_columns),
            how="left",
            validate="many_to_one",
        )
        .merge(
            lookup_state.climatology_year_totals,
            on=merge_columns,
            how="left",
            validate="many_to_one",
        )
    )
    numerator = merged["training_sum"].to_numpy(dtype=float) - merged["year_sum"].to_numpy(
        dtype=float
    )
    denominator = merged["training_count"].to_numpy(dtype=float) - merged["year_count"].to_numpy(
        dtype=float
    )
    predictions = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )
    reasons = np.where(
        np.isfinite(predictions),
        "cell_training_mean_leave_one_year_out",
        "missing_cell_training_history",
    )
    return (
        pd.Series(predictions, index=dataframe.index, dtype="float64"),
        pd.Series(reasons, index=dataframe.index, dtype="object"),
    )


def apply_climatology_fallbacks(
    dataframe: pd.DataFrame,
    predictions: pd.Series,
    reasons: pd.Series,
    lookup_state: ReferenceLookupState,
) -> tuple[pd.Series, pd.Series]:
    """Fill missing climatology predictions with label-source and global means."""
    missing = predictions.isna()
    if missing.any() and lookup_state.label_source_means and "label_source" in dataframe.columns:
        label_source_values = dataframe.loc[missing, "label_source"].fillna("unknown").astype(str)
        label_predictions = label_source_values.map(lookup_state.label_source_means)
        label_mask = label_predictions.notna()
        selected_index = label_predictions.loc[label_mask].index
        predictions.loc[selected_index] = label_predictions.loc[label_mask].to_numpy(dtype=float)
        reasons.loc[selected_index] = "label_source_training_mean"
    missing = predictions.isna()
    if missing.any():
        predictions.loc[missing] = lookup_state.global_mean
        reasons.loc[missing] = "global_training_mean"
    return predictions, reasons


def reference_prediction_frame(
    dataframe: pd.DataFrame,
    predictions: pd.Series,
    baseline_config: BaselineConfig,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Build prediction rows for reference baselines with missing rows omitted."""
    valid_mask = predictions.notna().to_numpy(dtype=bool)
    if not np.any(valid_mask):
        return pd.DataFrame(columns=prediction_output_columns(dataframe, baseline_config))
    return prediction_frame(
        dataframe.loc[valid_mask],
        baseline_config,
        model_name=model_name,
        alpha=math.nan,
        predictions=predictions.loc[valid_mask].to_numpy(dtype=float),
    )


def prediction_output_columns(
    dataframe: pd.DataFrame, baseline_config: BaselineConfig
) -> list[str]:
    """Return the standard prediction output columns for an empty frame."""
    identity_columns = prediction_identity_columns(dataframe, baseline_config)
    return [
        *identity_columns,
        "model_name",
        "alpha",
        "pred_kelp_fraction_y",
        "pred_kelp_fraction_y_clipped",
        "pred_kelp_max_y",
        "residual_kelp_fraction_y",
        "residual_kelp_fraction_y_clipped",
        "residual_kelp_max_y",
    ]


def fallback_summary_rows(
    dataframe: pd.DataFrame,
    reasons: pd.Series,
    *,
    evaluation_scope: str,
    model_name: str,
) -> list[dict[str, object]]:
    """Summarize fallback and missing-history behavior by split, year, and source."""
    frame = dataframe[["split", "year"]].copy()
    frame["label_source"] = label_source_values(dataframe)
    frame["fallback_reason"] = reasons.to_numpy(dtype=object)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        ["split", "year", "label_source", "fallback_reason"],
        sort=True,
        dropna=False,
    ):
        split, year, label_source, fallback_reason = cast(tuple[str, int, str, str], keys)
        rows.append(
            {
                "evaluation_scope": evaluation_scope,
                "model_name": model_name,
                "split": str(split),
                "year": int(year),
                "label_source": str(label_source),
                "fallback_reason": str(fallback_reason),
                "row_count": int(len(group)),
            }
        )
    return rows


def label_source_values(dataframe: pd.DataFrame) -> pd.Series:
    """Return label-source values with a stable fallback for older artifacts."""
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


def build_metric_rows(
    predictions: pd.DataFrame,
    baseline_config: BaselineConfig,
) -> list[dict[str, object]]:
    """Build wide regression and threshold metric rows for all models and splits."""
    rows: list[dict[str, object]] = []
    for model_name in ordered_model_names(predictions):
        model_rows = predictions.loc[predictions["model_name"] == model_name]
        for split in SPLIT_ORDER:
            split_rows = model_rows.loc[model_rows["split"] == split]
            if split_rows.empty:
                continue
            rows.extend(metric_rows_for_split(split_rows, baseline_config, model_name, split))
    return rows


def ordered_model_names(predictions: pd.DataFrame) -> list[str]:
    """Return model names in report-friendly order with unknown names appended."""
    available = [str(name) for name in predictions["model_name"].dropna().unique()]
    ordered = [name for name in REFERENCE_MODEL_ORDER if name in available]
    ordered.extend(sorted(name for name in available if name not in ordered))
    return ordered


def metric_rows_for_split(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    model_name: str,
    split: str,
) -> list[dict[str, object]]:
    """Build metric rows for overall and label-source groups."""
    alpha = prediction_alpha(dataframe)
    rows = [
        metric_row(
            dataframe,
            baseline_config.target_column,
            model_name,
            split,
            alpha,
            metric_group="overall",
            metric_group_value="all",
            weights=None,
        )
    ]
    weights = metric_weights(dataframe, baseline_config)
    if weights is not None:
        rows.append(
            metric_row(
                dataframe,
                baseline_config.target_column,
                model_name,
                split,
                alpha,
                metric_group="overall",
                metric_group_value="all",
                weights=weights,
            )
        )
    if "label_source" not in dataframe.columns:
        return rows
    for label_source, group in dataframe.groupby("label_source", sort=True):
        rows.append(
            metric_row(
                group,
                baseline_config.target_column,
                model_name,
                split,
                alpha,
                metric_group="label_source",
                metric_group_value=str(label_source),
                weights=None,
            )
        )
        group_weights = metric_weights(group, baseline_config)
        if group_weights is not None:
            rows.append(
                metric_row(
                    group,
                    baseline_config.target_column,
                    model_name,
                    split,
                    alpha,
                    metric_group="label_source",
                    metric_group_value=str(label_source),
                    weights=group_weights,
                )
            )
    return rows


def prediction_alpha(dataframe: pd.DataFrame) -> float:
    """Return the model alpha recorded on prediction rows, if any."""
    if "alpha" not in dataframe.columns:
        return math.nan
    values = pd.to_numeric(dataframe["alpha"], errors="coerce").dropna().unique()
    if len(values) == 0:
        return math.nan
    return float(values[0])


def metric_row(
    dataframe: pd.DataFrame,
    target_column: str,
    model_name: str,
    split: str,
    alpha: float,
    *,
    metric_group: str,
    metric_group_value: str,
    weights: np.ndarray | None,
) -> dict[str, object]:
    """Build one metric row for a model and split."""
    observed = dataframe[target_column].to_numpy(dtype=float)
    predicted = dataframe["pred_kelp_fraction_y"].to_numpy(dtype=float)
    predicted_clipped = dataframe["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = weighted_sum(dataframe["kelp_max_y"].to_numpy(dtype=float), weights)
    predicted_area = weighted_sum(dataframe["pred_kelp_max_y"].to_numpy(dtype=float), weights)
    row: dict[str, object] = {
        "model_name": model_name,
        "split": split,
        "metric_group": metric_group,
        "metric_group_value": metric_group_value,
        "weighted": weights is not None,
        "target": target_column,
        "alpha": alpha,
        "row_count": int(len(dataframe)),
        "mae": mean_absolute_error(observed, predicted, weights=weights),
        "rmse": root_mean_squared_error(observed, predicted, weights=weights),
        "r2": r2_score(observed, predicted, weights=weights),
        "pearson": correlation(observed, predicted, method="pearson"),
        "spearman": correlation(observed, predicted, method="spearman"),
        "observed_mean_fraction": weighted_mean(observed, weights),
        "predicted_mean_fraction": weighted_mean(predicted, weights),
        "predicted_mean_fraction_clipped": weighted_mean(predicted_clipped, weights),
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "area_bias": predicted_area - observed_area,
        "area_pct_bias": percent_bias(predicted_area, observed_area),
    }
    row.update(threshold_metrics(observed, predicted_clipped, weights=weights))
    return row


def threshold_metrics(
    observed: np.ndarray, predicted_clipped: np.ndarray, *, weights: np.ndarray | None
) -> dict[str, float]:
    """Build threshold diagnostics for clipped fraction predictions."""
    metrics: dict[str, float] = {}
    for threshold, label in THRESHOLDS:
        observed_positive = observed >= threshold
        predicted_positive = predicted_clipped >= threshold
        precision, recall, f1 = precision_recall_f1(
            observed_positive, predicted_positive, weights=weights
        )
        metrics[f"observed_positive_rate_ge_{label}"] = safe_bool_rate(
            observed_positive, weights=weights
        )
        metrics[f"predicted_positive_rate_ge_{label}"] = safe_bool_rate(
            predicted_positive, weights=weights
        )
        metrics[f"precision_ge_{label}"] = precision
        metrics[f"recall_ge_{label}"] = recall
        metrics[f"f1_ge_{label}"] = f1
    return metrics


def precision_recall_f1(
    observed_positive: np.ndarray,
    predicted_positive: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 with NaN for undefined ratios."""
    true_positive = weighted_sum_bool(observed_positive & predicted_positive, weights)
    false_positive = weighted_sum_bool(~observed_positive & predicted_positive, weights)
    false_negative = weighted_sum_bool(observed_positive & ~predicted_positive, weights)
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def mean_absolute_error(
    observed: np.ndarray, predicted: np.ndarray, *, weights: np.ndarray | None = None
) -> float:
    """Compute mean absolute error."""
    return weighted_mean(np.abs(observed - predicted), weights)


def root_mean_squared_error(
    observed: np.ndarray, predicted: np.ndarray, *, weights: np.ndarray | None = None
) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(weighted_mean((observed - predicted) ** 2, weights)))


def r2_score(
    observed: np.ndarray, predicted: np.ndarray, *, weights: np.ndarray | None = None
) -> float:
    """Compute R2 or NaN when the observed target is constant."""
    observed_mean = weighted_mean(observed, weights)
    residual_sum_squares = weighted_sum((observed - predicted) ** 2, weights)
    total_sum_squares = weighted_sum((observed - observed_mean) ** 2, weights)
    if total_sum_squares == 0:
        return math.nan
    return 1.0 - residual_sum_squares / total_sum_squares


def correlation(observed: np.ndarray, predicted: np.ndarray, *, method: str) -> float:
    """Compute a pandas correlation coefficient when both vectors vary."""
    if observed.size < 2 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return math.nan
    value = pd.Series(observed).corr(pd.Series(predicted), method=method)
    return float(value) if pd.notna(value) else math.nan


def safe_mean(values: np.ndarray) -> float:
    """Compute a mean or NaN for an empty vector."""
    if values.size == 0:
        return math.nan
    return float(np.nanmean(values))


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute an optionally weighted finite mean."""
    valid_mask = np.isfinite(values)
    if weights is not None:
        valid_mask &= np.isfinite(weights)
    if not np.any(valid_mask):
        return math.nan
    if weights is None:
        return float(np.nanmean(values[valid_mask]))
    return float(np.average(values[valid_mask], weights=weights[valid_mask]))


def weighted_sum(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute an optionally weighted finite sum."""
    valid_mask = np.isfinite(values)
    if weights is not None:
        valid_mask &= np.isfinite(weights)
    if not np.any(valid_mask):
        return 0.0
    if weights is None:
        return float(np.nansum(values[valid_mask]))
    return float(np.nansum(values[valid_mask] * weights[valid_mask]))


def percent_bias(predicted_area: float, observed_area: float) -> float:
    """Compute area percent bias or NaN when observed area is zero."""
    if observed_area == 0:
        return math.nan
    return (predicted_area - observed_area) / observed_area


def safe_bool_rate(values: np.ndarray, *, weights: np.ndarray | None = None) -> float:
    """Compute the mean of a boolean vector or NaN for an empty vector."""
    if values.size == 0:
        return math.nan
    if weights is None:
        return float(np.mean(values))
    return safe_ratio(weighted_sum_bool(values, weights), float(np.nansum(weights)))


def weighted_sum_bool(values: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute an optionally weighted boolean count."""
    if weights is None:
        return float(np.count_nonzero(values))
    return float(np.nansum(weights[values]))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a ratio or NaN for a zero denominator."""
    if denominator == 0:
        return math.nan
    return numerator / denominator


def write_split_manifest(split_manifest: pd.DataFrame, output_path: Path) -> None:
    """Write the row-level split manifest to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_parquet(output_path, index=False)


def write_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Write baseline predictions to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)


def write_prediction_part(predictions: pd.DataFrame, output_path: Path, part_index: int) -> None:
    """Write one streamed prediction part to a Parquet dataset directory."""
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path / f"part-{part_index:05d}.parquet", index=False)


def write_metrics(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write model metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_ridge_model(
    ridge_selection: RidgeSelection,
    baseline_config: BaselineConfig,
    *,
    train_mean: float,
) -> None:
    """Write the selected ridge model and compact metadata with joblib."""
    baseline_config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": ridge_selection.model,
        "model_name": RIDGE_MODEL_NAME,
        "sample_policy": baseline_config.sample_policy,
        "selected_alpha": ridge_selection.selected_alpha,
        "target_column": baseline_config.target_column,
        "feature_columns": list(baseline_config.feature_columns),
        "train_mean": train_mean,
    }
    joblib.dump(payload, baseline_config.model_output_path)


def write_geographic_model(
    geographic_selection: RidgeSelection,
    baseline_config: BaselineConfig,
) -> None:
    """Write the selected lat/lon/year geographic ridge model."""
    baseline_config.geographic_model_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": geographic_selection.model,
        "model_name": GEOGRAPHIC_MODEL_NAME,
        "sample_policy": baseline_config.sample_policy,
        "selected_alpha": geographic_selection.selected_alpha,
        "target_column": baseline_config.target_column,
        "feature_columns": list(GEOGRAPHIC_FEATURE_COLUMNS),
    }
    joblib.dump(payload, baseline_config.geographic_model_output_path)


def write_fallback_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write reference-baseline fallback counts to CSV."""
    rows = aggregate_fallback_summary_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FALLBACK_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(rows: list[dict[str, object]], output_path: Path, fields: tuple[str, ...]) -> None:
    """Write generic CSV rows with a stable schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_fallback_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate fallback-summary rows that share the same dimensions."""
    totals: dict[tuple[object, ...], int] = {}
    for row in rows:
        key = tuple(row[field] for field in FALLBACK_SUMMARY_FIELDS if field != "row_count")
        totals[key] = totals.get(key, 0) + int(cast(Any, row["row_count"]))
    output: list[dict[str, object]] = []
    key_fields = tuple(field for field in FALLBACK_SUMMARY_FIELDS if field != "row_count")
    sorted_totals = sorted(
        totals.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    )
    for key, row_count in sorted_totals:
        output.append({field: value for field, value in zip(key_fields, key, strict=True)})
        output[-1]["row_count"] = row_count
    return output


def write_reference_area_calibration(config_path: Path) -> list[dict[str, object]]:
    """Write compact full-grid area calibration rows for cached baseline models."""
    baseline_config = load_baseline_config(config_path)
    rows = build_reference_area_calibration_rows(baseline_config)
    write_csv(rows, baseline_config.area_calibration_path, AREA_CALIBRATION_FIELDS)
    LOGGER.info("Wrote reference area calibration: %s", baseline_config.area_calibration_path)
    return rows


def build_reference_area_calibration_rows(
    baseline_config: BaselineConfig,
) -> list[dict[str, object]]:
    """Build compact full-grid area summaries without row-level prediction output."""
    if baseline_config.inference_table_path is None:
        return []
    if not baseline_config.model_output_path.exists():
        return []
    inference = pl.scan_parquet(str(baseline_config.inference_table_path))
    inference = filter_polars_to_reporting_domain(
        inference,
        baseline_config.reporting_domain_mask,
    )
    schema_names = tuple(inference.collect_schema().names())
    key_columns = cell_key_columns_from_names(schema_names)
    base = full_grid_area_base_frame(inference, baseline_config, schema_names)
    ridge_payload = load_model_payload(baseline_config.model_output_path)
    model_frames = [
        area_model_frame(
            base,
            NO_SKILL_MODEL_NAME,
            pl.lit(float(ridge_payload.get("train_mean", math.nan))),
            baseline_config,
        ),
        area_model_frame(
            base,
            RIDGE_MODEL_NAME,
            linear_model_prediction_expr(ridge_payload, baseline_config.feature_columns),
            baseline_config,
        ),
        previous_year_area_frame(base, baseline_config, key_columns),
        climatology_area_frame(base, baseline_config, key_columns, ridge_payload),
    ]
    if baseline_config.geographic_model_output_path.exists():
        geographic_payload = load_model_payload(baseline_config.geographic_model_output_path)
        model_frames.append(
            area_model_frame(
                base,
                GEOGRAPHIC_MODEL_NAME,
                linear_model_prediction_expr(geographic_payload, GEOGRAPHIC_FEATURE_COLUMNS),
                baseline_config,
            )
        )
    summaries = [summarize_area_model_frame(frame, baseline_config) for frame in model_frames]
    if not summaries:
        return []
    summary = pl.concat(summaries, how="vertical").collect()
    return finalized_area_rows(summary, baseline_config)


def full_grid_area_base_frame(
    inference: pl.LazyFrame,
    baseline_config: BaselineConfig,
    schema_names: tuple[str, ...],
) -> pl.LazyFrame:
    """Return the shared lazy full-grid frame used for area calibration."""
    expressions = [
        split_expression(baseline_config),
        label_source_expression(schema_names),
        pl.col(baseline_config.target_column).cast(pl.Float64),
        pl.col("kelp_max_y").cast(pl.Float64),
    ]
    return inference.with_columns(expressions)


def split_expression(baseline_config: BaselineConfig) -> pl.Expr:
    """Build the configured split assignment expression for Polars scans."""
    return (
        pl.when(pl.col("year").is_in(list(baseline_config.train_years)))
        .then(pl.lit("train"))
        .when(pl.col("year").is_in(list(baseline_config.validation_years)))
        .then(pl.lit("validation"))
        .when(pl.col("year").is_in(list(baseline_config.test_years)))
        .then(pl.lit("test"))
        .otherwise(pl.lit("unassigned"))
        .alias("split")
    )


def label_source_expression(schema_names: tuple[str, ...]) -> pl.Expr:
    """Build a label-source expression with fallbacks for older artifacts."""
    columns = set(schema_names)
    if "label_source" in columns:
        return pl.col("label_source").fill_null("unknown").cast(pl.String).alias("label_source")
    if "is_kelpwatch_observed" in columns:
        return (
            pl.when(pl.col("is_kelpwatch_observed").fill_null(False))
            .then(pl.lit("kelpwatch_station"))
            .otherwise(pl.lit("assumed_background"))
            .alias("label_source")
        )
    return pl.lit("all").alias("label_source")


def linear_model_prediction_expr(
    payload: dict[str, Any], feature_columns: tuple[str, ...]
) -> pl.Expr:
    """Build a Polars expression for a fitted StandardScaler plus Ridge pipeline."""
    model = payload["model"]
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    coefficients = np.asarray(ridge.coef_, dtype=float)
    means = np.asarray(scaler.mean_, dtype=float)
    scales = np.asarray(scaler.scale_, dtype=float)
    expression = pl.lit(float(ridge.intercept_))
    for column, coefficient, mean, scale in zip(
        feature_columns,
        coefficients,
        means,
        scales,
        strict=True,
    ):
        divisor = float(scale) if float(scale) != 0.0 else 1.0
        expression = expression + (
            ((pl.col(column).cast(pl.Float64) - float(mean)) / divisor) * float(coefficient)
        )
    return expression


def area_model_frame(
    base: pl.LazyFrame,
    model_name: str,
    prediction_expression: pl.Expr,
    baseline_config: BaselineConfig,
) -> pl.LazyFrame:
    """Attach one model's prediction columns to the full-grid calibration frame."""
    return (
        base.with_columns(
            pl.lit(model_name).alias("model_name"),
            prediction_expression.alias("pred_kelp_fraction_y"),
        )
        .filter(pl.col("pred_kelp_fraction_y").is_not_null())
        .with_columns(
            pl.col("pred_kelp_fraction_y").clip(0.0, 1.0).alias("pred_kelp_fraction_y_clipped")
        )
        .with_columns(
            (pl.col("pred_kelp_fraction_y_clipped") * pl.lit(KELPWATCH_PIXEL_AREA_M2)).alias(
                "pred_kelp_max_y"
            ),
            (pl.col(baseline_config.target_column) - pl.col("pred_kelp_fraction_y_clipped")).alias(
                "residual_kelp_fraction_y"
            ),
        )
    )


def previous_year_area_frame(
    base: pl.LazyFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
) -> pl.LazyFrame:
    """Build previous-year full-grid prediction rows lazily for aggregation only."""
    previous = (
        base.group_by([*key_columns, "year"])
        .agg(pl.mean(baseline_config.target_column).alias("previous_prediction"))
        .with_columns((pl.col("year") + 1).alias("year"))
    )
    joined = base.join(previous, on=[*key_columns, "year"], how="left")
    return area_model_frame(
        joined,
        PREVIOUS_YEAR_MODEL_NAME,
        pl.col("previous_prediction"),
        baseline_config,
    )


def climatology_area_frame(
    base: pl.LazyFrame,
    baseline_config: BaselineConfig,
    key_columns: tuple[str, ...],
    ridge_payload: dict[str, Any],
) -> pl.LazyFrame:
    """Build grid-cell climatology prediction rows lazily for aggregation only."""
    train = base.filter(pl.col("year").is_in(list(baseline_config.train_years)))
    totals = train.group_by(list(key_columns)).agg(
        pl.sum(baseline_config.target_column).alias("training_sum"),
        pl.len().alias("training_count"),
    )
    year_totals = train.group_by([*key_columns, "year"]).agg(
        pl.sum(baseline_config.target_column).alias("year_sum"),
        pl.len().alias("year_count"),
    )
    label_source_means = train.group_by("label_source").agg(
        pl.mean(baseline_config.target_column).alias("label_source_training_mean")
    )
    joined = (
        base.join(totals, on=list(key_columns), how="left")
        .join(year_totals, on=[*key_columns, "year"], how="left")
        .join(label_source_means, on="label_source", how="left")
    )
    leave_one_out = (pl.col("training_sum") - pl.col("year_sum")) / (
        pl.col("training_count") - pl.col("year_count")
    )
    held_out_mean = pl.col("training_sum") / pl.col("training_count")
    prediction = pl.coalesce(
        [
            pl.when(pl.col("year").is_in(list(baseline_config.train_years)))
            .then(
                pl.when((pl.col("training_count") - pl.col("year_count")) > 0)
                .then(leave_one_out)
                .otherwise(None)
            )
            .otherwise(held_out_mean),
            pl.col("label_source_training_mean"),
            pl.lit(float(ridge_payload.get("train_mean", math.nan))),
        ]
    )
    return area_model_frame(joined, CLIMATOLOGY_MODEL_NAME, prediction, baseline_config)


def summarize_area_model_frame(
    dataframe: pl.LazyFrame,
    baseline_config: BaselineConfig,
) -> pl.LazyFrame:
    """Aggregate one lazy model frame to compact area-calibration rows."""
    by_source = dataframe.group_by(["model_name", "split", "year", "label_source"]).agg(
        area_summary_aggregations(baseline_config)
    )
    by_all_sources = (
        dataframe.with_columns(pl.lit("all").alias("label_source"))
        .group_by(["model_name", "split", "year", "label_source"])
        .agg(area_summary_aggregations(baseline_config))
    )
    return pl.concat([by_source, by_all_sources], how="vertical")


def area_summary_aggregations(baseline_config: BaselineConfig) -> list[pl.Expr]:
    """Return shared Polars aggregations for full-grid area-calibration rows."""
    observed_positive = pl.col(baseline_config.target_column) >= 0.10
    predicted_positive = pl.col("pred_kelp_fraction_y_clipped") >= 0.10
    return [
        pl.len().alias("row_count"),
        pl.sum("kelp_max_y").alias("observed_canopy_area"),
        pl.sum("pred_kelp_max_y").alias("predicted_canopy_area"),
        pl.col("residual_kelp_fraction_y").abs().sum().alias("absolute_error_sum"),
        (pl.col("residual_kelp_fraction_y") ** 2).sum().alias("squared_error_sum"),
        pl.sum(baseline_config.target_column).alias("observed_fraction_sum"),
        (pl.col(baseline_config.target_column) ** 2).sum().alias("observed_fraction_sq_sum"),
        (observed_positive & predicted_positive).cast(pl.Int64).sum().alias("true_positive"),
        (~observed_positive & predicted_positive).cast(pl.Int64).sum().alias("false_positive"),
        (observed_positive & ~predicted_positive).cast(pl.Int64).sum().alias("false_negative"),
    ]


def finalized_area_rows(
    summary: pl.DataFrame, baseline_config: BaselineConfig
) -> list[dict[str, object]]:
    """Convert aggregate Polars rows into the stable area-calibration schema."""
    rows: list[dict[str, object]] = []
    for row in summary.to_dicts():
        row_count = int(cast(Any, row["row_count"]))
        observed_area = float(cast(Any, row["observed_canopy_area"]))
        predicted_area = float(cast(Any, row["predicted_canopy_area"]))
        squared_error = float(cast(Any, row["squared_error_sum"]))
        observed_sum = float(cast(Any, row["observed_fraction_sum"]))
        observed_sq_sum = float(cast(Any, row["observed_fraction_sq_sum"]))
        total_sum_squares = observed_sq_sum - (observed_sum**2 / row_count)
        true_positive = float(cast(Any, row["true_positive"]))
        false_positive = float(cast(Any, row["false_positive"]))
        false_negative = float(cast(Any, row["false_negative"]))
        rows.append(
            {
                "model_name": row["model_name"],
                "split": row["split"],
                "year": int(cast(Any, row["year"])),
                "mask_status": mask_status(baseline_config.reporting_domain_mask),
                "evaluation_scope": evaluation_scope(baseline_config.reporting_domain_mask),
                "label_source": row["label_source"],
                "row_count": row_count,
                "observed_canopy_area": observed_area,
                "predicted_canopy_area": predicted_area,
                "area_bias": predicted_area - observed_area,
                "area_pct_bias": percent_bias(predicted_area, observed_area),
                "mae": safe_ratio(float(cast(Any, row["absolute_error_sum"])), row_count),
                "rmse": math.sqrt(safe_ratio(squared_error, row_count)),
                "r2": (
                    math.nan if total_sum_squares == 0 else 1.0 - squared_error / total_sum_squares
                ),
                "f1_ge_10pct": safe_ratio(
                    2.0 * true_positive,
                    2.0 * true_positive + false_positive + false_negative,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            str(item["model_name"]),
            str(item["split"]),
            int(cast(Any, item["year"])),
            str(item["label_source"]),
        ),
    )


def write_eval_manifest(
    *,
    baseline_config: BaselineConfig,
    prepared: PreparedData,
    ridge_selection: RidgeSelection,
    geographic_selection: RidgeSelection,
    train_mean: float,
    predictions: pd.DataFrame,
    fallback_rows: list[dict[str, object]],
) -> None:
    """Write a JSON manifest for the baseline training run."""
    payload = {
        "command": "train-baselines",
        "config_path": str(baseline_config.config_path),
        "sample_policy": baseline_config.sample_policy,
        "aligned_table": str(baseline_config.aligned_table_path),
        "split_manifest": str(baseline_config.split_manifest_path),
        "model_output": str(baseline_config.model_output_path),
        "sample_predictions": str(baseline_config.sample_predictions_path),
        "full_grid_predictions": str(baseline_config.predictions_path),
        "inference_table": (
            str(baseline_config.inference_table_path)
            if baseline_config.inference_table_path is not None
            else None
        ),
        "metrics": str(baseline_config.metrics_path),
        "manifest": str(baseline_config.eval_manifest_path),
        "geographic_model": str(baseline_config.geographic_model_output_path),
        "fallback_summary": str(baseline_config.fallback_summary_path),
        "reference_area_calibration": str(baseline_config.area_calibration_path),
        "target_column": baseline_config.target_column,
        "feature_columns": list(baseline_config.feature_columns),
        "geographic_feature_columns": list(GEOGRAPHIC_FEATURE_COLUMNS),
        "train_years": list(baseline_config.train_years),
        "validation_years": list(baseline_config.validation_years),
        "test_years": list(baseline_config.test_years),
        "alpha_grid": list(baseline_config.alpha_grid),
        "selected_alpha": ridge_selection.selected_alpha,
        "geographic_selected_alpha": geographic_selection.selected_alpha,
        "train_mean": train_mean,
        "dropped_counts_by_split": prepared.dropped_counts_by_split,
        "retained_row_count": int(len(prepared.retained_rows)),
        "prediction_row_count": int(len(predictions)),
        "validation_alpha_metrics": ridge_selection.validation_rows,
        "geographic_validation_alpha_metrics": geographic_selection.validation_rows,
        "prediction_models": sorted(predictions["model_name"].unique().tolist()),
        "fallback_summary_row_count": len(fallback_rows),
        "use_sample_weight": baseline_config.use_sample_weight,
        "sample_weight_column": baseline_config.sample_weight_column,
    }
    write_json(baseline_config.eval_manifest_path, payload)


def load_model_payload(path: Path) -> dict[str, Any]:
    """Load a trained baseline model payload from disk."""
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        msg = f"model payload is not a dictionary: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def model_weights(dataframe: pd.DataFrame, baseline_config: BaselineConfig) -> np.ndarray | None:
    """Return sample weights for model fitting when configured and available."""
    if not baseline_config.use_sample_weight:
        return None
    if baseline_config.sample_weight_column not in dataframe.columns:
        return None
    return cast(np.ndarray, dataframe[baseline_config.sample_weight_column].to_numpy(dtype=float))


def metric_weights(dataframe: pd.DataFrame, baseline_config: BaselineConfig) -> np.ndarray | None:
    """Return sample weights for metric reporting when available."""
    if not baseline_config.use_sample_weight:
        return None
    if baseline_config.sample_weight_column not in dataframe.columns:
        return None
    return cast(np.ndarray, dataframe[baseline_config.sample_weight_column].to_numpy(dtype=float))


def prediction_input_columns(baseline_config: BaselineConfig) -> list[str]:
    """Return columns needed for streamed full-grid prediction."""
    columns = [
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        baseline_config.target_column,
        "kelp_max_y",
        *baseline_config.feature_columns,
    ]
    for column in OPTIONAL_PROVENANCE_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def read_reference_source_table(path: Path, baseline_config: BaselineConfig) -> pd.DataFrame:
    """Read minimal full-grid columns needed for reference-baseline lookup tables."""
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    available_columns = tuple(dataset.schema.names)
    key_columns = cell_key_columns_from_names(available_columns)
    columns = [*key_columns, "year", baseline_config.target_column]
    if "label_source" in available_columns:
        columns.append("label_source")
    LOGGER.info("Loading reference-baseline lookup columns from %s", path)
    return cast(pd.DataFrame, pd.read_parquet(path, columns=columns))


def predict_with_missing_features(
    dataframe: pd.DataFrame, baseline_config: BaselineConfig, model: Any
) -> np.ndarray:
    """Predict rows with complete features and leave missing-feature rows as NaN."""
    feature_complete = dataframe.loc[:, list(baseline_config.feature_columns)].notna().all(axis=1)
    predictions = np.full(len(dataframe), np.nan, dtype=float)
    if feature_complete.any():
        complete_rows = dataframe.loc[feature_complete]
        predictions[feature_complete.to_numpy(dtype=bool)] = np.asarray(
            model.predict(feature_matrix(complete_rows, baseline_config.feature_columns)),
            dtype=float,
        )
    return predictions


def iter_parquet_batches(path: Path, columns: list[str], batch_size: int) -> Any:
    """Yield pandas DataFrames from a Parquet dataset in row batches."""
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    available_columns = set(dataset.schema.names)
    selected_columns = [column for column in columns if column in available_columns]
    for batch in dataset.to_batches(columns=selected_columns, batch_size=batch_size):
        yield batch.to_pandas()


def reset_output_path(path: Path) -> None:
    """Remove and recreate a Parquet dataset directory."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def suffix_path(path: Path, suffix: str) -> Path:
    """Insert a suffix before a path's final extension."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def write_prediction_manifest(
    *,
    baseline_config: BaselineConfig,
    inference_path: Path,
    output_path: Path,
    manifest_path: Path,
    row_count: int,
    part_count: int,
    label_source_counts: dict[str, int],
    fast: bool,
) -> None:
    """Write a manifest for streamed full-grid prediction."""
    payload = {
        "command": "predict-full-grid",
        "config_path": str(baseline_config.config_path),
        "sample_policy": baseline_config.sample_policy,
        "fast": fast,
        "inference_table": str(inference_path),
        "predictions": str(output_path),
        "model_output": str(baseline_config.model_output_path),
        "row_count": row_count,
        "part_count": part_count,
        "label_source_counts": label_source_counts,
        "prediction_models": [RIDGE_MODEL_NAME],
        "feature_columns": list(baseline_config.feature_columns),
        "target_column": baseline_config.target_column,
    }
    write_json(manifest_path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like dynamic value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))


def require_positive_float(value: object, name: str) -> float:
    """Validate a positive numeric dynamic value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"field must be numeric: {name}"
        raise ValueError(msg)
    parsed = float(value)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed
