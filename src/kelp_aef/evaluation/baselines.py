"""Train and evaluate first tabular baseline models."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

KELPWATCH_PIXEL_AREA_M2 = 900.0
NO_SKILL_MODEL_NAME = "no_skill_train_mean"
RIDGE_MODEL_NAME = "ridge_regression"
SPLIT_ORDER = ("train", "validation", "test")
THRESHOLDS = ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"))
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
OPTIONAL_PROVENANCE_COLUMNS = (
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "label_source",
    "is_kelpwatch_observed",
    "kelpwatch_station_count",
    "sample_weight",
)
FULL_GRID_PREDICTION_BATCH_SIZE = 100_000


@dataclass(frozen=True)
class BaselineConfig:
    """Resolved config values for baseline training and evaluation."""

    config_path: Path
    aligned_table_path: Path
    split_manifest_path: Path
    model_output_path: Path
    sample_predictions_path: Path
    predictions_path: Path
    prediction_manifest_path: Path
    inference_table_path: Path | None
    metrics_path: Path
    eval_manifest_path: Path
    target_column: str
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    drop_missing_features: bool
    use_sample_weight: bool
    sample_weight_column: str


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


def train_baselines(config_path: Path) -> int:
    """Train and evaluate the first no-skill and ridge baselines."""
    baseline_config = load_baseline_config(config_path)
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
    predictions = build_prediction_rows(
        prepared.retained_rows,
        baseline_config,
        train_mean=train_mean,
        ridge_model=ridge_selection.model,
        selected_alpha=ridge_selection.selected_alpha,
    )
    metrics = build_metric_rows(predictions, baseline_config, ridge_selection.selected_alpha)
    write_predictions(predictions, baseline_config.sample_predictions_path)
    write_metrics(metrics, baseline_config.metrics_path)
    write_ridge_model(
        ridge_selection,
        baseline_config,
        train_mean=train_mean,
    )
    write_eval_manifest(
        baseline_config=baseline_config,
        prepared=prepared,
        ridge_selection=ridge_selection,
        train_mean=train_mean,
        predictions=predictions,
    )
    LOGGER.info("Wrote split manifest: %s", baseline_config.split_manifest_path)
    LOGGER.info("Wrote baseline sample predictions: %s", baseline_config.sample_predictions_path)
    LOGGER.info("Wrote baseline metrics: %s", baseline_config.metrics_path)
    LOGGER.info("Wrote ridge model: %s", baseline_config.model_output_path)
    LOGGER.info("Wrote baseline evaluation manifest: %s", baseline_config.eval_manifest_path)
    return 0


def predict_full_grid(config_path: Path, *, fast: bool = False) -> int:
    """Apply the trained ridge model to the full-grid inference table in chunks."""
    baseline_config = load_baseline_config(config_path)
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
    return 0


def load_baseline_config(config_path: Path) -> BaselineConfig:
    """Load baseline training settings from the workflow config."""
    config = load_yaml_config(config_path)
    alignment = require_mapping(config.get("alignment"), "alignment")
    splits = require_mapping(config.get("splits"), "splits")
    features = require_mapping(config.get("features"), "features")
    models = require_mapping(config.get("models"), "models")
    baseline = require_mapping(models.get("baselines"), "models.baselines")
    predictions_path = Path(
        require_string(baseline.get("predictions"), "models.baselines.predictions")
    )
    sample_predictions_value = baseline.get("sample_predictions")
    inference_table_value = baseline.get("inference_table")
    prediction_manifest_value = baseline.get("prediction_manifest")
    return BaselineConfig(
        config_path=config_path,
        aligned_table_path=Path(
            require_string(
                baseline.get("input_table") or alignment.get("output_table"),
                "models.baselines.input_table or alignment.output_table",
            )
        ),
        split_manifest_path=Path(
            require_string(splits.get("output_manifest"), "splits.output_manifest")
        ),
        model_output_path=Path(
            require_string(baseline.get("ridge_model"), "models.baselines.ridge_model")
        ),
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
    )


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
) -> pd.DataFrame:
    """Build prediction rows for no-skill and selected ridge models."""
    x_all = feature_matrix(retained_rows, baseline_config.feature_columns)
    no_skill_predictions = np.full(len(retained_rows), train_mean, dtype=float)
    ridge_predictions = np.asarray(ridge_model.predict(x_all), dtype=float)
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


def build_metric_rows(
    predictions: pd.DataFrame,
    baseline_config: BaselineConfig,
    selected_alpha: float,
) -> list[dict[str, object]]:
    """Build wide regression and threshold metric rows for all models and splits."""
    rows: list[dict[str, object]] = []
    for model_name in (NO_SKILL_MODEL_NAME, RIDGE_MODEL_NAME):
        model_rows = predictions.loc[predictions["model_name"] == model_name]
        for split in SPLIT_ORDER:
            split_rows = model_rows.loc[model_rows["split"] == split]
            if split_rows.empty:
                continue
            alpha = selected_alpha if model_name == RIDGE_MODEL_NAME else math.nan
            rows.extend(
                metric_rows_for_split(split_rows, baseline_config, model_name, split, alpha)
            )
    return rows


def metric_rows_for_split(
    dataframe: pd.DataFrame,
    baseline_config: BaselineConfig,
    model_name: str,
    split: str,
    alpha: float,
) -> list[dict[str, object]]:
    """Build metric rows for overall and label-source groups."""
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
        "selected_alpha": ridge_selection.selected_alpha,
        "target_column": baseline_config.target_column,
        "feature_columns": list(baseline_config.feature_columns),
        "train_mean": train_mean,
    }
    joblib.dump(payload, baseline_config.model_output_path)


def write_eval_manifest(
    *,
    baseline_config: BaselineConfig,
    prepared: PreparedData,
    ridge_selection: RidgeSelection,
    train_mean: float,
    predictions: pd.DataFrame,
) -> None:
    """Write a JSON manifest for the baseline training run."""
    payload = {
        "command": "train-baselines",
        "config_path": str(baseline_config.config_path),
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
        "target_column": baseline_config.target_column,
        "feature_columns": list(baseline_config.feature_columns),
        "train_years": list(baseline_config.train_years),
        "validation_years": list(baseline_config.validation_years),
        "test_years": list(baseline_config.test_years),
        "alpha_grid": list(baseline_config.alpha_grid),
        "selected_alpha": ridge_selection.selected_alpha,
        "train_mean": train_mean,
        "dropped_counts_by_split": prepared.dropped_counts_by_split,
        "retained_row_count": int(len(prepared.retained_rows)),
        "prediction_row_count": int(len(predictions)),
        "validation_alpha_metrics": ridge_selection.validation_rows,
        "prediction_models": sorted(predictions["model_name"].unique().tolist()),
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
        "fast": fast,
        "inference_table": str(inference_path),
        "predictions": str(output_path),
        "model_output": str(baseline_config.model_output_path),
        "row_count": row_count,
        "part_count": part_count,
        "label_source_counts": label_source_counts,
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
