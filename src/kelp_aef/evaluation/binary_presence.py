"""Train and evaluate the Phase 1 balanced binary annual-max model."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
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


@dataclass(frozen=True)
class BinaryPresenceConfig:
    """Resolved config values for binary presence model training."""

    config_path: Path
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
    reporting_domain_mask: ReportingDomainMask | None


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


def train_binary_presence(config_path: Path) -> int:
    """Train the balanced binary annual-max model and write configured artifacts."""
    binary_config = load_binary_presence_config(config_path)
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
        reporting_domain_mask=reporting_domain_mask,
    )


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
        msg = f"split manifest is missing {missing_count} binary model rows"
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
