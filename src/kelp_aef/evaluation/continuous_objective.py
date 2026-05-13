"""Train direct continuous objective sidecar models for Phase 1 experiments."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import joblib  # type: ignore[import-untyped]
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
    KELPWATCH_PIXEL_AREA_M2,
    fit_ridge_model,
    iter_parquet_batches,
    make_ridge_pipeline,
    parse_bands,
    percent_bias,
    precision_recall_f1,
    reset_output_path,
    root_mean_squared_error,
    safe_ratio,
    write_prediction_part,
)

LOGGER = logging.getLogger(__name__)

MODEL_FAMILY = "continuous_objective"
DEFAULT_EXPERIMENT = "capped-weight"
DEFAULT_MODEL_NAME = "ridge_capped_weight"
DEFAULT_OBJECTIVE_POLICY = "capped_weighted_ridge"
DEFAULT_FIT_WEIGHT_POLICY = "capped_assumed_background_sample_weight"
STRATIFIED_FIT_WEIGHT_POLICY = "stratified_assumed_background_sample_weight"
DEFAULT_TARGET_COLUMN = "kelp_fraction_y"
DEFAULT_TARGET_AREA_COLUMN = "kelp_max_y"
DEFAULT_SAMPLE_WEIGHT_COLUMN = "sample_weight"
DEFAULT_BATCH_SIZE = 100_000
DEFAULT_FIT_WEIGHT_CAP = 5.0
DEFAULT_THRESHOLD_FRACTION = 0.10
DEFAULT_BACKGROUND_STRATUM_COLUMNS = (
    "year",
    "label_source",
    "domain_mask_reason",
    "depth_bin",
)
DEFAULT_STRATUM_BALANCE_GAMMA = 1.0
SUPPORTED_FIT_WEIGHT_POLICIES = (
    DEFAULT_FIT_WEIGHT_POLICY,
    STRATIFIED_FIT_WEIGHT_POLICY,
)
SPLIT_ORDER = ("train", "validation", "test")
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
PREDICTION_FIELDS = (
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
    "fit_weight_stratum",
    "fit_weight",
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
    "sample_policy",
    "model_name",
    "model_family",
    "objective_policy",
    "fit_weight_policy",
    "fit_weight_cap",
    "stratum_balance_gamma",
    "background_weight_budget_multiplier",
    "stratum_columns",
    "target",
    "target_area_column",
    "cell_area_m2",
    "alpha",
    "pred_kelp_fraction_y",
    "pred_kelp_fraction_y_clipped",
    "pred_kelp_max_y",
    "residual_kelp_fraction_y",
    "residual_kelp_fraction_y_clipped",
    "residual_kelp_max_y",
)
METRIC_FIELDS = (
    "model_name",
    "model_family",
    "objective_policy",
    "fit_weight_policy",
    "fit_weight_cap",
    "stratum_balance_gamma",
    "background_weight_budget_multiplier",
    "split",
    "year",
    "label_source",
    "mask_status",
    "evaluation_scope",
    "row_count",
    "weighted",
    "target",
    "alpha",
    "mae",
    "rmse",
    "r2",
    "spearman",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "precision_ge_10pct",
    "recall_ge_10pct",
    "f1_ge_10pct",
    "observed_positive_rate_ge_10pct",
    "predicted_positive_rate_ge_10pct",
    "fit_weight_min",
    "fit_weight_max",
    "fit_weight_mean",
)
AREA_CALIBRATION_FIELDS = (
    "model_name",
    "model_family",
    "objective_policy",
    "fit_weight_policy",
    "fit_weight_cap",
    "stratum_balance_gamma",
    "background_weight_budget_multiplier",
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
LEAKAGE_FIELDS = (
    "model_name",
    "model_family",
    "objective_policy",
    "fit_weight_policy",
    "fit_weight_cap",
    "stratum_balance_gamma",
    "background_weight_budget_multiplier",
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
)

MetricKey = tuple[str, int, str]


@dataclass(frozen=True)
class ContinuousObjectiveConfig:
    """Resolved config values for one direct continuous-objective experiment."""

    config_path: Path
    experiment: str
    sample_policy: str
    input_table_path: Path
    inference_table_path: Path
    model_path: Path
    sample_predictions_path: Path
    full_grid_predictions_path: Path
    metrics_path: Path
    area_calibration_path: Path
    assumed_background_leakage_path: Path
    manifest_path: Path
    model_name: str
    objective_policy: str
    fit_weight_policy: str
    fit_weight_cap: float | None
    stratum_balance_gamma: float
    background_weight_budget_multiplier: float | None
    stratum_columns: tuple[str, ...]
    sample_weight_column: str
    target_column: str
    target_area_column: str
    cell_area_m2: float
    feature_columns: tuple[str, ...]
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    alpha_grid: tuple[float, ...]
    batch_size: int
    drop_missing_features: bool
    reporting_domain_mask: ReportingDomainMask | None


@dataclass(frozen=True)
class ContinuousObjectiveSelection:
    """Selected ridge model and validation-only alpha diagnostics."""

    model: Any
    selected_alpha: float
    validation_rows: list[dict[str, object]]


def train_continuous_objective(config_path: Path, *, experiment: str) -> int:
    """Train and evaluate one configured direct continuous-objective experiment."""
    objective_config = load_continuous_objective_config(config_path, experiment)
    train_continuous_objective_config(objective_config)
    return 0


def train_continuous_objective_config(objective_config: ContinuousObjectiveConfig) -> None:
    """Fit one continuous-objective model and write predictions and diagnostics."""
    LOGGER.info("Loading continuous-objective sample: %s", objective_config.input_table_path)
    sample = cast(pd.DataFrame, pd.read_parquet(objective_config.input_table_path))
    validate_sample_table(sample, objective_config)
    retained, dropped_counts = prepare_sample_frame(sample, objective_config)
    retained["fit_weight_stratum"] = fit_weight_strata(retained, objective_config)
    retained["fit_weight"] = fit_weights(retained, objective_config)
    train_rows = rows_for_split(retained, "train")
    validation_rows = rows_for_split(retained, "validation")
    selection = fit_select_weighted_ridge(train_rows, validation_rows, objective_config)
    sample_predictions = build_prediction_frame(
        retained,
        objective_config,
        predictions=np.asarray(
            selection.model.predict(feature_matrix(retained, objective_config.feature_columns)),
            dtype=float,
        ),
        alpha=selection.selected_alpha,
        evaluation_scope_value="model_input_sample",
    )
    sample_metric_rows = metric_rows_from_predictions(sample_predictions, objective_config)
    write_parquet(sample_predictions, objective_config.sample_predictions_path)
    write_csv_rows(sample_metric_rows, objective_config.metrics_path, METRIC_FIELDS)
    write_model(selection, objective_config)

    full_grid_rows, part_count, full_grid_dropped, full_grid_metrics = write_full_grid_predictions(
        selection,
        objective_config,
    )
    area_rows = area_calibration_rows(full_grid_metrics, objective_config)
    leakage_rows = assumed_background_leakage_rows(full_grid_metrics, objective_config)
    write_csv_rows(area_rows, objective_config.area_calibration_path, AREA_CALIBRATION_FIELDS)
    write_csv_rows(
        leakage_rows,
        objective_config.assumed_background_leakage_path,
        LEAKAGE_FIELDS,
    )
    write_manifest(
        objective_config=objective_config,
        selection=selection,
        sample_rows=len(retained),
        sample_dropped_counts=dropped_counts,
        sample_metric_rows=sample_metric_rows,
        full_grid_rows=full_grid_rows,
        full_grid_part_count=part_count,
        full_grid_dropped_missing_feature_rows=full_grid_dropped,
        area_rows=area_rows,
        leakage_rows=leakage_rows,
    )
    LOGGER.info(
        "Wrote continuous-objective model and predictions for experiment %s",
        objective_config.experiment,
    )


def load_continuous_objective_config(
    config_path: Path,
    experiment: str,
) -> ContinuousObjectiveConfig:
    """Load one configured direct continuous-objective experiment."""
    config = load_yaml_config(config_path)
    features = require_mapping(config.get("features"), "features")
    splits = require_mapping(config.get("splits"), "splits")
    models = require_mapping(config.get("models"), "models")
    baselines = optional_mapping(models.get("baselines"), "models.baselines")
    objective = require_mapping(
        models.get("continuous_objective"),
        "models.continuous_objective",
    )
    experiments = require_mapping(
        objective.get("experiments"),
        "models.continuous_objective.experiments",
    )
    experiment_block = require_mapping(
        experiments.get(experiment),
        f"models.continuous_objective.experiments.{experiment}",
    )
    reporting_domain_mask = load_reporting_domain_mask(config)
    fit_weight_policy = str(
        experiment_block.get("fit_weight_policy")
        or objective.get("fit_weight_policy")
        or DEFAULT_FIT_WEIGHT_POLICY
    )
    validate_fit_weight_policy(fit_weight_policy)
    fit_weight_cap_default = (
        DEFAULT_FIT_WEIGHT_CAP if fit_weight_policy == DEFAULT_FIT_WEIGHT_POLICY else None
    )
    stratum_columns = read_stratum_columns(
        experiment_block.get("stratum_columns") or objective.get("stratum_columns"),
        f"models.continuous_objective.experiments.{experiment}.stratum_columns",
        default=DEFAULT_BACKGROUND_STRATUM_COLUMNS
        if fit_weight_policy == STRATIFIED_FIT_WEIGHT_POLICY
        else (),
    )
    return ContinuousObjectiveConfig(
        config_path=config_path,
        experiment=experiment,
        sample_policy=str(
            experiment_block.get("sample_policy")
            or objective.get("sample_policy")
            or baselines.get("sample_policy")
            or "current_masked_sample"
        ),
        input_table_path=Path(
            require_string(
                experiment_block.get("input_table")
                or objective.get("input_table")
                or baselines.get("input_table"),
                f"models.continuous_objective.experiments.{experiment}.input_table",
            )
        ),
        inference_table_path=Path(
            require_string(
                experiment_block.get("inference_table")
                or objective.get("inference_table")
                or baselines.get("inference_table"),
                f"models.continuous_objective.experiments.{experiment}.inference_table",
            )
        ),
        model_path=config_path_field(experiment_block, experiment, "model"),
        sample_predictions_path=config_path_field(
            experiment_block,
            experiment,
            "sample_predictions",
        ),
        full_grid_predictions_path=config_path_field(
            experiment_block,
            experiment,
            "full_grid_predictions",
        ),
        metrics_path=config_path_field(experiment_block, experiment, "metrics"),
        area_calibration_path=config_path_field(experiment_block, experiment, "area_calibration"),
        assumed_background_leakage_path=config_path_field(
            experiment_block,
            experiment,
            "assumed_background_leakage",
        ),
        manifest_path=config_path_field(experiment_block, experiment, "manifest"),
        model_name=str(experiment_block.get("model_name", DEFAULT_MODEL_NAME)),
        objective_policy=str(
            experiment_block.get("objective_policy")
            or objective.get("objective_policy")
            or DEFAULT_OBJECTIVE_POLICY
        ),
        fit_weight_policy=fit_weight_policy,
        fit_weight_cap=optional_positive_float(
            experiment_block.get("fit_weight_cap", objective.get("fit_weight_cap")),
            f"models.continuous_objective.experiments.{experiment}.fit_weight_cap",
            fit_weight_cap_default,
        ),
        stratum_balance_gamma=read_unit_interval_float(
            experiment_block.get(
                "stratum_balance_gamma",
                objective.get("stratum_balance_gamma"),
            ),
            f"models.continuous_objective.experiments.{experiment}.stratum_balance_gamma",
            DEFAULT_STRATUM_BALANCE_GAMMA,
        ),
        background_weight_budget_multiplier=optional_positive_float(
            experiment_block.get(
                "background_weight_budget_multiplier",
                objective.get("background_weight_budget_multiplier"),
            ),
            (
                "models.continuous_objective.experiments."
                f"{experiment}.background_weight_budget_multiplier"
            ),
            None,
        ),
        stratum_columns=stratum_columns,
        sample_weight_column=str(
            experiment_block.get("sample_weight_column")
            or objective.get("sample_weight_column")
            or DEFAULT_SAMPLE_WEIGHT_COLUMN
        ),
        target_column=str(
            experiment_block.get("target")
            or objective.get("target")
            or baselines.get("target")
            or DEFAULT_TARGET_COLUMN
        ),
        target_area_column=str(
            experiment_block.get("target_area_column")
            or objective.get("target_area_column")
            or DEFAULT_TARGET_AREA_COLUMN
        ),
        cell_area_m2=positive_float(
            experiment_block.get("cell_area_m2", objective.get("cell_area_m2")),
            f"models.continuous_objective.experiments.{experiment}.cell_area_m2",
            KELPWATCH_PIXEL_AREA_M2,
        ),
        feature_columns=parse_bands(
            experiment_block.get("features")
            or objective.get("features")
            or baselines.get("features")
            or features.get("bands")
        ),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
        alpha_grid=read_alpha_grid(
            experiment_block.get("alpha_grid") or objective.get("alpha_grid")
        ),
        batch_size=positive_int(
            experiment_block.get("batch_size", objective.get("batch_size")),
            f"models.continuous_objective.experiments.{experiment}.batch_size",
            DEFAULT_BATCH_SIZE,
        ),
        drop_missing_features=read_bool(
            experiment_block.get("drop_missing_features", objective.get("drop_missing_features")),
            f"models.continuous_objective.experiments.{experiment}.drop_missing_features",
            default=True,
        ),
        reporting_domain_mask=reporting_domain_mask,
    )


def config_path_field(experiment_block: dict[str, Any], experiment: str, key: str) -> Path:
    """Read one required path from a continuous-objective experiment block."""
    return Path(
        require_string(
            experiment_block.get(key),
            f"models.continuous_objective.experiments.{experiment}.{key}",
        )
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating missing values as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def validate_fit_weight_policy(policy: str) -> None:
    """Validate that an experiment declares a supported fit-weight policy."""
    if policy not in SUPPORTED_FIT_WEIGHT_POLICIES:
        msg = (
            "unsupported continuous-objective fit_weight_policy: "
            f"{policy}; expected one of {SUPPORTED_FIT_WEIGHT_POLICIES}"
        )
        raise ValueError(msg)


def read_stratum_columns(
    value: object,
    name: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Read optional background-stratum columns from config."""
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        msg = f"config field must be a non-empty list of column names: {name}"
        raise ValueError(msg)
    columns = tuple(str(column) for column in value)
    if any(not column for column in columns):
        msg = f"stratum columns must be non-empty strings: {name}"
        raise ValueError(msg)
    if len(set(columns)) != len(columns):
        msg = f"stratum columns contain duplicates: {name}"
        raise ValueError(msg)
    return columns


def read_bool(value: object, name: str, *, default: bool) -> bool:
    """Read an optional boolean config field."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"config field must be a boolean: {name}"
        raise ValueError(msg)
    return value


def read_year_list(config: dict[str, Any], key: str) -> tuple[int, ...]:
    """Read a non-empty list of split years from config."""
    values = config.get(key)
    if not isinstance(values, list) or not values:
        msg = f"config field must be a non-empty list of years: splits.{key}"
        raise ValueError(msg)
    if any(isinstance(value, bool) for value in values):
        msg = f"split years must be integers, not booleans: splits.{key}"
        raise ValueError(msg)
    return tuple(operator.index(cast(SupportsIndex, value)) for value in values)


def read_alpha_grid(value: object) -> tuple[float, ...]:
    """Read a non-empty ridge-alpha grid from config."""
    if not isinstance(value, list) or not value:
        msg = "models.continuous_objective.alpha_grid must be a non-empty list"
        raise ValueError(msg)
    alpha_grid = tuple(
        positive_float(item, "models.continuous_objective.alpha_grid[]", None) for item in value
    )
    if len(set(alpha_grid)) != len(alpha_grid):
        msg = "continuous-objective alpha grid contains duplicate values"
        raise ValueError(msg)
    return alpha_grid


def positive_float(value: object, name: str, default: float | None) -> float:
    """Read an optional positive float from config."""
    if value is None:
        if default is None:
            msg = f"missing required numeric field: {name}"
            raise ValueError(msg)
        return default
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    parsed = float(cast(Any, value))
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def optional_positive_float(value: object, name: str, default: float | None) -> float | None:
    """Read an optional positive float, preserving None when no default is set."""
    if value is None and default is None:
        return None
    return positive_float(value, name, default)


def read_unit_interval_float(value: object, name: str, default: float) -> float:
    """Read a positive float no larger than one."""
    parsed = positive_float(value, name, default)
    if parsed > 1.0:
        msg = f"field must be <= 1.0: {name}"
        raise ValueError(msg)
    return parsed


def positive_int(value: object, name: str, default: int) -> int:
    """Read an optional positive integer from config."""
    if value is None:
        return default
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    parsed = operator.index(cast(SupportsIndex, value))
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def validate_sample_table(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> None:
    """Validate that the sample table has the columns needed for fitting."""
    required = [
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        objective_config.target_column,
        objective_config.target_area_column,
        objective_config.sample_weight_column,
        *objective_config.feature_columns,
    ]
    if objective_config.fit_weight_policy == STRATIFIED_FIT_WEIGHT_POLICY:
        required.extend(objective_config.stratum_columns)
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        msg = f"continuous-objective sample table is missing required columns: {missing}"
        raise ValueError(msg)


def prepare_sample_frame(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign splits and retain sample rows with complete features and targets."""
    frame = dataframe.copy()
    frame["split"] = split_series_by_year(frame["year"], objective_config)
    feature_complete = frame.loc[:, list(objective_config.feature_columns)].notna().all(axis=1)
    target_complete = frame[objective_config.target_column].notna()
    retained_mask = (feature_complete & target_complete).to_numpy(dtype=bool)
    if not objective_config.drop_missing_features and not bool(retained_mask.all()):
        msg = "continuous-objective ridge cannot fit rows with missing features or targets"
        raise ValueError(msg)
    retained = frame.loc[retained_mask].copy()
    ensure_required_splits_present(retained)
    dropped_counts: dict[str, int] = {split: 0 for split in SPLIT_ORDER}
    dropped = frame.loc[~retained_mask]
    for split, group in dropped.groupby("split", sort=False):
        dropped_counts[str(split)] = int(len(group))
    LOGGER.info("Retained %s sample rows; dropped rows by split: %s", len(retained), dropped_counts)
    return retained, dropped_counts


def split_series_by_year(
    years: pd.Series,
    objective_config: ContinuousObjectiveConfig,
) -> pd.Series:
    """Assign split labels from configured year lists."""
    split_by_year = {
        **{year: "train" for year in objective_config.train_years},
        **{year: "validation" for year in objective_config.validation_years},
        **{year: "test" for year in objective_config.test_years},
    }
    split = years.astype(int).map(split_by_year).fillna("unassigned").astype(str)
    unassigned = sorted(int(year) for year in years.loc[split == "unassigned"].unique())
    if unassigned:
        msg = f"continuous-objective rows contain years not assigned to a split: {unassigned}"
        raise ValueError(msg)
    return cast(pd.Series, split)


def ensure_required_splits_present(dataframe: pd.DataFrame) -> None:
    """Validate that retained rows include train, validation, and test splits."""
    missing = [split for split in SPLIT_ORDER if split not in set(dataframe["split"])]
    if missing:
        msg = f"continuous-objective retained rows are missing splits: {missing}"
        raise ValueError(msg)


def rows_for_split(dataframe: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return rows for one split, raising when the split is empty."""
    rows = dataframe.loc[dataframe["split"] == split].copy()
    if rows.empty:
        msg = f"no continuous-objective rows for split: {split}"
        raise ValueError(msg)
    return rows


def fit_weights(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Return training weights for the configured continuous objective."""
    if objective_config.fit_weight_policy == DEFAULT_FIT_WEIGHT_POLICY:
        return capped_assumed_background_fit_weights(dataframe, objective_config)
    if objective_config.fit_weight_policy == STRATIFIED_FIT_WEIGHT_POLICY:
        return stratified_background_fit_weights(dataframe, objective_config)
    validate_fit_weight_policy(objective_config.fit_weight_policy)
    raise AssertionError("unreachable fit-weight policy validation branch")


def capped_assumed_background_fit_weights(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Return capped weights without upweighting observed Kelpwatch rows."""
    if objective_config.fit_weight_cap is None:
        msg = "capped continuous objective requires fit_weight_cap"
        raise ValueError(msg)
    sample_weights = dataframe[objective_config.sample_weight_column].to_numpy(dtype=float)
    weights = np.clip(sample_weights, 1.0, objective_config.fit_weight_cap)
    weights[kelpwatch_supported_mask(dataframe, objective_config)] = 1.0
    return cast(np.ndarray, weights)


def stratified_background_fit_weights(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Balance assumed-background total fit weight across strata within each year."""
    validate_stratum_columns(dataframe, objective_config)
    sample_weights = dataframe[objective_config.sample_weight_column].to_numpy(dtype=float)
    raw_weights = np.maximum(sample_weights, 1.0)
    weights = np.ones(len(dataframe), dtype=float)
    supported = kelpwatch_supported_mask(dataframe, objective_config)
    background_mask = ~supported
    if not np.any(background_mask):
        return weights
    background = dataframe.loc[background_mask, list(objective_config.stratum_columns)].copy()
    background["_raw_weight"] = raw_weights[background_mask]
    background["_source_index"] = np.flatnonzero(background_mask)
    for _, year_group in background.groupby("year", sort=False, dropna=False):
        stratum_totals = year_group.groupby(
            list(objective_config.stratum_columns), sort=False, dropna=False
        )["_raw_weight"].transform("sum")
        stratum_count = int(
            year_group.loc[:, list(objective_config.stratum_columns)].drop_duplicates().shape[0]
        )
        if stratum_count == 0:
            continue
        target_total = float(year_group["_raw_weight"].sum()) / stratum_count
        equalized = (
            year_group["_raw_weight"].to_numpy(dtype=float)
            * target_total
            / stratum_totals.to_numpy(dtype=float)
        )
        raw = year_group["_raw_weight"].to_numpy(dtype=float)
        scaled = raw * np.power(equalized / raw, objective_config.stratum_balance_gamma)
        scaled = apply_background_budget_for_year(
            scaled,
            supported,
            dataframe,
            objective_config,
            int(year_group["year"].iloc[0]),
        )
        weights[year_group["_source_index"].to_numpy(dtype=int)] = scaled
    return weights


def apply_background_budget_for_year(
    background_weights: np.ndarray,
    supported: np.ndarray,
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
    year: int,
) -> np.ndarray:
    """Cap total background fit weight for one year when a budget is configured."""
    multiplier = objective_config.background_weight_budget_multiplier
    if multiplier is None:
        return background_weights
    year_values = dataframe["year"].to_numpy(dtype=int)
    supported_total = float(np.count_nonzero(supported & (year_values == year)))
    if supported_total <= 0.0:
        return background_weights
    budget = supported_total * multiplier
    background_total = float(np.sum(background_weights))
    if background_total <= budget:
        return background_weights
    return background_weights * (budget / background_total)


def validate_stratum_columns(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> None:
    """Validate configured stratum columns before computing weights."""
    if not objective_config.stratum_columns:
        msg = "stratified continuous objective requires stratum_columns"
        raise ValueError(msg)
    if "year" not in objective_config.stratum_columns:
        msg = "stratified continuous objective must include year in stratum_columns"
        raise ValueError(msg)
    missing = [column for column in objective_config.stratum_columns if column not in dataframe]
    if missing:
        msg = f"stratified continuous objective is missing stratum columns: {missing}"
        raise ValueError(msg)


def fit_weight_strata(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Return a compact label describing each row's fit-weight stratum."""
    if objective_config.fit_weight_policy != STRATIFIED_FIT_WEIGHT_POLICY:
        return np.full(len(dataframe), "unstratified", dtype=object)
    validate_stratum_columns(dataframe, objective_config)
    labels = np.full(len(dataframe), "kelpwatch_supported", dtype=object)
    background_mask = ~kelpwatch_supported_mask(dataframe, objective_config)
    if not np.any(background_mask):
        return labels
    stratum_frame = dataframe.loc[background_mask, list(objective_config.stratum_columns)]
    labels[background_mask] = stratum_frame.apply(format_stratum_label, axis=1).to_numpy(
        dtype=object
    )
    return labels


def format_stratum_label(row: pd.Series) -> str:
    """Format one background stratum label from configured column values."""
    return "|".join(f"{column}={row[column]}" for column in row.index)


def kelpwatch_supported_mask(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Return rows supported by Kelpwatch or positive annual-max labels."""
    target_positive = dataframe[objective_config.target_column].to_numpy(dtype=float) > 0.0
    if "is_kelpwatch_observed" in dataframe.columns:
        observed = dataframe["is_kelpwatch_observed"].fillna(False).to_numpy(dtype=bool)
        return cast(np.ndarray, observed | target_positive)
    if "label_source" in dataframe.columns:
        label_source = dataframe["label_source"].fillna("").astype(str).to_numpy(dtype=object)
        return cast(np.ndarray, (label_source != "assumed_background") | target_positive)
    return cast(np.ndarray, target_positive)


def fit_select_weighted_ridge(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> ContinuousObjectiveSelection:
    """Fit ridge candidates with capped train weights and select by validation RMSE."""
    x_train = feature_matrix(train_rows, objective_config.feature_columns)
    y_train = target_vector(train_rows, objective_config)
    weights = train_rows["fit_weight"].to_numpy(dtype=float)
    x_validation = feature_matrix(validation_rows, objective_config.feature_columns)
    y_validation = target_vector(validation_rows, objective_config)
    validation_rows_for_grid: list[dict[str, object]] = []
    candidates: list[tuple[float, float, Any]] = []
    for alpha in objective_config.alpha_grid:
        model = make_ridge_pipeline(alpha)
        fit_ridge_model(model, x_train, y_train, weights)
        predictions = np.asarray(model.predict(x_validation), dtype=float)
        clipped = np.clip(predictions, 0.0, 1.0)
        rmse = root_mean_squared_error(y_validation, clipped)
        validation_rows_for_grid.append(
            {
                "alpha": alpha,
                "validation_rmse": rmse,
                "validation_mae": mean_absolute_error(y_validation, clipped),
            }
        )
        candidates.append((rmse, alpha, model))
        LOGGER.info("Continuous objective alpha=%s validation RMSE=%s", alpha, rmse)
    selected_rmse, selected_alpha, selected_model = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    LOGGER.info(
        "Selected continuous-objective alpha=%s with validation RMSE=%s",
        selected_alpha,
        selected_rmse,
    )
    return ContinuousObjectiveSelection(
        model=selected_model,
        selected_alpha=selected_alpha,
        validation_rows=validation_rows_for_grid,
    )


def feature_matrix(dataframe: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    """Return model features as a floating-point matrix."""
    return cast(np.ndarray, dataframe.loc[:, list(feature_columns)].to_numpy(dtype=float))


def target_vector(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> np.ndarray:
    """Return the continuous objective target vector."""
    return cast(np.ndarray, dataframe[objective_config.target_column].to_numpy(dtype=float))


def build_prediction_frame(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
    *,
    predictions: np.ndarray,
    alpha: float,
    evaluation_scope_value: str,
) -> pd.DataFrame:
    """Build row-level predictions with stable experiment metadata."""
    frame = dataframe[prediction_identity_columns(dataframe, objective_config)].copy()
    clipped = np.clip(predictions, 0.0, 1.0)
    frame["mask_status"] = mask_status(objective_config.reporting_domain_mask)
    frame["evaluation_scope"] = evaluation_scope_value
    frame["sample_policy"] = objective_config.sample_policy
    frame["model_name"] = objective_config.model_name
    frame["model_family"] = MODEL_FAMILY
    frame["objective_policy"] = objective_config.objective_policy
    frame["fit_weight_policy"] = objective_config.fit_weight_policy
    frame["fit_weight_cap"] = objective_config.fit_weight_cap
    frame["stratum_balance_gamma"] = objective_config.stratum_balance_gamma
    frame["background_weight_budget_multiplier"] = (
        objective_config.background_weight_budget_multiplier
    )
    frame["stratum_columns"] = ",".join(objective_config.stratum_columns)
    frame["target"] = objective_config.target_column
    frame["target_area_column"] = objective_config.target_area_column
    frame["cell_area_m2"] = objective_config.cell_area_m2
    frame["alpha"] = alpha
    frame["pred_kelp_fraction_y"] = predictions
    frame["pred_kelp_fraction_y_clipped"] = clipped
    frame["pred_kelp_max_y"] = clipped * objective_config.cell_area_m2
    observed = frame[objective_config.target_column].to_numpy(dtype=float)
    observed_area = frame[objective_config.target_area_column].to_numpy(dtype=float)
    frame["residual_kelp_fraction_y"] = observed - predictions
    frame["residual_kelp_fraction_y_clipped"] = observed - clipped
    frame["residual_kelp_max_y"] = observed_area - frame["pred_kelp_max_y"].to_numpy(dtype=float)
    return cast(pd.DataFrame, frame.loc[:, prediction_output_columns(frame)])


def prediction_identity_columns(
    dataframe: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> list[str]:
    """Return identity, provenance, and target columns for prediction output."""
    columns = [
        "year",
        "split",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        objective_config.target_column,
        objective_config.target_area_column,
    ]
    for column in OPTIONAL_ID_COLUMNS:
        if column in dataframe.columns and column not in columns:
            columns.append(column)
    if "fit_weight_stratum" in dataframe.columns and "fit_weight_stratum" not in columns:
        columns.append("fit_weight_stratum")
    if "fit_weight" in dataframe.columns and "fit_weight" not in columns:
        columns.append("fit_weight")
    return columns


def prediction_output_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return configured prediction columns that are present in a frame."""
    return [column for column in PREDICTION_FIELDS if column in dataframe.columns]


def metric_rows_from_predictions(
    predictions: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> list[dict[str, object]]:
    """Build unweighted sample metric rows by split, year, and label source."""
    rows: list[dict[str, object]] = []
    group_sets = [["split", "year"]]
    if "label_source" in predictions.columns:
        group_sets.append(["split", "year", "label_source"])
    for group_columns in group_sets:
        for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            label_source = "all" if "label_source" not in group_columns else str(key_tuple[-1])
            rows.append(
                metric_row(
                    group,
                    objective_config,
                    split=str(key_tuple[0]),
                    year=int(key_tuple[1]),
                    label_source=label_source,
                )
            )
    return rows


def metric_row(
    group: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
    *,
    split: str,
    year: int,
    label_source: str,
) -> dict[str, object]:
    """Build one unweighted metric row from row-level predictions."""
    observed = group[objective_config.target_column].to_numpy(dtype=float)
    predicted = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = group[objective_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    observed_positive = observed >= DEFAULT_THRESHOLD_FRACTION
    predicted_positive = predicted >= DEFAULT_THRESHOLD_FRACTION
    precision, recall, f1 = precision_recall_f1(observed_positive, predicted_positive)
    fit_weight = (
        group["fit_weight"].to_numpy(dtype=float) if "fit_weight" in group.columns else np.array([])
    )
    return {
        "model_name": objective_config.model_name,
        "model_family": MODEL_FAMILY,
        "objective_policy": objective_config.objective_policy,
        "fit_weight_policy": objective_config.fit_weight_policy,
        "fit_weight_cap": objective_config.fit_weight_cap,
        "stratum_balance_gamma": objective_config.stratum_balance_gamma,
        "background_weight_budget_multiplier": (
            objective_config.background_weight_budget_multiplier
        ),
        "split": split,
        "year": year,
        "label_source": label_source,
        "mask_status": mask_status(objective_config.reporting_domain_mask),
        "evaluation_scope": "model_input_sample"
        if label_source == "all"
        else f"{label_source}_sample",
        "row_count": int(len(group)),
        "weighted": False,
        "target": objective_config.target_column,
        "alpha": prediction_alpha(group),
        "mae": mean_absolute_error(observed, predicted),
        "rmse": root_mean_squared_error(observed, predicted),
        "r2": unweighted_r2(observed, predicted),
        "spearman": correlation(observed, predicted),
        "observed_canopy_area": float(np.nansum(observed_area)),
        "predicted_canopy_area": float(np.nansum(predicted_area)),
        "area_bias": float(np.nansum(predicted_area) - np.nansum(observed_area)),
        "area_pct_bias": percent_bias(
            float(np.nansum(predicted_area)), float(np.nansum(observed_area))
        ),
        "precision_ge_10pct": precision,
        "recall_ge_10pct": recall,
        "f1_ge_10pct": f1,
        "observed_positive_rate_ge_10pct": safe_ratio(
            float(np.count_nonzero(observed_positive)), len(group)
        ),
        "predicted_positive_rate_ge_10pct": safe_ratio(
            float(np.count_nonzero(predicted_positive)), len(group)
        ),
        "fit_weight_min": finite_min(fit_weight),
        "fit_weight_max": finite_max(fit_weight),
        "fit_weight_mean": finite_mean(fit_weight),
    }


def prediction_alpha(dataframe: pd.DataFrame) -> float:
    """Return the selected alpha recorded on prediction rows."""
    if "alpha" not in dataframe.columns:
        return math.nan
    values = pd.to_numeric(dataframe["alpha"], errors="coerce").dropna().unique()
    if len(values) == 0:
        return math.nan
    return float(values[0])


def write_full_grid_predictions(
    selection: ContinuousObjectiveSelection,
    objective_config: ContinuousObjectiveConfig,
) -> tuple[int, int, int, list[dict[str, object]]]:
    """Stream retained-domain full-grid predictions and aggregate diagnostics."""
    reset_output_path(objective_config.full_grid_predictions_path)
    metric_totals: dict[MetricKey, dict[str, float]] = {}
    row_count = 0
    part_count = 0
    dropped_missing_feature_rows = 0
    columns = full_grid_input_columns(objective_config)
    LOGGER.info(
        "Streaming continuous-objective full-grid inference from %s",
        objective_config.inference_table_path,
    )
    for batch in iter_parquet_batches(
        objective_config.inference_table_path,
        columns,
        objective_config.batch_size,
    ):
        batch["split"] = split_series_by_year(batch["year"], objective_config)
        masked = apply_reporting_domain_mask(batch, objective_config.reporting_domain_mask)
        complete = feature_complete_mask(masked, objective_config.feature_columns)
        dropped_missing_feature_rows += int((~complete).sum())
        retained = masked.loc[complete].copy()
        if retained.empty:
            continue
        predictions = np.asarray(
            selection.model.predict(feature_matrix(retained, objective_config.feature_columns)),
            dtype=float,
        )
        prediction_rows = build_prediction_frame(
            retained,
            objective_config,
            predictions=predictions,
            alpha=selection.selected_alpha,
            evaluation_scope_value=evaluation_scope(objective_config.reporting_domain_mask),
        )
        write_prediction_part(
            prediction_rows, objective_config.full_grid_predictions_path, part_count
        )
        update_metric_accumulators(metric_totals, prediction_rows, objective_config)
        row_count += len(prediction_rows)
        part_count += 1
        LOGGER.info(
            "Wrote continuous-objective full-grid part %s with %s rows",
            part_count,
            len(prediction_rows),
        )
    return (
        row_count,
        part_count,
        dropped_missing_feature_rows,
        metric_rows_from_accumulators(metric_totals, objective_config),
    )


def full_grid_input_columns(objective_config: ContinuousObjectiveConfig) -> list[str]:
    """Return full-grid columns needed for prediction and output identity."""
    dataset = ds.dataset(objective_config.inference_table_path, format="parquet")  # type: ignore[no-untyped-call]
    available = set(dataset.schema.names)
    candidates = [
        *REQUIRED_INPUT_COLUMNS,
        *objective_config.feature_columns,
        *OPTIONAL_ID_COLUMNS,
    ]
    return [column for column in dict.fromkeys(candidates) if column in available]


def feature_complete_mask(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.Series:
    """Return rows with complete feature values."""
    missing = [column for column in feature_columns if column not in dataframe.columns]
    if missing:
        msg = f"continuous-objective inference table is missing features: {missing}"
        raise ValueError(msg)
    return cast(pd.Series, dataframe.loc[:, list(feature_columns)].notna().all(axis=1))


def update_metric_accumulators(
    totals: dict[MetricKey, dict[str, float]],
    predictions: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> None:
    """Update full-grid metric totals from one prediction batch."""
    group_sets = [["split", "year"]]
    if "label_source" in predictions.columns:
        group_sets.append(["split", "year", "label_source"])
    for group_columns in group_sets:
        for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            label_source = "all" if "label_source" not in group_columns else str(key_tuple[-1])
            key = (str(key_tuple[0]), int(key_tuple[1]), label_source)
            update_metric_state(
                totals.setdefault(key, empty_metric_state()), group, objective_config
            )


def empty_metric_state() -> dict[str, float]:
    """Return an initialized full-grid metric accumulator."""
    return {
        "row_count": 0.0,
        "observed_fraction_sum": 0.0,
        "observed_fraction_square_sum": 0.0,
        "predicted_fraction_sum": 0.0,
        "absolute_error_fraction_sum": 0.0,
        "squared_error_fraction_sum": 0.0,
        "observed_area_sum": 0.0,
        "predicted_area_sum": 0.0,
        "true_positive_count": 0.0,
        "false_positive_count": 0.0,
        "false_negative_count": 0.0,
        "observed_positive_count": 0.0,
        "predicted_positive_count": 0.0,
        "assumed_background_count": 0.0,
        "assumed_background_predicted_positive_count": 0.0,
        "assumed_background_predicted_area_m2": 0.0,
        "assumed_background_max_predicted_area_m2": 0.0,
    }


def update_metric_state(
    state: dict[str, float],
    group: pd.DataFrame,
    objective_config: ContinuousObjectiveConfig,
) -> None:
    """Accumulate scalar full-grid metrics for one grouped prediction frame."""
    observed = group[objective_config.target_column].to_numpy(dtype=float)
    predicted = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = group[objective_config.target_area_column].to_numpy(dtype=float)
    predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
    observed_positive = observed >= DEFAULT_THRESHOLD_FRACTION
    predicted_positive = predicted >= DEFAULT_THRESHOLD_FRACTION
    label_sources = label_source_array(group)
    assumed_background = label_sources == "assumed_background"
    state["row_count"] += float(len(group))
    state["observed_fraction_sum"] += float(np.nansum(observed))
    state["observed_fraction_square_sum"] += float(np.nansum(observed**2))
    state["predicted_fraction_sum"] += float(np.nansum(predicted))
    state["absolute_error_fraction_sum"] += float(np.nansum(np.abs(observed - predicted)))
    state["squared_error_fraction_sum"] += float(np.nansum((observed - predicted) ** 2))
    state["observed_area_sum"] += float(np.nansum(observed_area))
    state["predicted_area_sum"] += float(np.nansum(predicted_area))
    state["true_positive_count"] += float(np.count_nonzero(observed_positive & predicted_positive))
    state["false_positive_count"] += float(
        np.count_nonzero(~observed_positive & predicted_positive)
    )
    state["false_negative_count"] += float(
        np.count_nonzero(observed_positive & ~predicted_positive)
    )
    state["observed_positive_count"] += float(np.count_nonzero(observed_positive))
    state["predicted_positive_count"] += float(np.count_nonzero(predicted_positive))
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


def label_source_array(dataframe: pd.DataFrame) -> np.ndarray:
    """Return label-source values with a stable fallback."""
    if "label_source" in dataframe.columns:
        return cast(
            np.ndarray,
            dataframe["label_source"].fillna("unknown").astype(str).to_numpy(dtype=object),
        )
    if "is_kelpwatch_observed" in dataframe.columns:
        observed = dataframe["is_kelpwatch_observed"].fillna(False).to_numpy(dtype=bool)
        return np.where(observed, "kelpwatch_station", "assumed_background")
    return np.full(len(dataframe), "all", dtype=object)


def metric_rows_from_accumulators(
    totals: dict[MetricKey, dict[str, float]],
    objective_config: ContinuousObjectiveConfig,
) -> list[dict[str, object]]:
    """Convert full-grid accumulator states to metric rows."""
    return [
        metric_row_from_state(key, state, objective_config)
        for key, state in sorted(totals.items(), key=lambda item: item[0])
    ]


def metric_row_from_state(
    key: MetricKey,
    state: dict[str, float],
    objective_config: ContinuousObjectiveConfig,
) -> dict[str, object]:
    """Build one full-grid metric row from an accumulated state."""
    split, year, label_source = key
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
        "model_name": objective_config.model_name,
        "model_family": MODEL_FAMILY,
        "objective_policy": objective_config.objective_policy,
        "fit_weight_policy": objective_config.fit_weight_policy,
        "fit_weight_cap": objective_config.fit_weight_cap,
        "stratum_balance_gamma": objective_config.stratum_balance_gamma,
        "background_weight_budget_multiplier": (
            objective_config.background_weight_budget_multiplier
        ),
        "split": split,
        "year": year,
        "label_source": label_source,
        "mask_status": mask_status(objective_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(objective_config.reporting_domain_mask),
        "row_count": int(row_count),
        "weighted": False,
        "target": objective_config.target_column,
        "alpha": math.nan,
        "mae": safe_ratio(state["absolute_error_fraction_sum"], row_count),
        "rmse": math.sqrt(safe_ratio(state["squared_error_fraction_sum"], row_count)),
        "r2": r2,
        "spearman": math.nan,
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "area_bias": predicted_area - observed_area,
        "area_pct_bias": percent_bias(predicted_area, observed_area),
        "precision_ge_10pct": precision,
        "recall_ge_10pct": recall,
        "f1_ge_10pct": f1,
        "observed_positive_rate_ge_10pct": safe_ratio(state["observed_positive_count"], row_count),
        "predicted_positive_rate_ge_10pct": safe_ratio(
            state["predicted_positive_count"], row_count
        ),
        "fit_weight_min": math.nan,
        "fit_weight_max": math.nan,
        "fit_weight_mean": math.nan,
        "assumed_background_count": int(state["assumed_background_count"]),
        "assumed_background_predicted_area_m2": state["assumed_background_predicted_area_m2"],
        "assumed_background_max_predicted_area_m2": state[
            "assumed_background_max_predicted_area_m2"
        ],
        "assumed_background_predicted_positive_count": int(
            state["assumed_background_predicted_positive_count"]
        ),
        "assumed_background_predicted_positive_rate": safe_ratio(
            state["assumed_background_predicted_positive_count"],
            state["assumed_background_count"],
        ),
    }


def area_calibration_rows(
    metric_rows: list[dict[str, object]],
    objective_config: ContinuousObjectiveConfig,
) -> list[dict[str, object]]:
    """Convert full-grid metric rows into compact area-calibration rows."""
    rows: list[dict[str, object]] = []
    for row in metric_rows:
        rows.append(
            {
                "model_name": row["model_name"],
                "model_family": MODEL_FAMILY,
                "objective_policy": objective_config.objective_policy,
                "fit_weight_policy": objective_config.fit_weight_policy,
                "fit_weight_cap": objective_config.fit_weight_cap,
                "stratum_balance_gamma": objective_config.stratum_balance_gamma,
                "background_weight_budget_multiplier": (
                    objective_config.background_weight_budget_multiplier
                ),
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
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "f1_ge_10pct": row["f1_ge_10pct"],
            }
        )
    return rows


def assumed_background_leakage_rows(
    metric_rows: list[dict[str, object]],
    objective_config: ContinuousObjectiveConfig,
) -> list[dict[str, object]]:
    """Build retained-domain assumed-background leakage diagnostics."""
    rows: list[dict[str, object]] = []
    for row in metric_rows:
        if row.get("label_source") != "assumed_background":
            continue
        background_count = float(cast(Any, row.get("assumed_background_count", 0.0)))
        rows.append(
            {
                "model_name": row["model_name"],
                "model_family": MODEL_FAMILY,
                "objective_policy": objective_config.objective_policy,
                "fit_weight_policy": objective_config.fit_weight_policy,
                "fit_weight_cap": objective_config.fit_weight_cap,
                "stratum_balance_gamma": objective_config.stratum_balance_gamma,
                "background_weight_budget_multiplier": (
                    objective_config.background_weight_budget_multiplier
                ),
                "split": row["split"],
                "year": row["year"],
                "mask_status": row["mask_status"],
                "evaluation_scope": row["evaluation_scope"],
                "label_source": row["label_source"],
                "assumed_background_count": int(background_count),
                "assumed_background_predicted_area_m2": row["assumed_background_predicted_area_m2"],
                "assumed_background_mean_predicted_area_m2": safe_ratio(
                    float(cast(Any, row["assumed_background_predicted_area_m2"])),
                    background_count,
                ),
                "assumed_background_max_predicted_area_m2": row[
                    "assumed_background_max_predicted_area_m2"
                ],
                "assumed_background_predicted_positive_count": row[
                    "assumed_background_predicted_positive_count"
                ],
                "assumed_background_predicted_positive_rate": row[
                    "assumed_background_predicted_positive_rate"
                ],
            }
        )
    return rows


def write_model(
    selection: ContinuousObjectiveSelection,
    objective_config: ContinuousObjectiveConfig,
) -> None:
    """Serialize the selected continuous-objective ridge model."""
    objective_config.model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": selection.model,
        "model_name": objective_config.model_name,
        "model_family": MODEL_FAMILY,
        "experiment": objective_config.experiment,
        "sample_policy": objective_config.sample_policy,
        "objective_policy": objective_config.objective_policy,
        "fit_weight_policy": objective_config.fit_weight_policy,
        "fit_weight_cap": objective_config.fit_weight_cap,
        "stratum_balance_gamma": objective_config.stratum_balance_gamma,
        "background_weight_budget_multiplier": (
            objective_config.background_weight_budget_multiplier
        ),
        "stratum_columns": list(objective_config.stratum_columns),
        "selected_alpha": selection.selected_alpha,
        "target_column": objective_config.target_column,
        "target_area_column": objective_config.target_area_column,
        "feature_columns": list(objective_config.feature_columns),
    }
    joblib.dump(payload, objective_config.model_path)


def fit_weight_formula(objective_config: ContinuousObjectiveConfig) -> str:
    """Return the manifest formula for the configured fit weights."""
    if objective_config.fit_weight_policy == STRATIFIED_FIT_WEIGHT_POLICY:
        return (
            "fit_weight = 1.0 for Kelpwatch-supported or positive rows; otherwise "
            "max(sample_weight, 1.0) is moved toward equal assumed-background "
            "stratum totals within each year using stratum_balance_gamma; an "
            "optional per-year background budget scales background weights after "
            "stratum balancing"
        )
    return (
        "fit_weight = 1.0 for Kelpwatch-supported or positive rows; "
        "otherwise min(max(sample_weight, 1.0), fit_weight_cap)"
    )


def background_stratum_balance_policy(
    objective_config: ContinuousObjectiveConfig,
) -> str | None:
    """Return the manifest policy for background stratum balancing."""
    if objective_config.fit_weight_policy != STRATIFIED_FIT_WEIGHT_POLICY:
        return None
    return "equal_total_assumed_background_fit_weight_per_stratum_within_year"


def write_manifest(
    *,
    objective_config: ContinuousObjectiveConfig,
    selection: ContinuousObjectiveSelection,
    sample_rows: int,
    sample_dropped_counts: dict[str, int],
    sample_metric_rows: list[dict[str, object]],
    full_grid_rows: int,
    full_grid_part_count: int,
    full_grid_dropped_missing_feature_rows: int,
    area_rows: list[dict[str, object]],
    leakage_rows: list[dict[str, object]],
) -> None:
    """Write a compact JSON manifest for the continuous-objective experiment."""
    objective_config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": "train-continuous-objective",
        "config_path": str(objective_config.config_path),
        "experiment": objective_config.experiment,
        "model_family": MODEL_FAMILY,
        "model_name": objective_config.model_name,
        "sample_policy": objective_config.sample_policy,
        "objective_policy": objective_config.objective_policy,
        "fit_weight_policy": objective_config.fit_weight_policy,
        "fit_weight_formula": fit_weight_formula(objective_config),
        "fit_weight_cap": objective_config.fit_weight_cap,
        "stratum_balance_gamma": objective_config.stratum_balance_gamma,
        "background_weight_budget_multiplier": (
            objective_config.background_weight_budget_multiplier
        ),
        "stratum_columns": list(objective_config.stratum_columns),
        "background_stratum_balance_policy": background_stratum_balance_policy(objective_config),
        "kelpwatch_supported_weight_policy": (
            "Kelpwatch-supported rows and positive annual-max rows keep fit_weight=1.0"
        ),
        "uses_global_background_cap": (
            objective_config.fit_weight_policy == DEFAULT_FIT_WEIGHT_POLICY
        ),
        "sample_weight_column": objective_config.sample_weight_column,
        "target_column": objective_config.target_column,
        "target_area_column": objective_config.target_area_column,
        "cell_area_m2": objective_config.cell_area_m2,
        "feature_columns": list(objective_config.feature_columns),
        "train_years": list(objective_config.train_years),
        "validation_years": list(objective_config.validation_years),
        "test_years": list(objective_config.test_years),
        "alpha_grid": list(objective_config.alpha_grid),
        "selected_alpha": selection.selected_alpha,
        "validation_alpha_metrics": selection.validation_rows,
        "mask_status": mask_status(objective_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(objective_config.reporting_domain_mask),
        "row_counts": {
            "sample_retained_rows": sample_rows,
            "sample_metric_rows": len(sample_metric_rows),
            "sample_dropped_counts_by_split": sample_dropped_counts,
            "full_grid_prediction_rows": full_grid_rows,
            "full_grid_prediction_part_count": full_grid_part_count,
            "full_grid_dropped_missing_feature_rows": full_grid_dropped_missing_feature_rows,
            "area_calibration_rows": len(area_rows),
            "assumed_background_leakage_rows": len(leakage_rows),
        },
        "inputs": {
            "input_table": str(objective_config.input_table_path),
            "inference_table": str(objective_config.inference_table_path),
        },
        "outputs": {
            "model": str(objective_config.model_path),
            "sample_predictions": str(objective_config.sample_predictions_path),
            "full_grid_predictions": str(objective_config.full_grid_predictions_path),
            "metrics": str(objective_config.metrics_path),
            "area_calibration": str(objective_config.area_calibration_path),
            "assumed_background_leakage": str(objective_config.assumed_background_leakage_path),
            "manifest": str(objective_config.manifest_path),
        },
        "evaluation_metrics_weighted": False,
        "full_grid_area_calibration_from_row_level_predictions": True,
        "test_rows_used_for_training_or_model_selection": False,
        "qa_notes": [
            "This is a direct continuous ridge model, not a binary or hurdle composition.",
            "Only 2021 validation rows select ridge alpha; 2022 test rows are report rows.",
        ],
    }
    objective_config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def write_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to a Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def write_csv_rows(
    rows: list[dict[str, object]],
    output_path: Path,
    fields: tuple[str, ...],
) -> None:
    """Write dictionaries to CSV with a stable header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_absolute_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute unweighted mean absolute error."""
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(finite):
        return math.nan
    return float(np.mean(np.abs(observed[finite] - predicted[finite])))


def unweighted_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute unweighted R2 or NaN when the observed target is constant."""
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if observed.size == 0:
        return math.nan
    total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_sum_squares == 0:
        return math.nan
    residual_sum_squares = float(np.sum((observed - predicted) ** 2))
    return 1.0 - residual_sum_squares / total_sum_squares


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute a Spearman correlation when both vectors vary."""
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if observed.size < 2 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return math.nan
    value = pd.Series(observed).corr(pd.Series(predicted), method="spearman")
    return float(value) if pd.notna(value) else math.nan


def finite_min(values: np.ndarray) -> float:
    """Return the finite minimum or NaN for an empty vector."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.min(finite))


def finite_max(values: np.ndarray) -> float:
    """Return the finite maximum or NaN for an empty vector."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))


def finite_mean(values: np.ndarray) -> float:
    """Return the finite mean or NaN for an empty vector."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))
