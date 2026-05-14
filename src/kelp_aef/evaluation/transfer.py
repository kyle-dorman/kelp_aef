"""Evaluate a frozen source-region model policy on a target-region config."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import evaluation_scope, mask_status
from kelp_aef.evaluation.baselines import (
    AREA_CALIBRATION_FIELDS,
    CLIMATOLOGY_MODEL_NAME,
    GEOGRAPHIC_MODEL_NAME,
    NO_SKILL_MODEL_NAME,
    PREVIOUS_YEAR_MODEL_NAME,
    RIDGE_MODEL_NAME,
    build_reference_area_calibration_rows,
    load_baseline_config,
    predict_full_grid_config,
    safe_ratio,
)
from kelp_aef.evaluation.binary_presence import (
    BINARY_METRIC_FIELDS,
    BINARY_MODEL_NAME,
    CALIBRATED_FULL_GRID_SUMMARY_FIELDS,
    CALIBRATED_MAX_F1_POLICY,
    PLATT_PROBABILITY_SOURCE,
    BinaryCalibrator,
    CalibrationThresholds,
    aggregate_calibrated_full_grid_summary_rows,
    apply_binary_calibrator,
    binary_auprc,
    binary_auroc,
    calibrated_full_grid_summary_rows_for_frame,
    label_source_series,
    load_binary_calibration_config,
    load_binary_presence_config,
    normalized_group_value,
    predict_binary_full_grid,
)
from kelp_aef.evaluation.hurdle import (
    compose_hurdle_config,
    load_hurdle_config,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TRANSFER_NAME = "monterey"
DEFAULT_TRAINING_REGIME = "monterey_transfer"
DEFAULT_MODEL_ORIGIN_REGION = "monterey"
TRANSFER_COMPARISON_FIELDS = (
    "training_regime",
    "model_origin_region",
    "evaluation_region",
    "model_name",
    "model_family",
    "composition_policy",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "row_count",
    "mae",
    "rmse",
    "r2",
    "f1_ge_10pct",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_pct_bias",
    "source_table",
)


@dataclass(frozen=True)
class TransferOutputConfig:
    """Resolved path and provenance settings for one transfer evaluation."""

    transfer_name: str
    training_regime: str
    model_origin_region: str
    evaluation_region: str
    baseline_predictions_path: Path
    baseline_manifest_path: Path
    binary_predictions_path: Path
    binary_manifest_path: Path
    binary_metrics_path: Path
    binary_area_summary_path: Path
    hurdle_predictions_path: Path
    hurdle_manifest_path: Path
    hurdle_metrics_path: Path
    hurdle_area_calibration_path: Path
    hurdle_model_comparison_path: Path
    hurdle_residual_by_observed_bin_path: Path
    hurdle_assumed_background_leakage_path: Path
    hurdle_map_figure_path: Path | None
    reference_area_calibration_path: Path
    model_comparison_path: Path
    primary_summary_path: Path
    manifest_path: Path


def evaluate_transfer(config_path: Path, source_config_path: Path) -> int:
    """Evaluate frozen source-region models on the target-region full grid."""
    outputs = load_transfer_output_config(config_path)
    LOGGER.info(
        "Evaluating %s policy transfer from %s to %s",
        outputs.model_origin_region,
        source_config_path,
        config_path,
    )
    target_baseline = load_baseline_config(config_path)
    source_baseline = load_baseline_config(source_config_path)
    target_binary = load_binary_presence_config(config_path)
    source_binary = load_binary_presence_config(source_config_path)
    source_calibration = load_binary_calibration_config(source_config_path)
    target_hurdle = load_hurdle_config(config_path)
    source_hurdle = load_hurdle_config(source_config_path)

    ridge_payload = load_joblib_payload(source_baseline.model_output_path)
    binary_payload = load_joblib_payload(source_binary.model_output_path)
    calibration_payload = load_joblib_payload(source_calibration.model_output_path)
    validate_feature_schema(
        ridge_payload,
        target_baseline.feature_columns,
        source_baseline.model_output_path,
    )
    validate_feature_schema(
        binary_payload,
        target_binary.feature_columns,
        source_binary.model_output_path,
    )

    transfer_baseline = replace(
        target_baseline,
        sample_policy=outputs.training_regime,
        model_output_path=source_baseline.model_output_path,
        geographic_model_output_path=source_baseline.geographic_model_output_path,
        predictions_path=outputs.baseline_predictions_path,
        prediction_manifest_path=outputs.baseline_manifest_path,
        area_calibration_path=outputs.reference_area_calibration_path,
    )
    predict_full_grid_config(transfer_baseline, fast=False)
    reference_rows = build_reference_area_calibration_rows(transfer_baseline)
    write_csv_rows(reference_rows, outputs.reference_area_calibration_path, AREA_CALIBRATION_FIELDS)

    transfer_binary = replace(
        target_binary,
        sample_policy=outputs.training_regime,
        model_output_path=source_binary.model_output_path,
        full_grid_predictions_path=outputs.binary_predictions_path,
        metrics_path=outputs.binary_metrics_path,
        full_grid_area_summary_path=outputs.binary_area_summary_path,
        prediction_manifest_path=outputs.binary_manifest_path,
    )
    raw_threshold = float(binary_payload.get("probability_threshold", math.nan))
    selected_c = float(binary_payload.get("selected_c", math.nan))
    binary_model = binary_payload.get("model")
    if binary_model is None:
        msg = f"binary model payload is missing model: {source_binary.model_output_path}"
        raise ValueError(msg)
    predict_binary_full_grid(binary_model, raw_threshold, selected_c, transfer_binary)

    calibrator = calibrator_from_payload(calibration_payload)
    calibrated_thresholds = thresholds_from_payload(calibration_payload, source_hurdle)
    binary_predictions = read_calibrated_binary_predictions(
        outputs.binary_predictions_path,
        calibrator,
        calibrated_thresholds.recommended_threshold,
        calibrated_thresholds.recommended_policy,
        source_calibration.method,
    )
    binary_metrics = build_calibrated_binary_metric_rows(
        binary_predictions,
        transfer_binary,
        calibrated_thresholds.recommended_threshold,
        calibrated_thresholds.recommended_policy,
        source_calibration.calibration_split,
        source_calibration.calibration_year,
    )
    transfer_calibration = replace(source_calibration, binary_config=transfer_binary)
    binary_area_rows = aggregate_calibrated_full_grid_summary_rows(
        calibrated_full_grid_summary_rows_for_frame(
            binary_predictions,
            transfer_calibration,
        )
    )
    write_csv_rows(binary_metrics, outputs.binary_metrics_path, BINARY_METRIC_FIELDS)
    write_csv_rows(
        binary_area_rows,
        outputs.binary_area_summary_path,
        CALIBRATED_FULL_GRID_SUMMARY_FIELDS,
    )
    write_binary_transfer_manifest(
        outputs=outputs,
        source_binary_model_path=source_binary.model_output_path,
        source_calibration_model_path=source_calibration.model_output_path,
        source_threshold_path=source_calibration.threshold_selection_path,
        raw_threshold=raw_threshold,
        calibrated_thresholds=calibrated_thresholds,
        prediction_rows=len(binary_predictions),
        metric_rows=len(binary_metrics),
        area_rows=len(binary_area_rows),
    )

    transfer_hurdle = replace(
        target_hurdle,
        sample_policy=outputs.training_regime,
        binary_full_grid_predictions_path=outputs.binary_predictions_path,
        binary_calibration_model_path=source_hurdle.binary_calibration_model_path,
        binary_threshold_selection_path=source_hurdle.binary_threshold_selection_path,
        conditional_model_path=source_hurdle.conditional_model_path,
        reference_area_calibration_path=outputs.reference_area_calibration_path,
        predictions_path=outputs.hurdle_predictions_path,
        manifest_path=outputs.hurdle_manifest_path,
        metrics_path=outputs.hurdle_metrics_path,
        area_calibration_path=outputs.hurdle_area_calibration_path,
        model_comparison_path=outputs.hurdle_model_comparison_path,
        residual_by_observed_bin_path=outputs.hurdle_residual_by_observed_bin_path,
        assumed_background_leakage_path=outputs.hurdle_assumed_background_leakage_path,
        map_figure_path=outputs.hurdle_map_figure_path,
        presence_threshold_policy=source_hurdle.presence_threshold_policy,
        presence_threshold=source_hurdle.presence_threshold,
    )
    compose_hurdle_config(transfer_hurdle)

    hurdle_area_rows = read_csv_dicts(outputs.hurdle_area_calibration_path)
    comparison_rows = build_transfer_model_comparison(
        reference_rows=reference_rows,
        hurdle_area_rows=hurdle_area_rows,
        outputs=outputs,
    )
    primary_rows = primary_transfer_rows(comparison_rows)
    write_csv_rows(comparison_rows, outputs.model_comparison_path, TRANSFER_COMPARISON_FIELDS)
    write_csv_rows(primary_rows, outputs.primary_summary_path, TRANSFER_COMPARISON_FIELDS)
    write_transfer_manifest(
        outputs=outputs,
        config_path=config_path,
        source_config_path=source_config_path,
        source_paths=[
            source_baseline.model_output_path,
            source_baseline.geographic_model_output_path,
            source_binary.model_output_path,
            source_calibration.model_output_path,
            source_calibration.threshold_selection_path,
            source_hurdle.conditional_model_path,
        ],
        target_paths=[
            target_baseline.inference_table_path,
            target_binary.inference_table_path,
            target_hurdle.inference_table_path,
            target_baseline.reporting_domain_mask.table_path
            if target_baseline.reporting_domain_mask is not None
            else None,
        ],
        output_paths=transfer_output_paths(outputs),
        reference_rows=len(reference_rows),
        binary_metric_rows=len(binary_metrics),
        binary_area_rows=len(binary_area_rows),
        hurdle_area_rows=len(hurdle_area_rows),
        comparison_rows=len(comparison_rows),
        primary_rows=len(primary_rows),
        calibrated_threshold=calibrated_thresholds.recommended_threshold,
    )
    LOGGER.info("Wrote transfer comparison: %s", outputs.model_comparison_path)
    LOGGER.info("Wrote transfer manifest: %s", outputs.manifest_path)
    return 0


def load_transfer_output_config(
    config_path: Path,
    transfer_name: str = DEFAULT_TRANSFER_NAME,
) -> TransferOutputConfig:
    """Load transfer sidecar paths and provenance labels from the target config."""
    config = load_yaml_config(config_path)
    models = require_mapping(config.get("models"), "models")
    transfer = require_mapping(models.get("transfer"), "models.transfer")
    entry = require_mapping(
        transfer.get(transfer_name),
        f"models.transfer.{transfer_name}",
    )
    return TransferOutputConfig(
        transfer_name=transfer_name,
        training_regime=str(entry.get("training_regime", DEFAULT_TRAINING_REGIME)),
        model_origin_region=str(entry.get("model_origin_region", DEFAULT_MODEL_ORIGIN_REGION)),
        evaluation_region=region_name(config),
        baseline_predictions_path=transfer_path(
            entry,
            transfer_name,
            "baseline_full_grid_predictions",
        ),
        baseline_manifest_path=transfer_path(entry, transfer_name, "baseline_prediction_manifest"),
        binary_predictions_path=transfer_path(entry, transfer_name, "binary_full_grid_predictions"),
        binary_manifest_path=transfer_path(entry, transfer_name, "binary_prediction_manifest"),
        binary_metrics_path=transfer_path(entry, transfer_name, "binary_metrics"),
        binary_area_summary_path=transfer_path(
            entry,
            transfer_name,
            "binary_full_grid_area_summary",
        ),
        hurdle_predictions_path=transfer_path(entry, transfer_name, "hurdle_full_grid_predictions"),
        hurdle_manifest_path=transfer_path(entry, transfer_name, "hurdle_prediction_manifest"),
        hurdle_metrics_path=transfer_path(entry, transfer_name, "hurdle_metrics"),
        hurdle_area_calibration_path=transfer_path(
            entry,
            transfer_name,
            "hurdle_area_calibration",
        ),
        hurdle_model_comparison_path=transfer_path(
            entry,
            transfer_name,
            "hurdle_model_comparison",
        ),
        hurdle_residual_by_observed_bin_path=transfer_path(
            entry,
            transfer_name,
            "hurdle_residual_by_observed_bin",
        ),
        hurdle_assumed_background_leakage_path=transfer_path(
            entry,
            transfer_name,
            "hurdle_assumed_background_leakage",
        ),
        hurdle_map_figure_path=optional_transfer_path(entry, "hurdle_map_figure"),
        reference_area_calibration_path=transfer_path(
            entry,
            transfer_name,
            "reference_area_calibration",
        ),
        model_comparison_path=transfer_path(entry, transfer_name, "model_comparison"),
        primary_summary_path=transfer_path(entry, transfer_name, "primary_summary"),
        manifest_path=transfer_path(entry, transfer_name, "manifest"),
    )


def transfer_path(entry: dict[str, Any], transfer_name: str, key: str) -> Path:
    """Read a required transfer path from config."""
    return Path(require_string(entry.get(key), f"models.transfer.{transfer_name}.{key}"))


def optional_transfer_path(entry: dict[str, Any], key: str) -> Path | None:
    """Read an optional transfer path from config."""
    value = entry.get(key)
    if value is None:
        return None
    return Path(require_string(value, f"models.transfer.{DEFAULT_TRANSFER_NAME}.{key}"))


def region_name(config: dict[str, Any]) -> str:
    """Return the configured region name, falling back to an unknown label."""
    region_value = config.get("region")
    if region_value is None:
        return "unknown"
    region = require_mapping(region_value, "region")
    return str(region.get("name", "unknown"))


def load_joblib_payload(path: Path) -> dict[str, Any]:
    """Load a serialized model payload and validate that it is a mapping."""
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        msg = f"model payload is not a dictionary: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def validate_feature_schema(
    payload: dict[str, Any],
    expected: tuple[str, ...],
    source_path: Path,
) -> None:
    """Validate that a frozen model payload uses the target feature schema."""
    payload_features = tuple(str(item) for item in payload.get("feature_columns", ()))
    if payload_features and payload_features != expected:
        msg = f"feature schema mismatch for {source_path}: {payload_features} vs target {expected}"
        raise ValueError(msg)


def calibrator_from_payload(payload: dict[str, Any]) -> BinaryCalibrator:
    """Build a BinaryCalibrator wrapper from a saved calibration payload."""
    return BinaryCalibrator(
        method=str(payload.get("calibration_method", "platt")),
        model=payload.get("calibrator"),
        status=str(payload.get("calibration_status", "loaded")),
        coefficient=float(payload.get("coefficient", math.nan)),
        intercept=float(payload.get("intercept", math.nan)),
    )


def thresholds_from_payload(
    payload: dict[str, Any],
    source_hurdle: Any,
) -> CalibrationThresholds:
    """Resolve the frozen calibrated threshold policy from source artifacts."""
    threshold = float(payload.get("recommended_threshold", source_hurdle.presence_threshold))
    policy = str(payload.get("recommended_policy", CALIBRATED_MAX_F1_POLICY))
    return CalibrationThresholds(
        recommended_threshold=threshold,
        recommended_policy=policy,
        rows=[],
        policy_thresholds={policy: (PLATT_PROBABILITY_SOURCE, threshold)},
    )


def read_calibrated_binary_predictions(
    predictions_path: Path,
    calibrator: BinaryCalibrator,
    probability_threshold: float,
    threshold_policy: str,
    calibration_method: str,
) -> pd.DataFrame:
    """Read raw binary predictions and attach frozen calibrated support decisions."""
    frame = pd.read_parquet(predictions_path)
    raw_probabilities = frame["pred_binary_probability"].to_numpy(dtype=float)
    calibrated = apply_binary_calibrator(calibrator, raw_probabilities)
    frame["calibration_method"] = calibration_method
    frame["probability_source"] = PLATT_PROBABILITY_SOURCE
    frame["threshold_policy"] = threshold_policy
    frame["probability_threshold"] = probability_threshold
    frame["calibrated_binary_probability"] = calibrated
    frame["pred_binary_class"] = calibrated >= probability_threshold
    return cast(pd.DataFrame, frame)


def build_calibrated_binary_metric_rows(
    predictions: pd.DataFrame,
    binary_config: Any,
    probability_threshold: float,
    threshold_policy: str,
    selection_split: str,
    selection_year: int,
) -> list[dict[str, object]]:
    """Build full-grid calibrated binary support metrics."""
    rows: list[dict[str, object]] = []
    rows.extend(
        grouped_calibrated_binary_metric_rows(
            predictions,
            binary_config,
            probability_threshold,
            threshold_policy,
            selection_split,
            selection_year,
            ["split", "year"],
        )
    )
    rows.extend(
        grouped_calibrated_binary_metric_rows(
            predictions,
            binary_config,
            probability_threshold,
            threshold_policy,
            selection_split,
            selection_year,
            ["split", "year", "label_source"],
        )
    )
    return rows


def grouped_calibrated_binary_metric_rows(
    predictions: pd.DataFrame,
    binary_config: Any,
    probability_threshold: float,
    threshold_policy: str,
    selection_split: str,
    selection_year: int,
    group_columns: list[str],
) -> list[dict[str, object]]:
    """Aggregate calibrated binary metrics for one grouping layout."""
    rows: list[dict[str, object]] = []
    frame = predictions.copy()
    if "label_source" not in frame.columns:
        frame["label_source"] = label_source_series(frame)
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values: dict[str, object] = {"split": "all", "year": "all", "label_source": "all"}
        for column, value in zip(group_columns, key_tuple, strict=True):
            group_values[column] = normalized_group_value(value)
        rows.append(
            calibrated_binary_metric_row(
                group,
                binary_config,
                probability_threshold,
                threshold_policy,
                selection_split,
                selection_year,
                group_values,
            )
        )
    return rows


def calibrated_binary_metric_row(
    group: pd.DataFrame,
    binary_config: Any,
    probability_threshold: float,
    threshold_policy: str,
    selection_split: str,
    selection_year: int,
    group_values: dict[str, object],
) -> dict[str, object]:
    """Build one calibrated binary support metric row."""
    probabilities = group["calibrated_binary_probability"].to_numpy(dtype=float)
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
    positive_count = int(np.count_nonzero(observed))
    negative_count = int(observed.size - positive_count)
    predicted_positive_count = int(np.count_nonzero(predicted))
    assumed_background_count = int(np.count_nonzero(assumed_background))
    assumed_background_false_positive = assumed_background & false_positive
    precision = safe_ratio(float(np.count_nonzero(true_positive)), float(predicted_positive_count))
    recall = safe_ratio(float(np.count_nonzero(true_positive)), float(positive_count))
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "model_name": BINARY_MODEL_NAME,
        "target_label": binary_config.target_label,
        "target_threshold_fraction": binary_config.target_threshold_fraction,
        "target_threshold_area": binary_config.target_threshold_area,
        "selection_split": selection_split,
        "selection_year": selection_year,
        "split": group_values["split"],
        "year": group_values["year"],
        "label_source": group_values["label_source"],
        "mask_status": mask_status(binary_config.reporting_domain_mask),
        "evaluation_scope": evaluation_scope(binary_config.reporting_domain_mask),
        "row_count": int(len(valid_group)),
        "positive_count": positive_count,
        "positive_rate": safe_ratio(float(positive_count), float(observed.size)),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": safe_ratio(
            float(predicted_positive_count),
            float(predicted.size),
        ),
        "probability_threshold": probability_threshold,
        "auroc": binary_auroc(observed, probabilities[valid_mask]),
        "auprc": binary_auprc(observed, probabilities[valid_mask]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": int(np.count_nonzero(true_positive)),
        "false_positive_count": int(np.count_nonzero(false_positive)),
        "false_positive_rate": safe_ratio(float(np.count_nonzero(false_positive)), negative_count),
        "false_negative_count": int(np.count_nonzero(false_negative)),
        "false_negative_rate": safe_ratio(float(np.count_nonzero(false_negative)), positive_count),
        "true_negative_count": int(np.count_nonzero(true_negative)),
        "assumed_background_count": assumed_background_count,
        "assumed_background_false_positive_count": int(
            np.count_nonzero(assumed_background_false_positive)
        ),
        "assumed_background_false_positive_rate": safe_ratio(
            float(np.count_nonzero(assumed_background_false_positive)),
            assumed_background_count,
        ),
        "threshold_policy": threshold_policy,
    }


def build_transfer_model_comparison(
    *,
    reference_rows: list[dict[str, object]],
    hurdle_area_rows: list[dict[str, object]],
    outputs: TransferOutputConfig,
) -> list[dict[str, object]]:
    """Build the compact transfer comparison table."""
    rows: list[dict[str, object]] = []
    for row in reference_rows:
        rows.append(transfer_comparison_row(row, outputs, outputs.reference_area_calibration_path))
    for row in hurdle_area_rows:
        rows.append(transfer_comparison_row(row, outputs, outputs.hurdle_area_calibration_path))
    return rows


def transfer_comparison_row(
    row: dict[str, object],
    outputs: TransferOutputConfig,
    source_table: Path,
) -> dict[str, object]:
    """Convert one area-calibration row into transfer comparison schema."""
    model_name = str(row.get("model_name", ""))
    return {
        "training_regime": outputs.training_regime,
        "model_origin_region": outputs.model_origin_region,
        "evaluation_region": outputs.evaluation_region,
        "model_name": model_name,
        "model_family": transfer_model_family(model_name),
        "composition_policy": row.get("composition_policy", ""),
        "split": row.get("split", ""),
        "year": row.get("year", ""),
        "mask_status": row.get("mask_status", ""),
        "evaluation_scope": row.get("evaluation_scope", ""),
        "label_source": row.get("label_source", ""),
        "row_count": row.get("row_count", ""),
        "mae": row.get("mae", math.nan),
        "rmse": row.get("rmse", math.nan),
        "r2": row.get("r2", math.nan),
        "f1_ge_10pct": row.get("f1_ge_10pct", math.nan),
        "observed_canopy_area": row.get("observed_canopy_area", math.nan),
        "predicted_canopy_area": row.get("predicted_canopy_area", math.nan),
        "area_pct_bias": row.get("area_pct_bias", math.nan),
        "source_table": str(source_table),
    }


def transfer_model_family(model_name: str) -> str:
    """Return a compact model-family label for transfer comparison rows."""
    if model_name in {NO_SKILL_MODEL_NAME, PREVIOUS_YEAR_MODEL_NAME, CLIMATOLOGY_MODEL_NAME}:
        return "reference_baseline"
    if model_name == GEOGRAPHIC_MODEL_NAME:
        return "geographic_reference"
    if model_name == RIDGE_MODEL_NAME:
        return "aef_ridge_transfer"
    if model_name.startswith("calibrated_"):
        return "hurdle_transfer"
    return "unknown"


def primary_transfer_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return the primary Big Sur held-out rows from the compact comparison."""
    output = []
    for row in rows:
        if (
            str(row.get("split")) == "test"
            and str(row.get("year")) == "2022"
            and str(row.get("evaluation_scope")) == "full_grid_masked"
            and str(row.get("label_source")) == "all"
        ):
            output.append(row)
    return output


def read_csv_dicts(path: Path) -> list[dict[str, object]]:
    """Read CSV rows as dictionaries when the path exists."""
    if not path.exists():
        return []
    return cast(list[dict[str, object]], pd.read_csv(path).to_dict("records"))


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


def write_binary_transfer_manifest(
    *,
    outputs: TransferOutputConfig,
    source_binary_model_path: Path,
    source_calibration_model_path: Path,
    source_threshold_path: Path,
    raw_threshold: float,
    calibrated_thresholds: CalibrationThresholds,
    prediction_rows: int,
    metric_rows: int,
    area_rows: int,
) -> None:
    """Write a binary-transfer manifest after calibrated summaries are complete."""
    payload = {
        "command": "evaluate-transfer",
        "transfer_name": outputs.transfer_name,
        "training_regime": outputs.training_regime,
        "model_origin_region": outputs.model_origin_region,
        "evaluation_region": outputs.evaluation_region,
        "source_binary_model": str(source_binary_model_path),
        "source_binary_calibration_model": str(source_calibration_model_path),
        "source_binary_threshold_selection": str(source_threshold_path),
        "raw_probability_threshold": raw_threshold,
        "calibrated_threshold_policy": calibrated_thresholds.recommended_policy,
        "calibrated_probability_threshold": calibrated_thresholds.recommended_threshold,
        "probability_source": PLATT_PROBABILITY_SOURCE,
        "outputs": {
            "binary_full_grid_predictions": str(outputs.binary_predictions_path),
            "binary_metrics": str(outputs.binary_metrics_path),
            "binary_full_grid_area_summary": str(outputs.binary_area_summary_path),
            "binary_prediction_manifest": str(outputs.binary_manifest_path),
        },
        "row_counts": {
            "prediction_rows": prediction_rows,
            "metric_rows": metric_rows,
            "area_summary_rows": area_rows,
        },
        "refit_binary_presence_model": False,
        "refit_binary_calibrator": False,
        "test_rows_used_for_training_calibration_or_threshold_selection": False,
    }
    write_json(outputs.binary_manifest_path, payload)


def write_transfer_manifest(
    *,
    outputs: TransferOutputConfig,
    config_path: Path,
    source_config_path: Path,
    source_paths: list[Path | None],
    target_paths: list[Path | None],
    output_paths: list[Path | None],
    reference_rows: int,
    binary_metric_rows: int,
    binary_area_rows: int,
    hurdle_area_rows: int,
    comparison_rows: int,
    primary_rows: int,
    calibrated_threshold: float,
) -> None:
    """Write the overall transfer evaluation manifest."""
    payload = {
        "command": "evaluate-transfer",
        "config_path": str(config_path),
        "source_config_path": str(source_config_path),
        "transfer_name": outputs.transfer_name,
        "training_regime": outputs.training_regime,
        "model_origin_region": outputs.model_origin_region,
        "evaluation_region": outputs.evaluation_region,
        "mask_status": "plausible_kelp_domain",
        "evaluation_scope": "full_grid_masked",
        "primary_split": "test",
        "primary_year": 2022,
        "primary_label_source": "all",
        "calibrated_probability_threshold": calibrated_threshold,
        "inputs": {
            "source_artifacts": [path_metadata(path) for path in source_paths if path is not None],
            "target_artifacts": [path_metadata(path) for path in target_paths if path is not None],
        },
        "outputs": [path_metadata(path) for path in output_paths if path is not None],
        "row_counts": {
            "reference_area_rows": reference_rows,
            "binary_metric_rows": binary_metric_rows,
            "binary_area_rows": binary_area_rows,
            "hurdle_area_rows": hurdle_area_rows,
            "comparison_rows": comparison_rows,
            "primary_summary_rows": primary_rows,
        },
        "refit_aef_ridge_model": False,
        "refit_binary_presence_model": False,
        "refit_binary_calibrator": False,
        "refit_conditional_canopy_model": False,
        "test_rows_used_for_training_calibration_or_threshold_selection": False,
    }
    write_json(outputs.manifest_path, payload)


def transfer_output_paths(outputs: TransferOutputConfig) -> list[Path | None]:
    """Return all configured transfer output paths for manifest metadata."""
    return [
        outputs.baseline_predictions_path,
        outputs.baseline_manifest_path,
        outputs.binary_predictions_path,
        outputs.binary_manifest_path,
        outputs.binary_metrics_path,
        outputs.binary_area_summary_path,
        outputs.hurdle_predictions_path,
        outputs.hurdle_manifest_path,
        outputs.hurdle_metrics_path,
        outputs.hurdle_area_calibration_path,
        outputs.hurdle_model_comparison_path,
        outputs.hurdle_residual_by_observed_bin_path,
        outputs.hurdle_assumed_background_leakage_path,
        outputs.hurdle_map_figure_path,
        outputs.reference_area_calibration_path,
        outputs.model_comparison_path,
        outputs.primary_summary_path,
        outputs.manifest_path,
    ]


def path_metadata(path: Path) -> dict[str, object]:
    """Return existence, size, and mtime metadata for a manifest path."""
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "file_size_bytes": stat.st_size if stat is not None and path.is_file() else None,
        "modified_time_ns": stat.st_mtime_ns if stat is not None else None,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON object with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
