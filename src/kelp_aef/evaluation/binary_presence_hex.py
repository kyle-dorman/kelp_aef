"""Build 1 km hex-aggregate maps for pooled binary-presence diagnostics."""
# mypy: disable-error-code="no-untyped-call,no-any-return"

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd  # type: ignore[import-untyped]
import joblib  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import RegularPolygon  # noqa: E402
from rasterio.warp import transform as warp_transform  # type: ignore[import-untyped]
from shapely.geometry import Polygon

from kelp_aef.config import require_mapping, require_string
from kelp_aef.evaluation.baselines import safe_ratio
from kelp_aef.evaluation.binary_presence import BinaryCalibrator, apply_binary_calibrator

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE_CRS = "EPSG:4326"
DEFAULT_TARGET_CRS = "EPSG:32610"
DEFAULT_HEX_FLAT_DIAMETER_M = 1000.0
DEFAULT_RATE_CLIP_QUANTILE = 0.98
DEFAULT_DIFFERENCE_CLIP_QUANTILE = 0.98
DEFAULT_THRESHOLD_POLICY = "validation_max_f1_calibrated"
DEFAULT_OBSERVED_COLUMN = "binary_observed_y"
DEFAULT_PROBABILITY_COLUMN = "pred_binary_probability"
DEFAULT_RATE_COLOR_MIN = 0.01
DEFAULT_DIFFERENCE_COLOR_MIN = 0.01
DEFAULT_BACKGROUND_COLOR = "#dceff7"
DEFAULT_HEX_EDGE_COLOR = "#f8fafc"
DEFAULT_COASTLINE_COLOR = "#111827"
DEFAULT_COASTLINE_LINEWIDTH = 0.6
POOLED_TRAINING_REGIME = "pooled_monterey_big_sur"
BINARY_HEX_FIELDS = (
    "evaluation_region",
    "training_regime",
    "model_origin_region",
    "split",
    "year",
    "mask_status",
    "evaluation_scope",
    "label_source",
    "target_label",
    "probability_source",
    "threshold_policy",
    "probability_threshold",
    "target_crs",
    "hex_flat_diameter_m",
    "hex_id",
    "hex_q",
    "hex_r",
    "hex_center_x_m",
    "hex_center_y_m",
    "hex_center_longitude",
    "hex_center_latitude",
    "hex_geometry_wkt",
    "n_cells",
    "observed_positive_count",
    "observed_positive_rate",
    "mean_calibrated_binary_probability",
    "predicted_positive_count",
    "predicted_positive_rate",
    "true_positive_count",
    "false_positive_count",
    "false_positive_rate",
    "false_negative_count",
    "false_negative_rate",
    "true_negative_count",
    "predicted_minus_observed_positive_rate",
)


@dataclass(frozen=True)
class BinaryPresenceHexInput:
    """Configured binary prediction input for one pooled evaluation region."""

    context_id: str
    binary_predictions_path: Path
    binary_calibration_model_path: Path
    training_regime: str
    model_origin_region: str
    evaluation_region: str
    threshold_policy: str
    required: bool


@dataclass(frozen=True)
class BinaryPresenceHexMapConfig:
    """Resolved settings for the Phase 2 pooled binary hex diagnostic."""

    inputs: tuple[BinaryPresenceHexInput, ...]
    figure_path: Path
    table_path: Path
    manifest_path: Path
    primary_split: str
    primary_year: int
    primary_mask_status: str
    primary_evaluation_scope: str
    primary_label_source: str
    source_crs: str
    target_crs: str
    hex_flat_diameter_m: float
    observed_column: str
    probability_column: str
    rate_clip_quantile: float
    difference_clip_quantile: float
    background_color: str
    hex_edge_color: str
    coastline_enabled: bool
    coastline_source_manifest_paths: tuple[Path, ...]
    coastline_color: str
    coastline_linewidth: float


@dataclass(frozen=True)
class BinaryPresenceHexMapTables:
    """Aggregated hex rows and diagnostics for the Phase 2 map outputs."""

    rows: list[dict[str, object]]
    input_row_counts: dict[str, int]
    color_scales: dict[str, object]


def load_binary_presence_hex_map_config(
    comparison: dict[str, Any],
    outputs: dict[str, Any],
    figures_dir: Path,
    tables_dir: Path,
    config_path: Path,
    *,
    primary_split: str,
    primary_year: int,
    primary_mask_status: str,
    primary_evaluation_scope: str,
    primary_label_source: str,
) -> BinaryPresenceHexMapConfig | None:
    """Load optional pooled binary hex-map diagnostics from the Phase 2 config."""
    settings = optional_mapping(
        comparison.get("pooled_binary_hex_map"),
        "training_regime_comparison.pooled_binary_hex_map",
    )
    if not settings:
        return None
    inputs = require_mapping(
        settings.get("inputs"),
        "training_regime_comparison.pooled_binary_hex_map.inputs",
    )
    return BinaryPresenceHexMapConfig(
        inputs=tuple(
            binary_presence_hex_input(name, value, config_path)
            for name, value in sorted(inputs.items(), key=lambda item: str(item[0]))
        ),
        figure_path=hex_output_path(
            settings,
            outputs,
            "figure",
            "model_analysis_phase2_pooled_binary_hex_map_figure",
            figures_dir / "monterey_big_sur_pooled_binary_presence_hex_1km.png",
            config_path,
        ),
        table_path=hex_output_path(
            settings,
            outputs,
            "table",
            "model_analysis_phase2_pooled_binary_hex_map_table",
            tables_dir / "monterey_big_sur_pooled_binary_presence_hex_1km.csv",
            config_path,
        ),
        manifest_path=hex_output_path(
            settings,
            outputs,
            "manifest",
            "model_analysis_phase2_pooled_binary_hex_map_manifest",
            tables_dir.parent.parent
            / "interim/monterey_big_sur_pooled_binary_presence_hex_manifest.json",
            config_path,
        ),
        primary_split=primary_split,
        primary_year=primary_year,
        primary_mask_status=primary_mask_status,
        primary_evaluation_scope=primary_evaluation_scope,
        primary_label_source=primary_label_source,
        source_crs=str(settings.get("source_crs", DEFAULT_SOURCE_CRS)),
        target_crs=str(settings.get("target_crs", DEFAULT_TARGET_CRS)),
        hex_flat_diameter_m=positive_float(
            settings.get("hex_flat_diameter_m", DEFAULT_HEX_FLAT_DIAMETER_M),
            "training_regime_comparison.pooled_binary_hex_map.hex_flat_diameter_m",
        ),
        observed_column=str(settings.get("observed_column", DEFAULT_OBSERVED_COLUMN)),
        probability_column=str(settings.get("probability_column", DEFAULT_PROBABILITY_COLUMN)),
        rate_clip_quantile=unit_interval_float(
            settings.get("rate_clip_quantile", DEFAULT_RATE_CLIP_QUANTILE),
            "training_regime_comparison.pooled_binary_hex_map.rate_clip_quantile",
        ),
        difference_clip_quantile=unit_interval_float(
            settings.get("difference_clip_quantile", DEFAULT_DIFFERENCE_CLIP_QUANTILE),
            "training_regime_comparison.pooled_binary_hex_map.difference_clip_quantile",
        ),
        background_color=str(settings.get("background_color", DEFAULT_BACKGROUND_COLOR)),
        hex_edge_color=str(settings.get("hex_edge_color", DEFAULT_HEX_EDGE_COLOR)),
        coastline_enabled=optional_bool(
            optional_mapping(
                settings.get("coastline"),
                "training_regime_comparison.pooled_binary_hex_map.coastline",
            ).get("enabled"),
            "training_regime_comparison.pooled_binary_hex_map.coastline.enabled",
            False,
        ),
        coastline_source_manifest_paths=coastline_source_manifest_paths(
            settings,
            config_path,
        ),
        coastline_color=str(
            optional_mapping(
                settings.get("coastline"),
                "training_regime_comparison.pooled_binary_hex_map.coastline",
            ).get("color", DEFAULT_COASTLINE_COLOR)
        ),
        coastline_linewidth=positive_float(
            optional_mapping(
                settings.get("coastline"),
                "training_regime_comparison.pooled_binary_hex_map.coastline",
            ).get("linewidth", DEFAULT_COASTLINE_LINEWIDTH),
            "training_regime_comparison.pooled_binary_hex_map.coastline.linewidth",
        ),
    )


def binary_presence_hex_input(
    name: object,
    value: object,
    config_path: Path,
) -> BinaryPresenceHexInput:
    """Load one pooled binary hex-map input entry."""
    context_id = str(name)
    entry = require_mapping(
        value,
        f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}",
    )
    return BinaryPresenceHexInput(
        context_id=context_id,
        binary_predictions_path=config_relative_path(
            entry.get("binary_predictions"),
            f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}."
            "binary_predictions",
            config_path,
        ),
        binary_calibration_model_path=config_relative_path(
            entry.get("binary_calibration_model"),
            f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}."
            "binary_calibration_model",
            config_path,
        ),
        training_regime=require_string(
            entry.get("training_regime"),
            f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}.training_regime",
        ),
        model_origin_region=require_string(
            entry.get("model_origin_region"),
            f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}."
            "model_origin_region",
        ),
        evaluation_region=require_string(
            entry.get("evaluation_region"),
            f"training_regime_comparison.pooled_binary_hex_map.inputs.{context_id}."
            "evaluation_region",
        ),
        threshold_policy=str(entry.get("threshold_policy", DEFAULT_THRESHOLD_POLICY)),
        required=bool(entry.get("required", True)),
    )


def hex_output_path(
    settings: dict[str, Any],
    outputs: dict[str, Any],
    setting_key: str,
    output_key: str,
    default: Path,
    config_path: Path,
) -> Path:
    """Resolve one binary hex-map output path from local or report outputs."""
    value = settings.get(setting_key, outputs.get(output_key))
    if value is None:
        return default
    return config_relative_path(value, setting_key, config_path)


def config_relative_path(value: object, name: str, config_path: Path) -> Path:
    """Read and resolve one path-like config value."""
    path = Path(require_string(value, name))
    if path.is_absolute():
        return path
    config_relative = config_path.parent / path
    if path.exists() and not config_relative.exists():
        return path
    return config_relative


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return a dynamic mapping or an empty mapping when omitted."""
    if value is None:
        return {}
    return require_mapping(value, name)


def optional_bool(value: object, name: str, default: bool) -> bool:
    """Read an optional boolean config value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"config field must be a boolean: {name}"
        raise ValueError(msg)
    return value


def coastline_source_manifest_paths(
    settings: dict[str, Any],
    config_path: Path,
) -> tuple[Path, ...]:
    """Read optional CUSP source manifest paths for coastline overlays."""
    coastline = optional_mapping(
        settings.get("coastline"),
        "training_regime_comparison.pooled_binary_hex_map.coastline",
    )
    values = coastline.get("source_manifests", [])
    if values is None:
        return ()
    if not isinstance(values, list):
        msg = (
            "config field must be a list: "
            "training_regime_comparison.pooled_binary_hex_map.coastline.source_manifests"
        )
        raise ValueError(msg)
    return tuple(
        config_relative_path(
            value,
            "training_regime_comparison.pooled_binary_hex_map.coastline.source_manifests",
            config_path,
        )
        for value in values
    )


def positive_float(value: object, name: str) -> float:
    """Read a positive floating-point config value."""
    parsed = float(cast(Any, value))
    if parsed <= 0:
        msg = f"config field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def unit_interval_float(value: object, name: str) -> float:
    """Read a floating-point config value in the closed unit interval."""
    parsed = float(cast(Any, value))
    if parsed < 0.0 or parsed > 1.0:
        msg = f"config field must be between 0 and 1: {name}"
        raise ValueError(msg)
    return parsed


def build_binary_presence_hex_map_tables(
    config: BinaryPresenceHexMapConfig,
) -> BinaryPresenceHexMapTables:
    """Build all pooled binary hex-map rows from configured inputs."""
    rows: list[dict[str, object]] = []
    input_row_counts: dict[str, int] = {}
    for input_config in config.inputs:
        frame = read_binary_presence_hex_frame(input_config, config)
        input_row_counts[input_config.context_id] = int(len(frame))
        if frame.empty:
            continue
        rows.extend(aggregate_binary_presence_hex_frame(frame, input_config, config))
    rows = sorted(rows, key=binary_presence_hex_sort_key)
    color_scales = binary_presence_hex_color_scales(rows, config)
    return BinaryPresenceHexMapTables(rows, input_row_counts, color_scales)


def read_binary_presence_hex_frame(
    input_config: BinaryPresenceHexInput,
    config: BinaryPresenceHexMapConfig,
) -> pd.DataFrame:
    """Read and calibrate primary split/year rows for one binary hex input."""
    path = input_config.binary_predictions_path
    if not path.exists():
        if input_config.required:
            msg = f"required pooled binary hex-map input is missing: {path}"
            raise FileNotFoundError(msg)
        return pd.DataFrame()
    dataset = ds.dataset(path, format="parquet")
    available = set(dataset.schema.names)
    columns = [
        column
        for column in (
            "split",
            "year",
            "longitude",
            "latitude",
            "label_source",
            "is_plausible_kelp_domain",
            "target_label",
            "aef_grid_cell_id",
            "aef_grid_row",
            "aef_grid_col",
            config.observed_column,
            config.probability_column,
        )
        if column in available
    ]
    expression = (dataset_field("split") == config.primary_split) & (
        dataset_field("year") == config.primary_year
    )
    if config.primary_label_source != "all" and "label_source" in available:
        expression = expression & (dataset_field("label_source") == config.primary_label_source)
    if (
        config.primary_mask_status == "plausible_kelp_domain"
        and "is_plausible_kelp_domain" in available
    ):
        expression = expression & (dataset_field("is_plausible_kelp_domain") == True)  # noqa: E712
    frame = cast(pd.DataFrame, dataset.to_table(columns=columns, filter=expression).to_pandas())
    required = {
        "split",
        "year",
        "longitude",
        "latitude",
        config.observed_column,
        config.probability_column,
    }
    if not required.issubset(frame.columns):
        msg = f"pooled binary hex-map input is missing required columns: {path}"
        raise ValueError(msg)
    if "label_source" not in frame.columns:
        frame["label_source"] = config.primary_label_source
    calibrated = calibrated_binary_probabilities(frame, input_config, config)
    frame["calibrated_binary_probability"] = calibrated.probabilities
    frame["calibrated_pred_binary_class"] = calibrated.predicted
    frame["probability_source"] = calibrated.probability_source
    frame["probability_threshold"] = calibrated.probability_threshold
    frame["threshold_policy"] = input_config.threshold_policy
    return valid_hex_input_rows(frame, config)


def dataset_field(name: str) -> Any:
    """Return a PyArrow dataset field expression with a typed wrapper."""
    return cast(Any, ds).field(name)


@dataclass(frozen=True)
class CalibratedBinaryProbabilities:
    """Calibrated probability arrays and threshold metadata for one input."""

    probabilities: np.ndarray
    predicted: np.ndarray
    probability_source: str
    probability_threshold: float


def calibrated_binary_probabilities(
    frame: pd.DataFrame,
    input_config: BinaryPresenceHexInput,
    config: BinaryPresenceHexMapConfig,
) -> CalibratedBinaryProbabilities:
    """Apply the configured calibration policy to raw binary probabilities."""
    payload = cast(dict[str, Any], joblib.load(input_config.binary_calibration_model_path))
    probability_source, threshold = binary_policy_threshold(payload, input_config.threshold_policy)
    raw_probabilities = frame[config.probability_column].to_numpy(dtype=float)
    probabilities = (
        raw_probabilities
        if probability_source == "raw_logistic"
        else apply_binary_calibrator(binary_calibrator_from_payload(payload), raw_probabilities)
    )
    predicted = probabilities >= float(threshold)
    return CalibratedBinaryProbabilities(
        probabilities=probabilities,
        predicted=predicted,
        probability_source=probability_source,
        probability_threshold=float(threshold),
    )


def binary_policy_threshold(
    payload: dict[str, Any],
    threshold_policy: str,
) -> tuple[str, float]:
    """Return the probability source and threshold for a calibration policy."""
    policy_thresholds = cast(dict[str, tuple[str, float]], payload.get("policy_thresholds", {}))
    if threshold_policy not in policy_thresholds:
        msg = f"binary calibration payload is missing threshold policy {threshold_policy}"
        raise ValueError(msg)
    source, threshold = policy_thresholds[threshold_policy]
    return str(source), float(threshold)


def binary_calibrator_from_payload(payload: dict[str, Any]) -> BinaryCalibrator:
    """Build a binary calibrator object from a serialized calibration payload."""
    return BinaryCalibrator(
        method=str(payload.get("calibration_method", "platt")),
        model=payload.get("calibrator"),
        status=str(payload.get("calibration_status", "")),
        coefficient=object_to_float(payload.get("coefficient")),
        intercept=object_to_float(payload.get("intercept")),
    )


def object_to_float(value: object) -> float:
    """Convert a dynamic payload scalar to float, returning NaN when missing."""
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return math.nan


def object_to_int(value: object) -> int:
    """Convert a dynamic payload scalar to int, returning zero when missing."""
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return 0


def valid_hex_input_rows(
    frame: pd.DataFrame,
    config: BinaryPresenceHexMapConfig,
) -> pd.DataFrame:
    """Drop rows that cannot be assigned to a finite projected hex."""
    valid = (
        np.isfinite(frame["longitude"].to_numpy(dtype=float))
        & np.isfinite(frame["latitude"].to_numpy(dtype=float))
        & np.isfinite(frame["calibrated_binary_probability"].to_numpy(dtype=float))
    )
    output = frame.loc[valid].copy()
    if config.primary_label_source != "all" and "label_source" in output.columns:
        source_matches = output["label_source"].astype(str) == config.primary_label_source
        output = output.loc[source_matches].copy()
    return output.reset_index(drop=True)


def aggregate_binary_presence_hex_frame(
    frame: pd.DataFrame,
    input_config: BinaryPresenceHexInput,
    config: BinaryPresenceHexMapConfig,
) -> list[dict[str, object]]:
    """Aggregate one calibrated binary frame into flat-top projected hex rows."""
    xs, ys = project_longitude_latitude(
        frame["longitude"].to_numpy(dtype=float),
        frame["latitude"].to_numpy(dtype=float),
        config,
    )
    hex_q, hex_r = hex_indices_for_projected_points(xs, ys, config.hex_flat_diameter_m)
    working = pd.DataFrame(
        {
            "hex_q": hex_q,
            "hex_r": hex_r,
            "observed": frame[config.observed_column].to_numpy(dtype=bool),
            "probability": frame["calibrated_binary_probability"].to_numpy(dtype=float),
            "predicted": frame["calibrated_pred_binary_class"].to_numpy(dtype=bool),
        }
    )
    working["true_positive"] = working["observed"] & working["predicted"]
    working["false_positive"] = ~working["observed"] & working["predicted"]
    working["false_negative"] = working["observed"] & ~working["predicted"]
    working["true_negative"] = ~working["observed"] & ~working["predicted"]
    grouped = (
        working.groupby(["hex_q", "hex_r"], sort=True)
        .agg(
            n_cells=("observed", "size"),
            observed_positive_count=("observed", "sum"),
            mean_calibrated_binary_probability=("probability", "mean"),
            predicted_positive_count=("predicted", "sum"),
            true_positive_count=("true_positive", "sum"),
            false_positive_count=("false_positive", "sum"),
            false_negative_count=("false_negative", "sum"),
            true_negative_count=("true_negative", "sum"),
        )
        .reset_index()
    )
    center_x, center_y = hex_centers(
        grouped["hex_q"].to_numpy(dtype=np.int64),
        grouped["hex_r"].to_numpy(dtype=np.int64),
        config.hex_flat_diameter_m,
    )
    center_longitudes, center_latitudes = longitude_latitude_for_projected_points(
        center_x,
        center_y,
        config,
    )
    target_label = first_string_value(frame, "target_label", "annual_max_ge_10pct")
    rows: list[dict[str, object]] = []
    for index, row in grouped.iterrows():
        n_cells = int(row["n_cells"])
        observed_positive_count = int(row["observed_positive_count"])
        predicted_positive_count = int(row["predicted_positive_count"])
        false_positive_count = int(row["false_positive_count"])
        false_negative_count = int(row["false_negative_count"])
        observed_negative_count = n_cells - observed_positive_count
        observed_positive_rate = safe_ratio(observed_positive_count, n_cells)
        predicted_positive_rate = safe_ratio(predicted_positive_count, n_cells)
        q = int(row["hex_q"])
        r = int(row["hex_r"])
        rows.append(
            {
                "evaluation_region": input_config.evaluation_region,
                "training_regime": input_config.training_regime,
                "model_origin_region": input_config.model_origin_region,
                "split": config.primary_split,
                "year": config.primary_year,
                "mask_status": config.primary_mask_status,
                "evaluation_scope": config.primary_evaluation_scope,
                "label_source": config.primary_label_source,
                "target_label": target_label,
                "probability_source": str(frame["probability_source"].iloc[0]),
                "threshold_policy": input_config.threshold_policy,
                "probability_threshold": float(frame["probability_threshold"].iloc[0]),
                "target_crs": config.target_crs,
                "hex_flat_diameter_m": config.hex_flat_diameter_m,
                "hex_id": f"{input_config.evaluation_region}:q{q}:r{r}",
                "hex_q": q,
                "hex_r": r,
                "hex_center_x_m": float(center_x[index]),
                "hex_center_y_m": float(center_y[index]),
                "hex_center_longitude": float(center_longitudes[index]),
                "hex_center_latitude": float(center_latitudes[index]),
                "hex_geometry_wkt": hex_polygon_wkt(
                    float(center_x[index]),
                    float(center_y[index]),
                    config.hex_flat_diameter_m,
                ),
                "n_cells": n_cells,
                "observed_positive_count": observed_positive_count,
                "observed_positive_rate": observed_positive_rate,
                "mean_calibrated_binary_probability": float(
                    row["mean_calibrated_binary_probability"]
                ),
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": predicted_positive_rate,
                "true_positive_count": int(row["true_positive_count"]),
                "false_positive_count": false_positive_count,
                "false_positive_rate": safe_ratio(false_positive_count, observed_negative_count),
                "false_negative_count": false_negative_count,
                "false_negative_rate": safe_ratio(false_negative_count, observed_positive_count),
                "true_negative_count": int(row["true_negative_count"]),
                "predicted_minus_observed_positive_rate": (
                    predicted_positive_rate - observed_positive_rate
                ),
            }
        )
    return rows


def project_longitude_latitude(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    config: BinaryPresenceHexMapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform longitude/latitude arrays into the configured projected CRS."""
    xs, ys = warp_transform(
        config.source_crs,
        config.target_crs,
        longitudes.tolist(),
        latitudes.tolist(),
    )
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def longitude_latitude_for_projected_points(
    xs: np.ndarray,
    ys: np.ndarray,
    config: BinaryPresenceHexMapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform projected coordinate arrays back to longitude/latitude."""
    longitudes, latitudes = warp_transform(
        config.target_crs,
        config.source_crs,
        xs.tolist(),
        ys.tolist(),
    )
    return np.asarray(longitudes, dtype=float), np.asarray(latitudes, dtype=float)


def hex_indices_for_projected_points(
    xs: np.ndarray,
    ys: np.ndarray,
    flat_diameter_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign projected points to deterministic flat-top hex axial indices."""
    side_length = hex_side_length(flat_diameter_m)
    fractional_q = (2.0 / 3.0 * xs) / side_length
    fractional_r = (-1.0 / 3.0 * xs + math.sqrt(3.0) / 3.0 * ys) / side_length
    return cube_round_axial(fractional_q, fractional_r)


def cube_round_axial(q_values: np.ndarray, r_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round fractional axial coordinates to the nearest hex coordinate."""
    x_values = q_values
    z_values = r_values
    y_values = -x_values - z_values
    rounded_x = np.rint(x_values)
    rounded_y = np.rint(y_values)
    rounded_z = np.rint(z_values)
    x_diff = np.abs(rounded_x - x_values)
    y_diff = np.abs(rounded_y - y_values)
    z_diff = np.abs(rounded_z - z_values)
    adjust_x = (x_diff > y_diff) & (x_diff > z_diff)
    adjust_y = (~adjust_x) & (y_diff > z_diff)
    adjust_z = ~(adjust_x | adjust_y)
    rounded_x[adjust_x] = -rounded_y[adjust_x] - rounded_z[adjust_x]
    rounded_y[adjust_y] = -rounded_x[adjust_y] - rounded_z[adjust_y]
    rounded_z[adjust_z] = -rounded_x[adjust_z] - rounded_y[adjust_z]
    return rounded_x.astype(np.int64), rounded_z.astype(np.int64)


def hex_centers(
    q_values: np.ndarray,
    r_values: np.ndarray,
    flat_diameter_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected center coordinates for flat-top axial hex indices."""
    side_length = hex_side_length(flat_diameter_m)
    center_x = side_length * 1.5 * q_values.astype(float)
    center_y = flat_diameter_m * (r_values.astype(float) + q_values.astype(float) / 2.0)
    return center_x, center_y


def hex_side_length(flat_diameter_m: float) -> float:
    """Return side length for a regular hex with the configured across-flats width."""
    return flat_diameter_m / math.sqrt(3.0)


def hex_polygon_wkt(center_x: float, center_y: float, flat_diameter_m: float) -> str:
    """Return projected WKT for a flat-top regular hexagon."""
    side_length = hex_side_length(flat_diameter_m)
    vertices = [
        (
            center_x + side_length * math.cos(math.pi / 3.0 * vertex),
            center_y + side_length * math.sin(math.pi / 3.0 * vertex),
        )
        for vertex in range(6)
    ]
    return str(Polygon(vertices).wkt)


def first_string_value(frame: pd.DataFrame, column: str, default: str) -> str:
    """Return the first non-null string value from a DataFrame column."""
    if column not in frame.columns:
        return default
    values = frame[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) else default


def binary_presence_hex_sort_key(row: dict[str, object]) -> tuple[int, int, int]:
    """Sort hex rows by region then axial coordinate."""
    region_order = {"big_sur": 0, "monterey": 1}
    return (
        region_order.get(str(row.get("evaluation_region", "")), 99),
        object_to_int(row.get("hex_q", 0)),
        object_to_int(row.get("hex_r", 0)),
    )


def binary_presence_hex_color_scales(
    rows: list[dict[str, object]],
    config: BinaryPresenceHexMapConfig,
) -> dict[str, object]:
    """Return robust plotting clip values for binary hex-map panels."""
    rate_values = np.asarray(
        [
            object_to_float(row.get(column, math.nan))
            for row in rows
            for column in ("observed_positive_rate", "predicted_positive_rate")
        ],
        dtype=float,
    )
    difference_values = np.asarray(
        [
            object_to_float(row.get("predicted_minus_observed_positive_rate", math.nan))
            for row in rows
        ],
        dtype=float,
    )
    rate_max = robust_positive_clip(
        rate_values,
        config.rate_clip_quantile,
        DEFAULT_RATE_COLOR_MIN,
        1.0,
    )
    difference_abs_max = robust_positive_clip(
        np.abs(difference_values),
        config.difference_clip_quantile,
        DEFAULT_DIFFERENCE_COLOR_MIN,
        1.0,
    )
    return {
        "observed_and_predicted_positive_rate": {
            "vmin": 0.0,
            "vmax": rate_max,
            "clip_quantile": config.rate_clip_quantile,
            "clip": True,
        },
        "predicted_minus_observed_positive_rate": {
            "vmin": -difference_abs_max,
            "vcenter": 0.0,
            "vmax": difference_abs_max,
            "clip_quantile": config.difference_clip_quantile,
            "clip": True,
        },
    }


def robust_positive_clip(
    values: np.ndarray,
    quantile: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite positive color-scale cap from a robust quantile."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return minimum
    cap = float(np.nanquantile(finite, quantile))
    return max(minimum, min(maximum, cap))


def write_binary_presence_hex_map_outputs(
    tables: BinaryPresenceHexMapTables,
    config: BinaryPresenceHexMapConfig,
) -> None:
    """Write binary hex-map CSV, PNG, and manifest outputs."""
    write_binary_presence_hex_csv(tables.rows, config.table_path)
    write_binary_presence_hex_figure(tables.rows, tables.color_scales, config)
    write_binary_presence_hex_manifest(tables, config)


def write_binary_presence_hex_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write binary hex rows to CSV with a stable field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BINARY_HEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_binary_presence_hex_figure(
    rows: list[dict[str, object]],
    color_scales: dict[str, object],
    config: BinaryPresenceHexMapConfig,
) -> None:
    """Write the pooled binary 1 km hex summary figure."""
    config.figure_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        constrained_layout=True,
        facecolor=config.background_color,
    )
    metrics = (
        ("observed_positive_rate", "Observed positive rate"),
        ("predicted_positive_rate", "Predicted positive rate"),
        ("predicted_minus_observed_positive_rate", "Predicted - observed rate"),
    )
    region_rows = (("monterey", "Monterey"), ("big_sur", "Big Sur"))
    collections: dict[str, PatchCollection] = {}
    coastline_layer = load_coastline_layer(config)
    for row_index, (region, region_label) in enumerate(region_rows):
        region_frame = frame.loc[frame.get("evaluation_region", pd.Series(dtype=str)) == region]
        for column_index, (metric, metric_label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            collection = plot_hex_metric_panel(
                axis,
                region_frame,
                metric,
                metric_label,
                region_label,
                color_scales,
                config,
                coastline_layer,
            )
            if collection is not None:
                collections[metric] = collection
    for column_index, (metric, _) in enumerate(metrics):
        if metric in collections:
            colorbar = fig.colorbar(
                collections[metric],
                ax=axes[:, column_index].ravel().tolist(),
                shrink=0.78,
                extend="both" if metric.startswith("predicted_minus") else "max",
            )
            colorbar.ax.set_ylabel(metric.replace("_", " "))
    fig.suptitle(
        "Pooled Monterey+Big Sur binary support, 1 km EPSG:32610 hex summaries",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        (
            f"Rows: split={config.primary_split}, year={config.primary_year}, "
            f"mask={config.primary_mask_status}, label_source={config.primary_label_source}. "
            "Color scales are clipped for readability; CSV preserves true values."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(config.figure_path, dpi=180)
    plt.close(fig)
    LOGGER.info("Wrote pooled binary hex map: %s", config.figure_path)


def plot_hex_metric_panel(
    axis: Any,
    frame: pd.DataFrame,
    metric: str,
    metric_label: str,
    region_label: str,
    color_scales: dict[str, object],
    config: BinaryPresenceHexMapConfig,
    coastline_layer: gpd.GeoDataFrame | None,
) -> PatchCollection | None:
    """Draw one region/metric panel for the binary hex map."""
    axis.set_title(f"{region_label}: {metric_label}")
    axis.set_facecolor(config.background_color)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    if frame.empty:
        axis.text(0.5, 0.5, "No rows", ha="center", va="center", transform=axis.transAxes)
        return None
    side_length = hex_side_length(config.hex_flat_diameter_m)
    patches = [
        RegularPolygon(
            (float(row["hex_center_x_m"]), float(row["hex_center_y_m"])),
            numVertices=6,
            radius=side_length,
            orientation=0.0,
        )
        for _, row in frame.iterrows()
    ]
    values = frame[metric].to_numpy(dtype=float)
    clipped_values, norm, cmap = clipped_metric_values(values, metric, color_scales)
    collection = PatchCollection(patches, cmap=cmap, norm=norm, linewidths=0.08, zorder=2)
    collection.set_array(clipped_values)
    collection.set_edgecolor(config.hex_edge_color)
    axis.add_collection(collection)
    x_span = float(frame["hex_center_x_m"].max() - frame["hex_center_x_m"].min())
    y_span = float(frame["hex_center_y_m"].max() - frame["hex_center_y_m"].min())
    pad = max(config.hex_flat_diameter_m * 4.0, max(x_span, y_span) * 0.06)
    axis.set_xlim(
        float(frame["hex_center_x_m"].min()) - pad,
        float(frame["hex_center_x_m"].max()) + pad,
    )
    axis.set_ylim(
        float(frame["hex_center_y_m"].min()) - pad,
        float(frame["hex_center_y_m"].max()) + pad,
    )
    draw_coastline_layer(axis, coastline_layer, config)
    return collection


def load_coastline_layer(config: BinaryPresenceHexMapConfig) -> gpd.GeoDataFrame | None:
    """Load configured local shoreline vectors in the hex target CRS."""
    if not config.coastline_enabled or not config.coastline_source_manifest_paths:
        return None
    vector_paths = coastline_vector_paths(config.coastline_source_manifest_paths)
    frames = []
    for vector_path in vector_paths:
        if not vector_path.exists():
            LOGGER.warning("Skipping missing coastline vector source: %s", vector_path)
            continue
        frame = gpd.read_file(coastline_dataset_uri(vector_path))
        frames.append(frame)
    if not frames:
        return None
    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    return cast(gpd.GeoDataFrame, combined.to_crs(config.target_crs))


def coastline_vector_paths(manifest_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Read unique local vector paths from CUSP source manifests."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            LOGGER.warning("Skipping missing coastline source manifest: %s", manifest_path)
            continue
        with manifest_path.open() as file:
            payload = json.load(file)
        records = payload.get("records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            value = record.get("local_path")
            if value is None:
                continue
            vector_path = Path(str(value))
            if vector_path not in seen:
                seen.add(vector_path)
                paths.append(vector_path)
    return tuple(paths)


def coastline_dataset_uri(path: Path) -> str:
    """Return a GeoPandas-readable URI for a local vector source package."""
    if path.suffix.lower() == ".zip":
        return f"zip://{path}"
    return str(path)


def draw_coastline_layer(
    axis: Any,
    coastline_layer: gpd.GeoDataFrame | None,
    config: BinaryPresenceHexMapConfig,
) -> None:
    """Draw configured coastline lines clipped to the current panel extent."""
    if coastline_layer is None or coastline_layer.empty:
        return
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    subset = coastline_layer.cx[x_min:x_max, y_min:y_max]
    if subset.empty:
        return
    subset.plot(
        ax=axis,
        color=config.coastline_color,
        linewidth=config.coastline_linewidth,
        zorder=4,
    )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)


def clipped_metric_values(
    values: np.ndarray,
    metric: str,
    color_scales: dict[str, object],
) -> tuple[np.ndarray, Normalize | TwoSlopeNorm, str]:
    """Return clipped plot values, color normalization, and cmap name."""
    if metric == "predicted_minus_observed_positive_rate":
        scale = cast(dict[str, object], color_scales[metric])
        vmin = object_to_float(scale.get("vmin"))
        vmax = object_to_float(scale.get("vmax"))
        clipped = np.clip(values, vmin, vmax)
        return clipped, TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), "RdBu_r"
    scale = cast(dict[str, object], color_scales["observed_and_predicted_positive_rate"])
    vmin = object_to_float(scale.get("vmin"))
    vmax = object_to_float(scale.get("vmax"))
    clipped = np.clip(values, vmin, vmax)
    return clipped, Normalize(vmin=vmin, vmax=vmax, clip=True), "viridis"


def write_binary_presence_hex_manifest(
    tables: BinaryPresenceHexMapTables,
    config: BinaryPresenceHexMapConfig,
) -> None:
    """Write a sidecar manifest for the pooled binary hex-map diagnostic."""
    payload = {
        "diagnostic": "pooled_binary_presence_hex_map",
        "target_framing": "Kelpwatch-style annual maximum reproduction",
        "primary_filters": {
            "split": config.primary_split,
            "year": config.primary_year,
            "mask_status": config.primary_mask_status,
            "evaluation_scope": config.primary_evaluation_scope,
            "label_source": config.primary_label_source,
            "training_regime": POOLED_TRAINING_REGIME,
        },
        "hex_definition": {
            "orientation": "flat_top",
            "target_crs": config.target_crs,
            "source_crs": config.source_crs,
            "flat_to_flat_diameter_m": config.hex_flat_diameter_m,
            "side_length_m": hex_side_length(config.hex_flat_diameter_m),
            "horizontal_center_spacing_m": 1.5 * hex_side_length(config.hex_flat_diameter_m),
            "vertical_center_spacing_m": config.hex_flat_diameter_m,
            "origin_x_m": 0.0,
            "origin_y_m": 0.0,
            "assignment": "cube_round_nearest_axial_hex_center",
        },
        "columns": {
            "observed": config.observed_column,
            "raw_probability": config.probability_column,
            "calibrated_probability": "calibrated_binary_probability",
            "selected_class": "calibrated_pred_binary_class",
        },
        "visual_style": {
            "background_color": config.background_color,
            "hex_edge_color": config.hex_edge_color,
            "coastline_enabled": config.coastline_enabled,
            "coastline_source_manifests": [
                str(path) for path in config.coastline_source_manifest_paths
            ],
            "coastline_color": config.coastline_color,
            "coastline_linewidth": config.coastline_linewidth,
        },
        "color_scales": tables.color_scales,
        "inputs": [
            {
                "context_id": input_config.context_id,
                "training_regime": input_config.training_regime,
                "model_origin_region": input_config.model_origin_region,
                "evaluation_region": input_config.evaluation_region,
                "binary_predictions": str(input_config.binary_predictions_path),
                "binary_calibration_model": str(input_config.binary_calibration_model_path),
                "threshold_policy": input_config.threshold_policy,
                "required": input_config.required,
                "row_count": tables.input_row_counts.get(input_config.context_id, 0),
            }
            for input_config in config.inputs
        ],
        "outputs": {
            "figure": str(config.figure_path),
            "table": str(config.table_path),
            "manifest": str(config.manifest_path),
        },
        "row_counts": {
            "hex_count": len(tables.rows),
            "cell_count": int(sum(object_to_int(row.get("n_cells", 0)) for row in tables.rows)),
        },
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def binary_presence_hex_row_counts(
    tables: BinaryPresenceHexMapTables | None,
) -> dict[str, int]:
    """Return row counts for sidecar and model-analysis manifests."""
    if tables is None:
        return {"pooled_binary_presence_hex": 0}
    return {"pooled_binary_presence_hex": len(tables.rows)}
