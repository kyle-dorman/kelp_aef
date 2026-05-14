"""Build a local interactive map viewer for model-result review."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    MASK_RETAIN_COLUMN,
    ReportingDomainMask,
    apply_reporting_domain_mask,
    evaluation_scope,
    load_reporting_domain_mask,
    mask_status,
)
from kelp_aef.viz.residual_maps import optional_int, optional_positive_int, read_footprint

LOGGER = logging.getLogger(__name__)

DEFAULT_SPLIT = "test"
DEFAULT_YEAR = 2022
DEFAULT_ROBUST_PERCENTILE = 99.0
DEFAULT_MAX_INSPECTION_POINTS = 50_000
DEFAULT_BASEMAP_NAME = "OpenStreetMap"
DEFAULT_BASEMAP_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
DEFAULT_BASEMAP_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)
DEFAULT_LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
DEFAULT_LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
DEFAULT_CONTINUOUS_MIN_AREA_M2 = 90.0
DEFAULT_LABEL_MIN_AREA_M2 = 1.0
DEFAULT_RESIDUAL_MIN_ABS_AREA_M2 = 90.0
DEFAULT_PROBABILITY_MIN = 0.10
DEFAULT_BINARY_OUTCOMES_VISIBLE = ("TP", "FP", "FN")
CONTINUOUS_LAYER_TYPE = "continuous"
CONTINUOUS_PREDICTION_LAYER_TYPE = "continuous_prediction"
BINARY_PROBABILITY_LAYER_TYPE = "binary_probability"
BINARY_OUTCOME_LAYER_TYPE = "binary_outcome"
OBSERVED_LABEL_LAYER_TYPE = "observed_label"
FILTER_MODE_MIN = "min"
FILTER_MODE_ABS_MIN = "abs_min"
FILTER_MODE_CLASSES = "classes"
HURDLE_REVIEW_MIN_AREA_M2 = 90.0
HURDLE_RESIDUAL_REVIEW_MIN_AREA_M2 = 90.0
CONDITIONAL_REVIEW_MIN_AREA_M2 = 450.0
SELECTION_BUCKET_COLUMNS = {
    "binary_non_true_negative": "selection_binary_non_true_negative",
    "binary_false_negative": "selection_binary_false_negative",
    "kelpwatch_positive": "selection_kelpwatch_positive",
    "kelpwatch_observed": "selection_kelpwatch_observed",
    "large_hurdle_residual": "selection_large_hurdle_residual",
    "high_hurdle_prediction": "selection_high_hurdle_prediction",
    "high_conditional_prediction": "selection_high_conditional_prediction",
    "true_negative": "selection_true_negative",
    "residual_fill": "selection_residual_fill",
    "nonzero_support_fill": "selection_nonzero_support_fill",
}

COMMON_COLUMNS = (
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
)
REQUIRED_DISPLAY_COLUMNS = (
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "longitude",
    "latitude",
    "kelp_max_y",
)


@dataclass(frozen=True)
class ResultsLayerConfig:
    """Resolved settings for one visualizer prediction layer."""

    layer_id: str
    display_name: str
    path: Path
    layer_type: str
    model_name: str | None
    prediction_column: str | None
    residual_column: str | None
    probability_column: str | None
    default_visible: bool


@dataclass(frozen=True)
class BasemapConfig:
    """Resolved external basemap settings for the local viewer."""

    enabled: bool
    name: str
    url_template: str
    attribution: str
    max_zoom: int


@dataclass(frozen=True)
class ResultsVisualizerConfig:
    """Resolved config values for the interactive results visualizer."""

    config_path: Path
    data_root: Path
    split: str
    year: int
    robust_percentile: float
    max_inspection_points: int
    html_path: Path
    asset_dir: Path
    manifest_path: Path
    inspection_points_path: Path
    footprint_path: Path
    leaflet_css_url: str
    leaflet_js_url: str
    basemap: BasemapConfig
    domain_mask: ReportingDomainMask | None
    layers: tuple[ResultsLayerConfig, ...]
    filters: VisualizerFilterConfig


@dataclass(frozen=True)
class LayerFilterDefaults:
    """Optional filter overrides for one configured source layer."""

    continuous_min_area_m2: float | None
    residual_min_abs_area_m2: float | None
    probability_min: float | None
    binary_outcomes_visible: tuple[str, ...] | None


@dataclass(frozen=True)
class VisualizerFilterConfig:
    """Resolved filter defaults for browser-side layer controls."""

    continuous_min_area_m2: float
    label_min_area_m2: float
    residual_min_abs_area_m2: float
    probability_min: float
    binary_outcomes_visible: tuple[str, ...]
    layer_overrides: dict[str, LayerFilterDefaults]


@dataclass(frozen=True)
class GridSpec:
    """Compact row/column extent and geographic bounds for selected cells."""

    min_row: int
    max_row: int
    min_col: int
    max_col: int
    west: float
    south: float
    east: float
    north: float

    @property
    def width(self) -> int:
        """Return the compact grid width in columns."""
        return self.max_col - self.min_col + 1

    @property
    def height(self) -> int:
        """Return the compact grid height in rows."""
        return self.max_row - self.min_row + 1

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        """Return Leaflet fit bounds as south-west and north-east corners."""
        return [[self.south, self.west], [self.north, self.east]]


@dataclass(frozen=True)
class PointLayer:
    """Browser point-layer metadata keyed to inspection-point properties."""

    layer_id: str
    display_name: str
    layer_type: str
    value_kind: str
    property_name: str
    popup_label: str
    scale_min: float
    scale_max: float
    diverging: bool
    default_visible: bool
    min_abs_display_value: float
    allowed_values: tuple[str, ...] | None = None
    filter_mode: str = FILTER_MODE_MIN
    filter_default_min: float | None = None
    filter_default_values: tuple[str, ...] = ()
    filter_label: str = ""
    legend_unit: str = ""
    legend_description: str = ""


@dataclass(frozen=True)
class InspectionSelection:
    """Selected inspection rows and row-count diagnostics for review buckets."""

    frame: pd.DataFrame
    bucket_counts: dict[str, dict[str, int]]
    cap_was_enough_for_priority: bool


def visualize_results(config_path: Path) -> int:
    """Write a Leaflet-based local results visualizer and supporting assets."""
    viewer_config = load_results_visualizer_config(config_path)
    LOGGER.info(
        "Building results visualizer for split=%s year=%s", viewer_config.split, viewer_config.year
    )
    layer_frames = read_visualizer_layers(viewer_config)
    if not layer_frames:
        msg = "no configured visualizer layers were available"
        raise ValueError(msg)
    primary_frame = primary_observed_frame(layer_frames)
    grid = grid_spec_from_frames(list(layer_frames.values()))
    prepare_asset_dir(viewer_config.asset_dir)

    canopy_scale = robust_scale(
        canopy_scale_values(primary_frame, layer_frames),
        viewer_config.robust_percentile,
    )
    residual_scale = robust_scale(
        residual_scale_values(layer_frames),
        viewer_config.robust_percentile,
    )
    inspection_selection = build_inspection_selection(
        layer_frames,
        viewer_config.max_inspection_points,
    )
    inspection_frame = inspection_selection.frame
    point_layers = build_point_layers(
        layer_frames=layer_frames,
        residual_scale=residual_scale,
        robust_percentile=viewer_config.robust_percentile,
        filter_config=viewer_config.filters,
    )
    inspection_geojson_path = viewer_config.asset_dir / "inspection_points.geojson"
    inspection_js_path = viewer_config.asset_dir / "inspection_points.js"
    write_inspection_outputs(
        inspection_frame,
        csv_path=viewer_config.inspection_points_path,
        geojson_path=inspection_geojson_path,
        js_path=inspection_js_path,
    )
    footprint = read_footprint(viewer_config.footprint_path)
    write_html(
        viewer_config=viewer_config,
        point_layers=point_layers,
        grid=grid,
        inspection_js_path=inspection_js_path,
        footprint_bounds=footprint.bounds if footprint is not None else None,
    )
    write_manifest(
        viewer_config=viewer_config,
        layer_frames=layer_frames,
        point_layers=point_layers,
        grid=grid,
        canopy_scale=canopy_scale,
        residual_scale=residual_scale,
        inspection_frame=inspection_frame,
        inspection_selection=inspection_selection,
        inspection_geojson_path=inspection_geojson_path,
        inspection_js_path=inspection_js_path,
    )
    LOGGER.info("Wrote results visualizer HTML: %s", viewer_config.html_path)
    LOGGER.info("Wrote results visualizer assets: %s", viewer_config.asset_dir)
    LOGGER.info("Wrote results visualizer manifest: %s", viewer_config.manifest_path)
    return 0


def prepare_asset_dir(asset_dir: Path) -> None:
    """Create the asset directory and remove stale generated layer assets."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.tif", "inspection_points.geojson", "inspection_points.js"):
        for path in asset_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def load_results_visualizer_config(config_path: Path) -> ResultsVisualizerConfig:
    """Load interactive results visualizer settings from the workflow config."""
    config = load_yaml_config(config_path)
    data_root = Path(require_string(config.get("data_root"), "data_root"))
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    reports = require_mapping(config.get("reports"), "reports")
    outputs = require_mapping(reports.get("outputs"), "reports.outputs")
    settings = optional_mapping(
        reports.get("results_visualizer"),
        "reports.results_visualizer",
    )
    output_defaults = visualizer_output_defaults(data_root)
    basemap = load_basemap_config(settings.get("basemap"))
    layers = load_layer_configs(config, settings)
    filters = load_visualizer_filter_config(settings)
    return ResultsVisualizerConfig(
        config_path=config_path,
        data_root=data_root,
        split=str(settings.get("split", DEFAULT_SPLIT)),
        year=optional_int(settings.get("year"), "reports.results_visualizer.year", DEFAULT_YEAR),
        robust_percentile=optional_float(
            settings.get("robust_percentile"),
            "reports.results_visualizer.robust_percentile",
            DEFAULT_ROBUST_PERCENTILE,
        ),
        max_inspection_points=optional_positive_int(
            settings.get("max_inspection_points"),
            "reports.results_visualizer.max_inspection_points",
            DEFAULT_MAX_INSPECTION_POINTS,
        ),
        html_path=output_path(
            outputs,
            "results_visualizer_html",
            output_defaults["html"],
        ),
        asset_dir=output_path(
            outputs,
            "results_visualizer_asset_dir",
            output_defaults["asset_dir"],
        ),
        manifest_path=output_path(
            outputs,
            "results_visualizer_manifest",
            output_defaults["manifest"],
        ),
        inspection_points_path=output_path(
            outputs,
            "results_visualizer_inspection_points",
            output_defaults["inspection_points"],
        ),
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        leaflet_css_url=str(settings.get("leaflet_css_url", DEFAULT_LEAFLET_CSS)),
        leaflet_js_url=str(settings.get("leaflet_js_url", DEFAULT_LEAFLET_JS)),
        basemap=basemap,
        domain_mask=load_reporting_domain_mask(config),
        layers=layers,
        filters=filters,
    )


def visualizer_output_defaults(data_root: Path) -> dict[str, Path]:
    """Return default output paths for generated visualizer artifacts."""
    interactive_root = data_root / "reports/interactive"
    return {
        "html": interactive_root / "monterey_results_visualizer.html",
        "asset_dir": interactive_root / "monterey_results_visualizer",
        "manifest": data_root / "interim/results_visualizer_manifest.json",
        "inspection_points": data_root / "reports/tables/results_visualizer_inspection_points.csv",
    }


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional mapping, treating missing values as an empty mapping."""
    if value is None:
        return {}
    return require_mapping(value, name)


def output_path(outputs: dict[str, Any], key: str, default: Path) -> Path:
    """Resolve an output path from report outputs with a fallback default."""
    value = outputs.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"reports.outputs.{key}"))


def optional_float(value: object, name: str, default: float) -> float:
    """Validate an optional floating-point config value."""
    if value is None:
        return default
    if not isinstance(value, str | int | float):
        msg = f"config field must be numeric: {name}"
        raise ValueError(msg)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        msg = f"config field must be numeric: {name}"
        raise ValueError(msg) from exc


def load_basemap_config(value: object) -> BasemapConfig:
    """Load external basemap settings for the generated Leaflet viewer."""
    settings = optional_mapping(value, "reports.results_visualizer.basemap")
    return BasemapConfig(
        enabled=bool(settings.get("enabled", True)),
        name=str(settings.get("name", DEFAULT_BASEMAP_NAME)),
        url_template=str(settings.get("url_template", DEFAULT_BASEMAP_URL)),
        attribution=str(settings.get("attribution", DEFAULT_BASEMAP_ATTRIBUTION)),
        max_zoom=optional_positive_int(
            settings.get("max_zoom"),
            "reports.results_visualizer.basemap.max_zoom",
            19,
        ),
    )


def load_visualizer_filter_config(settings: dict[str, Any]) -> VisualizerFilterConfig:
    """Load browser-side filter defaults for interactive point layers."""
    defaults = optional_mapping(
        settings.get("filter_defaults"),
        "reports.results_visualizer.filter_defaults",
    )
    return VisualizerFilterConfig(
        continuous_min_area_m2=optional_float(
            defaults.get("continuous_min_area_m2"),
            "reports.results_visualizer.filter_defaults.continuous_min_area_m2",
            DEFAULT_CONTINUOUS_MIN_AREA_M2,
        ),
        label_min_area_m2=optional_float(
            defaults.get("label_min_area_m2"),
            "reports.results_visualizer.filter_defaults.label_min_area_m2",
            DEFAULT_LABEL_MIN_AREA_M2,
        ),
        residual_min_abs_area_m2=optional_float(
            defaults.get("residual_min_abs_area_m2"),
            "reports.results_visualizer.filter_defaults.residual_min_abs_area_m2",
            DEFAULT_RESIDUAL_MIN_ABS_AREA_M2,
        ),
        probability_min=optional_float(
            defaults.get("probability_min"),
            "reports.results_visualizer.filter_defaults.probability_min",
            DEFAULT_PROBABILITY_MIN,
        ),
        binary_outcomes_visible=optional_string_tuple(
            defaults.get("binary_outcomes_visible"),
            "reports.results_visualizer.filter_defaults.binary_outcomes_visible",
            DEFAULT_BINARY_OUTCOMES_VISIBLE,
        ),
        layer_overrides=load_layer_filter_overrides(settings.get("layer_filter_defaults")),
    )


def load_layer_filter_overrides(value: object) -> dict[str, LayerFilterDefaults]:
    """Load optional layer-specific browser filter overrides by source layer id."""
    if value is None:
        return {}
    overrides = require_mapping(value, "reports.results_visualizer.layer_filter_defaults")
    return {
        slugify(str(layer_id)): layer_filter_defaults_from_mapping(layer_id, layer_value)
        for layer_id, layer_value in overrides.items()
    }


def layer_filter_defaults_from_mapping(
    layer_id: object,
    value: object,
) -> LayerFilterDefaults:
    """Validate one layer-specific filter override mapping."""
    name = f"reports.results_visualizer.layer_filter_defaults.{layer_id}"
    settings = require_mapping(value, name)
    return LayerFilterDefaults(
        continuous_min_area_m2=optional_float_or_none(
            settings.get("continuous_min_area_m2"),
            f"{name}.continuous_min_area_m2",
        ),
        residual_min_abs_area_m2=optional_float_or_none(
            settings.get("residual_min_abs_area_m2"),
            f"{name}.residual_min_abs_area_m2",
        ),
        probability_min=optional_float_or_none(
            settings.get("probability_min"),
            f"{name}.probability_min",
        ),
        binary_outcomes_visible=optional_string_tuple_or_none(
            settings.get("binary_outcomes_visible"),
            f"{name}.binary_outcomes_visible",
        ),
    )


def optional_float_or_none(value: object, name: str) -> float | None:
    """Validate an optional floating-point override that may be absent."""
    if value is None:
        return None
    return optional_float(value, name, 0.0)


def optional_string_tuple(
    value: object,
    name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate an optional string list and return uppercase labels."""
    if value is None:
        return default
    result = optional_string_tuple_or_none(value, name)
    return default if result is None else result


def optional_string_tuple_or_none(value: object, name: str) -> tuple[str, ...] | None:
    """Validate a string list override that may be absent."""
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        msg = f"config field must be a list: {name}"
        raise ValueError(msg)
    return tuple(str(item).strip().upper() for item in value if str(item).strip())


def load_layer_configs(
    config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[ResultsLayerConfig, ...]:
    """Load configured visualizer layers or build Monterey Phase 1 defaults."""
    layer_values = settings.get("layers")
    if layer_values is None:
        return default_layer_configs(config)
    if not isinstance(layer_values, list):
        msg = "config field must be a list: reports.results_visualizer.layers"
        raise ValueError(msg)
    return tuple(
        layer_config_from_mapping(value, index) for index, value in enumerate(layer_values)
    )


def default_layer_configs(config: dict[str, Any]) -> tuple[ResultsLayerConfig, ...]:
    """Build default result layers from the active Monterey Phase 1 model config."""
    models = require_mapping(config.get("models"), "models")
    layers: list[ResultsLayerConfig] = []
    hurdle = optional_mapping(models.get("hurdle"), "models.hurdle")
    if hurdle.get("predictions") is not None:
        layers.append(
            ResultsLayerConfig(
                layer_id="expected_value_hurdle",
                display_name="Expected-value hurdle",
                path=Path(require_string(hurdle.get("predictions"), "models.hurdle.predictions")),
                layer_type=CONTINUOUS_LAYER_TYPE,
                model_name=str(
                    hurdle.get("model_name", "calibrated_probability_x_conditional_canopy")
                ),
                prediction_column="pred_kelp_max_y",
                residual_column="residual_kelp_max_y",
                probability_column=None,
                default_visible=True,
            )
        )
    baselines = optional_mapping(models.get("baselines"), "models.baselines")
    if baselines.get("predictions") is not None:
        layers.append(
            ResultsLayerConfig(
                layer_id="ridge_baseline",
                display_name="AEF ridge baseline",
                path=Path(
                    require_string(baselines.get("predictions"), "models.baselines.predictions")
                ),
                layer_type=CONTINUOUS_LAYER_TYPE,
                model_name="ridge_regression",
                prediction_column="pred_kelp_max_y",
                residual_column="residual_kelp_max_y",
                probability_column=None,
                default_visible=False,
            )
        )
    binary = optional_mapping(models.get("binary_presence"), "models.binary_presence")
    if binary.get("full_grid_predictions") is not None:
        layers.append(
            ResultsLayerConfig(
                layer_id="binary_presence_probability",
                display_name="Binary presence probability",
                path=Path(
                    require_string(
                        binary.get("full_grid_predictions"),
                        "models.binary_presence.full_grid_predictions",
                    )
                ),
                layer_type=BINARY_PROBABILITY_LAYER_TYPE,
                model_name=str(binary.get("model_name", "logistic_annual_max_ge_10pct")),
                prediction_column=None,
                residual_column=None,
                probability_column="pred_binary_probability",
                default_visible=False,
            )
        )
    if not layers:
        msg = "no default visualizer layers could be resolved from models config"
        raise ValueError(msg)
    return tuple(layers)


def layer_config_from_mapping(value: object, index: int) -> ResultsLayerConfig:
    """Validate one configured visualizer layer mapping."""
    layer = require_mapping(value, f"reports.results_visualizer.layers[{index}]")
    layer_type = str(layer.get("type", CONTINUOUS_LAYER_TYPE))
    supported_types = {
        CONTINUOUS_LAYER_TYPE,
        CONTINUOUS_PREDICTION_LAYER_TYPE,
        BINARY_PROBABILITY_LAYER_TYPE,
        BINARY_OUTCOME_LAYER_TYPE,
    }
    if layer_type not in supported_types:
        msg = f"unsupported visualizer layer type: {layer_type}"
        raise ValueError(msg)
    return ResultsLayerConfig(
        layer_id=slugify(str(layer.get("id", layer.get("display_name", f"layer_{index}")))),
        display_name=str(layer.get("display_name", layer.get("id", f"Layer {index + 1}"))),
        path=Path(
            require_string(layer.get("path"), f"reports.results_visualizer.layers[{index}].path")
        ),
        layer_type=layer_type,
        model_name=optional_string(layer.get("model_name")),
        prediction_column=optional_string(layer.get("prediction_column")),
        residual_column=optional_string(layer.get("residual_column")),
        probability_column=optional_string(layer.get("probability_column")),
        default_visible=bool(layer.get("default_visible", index == 0)),
    )


def optional_string(value: object) -> str | None:
    """Return a string value or None for optional dynamic config fields."""
    if value is None:
        return None
    return str(value)


def read_visualizer_layers(
    viewer_config: ResultsVisualizerConfig,
) -> dict[ResultsLayerConfig, pd.DataFrame]:
    """Read and normalize every configured visualizer layer with available data."""
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame] = {}
    for layer in viewer_config.layers:
        if not layer.path.exists():
            LOGGER.warning("Skipping missing visualizer layer path: %s", layer.path)
            continue
        frame = read_layer_frame(layer, viewer_config)
        if frame.empty:
            LOGGER.warning("Skipping empty visualizer layer: %s", layer.display_name)
            continue
        layer_frames[layer] = frame
        LOGGER.info(
            "Loaded %s visualizer rows for %s",
            len(frame),
            layer.display_name,
        )
    return layer_frames


def read_layer_frame(
    layer: ResultsLayerConfig,
    viewer_config: ResultsVisualizerConfig,
) -> pd.DataFrame:
    """Read one prediction layer for the configured split and year."""
    dataset = ds.dataset(layer.path, format="parquet")  # type: ignore[no-untyped-call]
    schema_names = set(dataset.schema.names)
    value_columns = layer_value_columns(layer)
    requested_columns = [
        column
        for column in (
            *COMMON_COLUMNS,
            "model_name",
            *value_columns,
            *optional_layer_columns(layer),
        )
        if column in schema_names
    ]
    missing_display = [column for column in REQUIRED_DISPLAY_COLUMNS if column not in schema_names]
    if missing_display:
        msg = f"visualizer layer {layer.display_name} is missing columns: {missing_display}"
        raise ValueError(msg)
    missing_values = [column for column in value_columns if column not in schema_names]
    if missing_values:
        msg = f"visualizer layer {layer.display_name} is missing value columns: {missing_values}"
        raise ValueError(msg)
    expression = (dataset_field("split") == viewer_config.split) & (
        dataset_field("year") == viewer_config.year
    )
    if layer.model_name is not None and "model_name" in schema_names:
        expression = expression & (dataset_field("model_name") == layer.model_name)
    table = dataset.to_table(columns=requested_columns, filter=expression)
    frame = cast(pd.DataFrame, table.to_pandas())
    if frame.empty:
        return frame
    frame = apply_viewer_domain_mask(frame, viewer_config.domain_mask)
    return normalize_layer_frame(frame, layer)


def dataset_field(name: str) -> Any:
    """Return a pyarrow dataset field expression while keeping mypy contained."""
    return cast(Any, ds).field(name)


def layer_value_columns(layer: ResultsLayerConfig) -> tuple[str, ...]:
    """Return layer-specific prediction or probability columns."""
    if layer.layer_type == BINARY_PROBABILITY_LAYER_TYPE:
        return (layer.probability_column or "pred_binary_probability",)
    if layer.layer_type == BINARY_OUTCOME_LAYER_TYPE:
        columns = [layer.prediction_column or "pred_presence_class"]
        if layer.probability_column is not None:
            columns.append(layer.probability_column)
        return tuple(columns)
    prediction_column = layer.prediction_column or "pred_kelp_max_y"
    if layer.layer_type == CONTINUOUS_PREDICTION_LAYER_TYPE:
        return (prediction_column,)
    residual_column = layer.residual_column or "residual_kelp_max_y"
    return (prediction_column, residual_column)


def optional_layer_columns(layer: ResultsLayerConfig) -> tuple[str, ...]:
    """Return optional source columns that improve layer semantics when present."""
    if layer.layer_type == BINARY_OUTCOME_LAYER_TYPE:
        return ("presence_target_threshold_fraction", "target_threshold_fraction")
    return ()


def apply_viewer_domain_mask(
    frame: pd.DataFrame,
    domain_mask: ReportingDomainMask | None,
) -> pd.DataFrame:
    """Apply retained-domain filtering without duplicating existing mask columns."""
    if domain_mask is None:
        return frame.copy()
    if MASK_RETAIN_COLUMN in frame:
        retained = frame[MASK_RETAIN_COLUMN].astype(bool)
        return cast(pd.DataFrame, frame.loc[retained].copy())
    return apply_reporting_domain_mask(frame, domain_mask)


def normalize_layer_frame(frame: pd.DataFrame, layer: ResultsLayerConfig) -> pd.DataFrame:
    """Add stable visualizer columns for one layer regardless of source schema."""
    normalized = frame.copy()
    normalized["viewer_layer_id"] = layer.layer_id
    normalized["viewer_display_name"] = layer.display_name
    normalized["viewer_layer_type"] = layer.layer_type
    if layer.layer_type == BINARY_PROBABILITY_LAYER_TYPE:
        probability_column = layer.probability_column or "pred_binary_probability"
        normalized["viewer_probability"] = normalized[probability_column].astype(float)
    elif layer.layer_type == BINARY_OUTCOME_LAYER_TYPE:
        class_column = layer.prediction_column or "pred_presence_class"
        outcome_probability_column = layer.probability_column
        observed = observed_binary_target(normalized)
        predicted = boolean_series(normalized[class_column])
        normalized["viewer_binary_observed"] = observed
        normalized["viewer_binary_predicted"] = predicted
        normalized["viewer_binary_outcome"] = binary_outcome_labels(observed, predicted)
        if outcome_probability_column is not None and outcome_probability_column in normalized:
            normalized["viewer_probability"] = normalized[outcome_probability_column].astype(float)
    else:
        prediction_column = layer.prediction_column or "pred_kelp_max_y"
        normalized["viewer_prediction_area"] = normalized[prediction_column].astype(float)
        residual_column = layer.residual_column or "residual_kelp_max_y"
        if layer.layer_type == CONTINUOUS_LAYER_TYPE and residual_column in normalized:
            normalized["viewer_residual_area"] = normalized[residual_column].astype(float)
        elif layer.layer_type == CONTINUOUS_LAYER_TYPE:
            normalized["viewer_residual_area"] = (
                normalized["kelp_max_y"].astype(float) - normalized["viewer_prediction_area"]
            )
    add_default_context_columns(normalized)
    return normalized


def observed_binary_target(frame: pd.DataFrame) -> pd.Series:
    """Return the observed binary target using source threshold metadata when available."""
    if "presence_target_threshold_fraction" in frame:
        threshold = frame["presence_target_threshold_fraction"].astype(float)
    elif "target_threshold_fraction" in frame:
        threshold = frame["target_threshold_fraction"].astype(float)
    else:
        threshold = pd.Series(0.10, index=frame.index)
    return cast(pd.Series, frame["kelp_fraction_y"].astype(float) >= threshold)


def boolean_series(series: pd.Series) -> pd.Series:
    """Normalize boolean-like prediction values from parquet or CSV inputs."""
    if pd.api.types.is_bool_dtype(series):
        return cast(pd.Series, series.astype(bool))
    if pd.api.types.is_numeric_dtype(series):
        return cast(pd.Series, series.astype(float) != 0.0)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def binary_outcome_labels(observed: pd.Series, predicted: pd.Series) -> pd.Series:
    """Classify binary predictions as TP, FP, FN, or TN."""
    return pd.Series(
        np.select(
            [
                observed.to_numpy(dtype=bool) & predicted.to_numpy(dtype=bool),
                ~observed.to_numpy(dtype=bool) & predicted.to_numpy(dtype=bool),
                observed.to_numpy(dtype=bool) & ~predicted.to_numpy(dtype=bool),
            ],
            ["TP", "FP", "FN"],
            default="TN",
        ),
        index=observed.index,
    )


def add_default_context_columns(frame: pd.DataFrame) -> None:
    """Fill optional context columns expected by inspection exports."""
    defaults: dict[str, object] = {
        "label_source": "unknown",
        "is_kelpwatch_observed": False,
        "domain_mask_reason": "unknown",
        "depth_bin": "unknown",
        "mask_status": "unknown",
        "evaluation_scope": "unknown",
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value


def primary_observed_frame(
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
) -> pd.DataFrame:
    """Return the first continuous frame for observed-label rendering."""
    for layer, frame in layer_frames.items():
        if is_area_layer(layer):
            return frame
    first = next(iter(layer_frames.values()))
    return first


def is_area_layer(layer: ResultsLayerConfig) -> bool:
    """Return whether a visualizer layer is area-valued rather than probability-valued."""
    return layer.layer_type in {CONTINUOUS_LAYER_TYPE, CONTINUOUS_PREDICTION_LAYER_TYPE}


def grid_spec_from_frames(frames: list[pd.DataFrame]) -> GridSpec:
    """Compute compact image-grid and geographic bounds from loaded layer frames."""
    combined = pd.concat(
        [
            frame[["aef_grid_row", "aef_grid_col", "longitude", "latitude"]]
            for frame in frames
            if not frame.empty
        ],
        ignore_index=True,
    )
    if combined.empty:
        msg = "cannot build visualizer grid from empty frames"
        raise ValueError(msg)
    min_row = int(combined["aef_grid_row"].min())
    max_row = int(combined["aef_grid_row"].max())
    min_col = int(combined["aef_grid_col"].min())
    max_col = int(combined["aef_grid_col"].max())
    min_lon = float(combined["longitude"].min())
    max_lon = float(combined["longitude"].max())
    min_lat = float(combined["latitude"].min())
    max_lat = float(combined["latitude"].max())
    row_span = max(max_row - min_row, 1)
    col_span = max(max_col - min_col, 1)
    lon_pad = (max_lon - min_lon) / col_span / 2.0
    lat_pad = (max_lat - min_lat) / row_span / 2.0
    return GridSpec(
        min_row=min_row,
        max_row=max_row,
        min_col=min_col,
        max_col=max_col,
        west=min_lon - lon_pad,
        south=min_lat - lat_pad,
        east=max_lon + lon_pad,
        north=max_lat + lat_pad,
    )


def canopy_scale_values(
    primary_frame: pd.DataFrame,
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
) -> np.ndarray:
    """Return observed and predicted area values for shared canopy scaling."""
    arrays = [primary_frame["kelp_max_y"].to_numpy(dtype=float)]
    for layer, frame in layer_frames.items():
        if is_area_layer(layer):
            arrays.append(frame["viewer_prediction_area"].to_numpy(dtype=float))
    return np.concatenate(arrays)


def residual_scale_values(layer_frames: dict[ResultsLayerConfig, pd.DataFrame]) -> np.ndarray:
    """Return absolute residual area values for diverging residual scaling."""
    arrays = [
        np.abs(frame["viewer_residual_area"].to_numpy(dtype=float))
        for layer, frame in layer_frames.items()
        if layer.layer_type == CONTINUOUS_LAYER_TYPE and "viewer_residual_area" in frame
    ]
    if not arrays:
        return np.asarray([1.0], dtype=float)
    return np.concatenate(arrays)


def robust_scale(values: np.ndarray, percentile: float) -> float:
    """Return a positive robust scale from finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanpercentile(finite, percentile)), 1.0)


def build_point_layers(
    *,
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
    residual_scale: float,
    robust_percentile: float,
    filter_config: VisualizerFilterConfig,
) -> list[PointLayer]:
    """Build point-layer metadata for coordinate-safe browser rendering."""
    point_layers: list[PointLayer] = [
        observed_label_point_layer(layer_frames, robust_percentile, filter_config)
    ]
    for layer, frame in layer_frames.items():
        if layer.layer_type == BINARY_OUTCOME_LAYER_TYPE:
            outcome_values = ("TP", "FP", "FN", "TN")
            point_layers.append(
                PointLayer(
                    layer_id=f"{layer.layer_id}_outcome",
                    display_name=f"{layer.display_name} TP/FP/FN/TN",
                    layer_type=layer.layer_type,
                    value_kind="binary_outcome",
                    property_name=f"{layer.layer_id}_outcome",
                    popup_label=popup_label(layer, "outcome"),
                    scale_min=0.0,
                    scale_max=1.0,
                    diverging=False,
                    default_visible=layer.default_visible,
                    min_abs_display_value=0.0,
                    allowed_values=outcome_values,
                    filter_mode=FILTER_MODE_CLASSES,
                    filter_default_values=resolved_binary_filter_values(
                        filter_config, layer.layer_id, outcome_values
                    ),
                    filter_label="Outcome classes",
                    legend_unit="class",
                    legend_description="Binary outcome class",
                )
            )
            continue
        if layer.layer_type == BINARY_PROBABILITY_LAYER_TYPE:
            point_layers.append(
                PointLayer(
                    layer_id=f"{layer.layer_id}_probability",
                    display_name=f"{layer.display_name} probability",
                    layer_type=layer.layer_type,
                    value_kind="probability",
                    property_name=f"{layer.layer_id}_probability",
                    popup_label=popup_label(layer, "probability"),
                    scale_min=0.0,
                    scale_max=1.0,
                    diverging=False,
                    default_visible=layer.default_visible,
                    min_abs_display_value=0.01,
                    filter_mode=FILTER_MODE_MIN,
                    filter_default_min=resolved_probability_min(filter_config, layer.layer_id),
                    filter_label="Minimum probability",
                    legend_unit="probability",
                    legend_description="Predicted binary presence probability",
                )
            )
            continue
        prediction_min = resolved_continuous_min(filter_config, layer.layer_id)
        point_layers.append(
            PointLayer(
                layer_id=f"{layer.layer_id}_prediction",
                display_name=f"{layer.display_name} prediction",
                layer_type=layer.layer_type,
                value_kind="canopy_area_m2",
                property_name=f"{layer.layer_id}_prediction_m2",
                popup_label=popup_label(layer, "prediction"),
                scale_min=0.0,
                scale_max=robust_scale(
                    frame["viewer_prediction_area"].to_numpy(dtype=float),
                    robust_percentile,
                ),
                diverging=False,
                default_visible=layer.default_visible,
                min_abs_display_value=1.0,
                filter_mode=FILTER_MODE_MIN,
                filter_default_min=prediction_min,
                filter_label="Minimum area m2",
                legend_unit="m2",
                legend_description="Predicted canopy area",
            )
        )
        if "viewer_residual_area" in frame:
            point_layers.append(
                PointLayer(
                    layer_id=f"{layer.layer_id}_residual",
                    display_name=f"{layer.display_name} residual",
                    layer_type=layer.layer_type,
                    value_kind="observed_minus_predicted_m2",
                    property_name=f"{layer.layer_id}_residual_m2",
                    popup_label=popup_label(layer, "residual"),
                    scale_min=-residual_scale,
                    scale_max=residual_scale,
                    diverging=True,
                    default_visible=False,
                    min_abs_display_value=1.0,
                    filter_mode=FILTER_MODE_ABS_MIN,
                    filter_default_min=resolved_residual_min(filter_config, layer.layer_id),
                    filter_label="Minimum abs residual m2",
                    legend_unit="m2",
                    legend_description="Observed minus predicted canopy area",
                )
            )
    return point_layers


def observed_label_point_layer(
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
    robust_percentile: float,
    filter_config: VisualizerFilterConfig,
) -> PointLayer:
    """Build the observed Kelpwatch annual-max label point layer."""
    observed = primary_observed_frame(layer_frames)
    return PointLayer(
        layer_id="kelpwatch_observed_label",
        display_name="Kelpwatch observed label",
        layer_type=OBSERVED_LABEL_LAYER_TYPE,
        value_kind="observed_canopy_area_m2",
        property_name="observed_canopy_area_m2",
        popup_label="Observed m2",
        scale_min=0.0,
        scale_max=robust_scale(
            observed["kelp_max_y"].to_numpy(dtype=float),
            robust_percentile,
        ),
        diverging=False,
        default_visible=False,
        min_abs_display_value=1.0,
        filter_mode=FILTER_MODE_MIN,
        filter_default_min=filter_config.label_min_area_m2,
        filter_label="Minimum label area m2",
        legend_unit="m2",
        legend_description="Observed Kelpwatch annual-max canopy area",
    )


def layer_filter_override(
    filter_config: VisualizerFilterConfig,
    layer_id: str,
) -> LayerFilterDefaults | None:
    """Return layer-specific filter overrides for a source layer id."""
    return filter_config.layer_overrides.get(layer_id)


def resolved_continuous_min(filter_config: VisualizerFilterConfig, layer_id: str) -> float:
    """Return the browser default minimum for continuous prediction layers."""
    override = layer_filter_override(filter_config, layer_id)
    if override is not None and override.continuous_min_area_m2 is not None:
        return override.continuous_min_area_m2
    return filter_config.continuous_min_area_m2


def resolved_residual_min(filter_config: VisualizerFilterConfig, layer_id: str) -> float:
    """Return the browser default minimum absolute residual for residual layers."""
    override = layer_filter_override(filter_config, layer_id)
    if override is not None and override.residual_min_abs_area_m2 is not None:
        return override.residual_min_abs_area_m2
    return filter_config.residual_min_abs_area_m2


def resolved_probability_min(filter_config: VisualizerFilterConfig, layer_id: str) -> float:
    """Return the browser default minimum probability for probability layers."""
    override = layer_filter_override(filter_config, layer_id)
    if override is not None and override.probability_min is not None:
        return override.probability_min
    return filter_config.probability_min


def resolved_binary_filter_values(
    filter_config: VisualizerFilterConfig,
    layer_id: str,
    allowed_values: tuple[str, ...],
) -> tuple[str, ...]:
    """Return browser default visible binary classes for one outcome layer."""
    override = layer_filter_override(filter_config, layer_id)
    default_values = (
        override.binary_outcomes_visible
        if override is not None and override.binary_outcomes_visible is not None
        else filter_config.binary_outcomes_visible
    )
    values = tuple(value for value in allowed_values if value in default_values)
    return values or allowed_values


def popup_label(layer: ResultsLayerConfig, value_kind: str) -> str:
    """Return a compact popup label for one displayed layer value."""
    labels = {
        ("expected_value_hurdle", "prediction"): "Hurdle pred m2",
        ("expected_value_hurdle", "residual"): "Hurdle resid m2",
        ("conditional_ridge", "prediction"): "Cond ridge m2",
        ("binary_presence", "probability"): "Binary prob",
        ("binary_outcome", "outcome"): "Binary outcome",
    }
    if (layer.layer_id, value_kind) in labels:
        return labels[(layer.layer_id, value_kind)]
    if value_kind == "probability":
        return f"{layer.display_name} prob"
    if value_kind == "residual":
        return f"{layer.display_name} resid m2"
    if value_kind == "outcome":
        return f"{layer.display_name} outcome"
    return f"{layer.display_name} pred m2"


def build_inspection_frame(
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
    max_points: int,
) -> pd.DataFrame:
    """Build a bounded point table for click inspection in the browser."""
    return build_inspection_selection(layer_frames, max_points).frame


def build_inspection_selection(
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
    max_points: int,
) -> InspectionSelection:
    """Build selected inspection rows and manifest-ready bucket diagnostics."""
    continuous_items = [
        (layer, frame)
        for layer, frame in layer_frames.items()
        if layer.layer_type == CONTINUOUS_LAYER_TYPE
    ]
    if continuous_items:
        primary_layer, primary = continuous_items[0]
    else:
        primary_layer, primary = next(iter(layer_frames.items()))
    inspection = base_inspection_columns(primary).copy()
    inspection["primary_layer_id"] = primary_layer.layer_id
    for layer, frame in layer_frames.items():
        values = layer_inspection_values(layer, frame)
        inspection = inspection.merge(values, on="aef_grid_cell_id", how="left")
    inspection["abs_primary_residual"] = (
        inspection.get(
            f"{primary_layer.layer_id}_residual_m2", pd.Series(0.0, index=inspection.index)
        )
        .astype(float)
        .abs()
    )
    inspection["nonzero_support"] = inspection["observed_canopy_area_m2"].astype(
        float
    ).abs() + prediction_support_sum(inspection)
    add_selection_bucket_columns(inspection)
    selected = select_inspection_points(inspection, max_points)
    bucket_counts = selection_bucket_counts(inspection, selected)
    priority_columns = priority_selection_columns()
    priority_candidate_count = int(inspection[priority_columns].any(axis=1).sum())
    priority_included_count = int(selected[priority_columns].any(axis=1).sum())
    return InspectionSelection(
        frame=selected,
        bucket_counts=bucket_counts,
        cap_was_enough_for_priority=priority_candidate_count == priority_included_count,
    )


def base_inspection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return common cell-inspection columns from the primary display frame."""
    columns = [
        "aef_grid_cell_id",
        "aef_grid_row",
        "aef_grid_col",
        "longitude",
        "latitude",
        "year",
        "split",
        "label_source",
        "is_kelpwatch_observed",
        "domain_mask_reason",
        "depth_bin",
        "mask_status",
        "evaluation_scope",
        "kelp_max_y",
        "kelp_fraction_y",
    ]
    available = [column for column in columns if column in frame]
    result = frame[available].drop_duplicates("aef_grid_cell_id").copy()
    result = result.rename(
        columns={
            "kelp_max_y": "observed_canopy_area_m2",
            "kelp_fraction_y": "observed_fraction",
        }
    )
    return result


def layer_inspection_values(layer: ResultsLayerConfig, frame: pd.DataFrame) -> pd.DataFrame:
    """Return renamed layer-value columns keyed by grid cell id."""
    if layer.layer_type == BINARY_PROBABILITY_LAYER_TYPE:
        values = frame[["aef_grid_cell_id", "viewer_probability"]].copy()
        return values.rename(columns={"viewer_probability": f"{layer.layer_id}_probability"})
    if layer.layer_type == BINARY_OUTCOME_LAYER_TYPE:
        columns = [
            "aef_grid_cell_id",
            "viewer_binary_outcome",
            "viewer_binary_observed",
            "viewer_binary_predicted",
        ]
        if "viewer_probability" in frame:
            columns.append("viewer_probability")
        values = frame[columns].copy()
        rename_map = {
            "viewer_binary_outcome": f"{layer.layer_id}_outcome",
            "viewer_binary_observed": f"{layer.layer_id}_observed",
            "viewer_binary_predicted": f"{layer.layer_id}_predicted",
        }
        if "viewer_probability" in values:
            rename_map["viewer_probability"] = f"{layer.layer_id}_probability"
        return values.rename(columns=rename_map)
    columns = ["aef_grid_cell_id", "viewer_prediction_area"]
    if "viewer_residual_area" in frame:
        columns.append("viewer_residual_area")
    values = frame[columns].copy()
    rename_map = {"viewer_prediction_area": f"{layer.layer_id}_prediction_m2"}
    if "viewer_residual_area" in values:
        rename_map["viewer_residual_area"] = f"{layer.layer_id}_residual_m2"
    return values.rename(columns=rename_map)


def prediction_support_sum(inspection: pd.DataFrame) -> pd.Series:
    """Return per-row prediction support across all inspection prediction columns."""
    prediction_columns = [
        column for column in inspection.columns if column.endswith("_prediction_m2")
    ]
    if not prediction_columns:
        return pd.Series(0.0, index=inspection.index)
    return cast(pd.Series, inspection[prediction_columns].abs().sum(axis=1))


def add_selection_bucket_columns(inspection: pd.DataFrame) -> None:
    """Annotate inspection candidates with deterministic review-priority buckets."""
    observed = boolean_series(
        inspection.get("is_kelpwatch_observed", pd.Series(False, index=inspection.index))
    )
    observed_area = inspection["observed_canopy_area_m2"].fillna(0.0).astype(float)
    review_mask = binary_non_true_negative_or_unavailable_mask(inspection)
    inspection[SELECTION_BUCKET_COLUMNS["binary_non_true_negative"]] = (
        binary_non_true_negative_mask(inspection)
    )
    inspection[SELECTION_BUCKET_COLUMNS["true_negative"]] = true_negative_mask(inspection)
    inspection[SELECTION_BUCKET_COLUMNS["kelpwatch_observed"]] = observed & review_mask
    inspection[SELECTION_BUCKET_COLUMNS["kelpwatch_positive"]] = (
        observed & (observed_area > 0.0) & review_mask
    )
    inspection[SELECTION_BUCKET_COLUMNS["binary_false_negative"]] = binary_false_negative_mask(
        inspection
    )
    inspection[SELECTION_BUCKET_COLUMNS["large_hurdle_residual"]] = (
        inspection["abs_primary_residual"].fillna(0.0).astype(float)
        >= HURDLE_RESIDUAL_REVIEW_MIN_AREA_M2
    )
    inspection[SELECTION_BUCKET_COLUMNS["high_hurdle_prediction"]] = (
        high_prediction_mask(
            inspection,
            token="hurdle",
            threshold=HURDLE_REVIEW_MIN_AREA_M2,
        )
        & review_mask
    )
    inspection[SELECTION_BUCKET_COLUMNS["high_conditional_prediction"]] = (
        high_prediction_mask(
            inspection,
            token="conditional",
            threshold=CONDITIONAL_REVIEW_MIN_AREA_M2,
        )
        & review_mask
    )
    inspection[SELECTION_BUCKET_COLUMNS["residual_fill"]] = (
        inspection["abs_primary_residual"].fillna(0.0).astype(float) > 0.0
    ) & review_mask
    inspection[SELECTION_BUCKET_COLUMNS["nonzero_support_fill"]] = (
        inspection["nonzero_support"].fillna(0.0).astype(float) > 0.0
    ) & review_mask


def binary_outcome_columns(inspection: pd.DataFrame) -> list[str]:
    """Return inspection columns containing binary TP/FP/FN/TN labels."""
    return [column for column in inspection.columns if column.endswith("_outcome")]


def binary_non_true_negative_or_unavailable_mask(inspection: pd.DataFrame) -> pd.Series:
    """Return non-TN rows, or all rows when no binary outcome layer is available."""
    if not binary_outcome_columns(inspection):
        return pd.Series(True, index=inspection.index)
    return binary_non_true_negative_mask(inspection)


def binary_non_true_negative_mask(inspection: pd.DataFrame) -> pd.Series:
    """Return rows classified as TP, FP, or FN by any binary outcome layer."""
    outcome_columns = binary_outcome_columns(inspection)
    if not outcome_columns:
        return pd.Series(False, index=inspection.index)
    return cast(pd.Series, inspection[outcome_columns].isin({"TP", "FP", "FN"}).any(axis=1))


def binary_false_negative_mask(inspection: pd.DataFrame) -> pd.Series:
    """Return rows classified as binary false negatives by any outcome layer."""
    outcome_columns = binary_outcome_columns(inspection)
    if not outcome_columns:
        return pd.Series(False, index=inspection.index)
    return cast(pd.Series, inspection[outcome_columns].eq("FN").any(axis=1))


def true_negative_mask(inspection: pd.DataFrame) -> pd.Series:
    """Return rows classified as true negatives by any binary outcome layer."""
    outcome_columns = binary_outcome_columns(inspection)
    if not outcome_columns:
        return pd.Series(False, index=inspection.index)
    return cast(pd.Series, inspection[outcome_columns].eq("TN").any(axis=1))


def high_prediction_mask(
    inspection: pd.DataFrame,
    *,
    token: str,
    threshold: float,
) -> pd.Series:
    """Return rows whose named prediction columns meet a review threshold."""
    score = max_prediction_score(inspection, token)
    return cast(pd.Series, score >= threshold)


def max_prediction_score(inspection: pd.DataFrame, token: str) -> pd.Series:
    """Return the row-wise max prediction score for columns matching a name token."""
    prediction_columns = [
        column
        for column in inspection.columns
        if column.endswith("_prediction_m2") and token in column
    ]
    if not prediction_columns:
        return pd.Series(0.0, index=inspection.index)
    return cast(pd.Series, inspection[prediction_columns].fillna(0.0).max(axis=1))


def select_inspection_points(inspection: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Select a bounded priority mix without crowding out Kelpwatch/FN rows."""
    selected_indices: list[Any] = []
    selected_index_set: set[Any] = set()
    for bucket_name, score_column in selection_priority_order():
        if len(selected_indices) >= max_points:
            break
        mask = inspection[SELECTION_BUCKET_COLUMNS[bucket_name]].astype(bool)
        available = inspection.loc[mask & ~inspection.index.isin(selected_index_set)]
        if available.empty:
            continue
        scored = available.assign(
            _selection_score=selection_score(available, bucket_name, score_column)
        )
        ordered = scored.sort_values(
            ["_selection_score", "aef_grid_cell_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        remaining_slots = max_points - len(selected_indices)
        chosen = list(ordered.head(remaining_slots).index)
        selected_indices.extend(chosen)
        selected_index_set.update(chosen)
    selected = inspection.loc[selected_indices].copy()
    if len(selected) > max_points:
        selected = selected.head(max_points)
    selected["selection_reasons"] = selection_reasons(selected)
    return selected.copy()


def selection_score(
    candidates: pd.DataFrame,
    bucket_name: str,
    score_column: str,
) -> pd.Series:
    """Return a deterministic score for ordering candidates within a bucket."""
    if score_column in candidates:
        return cast(pd.Series, candidates[score_column].fillna(0.0).astype(float))
    if bucket_name == "high_hurdle_prediction":
        return max_prediction_score(candidates, "hurdle")
    if bucket_name == "high_conditional_prediction":
        return max_prediction_score(candidates, "conditional")
    return pd.Series(0.0, index=candidates.index)


def selection_priority_order() -> tuple[tuple[str, str], ...]:
    """Return bucket names and score columns in selection-priority order."""
    return (
        ("binary_non_true_negative", "abs_primary_residual"),
        ("binary_false_negative", "observed_canopy_area_m2"),
        ("kelpwatch_positive", "observed_canopy_area_m2"),
        ("kelpwatch_observed", "observed_canopy_area_m2"),
        ("large_hurdle_residual", "abs_primary_residual"),
        ("high_hurdle_prediction", "expected_value_hurdle_prediction_m2"),
        ("high_conditional_prediction", "conditional_ridge_prediction_m2"),
        ("residual_fill", "abs_primary_residual"),
        ("nonzero_support_fill", "nonzero_support"),
    )


def priority_selection_columns() -> list[str]:
    """Return bucket columns that should fit before residual/support fillers."""
    return [
        SELECTION_BUCKET_COLUMNS["binary_non_true_negative"],
        SELECTION_BUCKET_COLUMNS["binary_false_negative"],
        SELECTION_BUCKET_COLUMNS["kelpwatch_positive"],
        SELECTION_BUCKET_COLUMNS["kelpwatch_observed"],
        SELECTION_BUCKET_COLUMNS["large_hurdle_residual"],
        SELECTION_BUCKET_COLUMNS["high_hurdle_prediction"],
        SELECTION_BUCKET_COLUMNS["high_conditional_prediction"],
    ]


def selection_reasons(selection: pd.DataFrame) -> pd.Series:
    """Return pipe-delimited bucket names that each selected row satisfies."""
    reasons = []
    for _, row in selection.iterrows():
        row_reasons = [
            bucket_name
            for bucket_name, column in SELECTION_BUCKET_COLUMNS.items()
            if bool(row.get(column, False))
        ]
        reasons.append("|".join(row_reasons))
    return pd.Series(reasons, index=selection.index)


def selection_bucket_counts(
    inspection: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Return candidate, included, and omitted counts for each selection bucket."""
    counts: dict[str, dict[str, int]] = {}
    for bucket_name, column in SELECTION_BUCKET_COLUMNS.items():
        candidate_count = int(inspection[column].astype(bool).sum())
        included_count = int(selected[column].astype(bool).sum()) if column in selected else 0
        counts[bucket_name] = {
            "candidate_count": candidate_count,
            "included_count": included_count,
            "omitted_count": candidate_count - included_count,
        }
    return counts


def write_inspection_outputs(
    inspection: pd.DataFrame,
    *,
    csv_path: Path,
    geojson_path: Path,
    js_path: Path,
) -> None:
    """Write CSV, GeoJSON, and JavaScript inspection-point assets."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    inspection.to_csv(csv_path, index=False)
    geojson = inspection_geojson(inspection)
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    geojson_path.write_text(json.dumps(geojson, separators=(",", ":")) + "\n")
    js_path.write_text(
        "window.RESULTS_VISUALIZER_INSPECTION = "
        + json.dumps(geojson, separators=(",", ":"))
        + ";\n"
    )


def inspection_geojson(inspection: pd.DataFrame) -> dict[str, Any]:
    """Convert inspection rows to a GeoJSON feature collection."""
    clean = inspection.replace({np.nan: None})
    geometry = gpd.points_from_xy(clean["longitude"], clean["latitude"], crs="EPSG:4326")
    geo_frame = gpd.GeoDataFrame(clean, geometry=geometry, crs="EPSG:4326")
    return cast(dict[str, Any], json.loads(geo_frame.to_json()))


def write_html(
    *,
    viewer_config: ResultsVisualizerConfig,
    point_layers: list[PointLayer],
    grid: GridSpec,
    inspection_js_path: Path,
    footprint_bounds: tuple[float, float, float, float] | None,
) -> None:
    """Write the Leaflet HTML document for local result review."""
    viewer_config.html_path.parent.mkdir(parents=True, exist_ok=True)
    payload = html_payload(viewer_config, point_layers, grid, inspection_js_path, footprint_bounds)
    viewer_config.html_path.write_text(payload)


def html_payload(
    viewer_config: ResultsVisualizerConfig,
    point_layers: list[PointLayer],
    grid: GridSpec,
    inspection_js_path: Path,
    footprint_bounds: tuple[float, float, float, float] | None,
) -> str:
    """Build the full Leaflet HTML payload."""
    point_layers_payload = [point_layer_payload(layer) for layer in point_layers]
    inspection_js = relative_asset_path(viewer_config.html_path, inspection_js_path)
    payload = {
        "split": viewer_config.split,
        "year": viewer_config.year,
        "bounds": grid.leaflet_bounds,
        "footprintBounds": footprint_bounds,
        "pointLayers": point_layers_payload,
        "basemap": {
            "enabled": viewer_config.basemap.enabled,
            "name": viewer_config.basemap.name,
            "urlTemplate": viewer_config.basemap.url_template,
            "attribution": viewer_config.basemap.attribution,
            "maxZoom": viewer_config.basemap.max_zoom,
        },
        "maskStatus": mask_status(viewer_config.domain_mask),
        "evaluationScope": evaluation_scope(viewer_config.domain_mask),
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Monterey Results Visualizer</title>
<link rel="stylesheet" href="{html.escape(viewer_config.leaflet_css_url, quote=True)}" />
<style>
html, body, #map {{ height: 100%; margin: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; }}
#map {{ background: #f4f1e8; }}
.panel {{ position: absolute; z-index: 1000; top: 12px; right: 12px; width: 310px; max-height: calc(100vh - 24px); overflow: auto; background: rgba(255,255,255,0.94); border: 1px solid #c9d1d5; border-radius: 6px; box-shadow: 0 2px 12px rgba(0,0,0,0.16); }}
.panel header {{ padding: 10px 12px; border-bottom: 1px solid #dbe1e4; }}
.panel h1 {{ margin: 0; font-size: 16px; }}
.panel .meta {{ margin-top: 4px; color: #52616b; font-size: 12px; line-height: 1.35; }}
.legend {{ padding: 10px 12px 12px; color: #52616b; font-size: 12px; line-height: 1.45; }}
.legend h2 {{ color: #172026; font-size: 12px; margin: 0 0 8px; }}
.legend-title {{ color: #172026; font-weight: 700; margin-bottom: 7px; }}
.legend-ramp {{ border: 1px solid #6f7b80; border-radius: 4px; height: 12px; margin: 7px 0 5px; }}
.legend-scale {{ display: flex; justify-content: space-between; gap: 8px; font-variant-numeric: tabular-nums; }}
.legend-swatch-row {{ align-items: center; display: flex; gap: 7px; margin: 5px 0; }}
.legend-swatch {{ border: 1px solid #172026; border-radius: 50%; display: inline-block; height: 11px; width: 11px; }}
.filter-summary {{ border-top: 1px solid #dbe1e4; margin-top: 10px; padding-top: 8px; }}
.leaflet-popup-content {{ min-width: 245px; max-width: 310px; }}
.leaflet-popup-content table {{ border-collapse: collapse; font-size: 12px; table-layout: fixed; width: 100%; }}
.leaflet-popup-content th {{ text-align: left; color: #52616b; padding: 3px 8px 3px 0; white-space: normal; width: 48%; }}
.leaflet-popup-content td {{ text-align: right; padding: 3px 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; width: 52%; }}
.copy-btn {{ margin-top: 8px; width: 100%; border: 1px solid #9aa6ac; background: #f7fafb; border-radius: 4px; padding: 5px 7px; cursor: pointer; }}
.leaflet-control-layers {{ max-height: 70vh; overflow: auto; }}
.data-layer-control {{ background: rgba(255,255,255,0.94); border: 1px solid #c9d1d5; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.18); padding: 8px 10px; min-width: 205px; }}
.data-layer-control .control-title {{ color: #52616b; font-size: 11px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }}
.data-layer-control label {{ align-items: center; color: #172026; cursor: pointer; display: flex; font-size: 12px; gap: 6px; line-height: 1.3; margin: 4px 0; }}
.data-layer-control input {{ margin: 0; }}
.filter-control {{ background: rgba(255,255,255,0.94); border: 1px solid #c9d1d5; border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.18); margin-top: 8px; min-width: 205px; padding: 8px 10px; }}
.filter-control .control-title {{ color: #52616b; font-size: 11px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }}
.filter-control label {{ align-items: center; color: #172026; display: flex; font-size: 12px; gap: 6px; line-height: 1.3; margin: 5px 0; }}
.filter-control input[type="number"] {{ border: 1px solid #aab5ba; border-radius: 4px; box-sizing: border-box; font-size: 12px; padding: 4px 5px; width: 84px; }}
.filter-control input[type="checkbox"] {{ margin: 0; }}
.filter-value {{ color: #52616b; font-size: 11px; margin-top: 5px; }}
</style>
</head>
<body>
<div id="map"></div>
<aside class="panel">
  <header>
    <h1>Monterey Results Visualizer</h1>
    <div class="meta">Split {html.escape(viewer_config.split)} | {viewer_config.year} | {html.escape(mask_status(viewer_config.domain_mask))}</div>
  </header>
  <div class="legend">
    <h2>Active legend</h2>
    <div id="active-legend"></div>
    <div id="active-filter-summary" class="filter-summary"></div>
  </div>
</aside>
<script src="{html.escape(viewer_config.leaflet_js_url, quote=True)}"></script>
<script src="{html.escape(inspection_js, quote=True)}"></script>
<script>
const VIEWER = {json.dumps(payload, separators=(",", ":"))};
const blankBase = L.layerGroup();
const baseLayers = {{ "No basemap": blankBase }};
const initialLayers = [blankBase];
if (VIEWER.basemap.enabled) {{
  const base = L.tileLayer(VIEWER.basemap.urlTemplate, {{
    maxZoom: VIEWER.basemap.maxZoom,
    attribution: VIEWER.basemap.attribution
  }});
  baseLayers[VIEWER.basemap.name] = base;
  initialLayers[0] = base;
}}
const map = L.map("map", {{ preferCanvas: true, layers: initialLayers }});
map.fitBounds(VIEWER.bounds);
const dataLayers = {{}};
const filterState = {{}};
let activeDataLayerId = null;
for (const pointLayer of VIEWER.pointLayers) {{
  filterState[pointLayer.id] = initialFilterState(pointLayer);
  dataLayers[pointLayer.id] = {{ config: pointLayer, layer: null }};
  if (pointLayer.defaultVisible && activeDataLayerId === null) activeDataLayerId = pointLayer.id;
}}
if (activeDataLayerId === null && VIEWER.pointLayers.length > 0) activeDataLayerId = VIEWER.pointLayers[0].id;
const dataLayerControl = L.control({{ position: "topleft" }});
let dataLayerControlContainer = null;
let filterControlContainer = null;
dataLayerControl.onAdd = function() {{
  const container = L.DomUtil.create("div", "data-layer-control leaflet-control");
  dataLayerControlContainer = container;
  const title = document.createElement("div");
  title.className = "control-title";
  title.textContent = "Data layer";
  container.appendChild(title);
  for (const pointLayer of VIEWER.pointLayers) {{
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "results-visualizer-data-layer";
    input.value = pointLayer.id;
    input.checked = pointLayer.id === activeDataLayerId;
    input.addEventListener("change", () => {{
      if (input.checked) setActiveDataLayer(pointLayer.id);
    }});
    const text = document.createElement("span");
    text.textContent = pointLayer.displayName;
    label.appendChild(input);
    label.appendChild(text);
    container.appendChild(label);
  }}
  L.DomEvent.disableClickPropagation(container);
  L.DomEvent.disableScrollPropagation(container);
  return container;
}};
dataLayerControl.addTo(map);
const filterControl = L.control({{ position: "topleft" }});
filterControl.onAdd = function() {{
  const container = L.DomUtil.create("div", "filter-control leaflet-control");
  filterControlContainer = container;
  L.DomEvent.disableClickPropagation(container);
  L.DomEvent.disableScrollPropagation(container);
  return container;
}};
filterControl.addTo(map);
L.control.layers(baseLayers, {{}}, {{ collapsed: true, position: "bottomright" }}).addTo(map);
L.control.scale({{ imperial: false }}).addTo(map);
if (activeDataLayerId !== null) setActiveDataLayer(activeDataLayerId);
function initialFilterState(pointLayer) {{
  return {{
    minValue: pointLayer.filter.defaultMinValue,
    visibleValues: [...pointLayer.filter.defaultValues]
  }};
}}
function makeDataLayer(pointLayer) {{
  return L.geoJSON(window.RESULTS_VISUALIZER_INSPECTION, {{
    filter: (feature) => shouldDrawPoint(feature.properties, pointLayer),
    pointToLayer: (feature, latlng) => L.circleMarker(latlng, pointStyle(feature.properties, pointLayer)),
    onEachFeature: (feature, layer) => layer.bindPopup(popupHtml(feature.properties))
  }});
}}
function setActiveDataLayer(layerId) {{
  for (const entry of Object.values(dataLayers)) {{
    if (entry.layer !== null && map.hasLayer(entry.layer)) map.removeLayer(entry.layer);
  }}
  if (dataLayers[layerId]) {{
    activeDataLayerId = layerId;
    dataLayers[layerId].layer = makeDataLayer(dataLayers[layerId].config);
    dataLayers[layerId].layer.addTo(map);
    updateFilterControl(dataLayers[layerId].config);
    updateLegend(dataLayers[layerId].config);
  }}
}}
function refreshActiveDataLayer() {{
  if (activeDataLayerId === null || !dataLayers[activeDataLayerId]) return;
  const entry = dataLayers[activeDataLayerId];
  if (entry.layer !== null && map.hasLayer(entry.layer)) map.removeLayer(entry.layer);
  entry.layer = makeDataLayer(entry.config);
  entry.layer.addTo(map);
  updateLegend(entry.config);
}}
function updateFilterControl(pointLayer) {{
  if (filterControlContainer === null) return;
  filterControlContainer.replaceChildren();
  const title = document.createElement("div");
  title.className = "control-title";
  title.textContent = "Layer filter";
  filterControlContainer.appendChild(title);
  const layerName = document.createElement("div");
  layerName.className = "filter-value";
  layerName.textContent = pointLayer.displayName;
  filterControlContainer.appendChild(layerName);
  if (pointLayer.filter.mode === "classes") {{
    for (const value of pointLayer.filter.allowedValues) {{
      const label = document.createElement("label");
      label.className = "binary-outcome-filter";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = value;
      input.checked = filterState[pointLayer.id].visibleValues.includes(value);
      input.addEventListener("change", () => {{
        const state = filterState[pointLayer.id];
        if (input.checked && !state.visibleValues.includes(value)) state.visibleValues.push(value);
        if (!input.checked) state.visibleValues = state.visibleValues.filter(item => item !== value);
        refreshActiveDataLayer();
        updateFilterSummary(pointLayer);
      }});
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = outcomeColor(value);
      const text = document.createElement("span");
      text.textContent = value;
      label.appendChild(input);
      label.appendChild(swatch);
      label.appendChild(text);
      filterControlContainer.appendChild(label);
    }}
  }} else {{
    const label = document.createElement("label");
    const text = document.createElement("span");
    text.textContent = pointLayer.filter.label;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = pointLayer.valueKind === "probability" ? "0.01" : "1";
    input.value = String(filterState[pointLayer.id].minValue);
    input.addEventListener("input", () => {{
      const value = Number(input.value);
      filterState[pointLayer.id].minValue = Number.isFinite(value) ? value : 0;
      refreshActiveDataLayer();
      updateFilterSummary(pointLayer);
    }});
    label.appendChild(text);
    label.appendChild(input);
    filterControlContainer.appendChild(label);
  }}
  updateFilterSummary(pointLayer);
}}
function updateFilterSummary(pointLayer) {{
  const summary = document.getElementById("active-filter-summary");
  if (summary === null) return;
  const state = filterState[pointLayer.id];
  if (pointLayer.filter.mode === "classes") {{
    summary.textContent = `Filter: ${{state.visibleValues.length ? state.visibleValues.join(", ") : "none"}}`;
  }} else {{
    const comparator = pointLayer.filter.mode === "abs_min" ? "absolute value >= " : "value >= ";
    summary.textContent = `Filter: ${{comparator}}${{formatNumber(state.minValue, pointLayer.valueKind === "probability" ? 2 : 0)}} ${{pointLayer.legend.unit}}`;
  }}
}}
function shouldDrawPoint(props, pointLayer) {{
  const state = filterState[pointLayer.id];
  if (pointLayer.valueKind === "binary_outcome") {{
    const value = props[pointLayer.propertyName];
    return pointLayer.allowedValues.includes(value) && state.visibleValues.includes(value);
  }}
  const value = Number(props[pointLayer.propertyName]);
  if (!Number.isFinite(value)) return false;
  if (pointLayer.filter.mode === "abs_min") return Math.abs(value) >= state.minValue;
  return value >= state.minValue;
}}
function pointStyle(props, pointLayer) {{
  const value = pointLayer.valueKind === "binary_outcome" ? props[pointLayer.propertyName] : Number(props[pointLayer.propertyName]);
  return {{
    radius: pointRadius(value, pointLayer),
    color: "#172026",
    weight: 0.75,
    fillColor: pointColor(value, pointLayer),
    fillOpacity: 0.78,
    opacity: 0.82
  }};
}}
function pointRadius(value, pointLayer) {{
  if (pointLayer.valueKind === "binary_outcome") return 5;
  const span = Math.max(Math.abs(pointLayer.scaleMin), Math.abs(pointLayer.scaleMax), 1);
  const amount = Math.min(Math.abs(value) / span, 1);
  return 3 + Math.sqrt(amount) * 4;
}}
function pointColor(value, pointLayer) {{
  if (pointLayer.valueKind === "binary_outcome") {{
    return outcomeColor(value);
  }}
  if (pointLayer.diverging) {{
    const span = Math.max(Math.abs(pointLayer.scaleMin), Math.abs(pointLayer.scaleMax), 1);
    const amount = Math.min(Math.abs(value) / span, 1);
    return value >= 0 ? interpolateColor("#fdd49e", "#b30000", amount) : interpolateColor("#bdd7e7", "#08519c", amount);
  }}
  const amount = Math.min(Math.max((value - pointLayer.scaleMin) / Math.max(pointLayer.scaleMax - pointLayer.scaleMin, 1), 0), 1);
  return interpolateColor("#c7e9c0", "#006d2c", amount);
}}
function updateLegend(pointLayer) {{
  const container = document.getElementById("active-legend");
  if (container === null) return;
  container.replaceChildren();
  const title = document.createElement("div");
  title.className = "legend-title";
  title.textContent = pointLayer.displayName;
  container.appendChild(title);
  if (pointLayer.valueKind === "binary_outcome") {{
    for (const value of pointLayer.filter.allowedValues) {{
      const row = document.createElement("div");
      row.className = "legend-swatch-row";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = outcomeColor(value);
      const label = document.createElement("span");
      label.textContent = binaryOutcomeLabel(value);
      row.appendChild(swatch);
      row.appendChild(label);
      container.appendChild(row);
    }}
    return;
  }}
  const ramp = document.createElement("div");
  ramp.className = "legend-ramp";
  if (pointLayer.diverging) {{
    ramp.style.background = "linear-gradient(90deg, #08519c 0%, #bdd7e7 45%, #f7f7f7 50%, #fdd49e 55%, #b30000 100%)";
  }} else if (pointLayer.valueKind === "probability") {{
    ramp.style.background = "linear-gradient(90deg, #f7fcf5 0%, #c7e9c0 45%, #006d2c 100%)";
  }} else {{
    ramp.style.background = "linear-gradient(90deg, #f7fcf5 0%, #c7e9c0 45%, #006d2c 100%)";
  }}
  container.appendChild(ramp);
  const scale = document.createElement("div");
  scale.className = "legend-scale";
  if (pointLayer.diverging) {{
    scale.innerHTML = `<span>${{formatNumber(pointLayer.scaleMin, 0)}}</span><span>0</span><span>${{formatNumber(pointLayer.scaleMax, 0)}} ${{escapeHtml(pointLayer.legend.unit)}}</span>`;
  }} else {{
    const digits = pointLayer.valueKind === "probability" ? 2 : 0;
    scale.innerHTML = `<span>${{formatNumber(pointLayer.scaleMin, digits)}}</span><span>${{formatNumber(pointLayer.scaleMax, digits)}} ${{escapeHtml(pointLayer.legend.unit)}}</span>`;
  }}
  container.appendChild(scale);
  const description = document.createElement("div");
  description.className = "filter-value";
  description.textContent = pointLayer.legend.description;
  container.appendChild(description);
}}
function outcomeColor(value) {{
  const colors = {{ TP: "#238b45", FP: "#d95f02", FN: "#756bb1", TN: "#bdbdbd" }};
  return colors[value] || "#737373";
}}
function binaryOutcomeLabel(value) {{
  const labels = {{ TP: "TP observed and predicted", FP: "FP predicted only", FN: "FN observed only", TN: "TN background" }};
  return labels[value] || value;
}}
function interpolateColor(low, high, amount) {{
  const start = hexToRgb(low);
  const end = hexToRgb(high);
  const mix = start.map((value, index) => Math.round(value + (end[index] - value) * amount));
  return `rgb(${{mix[0]}}, ${{mix[1]}}, ${{mix[2]}})`;
}}
function hexToRgb(hex) {{
  return [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16));
}}
function popupHtml(props) {{
  const rows = [
    ["Cell", props.aef_grid_cell_id],
    ["Lon", formatNumber(props.longitude, 6)],
    ["Lat", formatNumber(props.latitude, 6)],
    ["Observed m2", formatNumber(props.observed_canopy_area_m2, 1)],
    ["Observed frac", formatNumber(props.observed_fraction, 3)],
    ["Depth bin", props.depth_bin]
  ];
  const popupProperties = new Set(["observed_canopy_area_m2"]);
  for (const pointLayer of VIEWER.pointLayers) {{
    if (popupProperties.has(pointLayer.propertyName)) continue;
    popupProperties.add(pointLayer.propertyName);
    const value = props[pointLayer.propertyName];
    if (value !== null && value !== undefined) {{
      if (pointLayer.valueKind === "binary_outcome") {{
        rows.push([pointLayer.popupLabel, value]);
      }} else if (!Number.isNaN(Number(value))) {{
        const digits = pointLayer.valueKind === "probability" ? 3 : 1;
        rows.push([pointLayer.popupLabel, formatNumber(value, digits)]);
      }}
    }}
  }}
  const table = rows.map(row => `<tr><th>${{escapeHtml(row[0])}}</th><td>${{escapeHtml(row[1])}}</td></tr>`).join("");
  const copyText = `${{props.latitude}},${{props.longitude}}`;
  return `<table>${{table}}</table><button class="copy-btn" onclick="navigator.clipboard.writeText('${{copyText}}')">Copy lat/lon</button>`;
}}
function formatNumber(value, digits) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  if (typeof value === "number") return value.toFixed(digits);
  return value;
}}
function escapeHtml(value) {{
  return String(value ?? "").replace(/[&<>"']/g, char => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[char]));
}}
</script>
</body>
</html>
"""


def point_layer_payload(layer: PointLayer) -> dict[str, object]:
    """Return JSON-friendly metadata for one coordinate-based point layer."""
    return {
        "id": layer.layer_id,
        "displayName": layer.display_name,
        "type": layer.layer_type,
        "valueKind": layer.value_kind,
        "propertyName": layer.property_name,
        "popupLabel": layer.popup_label,
        "scaleMin": layer.scale_min,
        "scaleMax": layer.scale_max,
        "diverging": layer.diverging,
        "defaultVisible": layer.default_visible,
        "minAbsDisplayValue": layer.min_abs_display_value,
        "allowedValues": list(layer.allowed_values) if layer.allowed_values is not None else [],
        "filter": point_layer_filter_payload(layer),
        "legend": point_layer_legend_payload(layer),
    }


def point_layer_filter_payload(layer: PointLayer) -> dict[str, object]:
    """Return JSON-friendly filter defaults for one point layer."""
    return {
        "mode": layer.filter_mode,
        "label": layer.filter_label,
        "defaultMinValue": layer.filter_default_min,
        "defaultValues": list(layer.filter_default_values),
        "allowedValues": list(layer.allowed_values) if layer.allowed_values is not None else [],
    }


def point_layer_legend_payload(layer: PointLayer) -> dict[str, object]:
    """Return JSON-friendly legend metadata for one point layer."""
    return {
        "kind": layer.value_kind,
        "unit": layer.legend_unit,
        "description": layer.legend_description,
        "scaleMin": layer.scale_min,
        "scaleMax": layer.scale_max,
        "diverging": layer.diverging,
    }


def filter_config_payload(filter_config: VisualizerFilterConfig) -> dict[str, object]:
    """Return JSON-friendly global and layer-specific filter defaults."""
    return {
        "continuous_min_area_m2": filter_config.continuous_min_area_m2,
        "label_min_area_m2": filter_config.label_min_area_m2,
        "residual_min_abs_area_m2": filter_config.residual_min_abs_area_m2,
        "probability_min": filter_config.probability_min,
        "binary_outcomes_visible": list(filter_config.binary_outcomes_visible),
        "layer_overrides": {
            layer_id: layer_filter_config_payload(override)
            for layer_id, override in filter_config.layer_overrides.items()
        },
    }


def layer_filter_config_payload(override: LayerFilterDefaults) -> dict[str, object]:
    """Return JSON-friendly layer-specific filter override values."""
    return {
        "continuous_min_area_m2": override.continuous_min_area_m2,
        "residual_min_abs_area_m2": override.residual_min_abs_area_m2,
        "probability_min": override.probability_min,
        "binary_outcomes_visible": list(override.binary_outcomes_visible)
        if override.binary_outcomes_visible is not None
        else None,
    }


def relative_asset_path(html_path: Path, asset_path: Path) -> str:
    """Return a POSIX relative path from the HTML file to a generated asset."""
    return Path(os.path.relpath(asset_path, html_path.parent)).as_posix()


def write_manifest(
    *,
    viewer_config: ResultsVisualizerConfig,
    layer_frames: dict[ResultsLayerConfig, pd.DataFrame],
    point_layers: list[PointLayer],
    grid: GridSpec,
    canopy_scale: float,
    residual_scale: float,
    inspection_frame: pd.DataFrame,
    inspection_selection: InspectionSelection,
    inspection_geojson_path: Path,
    inspection_js_path: Path,
) -> None:
    """Write a JSON manifest describing visualizer inputs, scales, and outputs."""
    payload = {
        "command": "visualize-results",
        "config_path": str(viewer_config.config_path),
        "split": viewer_config.split,
        "year": viewer_config.year,
        "mask_status": mask_status(viewer_config.domain_mask),
        "evaluation_scope": evaluation_scope(viewer_config.domain_mask),
        "domain_mask": (
            {
                "table": str(viewer_config.domain_mask.table_path),
                "manifest": str(viewer_config.domain_mask.manifest_path)
                if viewer_config.domain_mask.manifest_path is not None
                else None,
                "primary_domain": viewer_config.domain_mask.primary_domain,
            }
            if viewer_config.domain_mask is not None
            else None
        ),
        "grid": {
            "min_row": grid.min_row,
            "max_row": grid.max_row,
            "min_col": grid.min_col,
            "max_col": grid.max_col,
            "width": grid.width,
            "height": grid.height,
            "bounds": {
                "west": grid.west,
                "south": grid.south,
                "east": grid.east,
                "north": grid.north,
            },
            "bounds_source": "selected row longitude/latitude extent",
        },
        "color_scales": {
            "canopy_area_m2": {"min": 0.0, "max": canopy_scale},
            "residual_m2": {"min": -residual_scale, "max": residual_scale},
            "probability": {"min": 0.0, "max": 1.0},
            "robust_percentile": viewer_config.robust_percentile,
        },
        "basemap": {
            "enabled": viewer_config.basemap.enabled,
            "name": viewer_config.basemap.name,
            "url_template": viewer_config.basemap.url_template,
            "attribution": viewer_config.basemap.attribution,
            "runtime_dependency_only": True,
        },
        "filter_defaults": filter_config_payload(viewer_config.filters),
        "layers": [
            {
                "layer_id": layer.layer_id,
                "display_name": layer.display_name,
                "type": layer.layer_type,
                "model_name": layer.model_name,
                "input_path": str(layer.path),
                "row_count": int(len(frame)),
            }
            for layer, frame in layer_frames.items()
        ],
        "point_layers": [
            {
                "layer_id": layer.layer_id,
                "display_name": layer.display_name,
                "type": layer.layer_type,
                "value_kind": layer.value_kind,
                "property_name": layer.property_name,
                "popup_label": layer.popup_label,
                "scale_min": layer.scale_min,
                "scale_max": layer.scale_max,
                "min_abs_display_value": layer.min_abs_display_value,
                "allowed_values": list(layer.allowed_values)
                if layer.allowed_values is not None
                else [],
                "default_visible": layer.default_visible,
                "filter": point_layer_filter_payload(layer),
                "legend": point_layer_legend_payload(layer),
                "coordinate_based": True,
            }
            for layer in point_layers
        ],
        "inspection": {
            "row_count": int(len(inspection_frame)),
            "max_points": viewer_config.max_inspection_points,
            "cap_was_enough_for_priority_buckets": (
                inspection_selection.cap_was_enough_for_priority
            ),
            "selection_buckets": inspection_selection.bucket_counts,
            "csv": str(viewer_config.inspection_points_path),
            "geojson": str(inspection_geojson_path),
            "javascript": str(inspection_js_path),
        },
        "outputs": {
            "html": str(viewer_config.html_path),
            "asset_dir": str(viewer_config.asset_dir),
            "manifest": str(viewer_config.manifest_path),
        },
        "review_semantics": (
            "Qualitative visual QA only; Planet or Kelpwatch comparisons should not tune "
            "2022 test metrics, thresholds, masks, or model selection."
        ),
    }
    write_json(viewer_config.manifest_path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON document with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def slugify(value: str) -> str:
    """Return a filesystem- and JavaScript-friendly identifier."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "layer"


def write_inspection_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write inspection rows to CSV with a stable union field order."""
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
