"""Visual QA for downloaded Kelpwatch source data."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import logging
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import geopandas as gpd  # type: ignore[import-untyped]
import matplotlib
import numpy as np
import rasterio  # type: ignore[import-untyped]
import xarray as xr
from rasterio.features import geometry_mask  # type: ignore[import-untyped]
from rasterio.transform import Affine  # type: ignore[import-untyped]
from shapely.geometry.base import BaseGeometry

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.labels.kelpwatch import NETCDF_ENGINE, identify_label_source_variable
from kelp_aef.regions import region_display_name, region_output_slug

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_PREVIEW_MAX_PIXELS = 500_000
NODATA_VALUE = -9999.0
YEAR_DIM = "year"
SUMMARY_FIELDS = (
    "year",
    "pixel_count",
    "valid_count",
    "missing_count",
    "zero_count",
    "nonzero_count",
    "min",
    "median",
    "p95",
    "p99",
    "max",
    "aggregate_canopy_area",
)
X_COORD_CANDIDATES = ("x", "lon", "longitude", "easting")
Y_COORD_CANDIDATES = ("y", "lat", "latitude", "northing")
TIME_COORD_CANDIDATES = ("time", "date")


@dataclass(frozen=True)
class KelpwatchVisualizeConfig:
    """Resolved config values needed by the Kelpwatch visualization step."""

    config_path: Path
    region_name: str
    region_slug: str
    region_display_name: str
    years: tuple[int, ...]
    target: str
    aggregation: str
    footprint_path: Path
    footprint_crs: str
    source_manifest: Path
    figures_dir: Path
    tables_dir: Path
    interim_dir: Path
    contact_sheet_path: Path
    time_series_path: Path
    summary_table_path: Path
    geotiff_path: Path
    html_path: Path
    qa_json_path: Path


@dataclass(frozen=True)
class SpatialContext:
    """Spatial metadata for a prepared Monterey subset."""

    layout: str
    x_name: str
    y_name: str
    crs: str
    transform: Affine
    footprint: BaseGeometry
    footprint_bounds: tuple[float, float, float, float]
    raster_height: int
    raster_width: int
    pixel_count: int
    spatial_dims: tuple[str, ...]
    station_name: str | None = None
    station_rows: np.ndarray | None = None
    station_cols: np.ndarray | None = None


def load_kelpwatch_visualize_config(config_path: Path) -> KelpwatchVisualizeConfig:
    """Load Kelpwatch visualization settings from the workflow config."""
    config = load_yaml_config(config_path)
    years_config = require_mapping(config.get("years"), "years")
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    reports = require_mapping(config.get("reports"), "reports")

    raw_years = years_config.get("smoke")
    if not isinstance(raw_years, list):
        msg = "config field must be a list of years: years.smoke"
        raise ValueError(msg)
    years = tuple(require_int_value(year, "years.smoke[]") for year in raw_years)
    region_name = require_string(region.get("name"), "region.name")
    slug = region_output_slug(region_name)

    figures_dir = Path(require_string(reports.get("figures_dir"), "reports.figures_dir"))
    tables_dir = Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))
    source_manifest = Path(
        require_string(label_paths.get("source_manifest"), "labels.paths.source_manifest")
    )
    interim_dir = source_manifest.parent

    return KelpwatchVisualizeConfig(
        config_path=config_path,
        region_name=region_name,
        region_slug=slug,
        region_display_name=region_display_name(region_name),
        years=years,
        target=require_string(labels.get("target"), "labels.target"),
        aggregation=require_string(labels.get("aggregation"), "labels.aggregation"),
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        footprint_crs=str(region.get("crs", "EPSG:4326")),
        source_manifest=source_manifest,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        interim_dir=interim_dir,
        contact_sheet_path=figures_dir / f"kelpwatch_{slug}_annual_max_qa.png",
        time_series_path=figures_dir / f"kelpwatch_{slug}_quarterly_timeseries_qa.png",
        summary_table_path=tables_dir / f"kelpwatch_{slug}_source_qa.csv",
        geotiff_path=interim_dir / f"kelpwatch_{slug}_annual_max_{years[0]}_{years[-1]}.tif",
        html_path=figures_dir / f"kelpwatch_{slug}_interactive_qa.html",
        qa_json_path=interim_dir / f"kelpwatch_{slug}_source_qa.json",
    )


def visualize_kelpwatch(
    config_path: Path,
    *,
    variable: str | None = None,
    preview_max_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
) -> int:
    """Run the Kelpwatch source visual QA stage."""
    qa_config = load_kelpwatch_visualize_config(config_path)
    LOGGER.info("Loading Kelpwatch source manifest: %s", qa_config.source_manifest)
    manifest = load_json_object(qa_config.source_manifest)
    netcdf_path = source_netcdf_path_from_manifest(manifest)
    selected_variable = variable or label_variable_from_manifest(manifest, qa_config)

    LOGGER.info("Opening Kelpwatch NetCDF: %s", netcdf_path)
    with xr.open_dataset(netcdf_path, engine=NETCDF_ENGINE, decode_cf=True) as dataset:
        if selected_variable is None:
            selected_variable = infer_label_variable(dataset, qa_config)
        LOGGER.info("Using Kelpwatch variable for QA: %s", selected_variable)
        data = dataset[selected_variable]
        time_name = detect_time_name(data)
        spatial_subset, spatial_context = subset_to_footprint(data, dataset, qa_config)
        time_subset = subset_to_years(spatial_subset, time_name, qa_config.years)
        annual_max = build_annual_max(time_subset, time_name, qa_config.years)
        summary_rows = build_summary_rows(annual_max, spatial_context)
        time_series_rows = build_time_series_rows(time_subset, time_name, spatial_context)

        write_geotiff(annual_max, spatial_context, qa_config.geotiff_path)
        write_summary_csv(summary_rows, qa_config.summary_table_path)
        color_scale = color_scale_from_annual_max(annual_max)
        write_contact_sheet(
            annual_max=annual_max,
            spatial_context=spatial_context,
            output_path=qa_config.contact_sheet_path,
            color_scale=color_scale,
            region_display=qa_config.region_display_name,
        )
        write_time_series_figure(
            time_series_rows,
            qa_config.time_series_path,
            region_display=qa_config.region_display_name,
        )
        write_interactive_html(
            annual_max=annual_max,
            spatial_context=spatial_context,
            output_path=qa_config.html_path,
            color_scale=color_scale,
            preview_max_pixels=preview_max_pixels,
            region_display=qa_config.region_display_name,
        )
        write_qa_json(
            qa_config=qa_config,
            netcdf_path=netcdf_path,
            variable=selected_variable,
            spatial_context=spatial_context,
            color_scale=color_scale,
            output_path=qa_config.qa_json_path,
        )

    LOGGER.info("Wrote Kelpwatch annual-max contact sheet: %s", qa_config.contact_sheet_path)
    LOGGER.info("Wrote Kelpwatch quarterly time series: %s", qa_config.time_series_path)
    LOGGER.info("Wrote Kelpwatch GIS export: %s", qa_config.geotiff_path)
    LOGGER.info("Wrote Kelpwatch interactive QA viewer: %s", qa_config.html_path)
    LOGGER.info("Wrote Kelpwatch source QA table: %s", qa_config.summary_table_path)
    return 0


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"expected JSON object at {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def source_netcdf_path_from_manifest(manifest: dict[str, Any]) -> Path:
    """Read the downloaded NetCDF path from a Kelpwatch source manifest."""
    transfer = require_mapping(manifest.get("transfer"), "manifest.transfer")
    local_path = Path(require_string(transfer.get("local_path"), "manifest.transfer.local_path"))
    if not local_path.exists():
        msg = f"Kelpwatch NetCDF does not exist: {local_path}"
        raise FileNotFoundError(msg)
    return local_path


def label_variable_from_manifest(
    manifest: dict[str, Any], qa_config: KelpwatchVisualizeConfig
) -> str | None:
    """Read the selected Kelpwatch label source variable from a source manifest."""
    label_source = manifest.get("label_source")
    if not isinstance(label_source, dict):
        return None
    variable = label_source.get("selected_variable")
    if variable is None:
        return None
    selected_variable = require_string(variable, "manifest.label_source.selected_variable")
    preferred_variable = preferred_manifest_candidate(label_source, qa_config)
    if preferred_variable is not None and preferred_variable != selected_variable:
        LOGGER.warning(
            "Using preferred Kelpwatch label variable %s instead of manifest selection %s",
            preferred_variable,
            selected_variable,
        )
        return preferred_variable
    return selected_variable


def preferred_manifest_candidate(
    label_source: dict[str, Any], qa_config: KelpwatchVisualizeConfig
) -> str | None:
    """Re-score manifest candidates with current label-source heuristics."""
    candidates = label_source.get("candidates")
    if not isinstance(candidates, list):
        return None
    variables = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        variables.append(
            {
                "name": candidate.get("name"),
                "attrs": {
                    "units": candidate.get("units"),
                    "long_name": candidate.get("long_name"),
                },
            }
        )
    if not variables:
        return None
    rescored = identify_label_source_variable(
        netcdf_metadata={"variables": variables},
        target=qa_config.target,
        aggregation=qa_config.aggregation,
    )
    selected = rescored.get("selected_variable")
    return selected if isinstance(selected, str) else None


def infer_label_variable(dataset: xr.Dataset, qa_config: KelpwatchVisualizeConfig) -> str:
    """Infer the Kelpwatch label variable from live NetCDF metadata."""
    netcdf_metadata = {
        "variables": [
            {
                "name": str(name),
                "attrs": {str(key): value for key, value in variable.attrs.items()},
            }
            for name, variable in dataset.data_vars.items()
        ]
    }
    label_source = identify_label_source_variable(
        netcdf_metadata=netcdf_metadata,
        target=qa_config.target,
        aggregation=qa_config.aggregation,
    )
    selected = label_source.get("selected_variable")
    if not isinstance(selected, str):
        candidates = [str(name) for name in dataset.data_vars]
        msg = (
            "Could not infer Kelpwatch label variable. "
            f"Pass --variable explicitly. Data variables: {candidates}"
        )
        raise ValueError(msg)
    return selected


def detect_time_name(data: xr.DataArray) -> str:
    """Detect the time coordinate or dimension name for a Kelpwatch variable."""
    for name in TIME_COORD_CANDIDATES:
        if name in data.dims or name in data.coords:
            return name
    for dim_name in data.dims:
        if "time" in str(dim_name).lower() or "date" in str(dim_name).lower():
            return str(dim_name)
    msg = f"could not identify time dimension for variable {data.name!r}: {data.dims}"
    raise ValueError(msg)


def subset_to_footprint(
    data: xr.DataArray,
    dataset: xr.Dataset,
    qa_config: KelpwatchVisualizeConfig,
) -> tuple[xr.DataArray, SpatialContext]:
    """Subset a Kelpwatch variable to the configured footprint and mask geometry."""
    station_layout = detect_station_layout(data, dataset)
    if station_layout is not None:
        station_name, x_name, y_name = station_layout
        return subset_station_to_footprint(
            data=data,
            dataset=dataset,
            qa_config=qa_config,
            station_name=station_name,
            x_name=x_name,
            y_name=y_name,
        )

    return subset_rectilinear_to_footprint(data, dataset, qa_config)


def detect_station_layout(data: xr.DataArray, dataset: xr.Dataset) -> tuple[str, str, str] | None:
    """Detect Kelpwatch's point-station layout with lon/lat variables."""
    for dim_name in data.dims:
        x_name = find_one_dimensional_dataset_variable(dataset, X_COORD_CANDIDATES, str(dim_name))
        y_name = find_one_dimensional_dataset_variable(dataset, Y_COORD_CANDIDATES, str(dim_name))
        if x_name is not None and y_name is not None:
            return str(dim_name), x_name, y_name
    return None


def find_one_dimensional_dataset_variable(
    dataset: xr.Dataset, candidates: tuple[str, ...], dim_name: str
) -> str | None:
    """Find a one-dimensional dataset variable by candidate name and dimension."""
    lower_to_name = {str(name).lower(): str(name) for name in dataset.variables}
    for candidate in candidates:
        name = lower_to_name.get(candidate)
        if name is None:
            continue
        variable = dataset[name]
        if tuple(str(dim) for dim in variable.dims) == (dim_name,):
            return name
    return None


def subset_station_to_footprint(
    *,
    data: xr.DataArray,
    dataset: xr.Dataset,
    qa_config: KelpwatchVisualizeConfig,
    station_name: str,
    x_name: str,
    y_name: str,
) -> tuple[xr.DataArray, SpatialContext]:
    """Subset sparse station-style Kelpwatch data to footprint-contained stations."""
    dataset_crs = infer_dataset_crs(dataset, data, x_name, y_name)
    footprint = read_footprint(qa_config.footprint_path, qa_config.footprint_crs, dataset_crs)
    x_values = one_dimensional_variable_values(dataset, x_name)
    y_values = one_dimensional_variable_values(dataset, y_name)
    if x_values.size != data.sizes[station_name] or y_values.size != data.sizes[station_name]:
        msg = f"station coordinate lengths do not match data dimension: {station_name}"
        raise ValueError(msg)

    minx, miny, maxx, maxy = footprint.bounds
    bbox_mask = (x_values >= minx) & (x_values <= maxx) & (y_values >= miny) & (y_values <= maxy)
    candidate_indices = np.flatnonzero(bbox_mask)
    if candidate_indices.size == 0:
        msg = "Kelpwatch station footprint subset is empty before geometry filtering"
        raise ValueError(msg)

    candidate_frame = gpd.GeoDataFrame(
        {"station_index": candidate_indices},
        geometry=gpd.points_from_xy(x_values[candidate_indices], y_values[candidate_indices]),
        crs=dataset_crs,
    )
    inside_mask = candidate_frame.geometry.intersects(footprint)
    station_indices = np.asarray(candidate_frame.loc[inside_mask, "station_index"], dtype=int)
    if station_indices.size == 0:
        msg = "Kelpwatch station footprint subset is empty after geometry filtering"
        raise ValueError(msg)

    subset = data.isel({station_name: station_indices})
    station_x = x_values[station_indices]
    station_y = y_values[station_indices]
    pixel_width, pixel_height = infer_station_pixel_size(dataset, station_x, station_y)
    transform, raster_height, raster_width, rows, cols = station_raster_grid(
        station_x,
        station_y,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )
    spatial_context = SpatialContext(
        layout="station",
        x_name=x_name,
        y_name=y_name,
        crs=dataset_crs,
        transform=transform,
        footprint=footprint,
        footprint_bounds=footprint.bounds,
        raster_height=raster_height,
        raster_width=raster_width,
        pixel_count=int(station_indices.size),
        spatial_dims=(station_name,),
        station_name=station_name,
        station_rows=rows,
        station_cols=cols,
    )
    LOGGER.info(
        "Selected %s Kelpwatch stations inside footprint; QA raster grid is %sx%s",
        station_indices.size,
        raster_width,
        raster_height,
    )
    return subset, spatial_context


def subset_rectilinear_to_footprint(
    data: xr.DataArray,
    dataset: xr.Dataset,
    qa_config: KelpwatchVisualizeConfig,
) -> tuple[xr.DataArray, SpatialContext]:
    """Subset rectilinear Kelpwatch data to the configured footprint."""
    x_name = detect_spatial_name(data, X_COORD_CANDIDATES, "x")
    y_name = detect_spatial_name(data, Y_COORD_CANDIDATES, "y")
    dataset_crs = infer_dataset_crs(dataset, data, x_name, y_name)
    footprint = read_footprint(qa_config.footprint_path, qa_config.footprint_crs, dataset_crs)
    minx, miny, maxx, maxy = footprint.bounds

    subset = subset_by_coordinate_bounds(data, x_name, y_name, minx, miny, maxx, maxy)
    subset = subset.sortby(x_name, ascending=True).sortby(y_name, ascending=False)
    x_values = one_dimensional_coordinate_values(subset, x_name)
    y_values = one_dimensional_coordinate_values(subset, y_name)
    if x_values.size == 0 or y_values.size == 0:
        msg = "Kelpwatch footprint subset is empty"
        raise ValueError(msg)

    transform = transform_from_coordinates(x_values, y_values)
    mask = geometry_mask(
        [footprint.__geo_interface__],
        out_shape=(len(y_values), len(x_values)),
        transform=transform,
        invert=True,
    )
    mask_array = xr.DataArray(
        mask,
        dims=(y_name, x_name),
        coords={y_name: y_values, x_name: x_values},
    )
    masked = subset.where(mask_array)
    spatial_context = SpatialContext(
        layout="rectilinear",
        x_name=x_name,
        y_name=y_name,
        crs=dataset_crs,
        transform=transform,
        footprint=footprint,
        footprint_bounds=footprint.bounds,
        raster_height=len(y_values),
        raster_width=len(x_values),
        pixel_count=int(np.count_nonzero(mask)),
        spatial_dims=(y_name, x_name),
    )
    return masked, spatial_context


def detect_spatial_name(data: xr.DataArray, candidates: tuple[str, ...], axis_name: str) -> str:
    """Detect one spatial coordinate or dimension name from candidate names."""
    lower_to_name = {str(name).lower(): str(name) for name in (*data.dims, *data.coords)}
    for candidate in candidates:
        if candidate in lower_to_name:
            return lower_to_name[candidate]
    msg = f"could not identify {axis_name} coordinate for variable {data.name!r}: {data.dims}"
    raise ValueError(msg)


def infer_dataset_crs(dataset: xr.Dataset, data: xr.DataArray, x_name: str, y_name: str) -> str:
    """Infer the CRS for Kelpwatch spatial coordinates."""
    variable_attrs = (variable.attrs for variable in dataset.variables.values())
    for attrs in [dataset.attrs, data.attrs, *variable_attrs]:
        for key in (
            "crs",
            "spatial_ref",
            "crs_wkt",
            "epsg_code",
            "geospatial_bounds_crs",
            "coordinate_reference_frame",
        ):
            value = attrs.get(key)
            if value is not None:
                return normalize_crs(str(value))

    x_values = one_dimensional_variable_values(dataset, x_name, fallback=data)
    y_values = one_dimensional_variable_values(dataset, y_name, fallback=data)
    if coordinates_look_like_lon_lat(x_values, y_values):
        return "EPSG:4326"
    msg = "could not infer Kelpwatch NetCDF CRS; add CRS metadata before visualization"
    raise ValueError(msg)


def normalize_crs(value: str) -> str:
    """Normalize common CRS metadata strings into GDAL-readable CRS values."""
    if "EPSG:4326" in value.upper():
        return "EPSG:4326"
    return value


def coordinates_look_like_lon_lat(x_values: np.ndarray, y_values: np.ndarray) -> bool:
    """Return whether coordinate ranges are compatible with longitude/latitude."""
    if x_values.size == 0 or y_values.size == 0:
        return False
    return bool(
        np.nanmin(x_values) >= -180
        and np.nanmax(x_values) <= 180
        and np.nanmin(y_values) >= -90
        and np.nanmax(y_values) <= 90
    )


def read_footprint(path: Path, source_crs: str, target_crs: str) -> BaseGeometry:
    """Read and reproject the configured footprint geometry with GeoPandas."""
    dataframe = gpd.read_file(path)
    if dataframe.empty:
        msg = f"footprint GeoJSON contains no features: {path}"
        raise ValueError(msg)
    if dataframe.crs is None:
        dataframe = dataframe.set_crs(source_crs)
    dataframe = dataframe.to_crs(target_crs)
    geometry = dataframe.geometry.iloc[0]
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        msg = f"footprint geometry is empty or invalid: {path}"
        raise ValueError(msg)
    return geometry


def subset_by_coordinate_bounds(
    data: xr.DataArray,
    x_name: str,
    y_name: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> xr.DataArray:
    """Subset a DataArray by spatial coordinate bounds."""
    x_values = one_dimensional_coordinate_values(data, x_name)
    y_values = one_dimensional_coordinate_values(data, y_name)
    x_slice = slice(minx, maxx) if coordinate_is_ascending(x_values) else slice(maxx, minx)
    y_slice = slice(miny, maxy) if coordinate_is_ascending(y_values) else slice(maxy, miny)
    return data.sel({x_name: x_slice, y_name: y_slice})


def one_dimensional_coordinate_values(
    container: xr.DataArray | xr.Dataset, name: str
) -> np.ndarray:
    """Return a spatial coordinate as a one-dimensional NumPy array."""
    return one_dimensional_variable_values(container, name)


def one_dimensional_variable_values(
    container: xr.DataArray | xr.Dataset,
    name: str,
    *,
    fallback: xr.DataArray | xr.Dataset | None = None,
) -> np.ndarray:
    """Return a one-dimensional coordinate or variable as floating-point values."""
    source = container if name in container.coords or name in container else fallback
    if source is None:
        msg = f"coordinate or variable is not present: {name}"
        raise ValueError(msg)
    values = source.coords[name].values if name in source.coords else source[name].values
    array = np.asarray(values)
    if array.ndim != 1:
        msg = f"coordinate must be one-dimensional for this QA stage: {name}"
        raise ValueError(msg)
    return array.astype(float)


def coordinate_is_ascending(values: np.ndarray) -> bool:
    """Return whether a coordinate vector is ascending."""
    if values.size < 2:
        return True
    return bool(values[0] < values[-1])


def transform_from_coordinates(x_values: np.ndarray, y_values: np.ndarray) -> Affine:
    """Build a raster transform from center coordinates sorted x asc and y desc."""
    pixel_width = coordinate_spacing(x_values)
    pixel_height = abs(coordinate_spacing(y_values))
    left = float(x_values[0]) - pixel_width / 2
    top = float(y_values[0]) + pixel_height / 2
    return rasterio.transform.from_origin(left, top, pixel_width, pixel_height)


def station_raster_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    pixel_width: float | None = None,
    pixel_height: float | None = None,
) -> tuple[Affine, int, int, np.ndarray, np.ndarray]:
    """Build a dense raster grid index for sparse Kelpwatch station centers."""
    x_unique = np.unique(np.sort(x_values.astype(float)))
    y_unique = np.unique(np.sort(y_values.astype(float)))
    pixel_width = pixel_width or coordinate_spacing(x_unique)
    pixel_height = pixel_height or coordinate_spacing(y_unique)
    minx = float(np.nanmin(x_values))
    maxx = float(np.nanmax(x_values))
    miny = float(np.nanmin(y_values))
    maxy = float(np.nanmax(y_values))
    raster_width = int(round((maxx - minx) / pixel_width)) + 1
    raster_height = int(round((maxy - miny) / pixel_height)) + 1
    transform = rasterio.transform.from_origin(
        minx - pixel_width / 2,
        maxy + pixel_height / 2,
        pixel_width,
        pixel_height,
    )
    rows = np.rint((maxy - y_values) / pixel_height).astype(int)
    cols = np.rint((x_values - minx) / pixel_width).astype(int)
    if (
        np.any(rows < 0)
        or np.any(rows >= raster_height)
        or np.any(cols < 0)
        or np.any(cols >= raster_width)
    ):
        msg = "computed station raster indices outside raster extent"
        raise ValueError(msg)
    return transform, raster_height, raster_width, rows, cols


def infer_station_pixel_size(
    dataset: xr.Dataset, x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float]:
    """Infer sparse station pixel size from metadata before falling back to coordinates."""
    lon_resolution = positive_float_attr(dataset.attrs.get("geospatial_lon_resolution"))
    lat_resolution = positive_float_attr(dataset.attrs.get("geospatial_lat_resolution"))
    if lon_resolution is not None and lat_resolution is not None:
        return lon_resolution, lat_resolution
    return (
        coordinate_spacing(np.unique(np.sort(x_values.astype(float)))),
        coordinate_spacing(np.unique(np.sort(y_values.astype(float)))),
    )


def positive_float_attr(value: object) -> float | None:
    """Return a positive float metadata attribute value when available."""
    if value is None:
        return None
    try:
        parsed = abs(float(cast(Any, value)))
    except (TypeError, ValueError):
        return None
    if parsed <= 0 or not np.isfinite(parsed):
        return None
    return parsed


def coordinate_spacing(values: np.ndarray) -> float:
    """Return absolute spacing for a regular coordinate vector."""
    if values.size < 2:
        return 1.0
    diffs = np.diff(values.astype(float))
    return float(abs(np.nanmedian(diffs)))


def subset_to_years(data: xr.DataArray, time_name: str, years: tuple[int, ...]) -> xr.DataArray:
    """Subset a DataArray to requested years using the detected time coordinate."""
    source_values = (
        data.coords[time_name].values if time_name in data.coords else data[time_name].values
    )
    time_values = np.asarray(source_values)
    time_years = years_from_time_values(time_values)
    mask = np.isin(time_years, np.asarray(years))
    if not np.any(mask):
        msg = f"Kelpwatch data contains no requested years: {years}"
        raise ValueError(msg)
    return data.isel({time_name: np.flatnonzero(mask)})


def years_from_time_values(values: np.ndarray) -> np.ndarray:
    """Convert common time coordinate encodings into calendar years."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[Y]").astype(int) + 1970
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(int)
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(int)
    return np.asarray([int(str(value)[:4]) for value in values], dtype=int)


def build_annual_max(data: xr.DataArray, time_name: str, years: tuple[int, ...]) -> xr.DataArray:
    """Compute annual maximum canopy values for visualization only."""
    source_values = (
        data.coords[time_name].values if time_name in data.coords else data[time_name].values
    )
    time_values = np.asarray(source_values)
    time_years = years_from_time_values(time_values)
    arrays: list[xr.DataArray] = []
    selected_years: list[int] = []
    for year in years:
        indices = np.flatnonzero(time_years == year)
        if indices.size == 0:
            continue
        arrays.append(data.isel({time_name: indices}).max(dim=time_name, skipna=True))
        selected_years.append(year)
    if not arrays:
        msg = f"could not compute annual max; no requested years found: {years}"
        raise ValueError(msg)
    annual = xr.concat(arrays, dim=YEAR_DIM)
    return annual.assign_coords({YEAR_DIM: np.asarray(selected_years, dtype=np.int16)})


def build_summary_rows(
    annual_max: xr.DataArray, spatial_context: SpatialContext
) -> list[dict[str, object]]:
    """Build per-year value and missingness summary rows."""
    rows: list[dict[str, object]] = []
    pixel_count = spatial_context.pixel_count
    for year in annual_max.coords[YEAR_DIM].values:
        values = annual_layer_values(annual_max.sel({YEAR_DIM: year}), spatial_context)
        finite = values[np.isfinite(values)]
        nonzero = finite[finite > 0]
        rows.append(
            {
                "year": int(year),
                "pixel_count": int(pixel_count),
                "valid_count": int(finite.size),
                "missing_count": int(pixel_count - finite.size),
                "zero_count": int(np.count_nonzero(finite == 0)),
                "nonzero_count": int(nonzero.size),
                "min": safe_nan_stat(finite, np.nanmin),
                "median": safe_nan_stat(finite, np.nanmedian),
                "p95": safe_nan_percentile(finite, 95),
                "p99": safe_nan_percentile(finite, 99),
                "max": safe_nan_stat(finite, np.nanmax),
                "aggregate_canopy_area": float(np.nansum(finite)) if finite.size else math.nan,
            }
        )
    return rows


def safe_nan_stat(values: np.ndarray, fn: Any) -> float:
    """Return a NaN-aware statistic or NaN for empty arrays."""
    if values.size == 0:
        return math.nan
    return float(fn(values))


def safe_nan_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a percentile or NaN for empty arrays."""
    if values.size == 0:
        return math.nan
    return float(np.nanpercentile(values, percentile))


def build_time_series_rows(
    data: xr.DataArray, time_name: str, spatial_context: SpatialContext
) -> list[dict[str, object]]:
    """Build aggregate quarterly canopy time-series rows."""
    rows: list[dict[str, object]] = []
    source_values = (
        data.coords[time_name].values if time_name in data.coords else data[time_name].values
    )
    time_values = np.asarray(source_values)
    time_years = years_from_time_values(time_values)
    aggregate = data.sum(dim=spatial_context.spatial_dims, skipna=True)
    for index, value in enumerate(np.asarray(aggregate.values, dtype=float)):
        rows.append(
            {
                "index": index,
                "label": time_label(time_values[index]),
                "year": int(time_years[index]),
                "aggregate_canopy_area": float(value),
            }
        )
    return rows


def time_label(value: object) -> str:
    """Return a concise string label for a time coordinate value."""
    if isinstance(value, np.datetime64):
        return str(value)[:10]
    return str(value)


def color_scale_from_annual_max(annual_max: xr.DataArray) -> dict[str, float]:
    """Choose a shared color scale for annual maximum maps."""
    values = np.asarray(annual_max.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"vmin": 0.0, "vmax": 1.0}
    vmax = float(np.nanpercentile(finite, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(finite)) if np.nanmax(finite) > 0 else 1.0
    return {"vmin": 0.0, "vmax": vmax}


def write_geotiff(
    annual_max: xr.DataArray, spatial_context: SpatialContext, output_path: Path
) -> None:
    """Write annual maximum Kelpwatch QA values as a multiband GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(
        [
            annual_layer_to_grid(annual_max.sel({YEAR_DIM: year}), spatial_context)
            for year in annual_max.coords[YEAR_DIM].values
        ]
    ).astype(np.float32)
    data = np.where(np.isfinite(data), data, NODATA_VALUE).astype(np.float32)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=spatial_context.raster_height,
        width=spatial_context.raster_width,
        count=data.shape[0],
        dtype="float32",
        crs=spatial_context.crs,
        transform=spatial_context.transform,
        nodata=NODATA_VALUE,
        compress="deflate",
    ) as dataset:
        for band_index, year in enumerate(annual_max.coords[YEAR_DIM].values, start=1):
            dataset.write(data[band_index - 1], band_index)
            dataset.set_band_description(band_index, str(int(year)))


def annual_layer_values(layer: xr.DataArray, spatial_context: SpatialContext) -> np.ndarray:
    """Return the footprint-contained source values for one annual layer."""
    values = np.asarray(layer.values, dtype=float).ravel()
    if spatial_context.layout == "station":
        return values
    return cast(np.ndarray, values[np.isfinite(values)])


def annual_layer_to_grid(layer: xr.DataArray, spatial_context: SpatialContext) -> np.ndarray:
    """Render one annual source layer to the dense QA raster grid."""
    values = np.asarray(layer.values, dtype=float)
    if spatial_context.layout != "station":
        return values
    if spatial_context.station_rows is None or spatial_context.station_cols is None:
        msg = "station layout is missing raster row/column indices"
        raise ValueError(msg)
    grid = np.full(
        (spatial_context.raster_height, spatial_context.raster_width),
        np.nan,
        dtype=float,
    )
    grid[spatial_context.station_rows, spatial_context.station_cols] = values.ravel()
    return grid


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write per-year Kelpwatch QA summary rows as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_sheet(
    *,
    annual_max: xr.DataArray,
    spatial_context: SpatialContext,
    output_path: Path,
    color_scale: dict[str, float],
    region_display: str,
) -> None:
    """Write a static annual maximum map contact sheet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    years = [int(year) for year in annual_max.coords[YEAR_DIM].values]
    column_count = min(3, len(years))
    row_count = int(math.ceil(len(years) / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(4 * column_count, 4 * row_count))
    axes_array = np.asarray(axes).reshape(-1)
    image = None
    for axis, year in zip(axes_array, years, strict=False):
        values = annual_layer_to_grid(annual_max.sel({YEAR_DIM: year}), spatial_context)
        image = axis.imshow(
            values,
            extent=extent_from_context(spatial_context),
            origin="upper",
            vmin=color_scale["vmin"],
            vmax=color_scale["vmax"],
            cmap="viridis",
        )
        plot_footprint(axis, spatial_context)
        axis.set_title(str(year))
        axis.set_xlabel(spatial_context.x_name)
        axis.set_ylabel(spatial_context.y_name)
    for axis in axes_array[len(years) :]:
        axis.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes_array[: len(years)].tolist(), shrink=0.85, label="Canopy area")
    fig.suptitle(f"Kelpwatch {region_display} Annual Maximum QA")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_time_series_figure(
    rows: list[dict[str, object]], output_path: Path, *, region_display: str
) -> None:
    """Write the quarterly aggregate canopy time-series QA figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in rows]
    values = [float(cast(float, row["aggregate_canopy_area"])) for row in rows]
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(range(len(values)), values, marker="o", linewidth=1.5)
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_ylabel("Aggregate canopy area")
    axis.set_title(f"Kelpwatch {region_display} Quarterly Aggregate QA")
    axis.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_interactive_html(
    *,
    annual_max: xr.DataArray,
    spatial_context: SpatialContext,
    output_path: Path,
    color_scale: dict[str, float],
    preview_max_pixels: int,
    region_display: str,
) -> None:
    """Write a self-contained HTML viewer with annual map toggles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    years = [int(year) for year in annual_max.coords[YEAR_DIM].values]
    image_records = []
    for year in years:
        full_grid = annual_layer_to_grid(annual_max.sel({YEAR_DIM: year}), spatial_context)
        preview = preview_grid_array(full_grid, preview_max_pixels)
        image_records.append(
            {
                "year": year,
                "data_uri": render_year_data_uri(preview, year, color_scale),
            }
        )
    output_path.write_text(build_interactive_html(image_records, color_scale, region_display))


def preview_grid_array(data: np.ndarray, preview_max_pixels: int) -> np.ndarray:
    """Downsample a dense grid preview when it exceeds the HTML pixel budget."""
    total_pixels = data.size
    if total_pixels <= preview_max_pixels:
        return data
    stride = int(math.ceil(math.sqrt(total_pixels / preview_max_pixels)))
    return data[::stride, ::stride]


def render_year_data_uri(data: np.ndarray, year: int, color_scale: dict[str, float]) -> str:
    """Render one annual layer to a PNG data URI."""
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(
        np.asarray(data, dtype=float),
        origin="upper",
        vmin=color_scale["vmin"],
        vmax=color_scale["vmax"],
        cmap="viridis",
    )
    axis.set_axis_off()
    axis.set_title(f"{year}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_interactive_html(
    image_records: list[dict[str, object]],
    color_scale: dict[str, float],
    region_display: str,
) -> str:
    """Build a small self-contained HTML viewer for annual QA layers."""
    buttons = "\n".join(
        f'<button type="button" data-year="{record["year"]}">{record["year"]}</button>'
        for record in image_records
    )
    images = "\n".join(
        (
            f'<img class="layer" id="layer-{record["year"]}" '
            f'src="{html.escape(str(record["data_uri"]))}" alt="{record["year"]}" />'
        )
        for record in image_records
    )
    first_year = image_records[0]["year"] if image_records else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Kelpwatch {html.escape(region_display)} QA</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    button {{ margin-right: 8px; padding: 6px 10px; }}
    .layer {{ display: none; max-width: min(960px, 100%); border: 1px solid #ccc; }}
    .layer.active {{ display: block; }}
    .meta {{ color: #444; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Kelpwatch {html.escape(region_display)} Annual Maximum QA</h1>
  <p class="meta">Color scale: {color_scale["vmin"]:.3g} to {color_scale["vmax"]:.3g}</p>
  <div>{buttons}</div>
  <div>{images}</div>
  <script>
    const showYear = (year) => {{
      document.querySelectorAll('.layer').forEach((node) => node.classList.remove('active'));
      const selected = document.getElementById(`layer-${{year}}`);
      if (selected) selected.classList.add('active');
    }};
    document.querySelectorAll('button[data-year]').forEach((button) => {{
      button.addEventListener('click', () => showYear(button.dataset.year));
    }});
    showYear('{first_year}');
  </script>
</body>
</html>
"""


def extent_from_context(spatial_context: SpatialContext) -> list[float]:
    """Return imshow extent from spatial coordinates."""
    left, top = spatial_context.transform * (0, 0)
    right, bottom = spatial_context.transform * (
        spatial_context.raster_width,
        spatial_context.raster_height,
    )
    return [
        float(left),
        float(right),
        float(bottom),
        float(top),
    ]


def plot_footprint(axis: Any, spatial_context: SpatialContext) -> None:
    """Draw the configured footprint boundary on a matplotlib axis."""
    boundary = spatial_context.footprint.boundary
    if hasattr(boundary, "geoms"):
        for line in boundary.geoms:
            x_values, y_values = line.xy
            axis.plot(x_values, y_values, color="white", linewidth=0.8)
    else:
        x_values, y_values = boundary.xy
        axis.plot(x_values, y_values, color="white", linewidth=0.8)


def write_qa_json(
    *,
    qa_config: KelpwatchVisualizeConfig,
    netcdf_path: Path,
    variable: str,
    spatial_context: SpatialContext,
    color_scale: dict[str, float],
    output_path: Path,
) -> None:
    """Write a small JSON supplement describing Kelpwatch QA outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "visualize-kelpwatch",
        "config_path": str(qa_config.config_path),
        "region_name": qa_config.region_name,
        "region_slug": qa_config.region_slug,
        "region_display_name": qa_config.region_display_name,
        "source_manifest": str(qa_config.source_manifest),
        "netcdf_path": str(netcdf_path),
        "variable": variable,
        "years": list(qa_config.years),
        "spatial": {
            "layout": spatial_context.layout,
            "x_name": spatial_context.x_name,
            "y_name": spatial_context.y_name,
            "crs": spatial_context.crs,
            "footprint_bounds": list(spatial_context.footprint_bounds),
            "pixel_count": spatial_context.pixel_count,
            "raster_height": spatial_context.raster_height,
            "raster_width": spatial_context.raster_width,
            "transform_gdal": list(spatial_context.transform.to_gdal()),
        },
        "color_scale": color_scale,
        "outputs": {
            "contact_sheet": str(qa_config.contact_sheet_path),
            "quarterly_timeseries": str(qa_config.time_series_path),
            "summary_table": str(qa_config.summary_table_path),
            "geotiff": str(qa_config.geotiff_path),
            "html": str(qa_config.html_path),
        },
    }
    with output_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))
