"""Build production annual Kelpwatch label tables."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from shapely.geometry.base import BaseGeometry

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.labels.kelpwatch import NETCDF_ENGINE

LOGGER = logging.getLogger(__name__)

KELPWATCH_PIXEL_AREA_M2 = 900.0
LABEL_VARIABLE = "area"
X_COORD_CANDIDATES = ("x", "lon", "longitude", "easting")
Y_COORD_CANDIDATES = ("y", "lat", "latitude", "northing")
REQUIRED_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_max_y",
    "kelp_fraction_y",
    "area_q1",
    "area_q2",
    "area_q3",
    "area_q4",
    "max_area_quarter",
    "valid_quarter_count",
    "nonzero_quarter_count",
    "kelp_present_gt0_y",
    "kelp_present_ge_1pct_y",
    "kelp_present_ge_5pct_y",
    "kelp_present_ge_10pct_y",
    "source_variable",
    "source_units",
    "source_package_id",
    "source_revision",
    "label_aggregation",
    "region_name",
)
SUMMARY_FIELDS = (
    "year",
    "row_count",
    "station_count",
    "valid_count",
    "missing_count",
    "zero_count",
    "gt0_count",
    "ge_1pct_count",
    "ge_5pct_count",
    "ge_10pct_count",
    "min",
    "median",
    "p95",
    "p99",
    "max",
    "aggregate_canopy_area",
)


@dataclass(frozen=True)
class KelpwatchLabelConfig:
    """Resolved config values for annual label derivation."""

    config_path: Path
    years: tuple[int, ...]
    target: str
    aggregation: str
    region_name: str
    footprint_path: Path
    footprint_crs: str
    source_manifest: Path
    annual_labels_path: Path
    summary_table_path: Path
    label_manifest_path: Path


@dataclass(frozen=True)
class StationSelection:
    """Kelpwatch station subset selected by the configured footprint."""

    station_name: str
    x_name: str
    y_name: str
    crs: str
    station_indices: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    footprint_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceMetadata:
    """Small source metadata block copied from the Kelpwatch source manifest."""

    package_id: str
    revision: int
    object_name: str | None


def build_annual_labels(config_path: Path) -> int:
    """Build annual Kelpwatch labels for the configured smoke region."""
    label_config = load_kelpwatch_label_config(config_path)
    LOGGER.info("Loading Kelpwatch source manifest: %s", label_config.source_manifest)
    manifest = load_json_object(label_config.source_manifest)
    source_metadata = source_metadata_from_manifest(manifest)
    netcdf_path = source_netcdf_path_from_manifest(manifest)
    selected_variable = selected_label_variable_from_manifest(manifest)
    if selected_variable != LABEL_VARIABLE:
        msg = (
            f"expected Kelpwatch label variable {LABEL_VARIABLE!r}, "
            f"found {selected_variable!r}; rerun inspect-kelpwatch or inspect the manifest"
        )
        raise ValueError(msg)

    LOGGER.info("Opening Kelpwatch NetCDF: %s", netcdf_path)
    with xr.open_dataset(netcdf_path, engine=NETCDF_ENGINE, decode_cf=True) as dataset:
        data = dataset[selected_variable]
        station_selection = select_stations_for_footprint(data, dataset, label_config)
        time_years, quarters = years_and_quarters(dataset, data)
        LOGGER.info(
            "Building annual labels for %s stations and years %s",
            station_selection.station_indices.size,
            list(label_config.years),
        )
        labels = annual_label_frame(
            data=data,
            dataset=dataset,
            station_selection=station_selection,
            time_years=time_years,
            quarters=quarters,
            label_config=label_config,
            source_metadata=source_metadata,
        )

    write_labels(labels, label_config.annual_labels_path)
    summary_rows = build_summary_rows(labels)
    write_summary_csv(summary_rows, label_config.summary_table_path)
    write_label_manifest(
        labels=labels,
        label_config=label_config,
        source_metadata=source_metadata,
        netcdf_path=netcdf_path,
        station_selection=station_selection,
        output_path=label_config.label_manifest_path,
    )
    LOGGER.info("Wrote annual Kelpwatch labels: %s", label_config.annual_labels_path)
    LOGGER.info("Wrote annual Kelpwatch label summary: %s", label_config.summary_table_path)
    LOGGER.info("Wrote annual Kelpwatch label manifest: %s", label_config.label_manifest_path)
    return 0


def load_kelpwatch_label_config(config_path: Path) -> KelpwatchLabelConfig:
    """Load annual label derivation settings from the workflow config."""
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

    annual_labels_path = Path(
        require_string(label_paths.get("annual_labels"), "labels.paths.annual_labels")
    )
    tables_dir = Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))
    return KelpwatchLabelConfig(
        config_path=config_path,
        years=years,
        target=require_string(labels.get("target"), "labels.target"),
        aggregation=require_string(labels.get("aggregation"), "labels.aggregation"),
        region_name=require_string(region.get("name"), "region.name"),
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        footprint_crs=str(region.get("crs", "EPSG:4326")),
        source_manifest=Path(
            require_string(label_paths.get("source_manifest"), "labels.paths.source_manifest")
        ),
        annual_labels_path=annual_labels_path,
        summary_table_path=tables_dir / "labels_annual_summary.csv",
        label_manifest_path=annual_labels_path.parent / "labels_annual_manifest.json",
    )


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


def source_metadata_from_manifest(manifest: dict[str, Any]) -> SourceMetadata:
    """Extract compact source metadata from a Kelpwatch source manifest."""
    source = require_mapping(manifest.get("source"), "manifest.source")
    object_name = source.get("object_name")
    return SourceMetadata(
        package_id=require_string(source.get("package_id"), "manifest.source.package_id"),
        revision=require_int_value(source.get("revision"), "manifest.source.revision"),
        object_name=str(object_name) if object_name is not None else None,
    )


def selected_label_variable_from_manifest(manifest: dict[str, Any]) -> str:
    """Read the selected label variable from a Kelpwatch source manifest."""
    label_source = require_mapping(manifest.get("label_source"), "manifest.label_source")
    return require_string(
        label_source.get("selected_variable"),
        "manifest.label_source.selected_variable",
    )


def select_stations_for_footprint(
    data: xr.DataArray,
    dataset: xr.Dataset,
    label_config: KelpwatchLabelConfig,
) -> StationSelection:
    """Select Kelpwatch stations whose lon/lat points intersect the configured footprint."""
    station_name, x_name, y_name = detect_station_layout(data, dataset)
    dataset_crs = infer_dataset_crs(dataset, data, x_name, y_name)
    footprint = read_footprint(label_config.footprint_path, label_config.footprint_crs, dataset_crs)
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

    LOGGER.info("Selected %s Kelpwatch stations inside footprint", station_indices.size)
    return StationSelection(
        station_name=station_name,
        x_name=x_name,
        y_name=y_name,
        crs=dataset_crs,
        station_indices=station_indices,
        longitude=x_values[station_indices],
        latitude=y_values[station_indices],
        footprint_bounds=footprint.bounds,
    )


def detect_station_layout(data: xr.DataArray, dataset: xr.Dataset) -> tuple[str, str, str]:
    """Detect Kelpwatch's station dimension and longitude/latitude variable names."""
    for dim_name in data.dims:
        x_name = find_one_dimensional_dataset_variable(dataset, X_COORD_CANDIDATES, str(dim_name))
        y_name = find_one_dimensional_dataset_variable(dataset, Y_COORD_CANDIDATES, str(dim_name))
        if x_name is not None and y_name is not None:
            return str(dim_name), x_name, y_name
    msg = f"could not identify station lon/lat layout for variable {data.name!r}: {data.dims}"
    raise ValueError(msg)


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
    msg = "could not infer Kelpwatch NetCDF CRS; add CRS metadata before label derivation"
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
        msg = f"coordinate must be one-dimensional for label derivation: {name}"
        raise ValueError(msg)
    return array.astype(float)


def years_and_quarters(dataset: xr.Dataset, data: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Read or infer one calendar year and quarter value for each time step."""
    time_name = detect_time_name(data)
    time_size = data.sizes[time_name]
    years = one_dimensional_time_values(dataset, "year", time_name)
    quarters = one_dimensional_time_values(dataset, "quarter", time_name)
    if years is None:
        years = years_from_time_values(time_values_for_name(data, time_name))
    if quarters is None:
        quarters = quarters_from_time_values(time_values_for_name(data, time_name))
    if years.size != time_size or quarters.size != time_size:
        msg = f"year/quarter coordinate lengths do not match time dimension: {time_name}"
        raise ValueError(msg)
    return years.astype(int), quarters.astype(int)


def detect_time_name(data: xr.DataArray) -> str:
    """Detect the time coordinate or dimension name for a Kelpwatch variable."""
    for name in ("time", "date"):
        if name in data.dims or name in data.coords:
            return name
    for dim_name in data.dims:
        if "time" in str(dim_name).lower() or "date" in str(dim_name).lower():
            return str(dim_name)
    msg = f"could not identify time dimension for variable {data.name!r}: {data.dims}"
    raise ValueError(msg)


def one_dimensional_time_values(
    dataset: xr.Dataset, name: str, time_name: str
) -> np.ndarray | None:
    """Return a one-dimensional time variable when it matches the time dimension."""
    if name not in dataset:
        return None
    variable = dataset[name]
    if tuple(str(dim) for dim in variable.dims) != (time_name,):
        return None
    return np.asarray(variable.values)


def time_values_for_name(data: xr.DataArray, time_name: str) -> np.ndarray:
    """Return raw values for a time coordinate or dimension."""
    values = data.coords[time_name].values if time_name in data.coords else data[time_name].values
    return np.asarray(values)


def years_from_time_values(values: np.ndarray) -> np.ndarray:
    """Convert common time coordinate encodings into calendar years."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[Y]").astype(int) + 1970
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(int)
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(int)
    return np.asarray([int(str(value)[:4]) for value in values], dtype=int)


def quarters_from_time_values(values: np.ndarray) -> np.ndarray:
    """Convert common time coordinate encodings into calendar quarters."""
    if np.issubdtype(values.dtype, np.datetime64):
        months = values.astype("datetime64[M]").astype(int) % 12 + 1
        return ((months - 1) // 3 + 1).astype(int)
    return np.asarray([quarter_from_label(str(value)) for value in values], dtype=int)


def quarter_from_label(value: str) -> int:
    """Infer a calendar quarter from a string-like time value."""
    if "-Q" in value:
        return int(value.rsplit("-Q", maxsplit=1)[1][0])
    if len(value) >= 7 and value[4] == "-":
        month = int(value[5:7])
        return (month - 1) // 3 + 1
    msg = f"could not infer quarter from time value: {value!r}"
    raise ValueError(msg)


def annual_label_frame(
    *,
    data: xr.DataArray,
    dataset: xr.Dataset,
    station_selection: StationSelection,
    time_years: np.ndarray,
    quarters: np.ndarray,
    label_config: KelpwatchLabelConfig,
    source_metadata: SourceMetadata,
) -> pd.DataFrame:
    """Build the annual station-by-year label table."""
    selected = data.isel({station_selection.station_name: station_selection.station_indices})
    values = clean_area_values(np.asarray(selected.values, dtype=float))
    rows: list[pd.DataFrame] = []
    source_units = str(data.attrs.get("units", ""))
    for year in label_config.years:
        quarter_values = quarter_matrix_for_year(values, time_years, quarters, year)
        rows.append(
            annual_rows_for_year(
                year=year,
                quarter_values=quarter_values,
                station_selection=station_selection,
                label_config=label_config,
                source_metadata=source_metadata,
                source_units=source_units,
            )
        )
    labels = pd.concat(rows, ignore_index=True)
    return labels.loc[:, list(REQUIRED_COLUMNS)]


def clean_area_values(values: np.ndarray) -> np.ndarray:
    """Convert Kelpwatch fill and invalid area values to NaN."""
    cleaned = values.astype(float, copy=True)
    cleaned[cleaned < 0] = np.nan
    return cleaned


def quarter_matrix_for_year(
    values: np.ndarray,
    time_years: np.ndarray,
    quarters: np.ndarray,
    year: int,
) -> np.ndarray:
    """Return a station-by-quarter matrix for one year."""
    station_count = values.shape[1]
    quarter_values = np.full((station_count, 4), np.nan, dtype=float)
    for quarter in range(1, 5):
        indices = np.flatnonzero((time_years == year) & (quarters == quarter))
        if indices.size == 0:
            continue
        quarter_values[:, quarter - 1] = nanmax_by_row(values[indices, :].T)
    return quarter_values


def nanmax_by_row(values: np.ndarray) -> np.ndarray:
    """Return row-wise maxima while preserving all-NaN rows as NaN."""
    result = np.full(values.shape[0], np.nan, dtype=float)
    valid_rows = np.any(np.isfinite(values), axis=1)
    if np.any(valid_rows):
        result[valid_rows] = np.nanmax(values[valid_rows], axis=1)
    return result


def annual_rows_for_year(
    *,
    year: int,
    quarter_values: np.ndarray,
    station_selection: StationSelection,
    label_config: KelpwatchLabelConfig,
    source_metadata: SourceMetadata,
    source_units: str,
) -> pd.DataFrame:
    """Build annual label rows for all selected stations in one year."""
    kelp_max = nanmax_by_row(quarter_values)
    fraction = kelp_max / KELPWATCH_PIXEL_AREA_M2
    valid_any = np.any(np.isfinite(quarter_values), axis=1)
    max_quarter = max_quarter_by_row(quarter_values, valid_any)
    frame = pd.DataFrame(
        {
            "year": year,
            "kelpwatch_station_id": station_selection.station_indices.astype(int),
            "longitude": station_selection.longitude,
            "latitude": station_selection.latitude,
            "kelp_max_y": kelp_max,
            "kelp_fraction_y": fraction,
            "area_q1": quarter_values[:, 0],
            "area_q2": quarter_values[:, 1],
            "area_q3": quarter_values[:, 2],
            "area_q4": quarter_values[:, 3],
            "max_area_quarter": pd.Series(max_quarter, dtype="Int64"),
            "valid_quarter_count": np.count_nonzero(np.isfinite(quarter_values), axis=1),
            "nonzero_quarter_count": np.count_nonzero(quarter_values > 0, axis=1),
            "source_variable": LABEL_VARIABLE,
            "source_units": source_units,
            "source_package_id": source_metadata.package_id,
            "source_revision": source_metadata.revision,
            "label_aggregation": label_config.aggregation,
            "region_name": label_config.region_name,
        }
    )
    frame["kelp_present_gt0_y"] = frame["kelp_fraction_y"] > 0
    frame["kelp_present_ge_1pct_y"] = frame["kelp_fraction_y"] >= 0.01
    frame["kelp_present_ge_5pct_y"] = frame["kelp_fraction_y"] >= 0.05
    frame["kelp_present_ge_10pct_y"] = frame["kelp_fraction_y"] >= 0.10
    return frame


def max_quarter_by_row(values: np.ndarray, valid_any: np.ndarray) -> np.ndarray:
    """Return the quarter containing each row's maximum value."""
    filled = np.where(np.isfinite(values), values, -np.inf)
    quarters = np.argmax(filled, axis=1).astype(float) + 1
    quarters[~valid_any] = np.nan
    return cast(np.ndarray, quarters)


def build_summary_rows(labels: pd.DataFrame) -> list[dict[str, object]]:
    """Build per-year annual label summary rows."""
    rows: list[dict[str, object]] = []
    for year, group in labels.groupby("year", sort=True):
        values = group["kelp_max_y"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "year": int(year),
                "row_count": int(len(group)),
                "station_count": int(group["kelpwatch_station_id"].nunique()),
                "valid_count": int(finite.size),
                "missing_count": int(len(group) - finite.size),
                "zero_count": int(np.count_nonzero(finite == 0)),
                "gt0_count": int(group["kelp_present_gt0_y"].sum()),
                "ge_1pct_count": int(group["kelp_present_ge_1pct_y"].sum()),
                "ge_5pct_count": int(group["kelp_present_ge_5pct_y"].sum()),
                "ge_10pct_count": int(group["kelp_present_ge_10pct_y"].sum()),
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


def write_labels(labels: pd.DataFrame, output_path: Path) -> None:
    """Write annual label rows to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write annual label summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_label_manifest(
    *,
    labels: pd.DataFrame,
    label_config: KelpwatchLabelConfig,
    source_metadata: SourceMetadata,
    netcdf_path: Path,
    station_selection: StationSelection,
    output_path: Path,
) -> None:
    """Write a small JSON manifest for the annual label artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "build-labels",
        "config_path": str(label_config.config_path),
        "source_manifest": str(label_config.source_manifest),
        "netcdf_path": str(netcdf_path),
        "annual_labels": str(label_config.annual_labels_path),
        "summary_table": str(label_config.summary_table_path),
        "label_variable": LABEL_VARIABLE,
        "label_aggregation": label_config.aggregation,
        "years": list(label_config.years),
        "row_count": int(len(labels)),
        "station_count": int(station_selection.station_indices.size),
        "source": {
            "package_id": source_metadata.package_id,
            "revision": source_metadata.revision,
            "object_name": source_metadata.object_name,
        },
        "spatial": {
            "station_name": station_selection.station_name,
            "x_name": station_selection.x_name,
            "y_name": station_selection.y_name,
            "crs": station_selection.crs,
            "footprint_bounds": list(station_selection.footprint_bounds),
            "support": "kelpwatch_30m_station",
        },
        "schema": list(labels.columns),
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
