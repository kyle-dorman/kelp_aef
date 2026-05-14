"""Align NOAA CRM topo-bathy sources to the existing AEF target grid."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import rasterio  # type: ignore[import-untyped]
import xarray as xr
from rasterio.warp import transform as warp_transform  # type: ignore[import-untyped]
from rasterio.warp import transform_bounds

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.noaa_crm import NoaaCrmProduct, load_noaa_crm_config

LOGGER = logging.getLogger(__name__)

DEFAULT_RESAMPLING_METHOD = "nearest_center_sample"
DEFAULT_PRODUCT_BOUNDARY_LATITUDE = 37.0
DEFAULT_ROW_CHUNK_SIZE = 100_000
TARGET_GRID_COLUMNS = (
    "year",
    "aef_grid_row",
    "aef_grid_col",
    "aef_grid_cell_id",
    "longitude",
    "latitude",
)
OUTPUT_COLUMNS = (
    "aef_grid_row",
    "aef_grid_col",
    "aef_grid_cell_id",
    "longitude",
    "latitude",
    "crm_elevation_m",
    "crm_depth_m",
    "crm_source_product_id",
    "crm_source_product_name",
    "crm_source_path",
    "crm_vertical_datum",
    "crm_alignment_method",
    "crm_value_status",
    "cudem_elevation_m",
    "cudem_depth_m",
    "cudem_source_tile_id",
    "cudem_source_path",
    "cudem_value_status",
    "usgs_3dep_elevation_m",
    "usgs_3dep_source_id",
    "usgs_3dep_source_path",
    "usgs_3dep_value_status",
)
SUMMARY_FIELDS = ("source", "metric", "category", "value")
COMPARISON_FIELDS = ("source_pair", "metric", "value")


@dataclass(frozen=True)
class ProductBoundaryRule:
    """Latitude split used to build the two-product CRM mosaic."""

    latitude: float
    south_product_id: str
    north_product_id: str


@dataclass(frozen=True)
class CrmAlignmentConfig:
    """Resolved config values for NOAA CRM target-grid alignment."""

    config_path: Path
    region_name: str
    source_manifest: Path
    query_manifest: Path
    cudem_tile_manifest: Path | None
    usgs_3dep_source_manifest: Path | None
    cusp_source_manifest: Path | None
    target_grid_table: Path
    target_grid_manifest: Path | None
    output_table: Path
    output_manifest: Path
    qa_summary_table: Path
    comparison_table: Path
    resampling_method: str
    product_boundary: ProductBoundaryRule
    row_chunk_size: int
    target_grid_year: int
    fast: bool
    row_window: tuple[int, int] | None
    col_window: tuple[int, int] | None
    products: dict[str, NoaaCrmProduct]


@dataclass(frozen=True)
class CrmProductSource:
    """Open xarray source for one configured CRM product."""

    product_id: str
    product_name: str
    local_path: Path
    data_variable: str
    vertical_datum: str
    dataset: Any
    data_array: Any
    lon_name: str
    lat_name: str
    bounds: tuple[float, float, float, float]
    nodata_values: tuple[float, ...]


@dataclass(frozen=True)
class RasterQaSource:
    """Open Rasterio source for one optional QA raster."""

    source_id: str
    local_path: Path
    dataset: Any
    bounds_wgs84: tuple[float, float, float, float]
    nodata_values: tuple[float, ...]


@dataclass(frozen=True)
class RasterSourceGroup:
    """Resolved optional raster QA sources and their availability status."""

    manifest_path: Path | None
    manifest_status: str
    sources: tuple[RasterQaSource, ...]


@dataclass
class DiffStats:
    """Running summary for one source-to-source difference metric."""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None


@dataclass
class AlignmentAccumulator:
    """Mutable counters used to build QA summary and comparison tables."""

    total_cells: int = 0
    crm_status_counts: Counter[str] = field(default_factory=Counter)
    crm_product_counts: Counter[str] = field(default_factory=Counter)
    crm_depth_bin_counts: Counter[str] = field(default_factory=Counter)
    cudem_status_counts: Counter[str] = field(default_factory=Counter)
    usgs_status_counts: Counter[str] = field(default_factory=Counter)
    crm_elevation_min: float | None = None
    crm_elevation_max: float | None = None
    crm_depth_min: float | None = None
    crm_depth_max: float | None = None
    diff_stats: dict[str, DiffStats] = field(default_factory=dict)
    sign_disagreement_counts: Counter[str] = field(default_factory=Counter)


def align_noaa_crm(config_path: Path, *, fast: bool = False) -> int:
    """Align CRM and optional QA sources to the static target grid."""
    align_config = load_crm_alignment_config(config_path, fast=fast)
    LOGGER.info(
        "Aligning NOAA CRM to target grid for %s with %s mode",
        align_config.region_name,
        "fast" if fast else "full",
    )
    LOGGER.info("Output overwrite policy: replace existing alignment artifacts.")
    target_grid = load_static_target_grid(align_config)
    target_bounds = bounds_from_target_grid(target_grid)
    crm_sources = open_crm_sources(align_config, target_bounds)
    cudem_sources = open_raster_group(
        manifest_path=align_config.cudem_tile_manifest,
        record_id_field="tile_id",
        source_name="noaa_cudem",
    )
    usgs_sources = open_raster_group(
        manifest_path=align_config.usgs_3dep_source_manifest,
        record_id_field="artifact_id",
        source_name="usgs_3dep",
    )
    cusp_validation = validate_cusp_manifest(align_config.cusp_source_manifest, target_bounds)
    accumulator = AlignmentAccumulator()
    reset_output_path(align_config.output_table)
    schema = output_arrow_schema()

    try:
        with parquet_writer(align_config.output_table) as writer_context:
            writer_context.open(schema)
            for chunk in target_grid_chunks(target_grid, align_config.row_chunk_size):
                aligned_chunk = align_target_chunk(
                    chunk=chunk,
                    align_config=align_config,
                    crm_sources=crm_sources,
                    cudem_sources=cudem_sources,
                    usgs_sources=usgs_sources,
                )
                update_accumulator(accumulator, aligned_chunk)
                table = pa.Table.from_pandas(
                    aligned_chunk,
                    schema=schema,
                    preserve_index=False,
                )
                writer_context.write_table(table)
                LOGGER.info(
                    "Aligned %s / %s target cells", accumulator.total_cells, len(target_grid)
                )
    finally:
        close_crm_sources(crm_sources)
        close_raster_group(cudem_sources)
        close_raster_group(usgs_sources)

    write_summary_table(accumulator, align_config.qa_summary_table)
    write_comparison_table(accumulator, align_config.comparison_table)
    manifest = build_alignment_manifest(
        align_config=align_config,
        target_grid=target_grid,
        target_bounds=target_bounds,
        accumulator=accumulator,
        cudem_sources=cudem_sources,
        usgs_sources=usgs_sources,
        cusp_validation=cusp_validation,
        schema=schema,
    )
    write_json(align_config.output_manifest, manifest)
    LOGGER.info("Wrote aligned CRM table: %s", align_config.output_table)
    LOGGER.info("Wrote aligned CRM manifest: %s", align_config.output_manifest)
    return 0


def load_crm_alignment_config(config_path: Path, *, fast: bool) -> CrmAlignmentConfig:
    """Load CRM alignment settings from the workflow config."""
    crm_config = load_noaa_crm_config(config_path)
    config = load_yaml_config(config_path)
    years = require_years(config)
    domain = require_mapping(config.get("domain"), "domain")
    crm = require_mapping(domain.get("noaa_crm"), "domain.noaa_crm")
    crm_alignment = require_mapping(crm.get("alignment"), "domain.noaa_crm.alignment")
    alignment = require_mapping(config.get("alignment"), "alignment")
    full_grid = require_mapping(alignment.get("full_grid"), "alignment.full_grid")
    fast_grid = optional_mapping(full_grid.get("fast"), "alignment.full_grid.fast")
    fast_outputs = optional_mapping(crm_alignment.get("fast"), "domain.noaa_crm.alignment.fast")
    product_boundary = load_product_boundary_rule(crm_alignment)
    return CrmAlignmentConfig(
        config_path=config_path,
        region_name=crm_config.region_name,
        source_manifest=path_field(
            crm_alignment, "source_manifest", "domain.noaa_crm.alignment.source_manifest"
        ),
        query_manifest=path_field(
            crm_alignment, "query_manifest", "domain.noaa_crm.alignment.query_manifest"
        ),
        cudem_tile_manifest=optional_path_field(
            crm_alignment,
            "cudem_tile_manifest",
            "domain.noaa_crm.alignment.cudem_tile_manifest",
        ),
        usgs_3dep_source_manifest=optional_path_field(
            crm_alignment,
            "usgs_3dep_source_manifest",
            "domain.noaa_crm.alignment.usgs_3dep_source_manifest",
        ),
        cusp_source_manifest=optional_path_field(
            crm_alignment,
            "cusp_source_manifest",
            "domain.noaa_crm.alignment.cusp_source_manifest",
        ),
        target_grid_table=path_field(
            crm_alignment, "target_grid_table", "domain.noaa_crm.alignment.target_grid_table"
        ),
        target_grid_manifest=optional_path_field(
            crm_alignment,
            "target_grid_manifest",
            "domain.noaa_crm.alignment.target_grid_manifest",
        ),
        output_table=fast_path(
            path_field(crm_alignment, "output_table", "domain.noaa_crm.alignment.output_table"),
            fast,
            fast_outputs,
            "output_table",
        ),
        output_manifest=fast_path(
            path_field(
                crm_alignment,
                "output_manifest",
                "domain.noaa_crm.alignment.output_manifest",
            ),
            fast,
            fast_outputs,
            "output_manifest",
        ),
        qa_summary_table=fast_path(
            path_field(
                crm_alignment,
                "qa_summary_table",
                "domain.noaa_crm.alignment.qa_summary_table",
            ),
            fast,
            fast_outputs,
            "qa_summary_table",
        ),
        comparison_table=fast_path(
            path_field(
                crm_alignment,
                "comparison_table",
                "domain.noaa_crm.alignment.comparison_table",
            ),
            fast,
            fast_outputs,
            "comparison_table",
        ),
        resampling_method=optional_string(
            crm_alignment.get("resampling_method"),
            "domain.noaa_crm.alignment.resampling_method",
            DEFAULT_RESAMPLING_METHOD,
        ),
        product_boundary=product_boundary,
        row_chunk_size=optional_positive_int(
            crm_alignment.get("row_chunk_size"),
            "domain.noaa_crm.alignment.row_chunk_size",
            DEFAULT_ROW_CHUNK_SIZE,
        ),
        target_grid_year=target_grid_year(crm_alignment, years, fast_grid, fast),
        fast=fast,
        row_window=window_from_config(
            fast_grid.get("row_window"), "alignment.full_grid.fast.row_window"
        )
        if fast
        else None,
        col_window=window_from_config(
            fast_grid.get("col_window"), "alignment.full_grid.fast.col_window"
        )
        if fast
        else None,
        products={product.product_id: product for product in crm_config.products},
    )


def require_years(config: dict[str, Any]) -> tuple[int, ...]:
    """Read the configured smoke years as integers."""
    years_config = require_mapping(config.get("years"), "years")
    years = years_config.get("smoke")
    if not isinstance(years, list) or not years:
        msg = "config field must be a non-empty list of years: years.smoke"
        raise ValueError(msg)
    return tuple(require_int(year, "years.smoke[]") for year in years)


def load_product_boundary_rule(crm_alignment: dict[str, Any]) -> ProductBoundaryRule:
    """Load the configured CRM product boundary rule."""
    value = crm_alignment.get("product_boundary")
    if value is None:
        msg = "config field is required: domain.noaa_crm.alignment.product_boundary"
        raise ValueError(msg)
    boundary = require_mapping(value, "domain.noaa_crm.alignment.product_boundary")
    return ProductBoundaryRule(
        latitude=optional_float(
            boundary.get("latitude"),
            "domain.noaa_crm.alignment.product_boundary.latitude",
            DEFAULT_PRODUCT_BOUNDARY_LATITUDE,
        ),
        south_product_id=require_string(
            boundary.get("south_product_id"),
            "domain.noaa_crm.alignment.product_boundary.south_product_id",
        ),
        north_product_id=require_string(
            boundary.get("north_product_id"),
            "domain.noaa_crm.alignment.product_boundary.north_product_id",
        ),
    )


def path_field(mapping: dict[str, Any], key: str, name: str) -> Path:
    """Read a required path-valued string field."""
    return Path(require_string(mapping.get(key), name))


def optional_path_field(mapping: dict[str, Any], key: str, name: str) -> Path | None:
    """Read an optional path-valued string field."""
    value = mapping.get(key)
    if value is None:
        return None
    return Path(require_string(value, name))


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional mapping config value."""
    if value is None:
        return {}
    return require_mapping(value, name)


def optional_string(value: object, name: str, default: str) -> str:
    """Read an optional string config value with a default."""
    if value is None:
        return default
    return require_string(value, name)


def require_int(value: object, name: str) -> int:
    """Validate a dynamic config value as an integer."""
    if not isinstance(value, int):
        msg = f"config field must be an integer: {name}"
        raise ValueError(msg)
    return value


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Read an optional positive integer config value."""
    if value is None:
        return default
    parsed = require_int(value, name)
    if parsed <= 0:
        msg = f"config field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def optional_float(value: object, name: str, default: float) -> float:
    """Read an optional numeric config value."""
    if value is None:
        return default
    if not isinstance(value, int | float):
        msg = f"config field must be a number: {name}"
        raise ValueError(msg)
    return float(value)


def target_grid_year(
    crm_alignment: dict[str, Any],
    years: tuple[int, ...],
    fast_grid: dict[str, Any],
    fast: bool,
) -> int:
    """Choose the full-grid year used to extract static target cells."""
    configured = crm_alignment.get("target_grid_year")
    if configured is not None:
        return require_int(configured, "domain.noaa_crm.alignment.target_grid_year")
    if fast:
        fast_years = fast_grid.get("years")
        if isinstance(fast_years, list) and fast_years:
            return require_int(fast_years[0], "alignment.full_grid.fast.years[0]")
    return years[0]


def window_from_config(value: object, name: str) -> tuple[int, int] | None:
    """Parse a two-element half-open row or column window."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        msg = f"config field must be a two-element list: {name}"
        raise ValueError(msg)
    start = require_int(value[0], f"{name}[0]")
    stop = require_int(value[1], f"{name}[1]")
    if stop <= start:
        msg = f"window stop must be greater than start: {name}"
        raise ValueError(msg)
    return start, stop


def fast_path(path: Path, fast: bool, fast_outputs: dict[str, Any], key: str) -> Path:
    """Resolve a fast-path output if configured."""
    if not fast:
        return path
    if key in fast_outputs:
        return Path(require_string(fast_outputs.get(key), f"domain.noaa_crm.alignment.fast.{key}"))
    return path.with_name(f"{path.stem}.fast{path.suffix}")


def load_static_target_grid(align_config: CrmAlignmentConfig) -> pd.DataFrame:
    """Load one unique static target-grid row per AEF cell."""
    if not align_config.target_grid_table.exists():
        msg = f"target-grid table does not exist: {align_config.target_grid_table}"
        raise FileNotFoundError(msg)
    dataset = pds.dataset(align_config.target_grid_table)  # type: ignore[no-untyped-call]
    filter_expression = (
        pds.field("year") == align_config.target_grid_year  # type: ignore[attr-defined,no-untyped-call]
    )
    if align_config.row_window is not None:
        filter_expression = (
            filter_expression
            & (
                pds.field("aef_grid_row")  # type: ignore[attr-defined,no-untyped-call]
                >= align_config.row_window[0]
            )
            & (
                pds.field("aef_grid_row")  # type: ignore[attr-defined,no-untyped-call]
                < align_config.row_window[1]
            )
        )
    if align_config.col_window is not None:
        filter_expression = (
            filter_expression
            & (
                pds.field("aef_grid_col")  # type: ignore[attr-defined,no-untyped-call]
                >= align_config.col_window[0]
            )
            & (
                pds.field("aef_grid_col")  # type: ignore[attr-defined,no-untyped-call]
                < align_config.col_window[1]
            )
        )
    table = dataset.to_table(columns=list(TARGET_GRID_COLUMNS), filter=filter_expression)
    frame = table.to_pandas()
    if frame.empty:
        msg = (
            "target-grid table produced no rows for "
            f"year={align_config.target_grid_year}, row_window={align_config.row_window}, "
            f"col_window={align_config.col_window}"
        )
        raise ValueError(msg)
    frame = (
        frame.sort_values(["aef_grid_cell_id"])
        .drop_duplicates("aef_grid_cell_id", keep="first")
        .reset_index(drop=True)
    )
    return frame.astype(
        {
            "aef_grid_row": "int32",
            "aef_grid_col": "int32",
            "aef_grid_cell_id": "int64",
            "longitude": "float64",
            "latitude": "float64",
        }
    )[list(TARGET_GRID_COLUMNS[1:])]


def bounds_from_target_grid(target_grid: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return WGS84 bounds from target-grid center coordinates."""
    return (
        float(target_grid["longitude"].min()),
        float(target_grid["latitude"].min()),
        float(target_grid["longitude"].max()),
        float(target_grid["latitude"].max()),
    )


def open_crm_sources(
    align_config: CrmAlignmentConfig,
    target_bounds: tuple[float, float, float, float],
) -> dict[str, CrmProductSource]:
    """Open required CRM product sources from the source manifest."""
    source_manifest = load_json_object(align_config.source_manifest, "NOAA CRM source manifest")
    records = cast(list[dict[str, Any]], source_manifest.get("records", []))
    records_by_id = {str(record.get("product_id")): record for record in records}
    required_ids = required_crm_product_ids(align_config, target_bounds)
    missing_ids = sorted(required_ids - set(records_by_id))
    if missing_ids:
        msg = f"NOAA CRM source manifest is missing required products: {missing_ids}"
        raise ValueError(msg)
    sources: dict[str, CrmProductSource] = {}
    for product_id in sorted(required_ids):
        product = align_config.products.get(product_id)
        if product is None:
            msg = f"CRM product is not configured in domain.noaa_crm.products: {product_id}"
            raise ValueError(msg)
        record = records_by_id[product_id]
        local_path = Path(
            require_string(record.get("local_path"), f"records[{product_id}].local_path")
        )
        if not local_path.exists():
            msg = f"NOAA CRM source file does not exist for {product_id}: {local_path}"
            raise FileNotFoundError(msg)
        sources[product_id] = open_crm_product_source(product, record, local_path)
    return sources


def required_crm_product_ids(
    align_config: CrmAlignmentConfig,
    target_bounds: tuple[float, float, float, float],
) -> set[str]:
    """Return CRM product IDs needed for the target-grid latitude extent."""
    _west, south, _east, north = target_bounds
    boundary_latitude = align_config.product_boundary.latitude
    required_ids: set[str] = set()
    if south < boundary_latitude:
        required_ids.add(align_config.product_boundary.south_product_id)
    if north >= boundary_latitude:
        required_ids.add(align_config.product_boundary.north_product_id)
    return required_ids


def open_crm_product_source(
    product: NoaaCrmProduct,
    record: dict[str, Any],
    local_path: Path,
) -> CrmProductSource:
    """Open one CRM product with xarray coordinate arrays as sampling truth."""
    dataset = open_xarray_dataset(local_path)
    data_variable = product.data_variable
    if data_variable not in dataset:
        data_variable = single_data_variable_name(dataset, product.product_id)
    data_array = dataset[data_variable]
    lon_name = coordinate_name(data_array, ("lon", "longitude", "x"), product.product_id)
    lat_name = coordinate_name(data_array, ("lat", "latitude", "y"), product.product_id)
    bounds = coordinate_bounds(dataset, lon_name, lat_name)
    return CrmProductSource(
        product_id=product.product_id,
        product_name=product.product_name,
        local_path=local_path,
        data_variable=data_variable,
        vertical_datum=str(record.get("vertical_datum") or product.vertical_datum),
        dataset=dataset,
        data_array=data_array,
        lon_name=lon_name,
        lat_name=lat_name,
        bounds=bounds,
        nodata_values=nodata_values(data_array),
    )


def open_xarray_dataset(path: Path) -> Any:
    """Open a NOAA CRM NetCDF with the installed xarray engines."""
    errors: list[str] = []
    for engine in ("h5netcdf", "scipy"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{engine}: {exc}")
    msg = f"could not open NetCDF with xarray: {path}; " + "; ".join(errors)
    raise OSError(msg)


def single_data_variable_name(dataset: Any, product_id: str) -> str:
    """Return the only non-CRS data variable in a dataset."""
    candidates = [name for name in dataset.data_vars if str(name).lower() != "crs"]
    if len(candidates) != 1:
        msg = f"CRM product {product_id} does not contain one sampleable data variable"
        raise ValueError(msg)
    return str(candidates[0])


def coordinate_name(data_array: Any, candidates: tuple[str, ...], product_id: str) -> str:
    """Find a coordinate name on a data array."""
    for candidate in candidates:
        if candidate in data_array.coords:
            return candidate
    msg = f"CRM product {product_id} is missing coordinate candidates: {candidates}"
    raise ValueError(msg)


def coordinate_bounds(
    dataset: Any, lon_name: str, lat_name: str
) -> tuple[float, float, float, float]:
    """Return sampleable bounds from xarray lon/lat center coordinates."""
    lon_values = np.asarray(dataset[lon_name].values, dtype=np.float64)
    lat_values = np.asarray(dataset[lat_name].values, dtype=np.float64)
    lon_padding = coordinate_padding(lon_values)
    lat_padding = coordinate_padding(lat_values)
    return (
        float(np.nanmin(lon_values) - lon_padding),
        float(np.nanmin(lat_values) - lat_padding),
        float(np.nanmax(lon_values) + lon_padding),
        float(np.nanmax(lat_values) + lat_padding),
    )


def coordinate_padding(values: np.ndarray) -> float:
    """Return a one-cell coordinate padding for edge/seam nearest sampling."""
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 2:
        return 0.0
    sorted_values = np.sort(np.unique(finite_values))
    differences = np.diff(sorted_values)
    positive_differences = differences[differences > 0]
    if len(positive_differences) == 0:
        return 0.0
    return float(np.median(positive_differences))


def nodata_values(data_array: Any) -> tuple[float, ...]:
    """Collect declared nodata sentinels from xarray metadata."""
    values: list[float] = []
    for mapping in (data_array.attrs, data_array.encoding):
        for key in ("_FillValue", "missing_value"):
            value = mapping.get(key)
            if isinstance(value, int | float):
                values.append(float(value))
    return tuple(values)


def close_crm_sources(sources: dict[str, CrmProductSource]) -> None:
    """Close all open xarray CRM datasets."""
    for source in sources.values():
        source.dataset.close()


def open_raster_group(
    *,
    manifest_path: Path | None,
    record_id_field: str,
    source_name: str,
) -> RasterSourceGroup:
    """Open optional raster QA sources from a manifest."""
    if manifest_path is None:
        return RasterSourceGroup(None, "not_configured", ())
    if not manifest_path.exists():
        LOGGER.warning("Optional %s manifest is missing: %s", source_name, manifest_path)
        return RasterSourceGroup(manifest_path, "manifest_missing", ())
    manifest = load_json_object(manifest_path, f"{source_name} manifest")
    sources: list[RasterQaSource] = []
    for record in cast(list[dict[str, Any]], manifest.get("records", [])):
        local_path_value = record.get("local_path")
        if not isinstance(local_path_value, str):
            continue
        local_path = Path(local_path_value)
        if not local_path.exists():
            continue
        try:
            dataset = rasterio.open(local_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping invalid %s raster %s: %s", source_name, local_path, exc)
            continue
        sources.append(
            RasterQaSource(
                source_id=str(
                    record.get(record_id_field) or record.get("source_id") or local_path.stem
                ),
                local_path=local_path,
                dataset=dataset,
                bounds_wgs84=raster_bounds_wgs84(dataset),
                nodata_values=raster_nodata_values(dataset),
            )
        )
    status = "available" if sources else "no_valid_sources"
    return RasterSourceGroup(manifest_path, status, tuple(sources))


def raster_bounds_wgs84(dataset: Any) -> tuple[float, float, float, float]:
    """Return a Rasterio dataset's bounds in WGS84."""
    bounds = tuple(float(value) for value in dataset.bounds)
    if dataset.crs is None or str(dataset.crs).upper() in {"EPSG:4326", "EPSG:4269"}:
        return cast(tuple[float, float, float, float], bounds)
    transformed = transform_bounds(dataset.crs, "EPSG:4326", *bounds, densify_pts=21)
    return cast(tuple[float, float, float, float], tuple(float(value) for value in transformed))


def raster_nodata_values(dataset: Any) -> tuple[float, ...]:
    """Collect nodata values from a Rasterio dataset."""
    values: list[float] = []
    for value in dataset.nodatavals:
        if value is not None:
            values.append(float(value))
    return tuple(values)


def close_raster_group(group: RasterSourceGroup) -> None:
    """Close all open raster QA datasets."""
    for source in group.sources:
        source.dataset.close()


def validate_cusp_manifest(
    manifest_path: Path | None,
    target_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Validate CUSP as shoreline vector metadata without raster alignment."""
    if manifest_path is None:
        return {"status": "not_configured"}
    if not manifest_path.exists():
        return {"status": "manifest_missing", "manifest_path": str(manifest_path)}
    manifest = load_json_object(manifest_path, "NOAA CUSP source manifest")
    records = cast(list[dict[str, Any]], manifest.get("records", []))
    if not records:
        return {"status": "no_records", "manifest_path": str(manifest_path)}
    record = records[0]
    vector = cast(dict[str, Any], record.get("vector", {}))
    vector_bounds = bounds_from_mapping(vector.get("bounds"))
    return {
        "status": vector.get("validation_status", "unknown"),
        "manifest_path": str(manifest_path),
        "local_path": record.get("local_path"),
        "crs": vector.get("crs"),
        "geometry_types": vector.get("geometry_types"),
        "feature_count": vector.get("feature_count"),
        "bounds": vector.get("bounds"),
        "covers_target_bounds": bounds_intersect(vector_bounds, target_bounds)
        if vector_bounds is not None
        else None,
        "alignment_role": "shoreline_vector_validation_only",
    }


def bounds_from_mapping(value: object) -> tuple[float, float, float, float] | None:
    """Convert a manifest bounds mapping to a tuple when available."""
    if not isinstance(value, dict):
        return None
    keys = ("west", "south", "east", "north")
    if not all(isinstance(value.get(key), int | float) for key in keys):
        return None
    west, south, east, north = (float(value[key]) for key in keys)
    return west, south, east, north


def bounds_intersect(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    """Return whether two WGS84 bounds intersect."""
    west_a, south_a, east_a, north_a = first
    west_b, south_b, east_b, north_b = second
    return west_a <= east_b and east_a >= west_b and south_a <= north_b and north_a >= south_b


def reset_output_path(path: Path) -> None:
    """Delete any existing output file or directory and recreate its parent."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


class DeferredParquetWriter:
    """Small context manager that opens a Parquet writer after schema discovery."""

    def __init__(self, path: Path) -> None:
        """Store the output path until the schema is known."""
        self.path = path
        self.writer: Any | None = None

    def __enter__(self) -> DeferredParquetWriter:
        """Enter the deferred writer context."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the writer if it was opened."""
        if self.writer is not None:
            self.writer.close()

    def open(self, schema: pa.Schema) -> None:
        """Open the underlying Parquet writer."""
        self.writer = pq.ParquetWriter(self.path, schema)  # type: ignore[no-untyped-call]

    def write_table(self, table: pa.Table) -> None:
        """Write one table to the underlying Parquet writer."""
        if self.writer is None:
            msg = "Parquet writer must be opened before writing"
            raise RuntimeError(msg)
        self.writer.write_table(table)


def parquet_writer(path: Path) -> DeferredParquetWriter:
    """Return a deferred Parquet writer for chunked output."""
    return DeferredParquetWriter(path)


def output_arrow_schema() -> pa.Schema:
    """Return the fixed Parquet schema for the aligned CRM table."""
    return pa.schema(
        [
            pa.field("aef_grid_row", pa.int32()),
            pa.field("aef_grid_col", pa.int32()),
            pa.field("aef_grid_cell_id", pa.int64()),
            pa.field("longitude", pa.float64()),
            pa.field("latitude", pa.float64()),
            pa.field("crm_elevation_m", pa.float32()),
            pa.field("crm_depth_m", pa.float32()),
            pa.field("crm_source_product_id", pa.string()),
            pa.field("crm_source_product_name", pa.string()),
            pa.field("crm_source_path", pa.string()),
            pa.field("crm_vertical_datum", pa.string()),
            pa.field("crm_alignment_method", pa.string()),
            pa.field("crm_value_status", pa.string()),
            pa.field("cudem_elevation_m", pa.float32()),
            pa.field("cudem_depth_m", pa.float32()),
            pa.field("cudem_source_tile_id", pa.string()),
            pa.field("cudem_source_path", pa.string()),
            pa.field("cudem_value_status", pa.string()),
            pa.field("usgs_3dep_elevation_m", pa.float32()),
            pa.field("usgs_3dep_source_id", pa.string()),
            pa.field("usgs_3dep_source_path", pa.string()),
            pa.field("usgs_3dep_value_status", pa.string()),
        ]
    )


def target_grid_chunks(frame: pd.DataFrame, row_chunk_size: int) -> list[pd.DataFrame]:
    """Split the target grid into row chunks."""
    return [
        frame.iloc[start : start + row_chunk_size].copy()
        for start in range(0, len(frame), row_chunk_size)
    ]


def align_target_chunk(
    *,
    chunk: pd.DataFrame,
    align_config: CrmAlignmentConfig,
    crm_sources: dict[str, CrmProductSource],
    cudem_sources: RasterSourceGroup,
    usgs_sources: RasterSourceGroup,
) -> pd.DataFrame:
    """Align all configured domain sources for one target-grid chunk."""
    result = chunk[
        ["aef_grid_row", "aef_grid_col", "aef_grid_cell_id", "longitude", "latitude"]
    ].copy()
    crm_values = sample_crm_sources(chunk, align_config, crm_sources)
    for column, values in crm_values.items():
        result[column] = values
    cudem_values = sample_raster_sources(
        chunk,
        cudem_sources,
        elevation_column="cudem_elevation_m",
        depth_column="cudem_depth_m",
        id_column="cudem_source_tile_id",
        path_column="cudem_source_path",
        status_column="cudem_value_status",
        derive_depth=True,
    )
    for column, values in cudem_values.items():
        result[column] = values
    usgs_values = sample_raster_sources(
        chunk,
        usgs_sources,
        elevation_column="usgs_3dep_elevation_m",
        depth_column=None,
        id_column="usgs_3dep_source_id",
        path_column="usgs_3dep_source_path",
        status_column="usgs_3dep_value_status",
        derive_depth=False,
    )
    for column, values in usgs_values.items():
        result[column] = values
    return result[list(OUTPUT_COLUMNS)]


def sample_crm_sources(
    chunk: pd.DataFrame,
    align_config: CrmAlignmentConfig,
    crm_sources: dict[str, CrmProductSource],
) -> dict[str, np.ndarray]:
    """Sample the configured CRM mosaic at target-cell centers."""
    longitudes = chunk["longitude"].to_numpy(dtype=np.float64)
    latitudes = chunk["latitude"].to_numpy(dtype=np.float64)
    count = len(chunk)
    elevation = np.full(count, np.nan, dtype=np.float32)
    depth = np.full(count, np.nan, dtype=np.float32)
    product_id_values = empty_object_array(count)
    product_name_values = empty_object_array(count)
    path_values = empty_object_array(count)
    vertical_datum_values = empty_object_array(count)
    method_values = np.full(count, align_config.resampling_method, dtype=object)
    status_values = np.full(count, "outside_product_boundary", dtype=object)

    south_mask = latitudes < align_config.product_boundary.latitude
    product_masks = (
        (align_config.product_boundary.south_product_id, south_mask),
        (align_config.product_boundary.north_product_id, ~south_mask),
    )
    for product_id, product_mask in product_masks:
        indices = np.flatnonzero(product_mask)
        if len(indices) == 0:
            continue
        source = crm_sources[product_id]
        product_id_values[indices] = source.product_id
        product_name_values[indices] = source.product_name
        path_values[indices] = str(source.local_path)
        vertical_datum_values[indices] = source.vertical_datum
        in_bounds = bounds_mask(longitudes[indices], latitudes[indices], source.bounds)
        status_values[indices] = "outside_source_bounds"
        sample_indices = indices[in_bounds]
        if len(sample_indices) == 0:
            continue
        values = sample_xarray_points(
            source,
            longitudes[sample_indices],
            latitudes[sample_indices],
        )
        valid_values = valid_numeric_values(values, source.nodata_values)
        elevation[sample_indices[valid_values]] = values[valid_values]
        depth[sample_indices[valid_values]] = np.maximum(0.0, -values[valid_values])
        status_values[sample_indices] = "no_data"
        status_values[sample_indices[valid_values]] = "valid"

    return {
        "crm_elevation_m": elevation,
        "crm_depth_m": depth,
        "crm_source_product_id": product_id_values,
        "crm_source_product_name": product_name_values,
        "crm_source_path": path_values,
        "crm_vertical_datum": vertical_datum_values,
        "crm_alignment_method": method_values,
        "crm_value_status": status_values,
    }


def empty_object_array(count: int) -> np.ndarray:
    """Return an object array initialized with pandas NA values."""
    values = np.empty(count, dtype=object)
    values[:] = pd.NA
    return values


def bounds_mask(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Return points whose lon/lat centers fall within bounds."""
    west, south, east, north = bounds
    return (longitudes >= west) & (longitudes <= east) & (latitudes >= south) & (latitudes <= north)


def sample_xarray_points(
    source: CrmProductSource,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Nearest-sample xarray data at lon/lat point arrays."""
    lon_selector = xr.DataArray(longitudes, dims="points")
    lat_selector = xr.DataArray(latitudes, dims="points")
    sampled = source.data_array.sel(
        {source.lon_name: lon_selector, source.lat_name: lat_selector},
        method="nearest",
    )
    return np.asarray(sampled.to_numpy(), dtype=np.float32)


def valid_numeric_values(values: np.ndarray, nodata: tuple[float, ...]) -> np.ndarray:
    """Return a mask of finite values that are not declared nodata sentinels."""
    valid = np.isfinite(values)
    for nodata_value in nodata:
        valid &= ~np.isclose(values, nodata_value)
    return cast(np.ndarray, valid)


def sample_raster_sources(
    chunk: pd.DataFrame,
    group: RasterSourceGroup,
    *,
    elevation_column: str,
    depth_column: str | None,
    id_column: str,
    path_column: str,
    status_column: str,
    derive_depth: bool,
) -> dict[str, np.ndarray]:
    """Sample an optional raster QA source group at target-cell centers."""
    count = len(chunk)
    elevation = np.full(count, np.nan, dtype=np.float32)
    depth = np.full(count, np.nan, dtype=np.float32) if depth_column is not None else None
    source_ids = empty_object_array(count)
    source_paths = empty_object_array(count)
    status = np.full(count, skipped_status_for_group(group), dtype=object)
    if not group.sources:
        result: dict[str, np.ndarray] = {
            elevation_column: elevation,
            id_column: source_ids,
            path_column: source_paths,
            status_column: status,
        }
        if depth_column is not None and depth is not None:
            result[depth_column] = depth
        return result

    longitudes = chunk["longitude"].to_numpy(dtype=np.float64)
    latitudes = chunk["latitude"].to_numpy(dtype=np.float64)
    unresolved = np.ones(count, dtype=bool)
    status[:] = "outside_coverage"
    for source in group.sources:
        candidate = unresolved & bounds_mask(longitudes, latitudes, source.bounds_wgs84)
        indices = np.flatnonzero(candidate)
        if len(indices) == 0:
            continue
        values = sample_raster_points(source, longitudes[indices], latitudes[indices])
        valid = valid_numeric_values(values, source.nodata_values)
        elevation[indices[valid]] = values[valid]
        if derive_depth and depth is not None:
            depth[indices[valid]] = np.maximum(0.0, -values[valid])
        source_ids[indices[valid]] = source.source_id
        source_paths[indices[valid]] = str(source.local_path)
        status[indices] = "no_data"
        status[indices[valid]] = "valid"
        unresolved[indices] = False

    result = {
        elevation_column: elevation,
        id_column: source_ids,
        path_column: source_paths,
        status_column: status,
    }
    if depth_column is not None and depth is not None:
        result[depth_column] = depth
    return result


def skipped_status_for_group(group: RasterSourceGroup) -> str:
    """Return the value-status string for a skipped optional source group."""
    if group.manifest_status == "manifest_missing":
        return "skipped_manifest_missing"
    if group.manifest_status == "not_configured":
        return "skipped_not_configured"
    return "skipped_no_valid_sources"


def sample_raster_points(
    source: RasterQaSource,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Sample one Rasterio raster at lon/lat point arrays."""
    if source.dataset.crs is None or str(source.dataset.crs).upper() in {"EPSG:4326", "EPSG:4269"}:
        xs = longitudes
        ys = latitudes
    else:
        transformed = warp_transform("EPSG:4326", source.dataset.crs, longitudes, latitudes)
        xs = np.asarray(transformed[0], dtype=np.float64)
        ys = np.asarray(transformed[1], dtype=np.float64)
    coordinates = list(zip(xs, ys, strict=True))
    samples = source.dataset.sample(coordinates, indexes=1, masked=True)
    values = [np.nan if np.ma.is_masked(sample[0]) else float(sample[0]) for sample in samples]
    return np.asarray(values, dtype=np.float32)


def update_accumulator(accumulator: AlignmentAccumulator, frame: pd.DataFrame) -> None:
    """Update running QA counters from one aligned chunk."""
    accumulator.total_cells += len(frame)
    accumulator.crm_status_counts.update(frame["crm_value_status"].astype(str).to_list())
    accumulator.cudem_status_counts.update(frame["cudem_value_status"].astype(str).to_list())
    accumulator.usgs_status_counts.update(frame["usgs_3dep_value_status"].astype(str).to_list())
    product_values = frame.loc[
        frame["crm_source_product_id"].notna(), "crm_source_product_id"
    ].astype(str)
    accumulator.crm_product_counts.update(product_values.to_list())
    valid_crm = frame["crm_value_status"] == "valid"
    update_min_max(
        accumulator,
        "crm_elevation",
        frame.loc[valid_crm, "crm_elevation_m"].to_numpy(dtype=np.float64),
    )
    update_min_max(
        accumulator,
        "crm_depth",
        frame.loc[valid_crm, "crm_depth_m"].to_numpy(dtype=np.float64),
    )
    accumulator.crm_depth_bin_counts.update(
        depth_bins(
            frame.loc[valid_crm, "crm_elevation_m"].to_numpy(dtype=np.float64),
            frame.loc[valid_crm, "crm_depth_m"].to_numpy(dtype=np.float64),
        )
    )
    update_comparison_stats(
        accumulator,
        "crm_vs_cudem",
        frame,
        other_elevation_column="cudem_elevation_m",
        other_depth_column="cudem_depth_m",
        other_status_column="cudem_value_status",
    )
    update_comparison_stats(
        accumulator,
        "crm_vs_usgs_3dep",
        frame,
        other_elevation_column="usgs_3dep_elevation_m",
        other_depth_column=None,
        other_status_column="usgs_3dep_value_status",
    )


def update_min_max(
    accumulator: AlignmentAccumulator,
    metric: str,
    values: np.ndarray,
) -> None:
    """Update min/max elevation or depth values."""
    if len(values) == 0:
        return
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if metric == "crm_elevation":
        accumulator.crm_elevation_min = combine_min(accumulator.crm_elevation_min, minimum)
        accumulator.crm_elevation_max = combine_max(accumulator.crm_elevation_max, maximum)
    elif metric == "crm_depth":
        accumulator.crm_depth_min = combine_min(accumulator.crm_depth_min, minimum)
        accumulator.crm_depth_max = combine_max(accumulator.crm_depth_max, maximum)


def combine_min(current: float | None, candidate: float) -> float:
    """Return the smaller non-null minimum."""
    return candidate if current is None else min(current, candidate)


def combine_max(current: float | None, candidate: float) -> float:
    """Return the larger non-null maximum."""
    return candidate if current is None else max(current, candidate)


def depth_bins(elevation: np.ndarray, depth: np.ndarray) -> Counter[str]:
    """Build broad CRM elevation/depth-bin counts."""
    counts: Counter[str] = Counter()
    counts["land_elevation_positive"] = int(np.sum(elevation > 0))
    counts["0_to_40m"] = int(np.sum((elevation <= 0) & (depth <= 40)))
    counts["40_to_50m"] = int(np.sum((depth > 40) & (depth <= 50)))
    counts["50_to_100m"] = int(np.sum((depth > 50) & (depth <= 100)))
    counts["100m_plus"] = int(np.sum(depth > 100))
    return counts


def update_comparison_stats(
    accumulator: AlignmentAccumulator,
    pair_name: str,
    frame: pd.DataFrame,
    *,
    other_elevation_column: str,
    other_depth_column: str | None,
    other_status_column: str,
) -> None:
    """Update CRM-vs-QA overlap and difference metrics."""
    overlap = (frame["crm_value_status"] == "valid") & (frame[other_status_column] == "valid")
    if not bool(overlap.any()):
        return
    crm_elevation = frame.loc[overlap, "crm_elevation_m"].to_numpy(dtype=np.float64)
    other_elevation = frame.loc[overlap, other_elevation_column].to_numpy(dtype=np.float64)
    update_diff_stat(accumulator, f"{pair_name}:elevation_diff_m", crm_elevation - other_elevation)
    if other_depth_column is not None:
        crm_depth = frame.loc[overlap, "crm_depth_m"].to_numpy(dtype=np.float64)
        other_depth = frame.loc[overlap, other_depth_column].to_numpy(dtype=np.float64)
        update_diff_stat(accumulator, f"{pair_name}:depth_diff_m", crm_depth - other_depth)
    sign_disagreement = np.signbit(crm_elevation) != np.signbit(other_elevation)
    accumulator.sign_disagreement_counts[pair_name] += int(np.sum(sign_disagreement))


def update_diff_stat(
    accumulator: AlignmentAccumulator,
    metric_name: str,
    values: np.ndarray,
) -> None:
    """Update a running difference metric."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    stats = accumulator.diff_stats.setdefault(metric_name, DiffStats())
    stats.count += int(len(finite))
    stats.total += float(np.sum(finite))
    stats.minimum = combine_min(stats.minimum, float(np.min(finite)))
    stats.maximum = combine_max(stats.maximum, float(np.max(finite)))


def write_summary_table(accumulator: AlignmentAccumulator, path: Path) -> None:
    """Write a tall QA summary CSV table."""
    rows = summary_rows(accumulator)
    write_csv(path, SUMMARY_FIELDS, rows)


def summary_rows(accumulator: AlignmentAccumulator) -> list[dict[str, object]]:
    """Build the QA summary rows."""
    rows: list[dict[str, object]] = [
        {
            "source": "target_grid",
            "metric": "total_cells",
            "category": "",
            "value": accumulator.total_cells,
        },
        {
            "source": "noaa_crm",
            "metric": "valid_cells",
            "category": "",
            "value": accumulator.crm_status_counts.get("valid", 0),
        },
        {
            "source": "noaa_crm",
            "metric": "missing_cells",
            "category": "",
            "value": accumulator.total_cells - accumulator.crm_status_counts.get("valid", 0),
        },
        {
            "source": "noaa_crm",
            "metric": "min_elevation_m",
            "category": "",
            "value": accumulator.crm_elevation_min,
        },
        {
            "source": "noaa_crm",
            "metric": "max_elevation_m",
            "category": "",
            "value": accumulator.crm_elevation_max,
        },
        {
            "source": "noaa_crm",
            "metric": "min_depth_m",
            "category": "",
            "value": accumulator.crm_depth_min,
        },
        {
            "source": "noaa_crm",
            "metric": "max_depth_m",
            "category": "",
            "value": accumulator.crm_depth_max,
        },
    ]
    rows.extend(counter_rows("noaa_crm", "value_status_cells", accumulator.crm_status_counts))
    rows.extend(counter_rows("noaa_crm", "source_product_cells", accumulator.crm_product_counts))
    rows.extend(counter_rows("noaa_crm", "depth_bin_cells", accumulator.crm_depth_bin_counts))
    rows.extend(counter_rows("noaa_cudem", "value_status_cells", accumulator.cudem_status_counts))
    rows.extend(counter_rows("usgs_3dep", "value_status_cells", accumulator.usgs_status_counts))
    return rows


def counter_rows(source: str, metric: str, counter: Counter[str]) -> list[dict[str, object]]:
    """Convert a counter into tall summary rows."""
    return [
        {"source": source, "metric": metric, "category": key, "value": value}
        for key, value in sorted(counter.items())
    ]


def write_comparison_table(accumulator: AlignmentAccumulator, path: Path) -> None:
    """Write CRM-to-QA cross-source comparison metrics."""
    rows: list[dict[str, object]] = []
    for metric_name, stats in sorted(accumulator.diff_stats.items()):
        pair_name, metric = metric_name.split(":", maxsplit=1)
        rows.extend(
            [
                {
                    "source_pair": pair_name,
                    "metric": f"{metric}:overlap_cells",
                    "value": stats.count,
                },
                {
                    "source_pair": pair_name,
                    "metric": f"{metric}:mean",
                    "value": stats.total / stats.count if stats.count else None,
                },
                {"source_pair": pair_name, "metric": f"{metric}:min", "value": stats.minimum},
                {"source_pair": pair_name, "metric": f"{metric}:max", "value": stats.maximum},
            ]
        )
    for pair_name, count in sorted(accumulator.sign_disagreement_counts.items()):
        rows.append(
            {
                "source_pair": pair_name,
                "metric": "elevation_sign_disagreement_cells",
                "value": count,
            }
        )
    rows.extend(coverage_comparison_rows(accumulator))
    write_csv(path, COMPARISON_FIELDS, rows)


def coverage_comparison_rows(accumulator: AlignmentAccumulator) -> list[dict[str, object]]:
    """Build source coverage percentage rows for the comparison table."""
    total = max(accumulator.total_cells, 1)
    return [
        {
            "source_pair": "crm_vs_cudem",
            "metric": "cudem_valid_cell_fraction",
            "value": accumulator.cudem_status_counts.get("valid", 0) / total,
        },
        {
            "source_pair": "crm_vs_usgs_3dep",
            "metric": "usgs_3dep_valid_cell_fraction",
            "value": accumulator.usgs_status_counts.get("valid", 0) / total,
        },
    ]


def build_alignment_manifest(
    *,
    align_config: CrmAlignmentConfig,
    target_grid: pd.DataFrame,
    target_bounds: tuple[float, float, float, float],
    accumulator: AlignmentAccumulator,
    cudem_sources: RasterSourceGroup,
    usgs_sources: RasterSourceGroup,
    cusp_validation: dict[str, Any],
    schema: pa.Schema | None,
) -> dict[str, Any]:
    """Build the JSON manifest for the aligned CRM support layer."""
    return {
        "command": "align-noaa-crm",
        "config_path": str(align_config.config_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "fast": align_config.fast,
        "overwrite_policy": "replace_existing_outputs",
        "target_grid": {
            "table": str(align_config.target_grid_table),
            "manifest": str(align_config.target_grid_manifest)
            if align_config.target_grid_manifest is not None
            else None,
            "source_year": align_config.target_grid_year,
            "row_count": int(len(target_grid)),
            "bounds": bounds_dict(target_bounds),
            "row_window": align_config.row_window,
            "col_window": align_config.col_window,
        },
        "source_manifests": {
            "noaa_crm": str(align_config.source_manifest),
            "noaa_crm_query": str(align_config.query_manifest),
            "noaa_cudem": str(align_config.cudem_tile_manifest)
            if align_config.cudem_tile_manifest is not None
            else None,
            "usgs_3dep": str(align_config.usgs_3dep_source_manifest)
            if align_config.usgs_3dep_source_manifest is not None
            else None,
            "noaa_cusp": str(align_config.cusp_source_manifest)
            if align_config.cusp_source_manifest is not None
            else None,
        },
        "crm_alignment": {
            "method": align_config.resampling_method,
            "product_boundary": {
                "latitude": align_config.product_boundary.latitude,
                "south_product_id": align_config.product_boundary.south_product_id,
                "north_product_id": align_config.product_boundary.north_product_id,
            },
            "elevation_sign_convention": "positive land elevation and negative ocean depth",
            "derived_depth_formula": "crm_depth_m = max(0, -crm_elevation_m)",
            "row_chunk_size": align_config.row_chunk_size,
        },
        "qa_source_status": {
            "noaa_cudem": raster_group_manifest_status(cudem_sources),
            "usgs_3dep": raster_group_manifest_status(usgs_sources),
            "noaa_cusp": cusp_validation,
        },
        "coverage_counts": {
            "noaa_crm": dict(sorted(accumulator.crm_status_counts.items())),
            "noaa_cudem": dict(sorted(accumulator.cudem_status_counts.items())),
            "usgs_3dep": dict(sorted(accumulator.usgs_status_counts.items())),
        },
        "crm_source_product_counts": dict(sorted(accumulator.crm_product_counts.items())),
        "outputs": {
            "table": str(align_config.output_table),
            "manifest": str(align_config.output_manifest),
            "qa_summary_table": str(align_config.qa_summary_table),
            "comparison_table": str(align_config.comparison_table),
        },
        "output_schema": schema_for_manifest(schema),
    }


def raster_group_manifest_status(group: RasterSourceGroup) -> dict[str, Any]:
    """Summarize one optional raster source group for the manifest."""
    return {
        "manifest_path": str(group.manifest_path) if group.manifest_path is not None else None,
        "status": group.manifest_status,
        "open_source_count": len(group.sources),
        "source_paths": [str(source.local_path) for source in group.sources],
    }


def schema_for_manifest(schema: pa.Schema | None) -> list[dict[str, str]]:
    """Convert a PyArrow schema to JSON-friendly records."""
    if schema is None:
        return []
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def bounds_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    """Convert bounds tuple into a named mapping."""
    west, south, east, north = bounds
    return {"west": west, "south": south, "east": east, "north": north}


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Write a CSV file with stable field ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    """Load a JSON file and validate that it is an object."""
    if not path.exists():
        msg = f"{description} does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        msg = f"{description} must contain a JSON object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
