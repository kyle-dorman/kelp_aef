"""Align AEF raster features with annual Kelpwatch labels."""

from __future__ import annotations

import csv
import json
import logging
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import rasterio  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.transform import rowcol  # type: ignore[import-untyped]
from rasterio.vrt import WarpedVRT  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

RASTERIO_AVERAGE_METHOD = "rasterio_average_10m_to_30m"
STATION_CENTERED_METHOD = "station_centered_3x3_mean"
LEGACY_STATION_CENTERED_METHOD = "mean_10m_to_kelpwatch_30m"
DEFAULT_ALIGNMENT_METHOD = RASTERIO_AVERAGE_METHOD
DEFAULT_FAST_MAX_STATIONS = 500
DEFAULT_TARGET_ROW_CHUNK_SIZE = 256
DEFAULT_LABEL_CRS = "EPSG:4326"
LABEL_REQUIRED_COLUMNS = (
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_max_y",
    "kelp_fraction_y",
)
SUMMARY_FIELDS = (
    "year",
    "row_count",
    "station_count",
    "complete_feature_row_count",
    "missing_feature_row_count",
    "expected_pixel_count_min",
    "expected_pixel_count_median",
    "expected_pixel_count_max",
    "valid_pixel_count_min",
    "valid_pixel_count_median",
    "valid_pixel_count_max",
    "kelp_valid_count",
    "kelp_gt0_count",
    "kelp_ge_1pct_count",
    "kelp_ge_5pct_count",
    "kelp_ge_10pct_count",
)
COMPARISON_FIELDS = (
    "year",
    "candidate_method",
    "reference_method",
    "row_count",
    "band_count",
    "compared_value_count",
    "mean_abs_diff",
    "median_abs_diff",
    "p95_abs_diff",
    "max_abs_diff",
    "row_max_abs_diff_median",
    "row_max_abs_diff_p95",
)


@dataclass(frozen=True)
class AlignmentConfig:
    """Resolved config values for feature/label alignment."""

    config_path: Path
    years: tuple[int, ...]
    label_path: Path
    label_manifest_path: Path
    label_crs: str
    tile_manifest_path: Path
    output_table_path: Path
    output_manifest_path: Path
    summary_table_path: Path
    comparison_table_path: Path | None
    bands: tuple[str, ...]
    method: str
    support_cells_per_side: int
    target_row_chunk_size: int
    fast: bool
    max_stations: int | None


@dataclass(frozen=True)
class AefYearAsset:
    """Local AEF raster asset for one aligned year."""

    year: int
    preferred_read_path: Path
    source_href: str | None
    raster_metadata: dict[str, Any]


@dataclass(frozen=True)
class SupportPixels:
    """Raster support pixel indices for station-centered aggregation windows."""

    rows: np.ndarray
    cols: np.ndarray
    expected_mask: np.ndarray


@dataclass(frozen=True)
class TargetGrid:
    """Rasterio target grid metadata for the AEF-aligned 30 m average path."""

    transform: Affine
    width: int
    height: int


def align_features_labels(
    config_path: Path,
    *,
    fast: bool = False,
    years: tuple[int, ...] | None = None,
    max_stations: int | None = None,
    output_table: Path | None = None,
    summary_output: Path | None = None,
    manifest_output: Path | None = None,
    comparison_output: Path | None = None,
) -> int:
    """Align configured AEF features and Kelpwatch labels into a parquet table."""
    align_config = load_alignment_config(
        config_path,
        fast=fast,
        years_override=years,
        max_stations_override=max_stations,
        output_table_override=output_table,
        summary_output_override=summary_output,
        manifest_output_override=manifest_output,
        comparison_output_override=comparison_output,
    )
    LOGGER.info(
        "Aligning features and labels for years %s with method %s",
        list(align_config.years),
        align_config.method,
    )
    labels = load_label_rows(align_config)
    if align_config.max_stations is not None:
        labels = spatial_station_subset(labels, align_config.max_stations)
    assets = load_aef_year_assets(align_config.tile_manifest_path, align_config.years)
    aligned_frames = [
        align_one_year(labels_for_year, assets[int(year)], align_config)
        for year, labels_for_year in labels.groupby("year", sort=True)
    ]
    aligned = pd.concat(aligned_frames, ignore_index=True)
    write_aligned_table(aligned, align_config.output_table_path)
    summary_rows = build_summary_rows(aligned, align_config.bands)
    write_summary_csv(summary_rows, align_config.summary_table_path)
    if should_write_fast_method_comparison(align_config):
        comparison_rows = build_fast_method_comparison(labels, assets, aligned, align_config)
        if align_config.comparison_table_path is not None:
            write_comparison_csv(comparison_rows, align_config.comparison_table_path)
            LOGGER.info(
                "Wrote fast alignment method comparison: %s", align_config.comparison_table_path
            )
    write_alignment_manifest(aligned, align_config, assets)
    LOGGER.info("Wrote aligned training table: %s", align_config.output_table_path)
    LOGGER.info("Wrote aligned training summary: %s", align_config.summary_table_path)
    LOGGER.info("Wrote aligned training manifest: %s", align_config.output_manifest_path)
    return 0


def load_alignment_config(
    config_path: Path,
    *,
    fast: bool,
    years_override: tuple[int, ...] | None,
    max_stations_override: int | None,
    output_table_override: Path | None,
    summary_output_override: Path | None,
    manifest_output_override: Path | None,
    comparison_output_override: Path | None,
) -> AlignmentConfig:
    """Load feature/label alignment settings from the workflow config."""
    config = load_yaml_config(config_path)
    years_config = require_mapping(config.get("years"), "years")
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    features = require_mapping(config.get("features"), "features")
    feature_paths = require_mapping(features.get("paths"), "features.paths")
    alignment = require_mapping(config.get("alignment"), "alignment")
    reports = require_mapping(config.get("reports"), "reports")

    configured_year_values = years_config.get("smoke")
    if not isinstance(configured_year_values, list):
        msg = "config field must be a list of integer years: years.smoke"
        raise ValueError(msg)
    configured_years = tuple(
        require_int_value(year, "years.smoke[]") for year in configured_year_values
    )
    fast_config = optional_mapping(alignment.get("fast"), "alignment.fast")
    selected_years = resolve_alignment_years(configured_years, fast_config, fast, years_override)
    selected_max_stations = resolve_max_stations(fast_config, fast, max_stations_override)

    label_path = Path(
        require_string(label_paths.get("annual_labels"), "labels.paths.annual_labels")
    )
    raw_label_manifest_path = label_paths.get("annual_label_manifest")
    label_manifest_path = (
        label_path.parent / "labels_annual_manifest.json"
        if raw_label_manifest_path is None
        else Path(
            require_string(
                raw_label_manifest_path,
                "labels.paths.annual_label_manifest",
            )
        )
    )
    label_crs = label_crs_from_manifest(label_manifest_path)
    output_table_path = resolve_output_path(
        alignment=alignment,
        fast_config=fast_config,
        fast=fast,
        override=output_table_override,
        config_key="output_table",
        fallback_path=Path(require_string(alignment.get("output_table"), "alignment.output_table")),
    )
    summary_table_path = resolve_output_path(
        alignment=alignment,
        fast_config=fast_config,
        fast=fast,
        override=summary_output_override,
        config_key="summary_table",
        fallback_path=Path(
            str(
                alignment.get("summary_table")
                or Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))
                / "aligned_training_table_summary.csv"
            )
        ),
    )
    output_manifest_path = resolve_output_path(
        alignment=alignment,
        fast_config=fast_config,
        fast=fast,
        override=manifest_output_override,
        config_key="output_manifest",
        fallback_path=Path(
            str(
                alignment.get("output_manifest")
                or output_table_path.parent / "aligned_training_table_manifest.json"
            )
        ),
    )
    comparison_table_path = resolve_optional_fast_output_path(
        fast_config=fast_config,
        fast=fast,
        override=comparison_output_override,
        config_key="comparison_table",
        fallback_path=suffix_path(summary_table_path, "_method_comparison"),
    )
    support_cells_per_side = support_cells_from_resolution(
        label_resolution_m=require_float_value(
            labels.get("native_resolution_m"),
            "labels.native_resolution_m",
        ),
        feature_resolution_m=require_float_value(
            features.get("native_resolution_m"),
            "features.native_resolution_m",
        ),
    )
    return AlignmentConfig(
        config_path=config_path,
        years=selected_years,
        label_path=label_path,
        label_manifest_path=label_manifest_path,
        label_crs=label_crs,
        tile_manifest_path=Path(
            require_string(feature_paths.get("tile_manifest"), "features.paths.tile_manifest")
        ),
        output_table_path=output_table_path,
        output_manifest_path=output_manifest_path,
        summary_table_path=summary_table_path,
        comparison_table_path=comparison_table_path,
        bands=parse_bands(features.get("bands")),
        method=alignment_method(alignment),
        support_cells_per_side=support_cells_per_side,
        target_row_chunk_size=optional_positive_int(
            alignment.get("target_row_chunk_size"),
            "alignment.target_row_chunk_size",
            DEFAULT_TARGET_ROW_CHUNK_SIZE,
        ),
        fast=fast,
        max_stations=selected_max_stations,
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating missing values as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def resolve_alignment_years(
    configured_years: tuple[int, ...],
    fast_config: dict[str, Any],
    fast: bool,
    years_override: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve alignment years from CLI overrides, fast config, or smoke config."""
    if years_override is not None:
        return years_override
    if not fast:
        return configured_years
    fast_year_values = fast_config.get("years")
    if isinstance(fast_year_values, list):
        return tuple(require_int_value(year, "alignment.fast.years[]") for year in fast_year_values)
    fast_year_value = fast_config.get("year")
    if fast_year_value is not None:
        return (require_int_value(fast_year_value, "alignment.fast.year"),)
    return (configured_years[-1],)


def resolve_max_stations(
    fast_config: dict[str, Any],
    fast: bool,
    max_stations_override: int | None,
) -> int | None:
    """Resolve the station cap for full or fast alignment runs."""
    if max_stations_override is not None:
        return max_stations_override
    if not fast:
        return None
    configured_value = fast_config.get("max_stations")
    if configured_value is None:
        return DEFAULT_FAST_MAX_STATIONS
    return require_int_value(configured_value, "alignment.fast.max_stations")


def resolve_output_path(
    *,
    alignment: dict[str, Any],
    fast_config: dict[str, Any],
    fast: bool,
    override: Path | None,
    config_key: str,
    fallback_path: Path,
) -> Path:
    """Resolve an output artifact path for full or fast alignment mode."""
    if override is not None:
        return override
    if fast:
        if config_key in fast_config:
            return Path(require_string(fast_config.get(config_key), f"alignment.fast.{config_key}"))
        return suffix_path(fallback_path, ".fast")
    if config_key in alignment:
        return Path(require_string(alignment.get(config_key), f"alignment.{config_key}"))
    return fallback_path


def resolve_optional_fast_output_path(
    *,
    fast_config: dict[str, Any],
    fast: bool,
    override: Path | None,
    config_key: str,
    fallback_path: Path,
) -> Path | None:
    """Resolve an optional output path that is only written for fast QA runs."""
    if override is not None:
        return override
    if not fast:
        return None
    if config_key in fast_config:
        return Path(require_string(fast_config.get(config_key), f"alignment.fast.{config_key}"))
    return fallback_path


def alignment_method(alignment: dict[str, Any]) -> str:
    """Resolve the configured alignment method name."""
    value = alignment.get("method")
    if value is None:
        return DEFAULT_ALIGNMENT_METHOD
    method = require_string(value, "alignment.method")
    return normalized_alignment_method(method)


def normalized_alignment_method(method: str) -> str:
    """Normalize known alignment method aliases to implementation names."""
    if method == LEGACY_STATION_CENTERED_METHOD:
        return STATION_CENTERED_METHOD
    if method in {RASTERIO_AVERAGE_METHOD, STATION_CENTERED_METHOD}:
        return method
    msg = (
        f"unsupported alignment method {method!r}; expected one of "
        f"{RASTERIO_AVERAGE_METHOD!r}, {STATION_CENTERED_METHOD!r}, "
        f"or legacy alias {LEGACY_STATION_CENTERED_METHOD!r}"
    )
    raise ValueError(msg)


def suffix_path(path: Path, suffix: str) -> Path:
    """Insert a suffix before a path's final extension."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def label_crs_from_manifest(path: Path) -> str:
    """Read the Kelpwatch label coordinate CRS from the annual label manifest."""
    if not path.exists():
        msg = f"annual label manifest does not exist: {path}"
        raise FileNotFoundError(msg)
    manifest = load_json_object(path)
    spatial = require_mapping(manifest.get("spatial"), "labels_annual_manifest.spatial")
    crs = spatial.get("crs")
    return DEFAULT_LABEL_CRS if crs is None else str(crs)


def support_cells_from_resolution(label_resolution_m: float, feature_resolution_m: float) -> int:
    """Convert native label and feature resolutions into an odd pixel window size."""
    ratio = label_resolution_m / feature_resolution_m
    cells_per_side = int(round(ratio))
    if cells_per_side < 1:
        msg = "alignment support must include at least one feature pixel"
        raise ValueError(msg)
    if abs(ratio - cells_per_side) > 1e-6:
        msg = (
            "label and feature resolutions must form an integer support ratio for "
            f"this first alignment path: {label_resolution_m}/{feature_resolution_m}"
        )
        raise ValueError(msg)
    if cells_per_side % 2 == 0:
        msg = f"support cells per side must be odd for station-centered windows: {cells_per_side}"
        raise ValueError(msg)
    return cells_per_side


def parse_bands(value: object) -> tuple[str, ...]:
    """Parse configured AEF band names from a range string or list."""
    if isinstance(value, list):
        bands = tuple(str(item) for item in value)
    elif isinstance(value, str):
        bands = parse_band_string(value)
    else:
        msg = "config field must be a band range string or list: features.bands"
        raise ValueError(msg)
    if not bands:
        msg = "at least one AEF band must be configured: features.bands"
        raise ValueError(msg)
    return bands


def parse_band_string(value: str) -> tuple[str, ...]:
    """Parse an AEF band string such as A00-A63 or A00,A01."""
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


def load_label_rows(align_config: AlignmentConfig) -> pd.DataFrame:
    """Load and validate annual label rows for the selected years."""
    if not align_config.label_path.exists():
        msg = f"annual label parquet does not exist: {align_config.label_path}"
        raise FileNotFoundError(msg)
    labels = pd.read_parquet(align_config.label_path)
    validate_label_columns(labels)
    selected = labels.loc[labels["year"].isin(align_config.years)].copy()
    if selected.empty:
        msg = f"annual labels contain no rows for years: {list(align_config.years)}"
        raise ValueError(msg)
    missing_years = sorted(
        set(align_config.years) - set(int(year) for year in selected["year"].unique())
    )
    if missing_years:
        msg = f"annual labels are missing configured years: {missing_years}"
        raise ValueError(msg)
    selected = selected.sort_values(["year", "kelpwatch_station_id"]).reset_index(drop=True)
    LOGGER.info(
        "Loaded %s annual label rows for %s stations",
        len(selected),
        selected["kelpwatch_station_id"].nunique(),
    )
    return selected


def validate_label_columns(labels: pd.DataFrame) -> None:
    """Validate that required label columns are present."""
    missing = [column for column in LABEL_REQUIRED_COLUMNS if column not in labels.columns]
    if missing:
        msg = f"annual labels are missing required columns: {missing}"
        raise ValueError(msg)


def spatial_station_subset(labels: pd.DataFrame, max_stations: int) -> pd.DataFrame:
    """Select a spatially coherent station subset around the label centroid."""
    station_frame = labels.drop_duplicates("kelpwatch_station_id")[
        ["kelpwatch_station_id", "longitude", "latitude"]
    ].copy()
    if len(station_frame) <= max_stations:
        LOGGER.info(
            "Station subset cap %s exceeds station count; keeping all stations", max_stations
        )
        return labels
    center_lon = float(station_frame["longitude"].median())
    center_lat = float(station_frame["latitude"].median())
    distance_squared = (station_frame["longitude"] - center_lon) ** 2 + (
        station_frame["latitude"] - center_lat
    ) ** 2
    selected_station_ids = set(
        station_frame.assign(_distance_squared=distance_squared)
        .nsmallest(max_stations, "_distance_squared")["kelpwatch_station_id"]
        .to_list()
    )
    selected = labels.loc[labels["kelpwatch_station_id"].isin(selected_station_ids)].copy()
    LOGGER.info(
        "Selected spatial fast subset: %s stations, %s rows",
        selected["kelpwatch_station_id"].nunique(),
        len(selected),
    )
    return selected.reset_index(drop=True)


def load_aef_year_assets(manifest_path: Path, years: tuple[int, ...]) -> dict[int, AefYearAsset]:
    """Load local AEF asset records from the tile manifest."""
    if not manifest_path.exists():
        msg = f"AEF tile manifest does not exist: {manifest_path}"
        raise FileNotFoundError(msg)
    manifest = load_json_object(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        msg = f"AEF tile manifest does not contain records: {manifest_path}"
        raise ValueError(msg)
    assets: dict[int, AefYearAsset] = {}
    for raw_record in records:
        record = require_mapping(raw_record, "aef_tile_manifest.records[]")
        year = require_int_value(record.get("year"), "aef_tile_manifest.records[].year")
        if year not in years:
            continue
        preferred_read_path = Path(
            require_string(
                record.get("preferred_read_path"),
                "aef_tile_manifest.records[].preferred_read_path",
            )
        )
        if not preferred_read_path.exists():
            msg = f"preferred AEF read path does not exist for {year}: {preferred_read_path}"
            raise FileNotFoundError(msg)
        source_href = record.get("source_href")
        assets[year] = AefYearAsset(
            year=year,
            preferred_read_path=preferred_read_path,
            source_href=str(source_href) if source_href is not None else None,
            raster_metadata=require_mapping(
                record.get("raster"), "aef_tile_manifest.records[].raster"
            ),
        )
    missing_years = sorted(set(years) - set(assets))
    if missing_years:
        msg = f"AEF tile manifest is missing selected years: {missing_years}"
        raise ValueError(msg)
    return assets


def align_one_year(
    labels_for_year: pd.DataFrame,
    asset: AefYearAsset,
    align_config: AlignmentConfig,
) -> pd.DataFrame:
    """Align one year of label rows against one AEF raster asset."""
    if align_config.method == RASTERIO_AVERAGE_METHOD:
        return align_one_year_rasterio_average(labels_for_year, asset, align_config)
    if align_config.method == STATION_CENTERED_METHOD:
        return align_one_year_station_centered(labels_for_year, asset, align_config)
    msg = f"unsupported alignment method: {align_config.method}"
    raise ValueError(msg)


def align_one_year_station_centered(
    labels_for_year: pd.DataFrame,
    asset: AefYearAsset,
    align_config: AlignmentConfig,
) -> pd.DataFrame:
    """Align one year with exact station-centered support windows."""
    LOGGER.info(
        "Aligning %s rows for %s from %s with %s",
        len(labels_for_year),
        asset.year,
        asset.preferred_read_path,
        align_config.method,
    )
    with rasterio.open(asset.preferred_read_path) as dataset:
        band_indexes = resolve_band_indexes(dataset, align_config.bands)
        x_values, y_values = transformed_label_points(
            labels_for_year, align_config.label_crs, dataset.crs
        )
        support = support_pixel_indices(
            dataset=dataset,
            x_values=x_values,
            y_values=y_values,
            cells_per_side=align_config.support_cells_per_side,
        )
        support_values = read_support_values(dataset, band_indexes, support)
        feature_matrix, valid_pixel_counts = mean_features_from_support(
            support_values,
            support.expected_mask,
        )
    expected_pixel_counts = support.expected_mask.sum(axis=1).astype(int)
    features = pd.DataFrame(feature_matrix, columns=list(align_config.bands))
    features["aef_x"] = x_values
    features["aef_y"] = y_values
    features["aef_expected_pixel_count"] = expected_pixel_counts
    features["aef_valid_pixel_count"] = valid_pixel_counts
    features["aef_missing_pixel_count"] = expected_pixel_counts - valid_pixel_counts
    features["aef_alignment_method"] = align_config.method
    features["aef_source_path"] = str(asset.preferred_read_path)
    features["aef_source_href"] = asset.source_href
    aligned = pd.concat(
        [labels_for_year.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    missing_rows = int((aligned["aef_valid_pixel_count"] == 0).sum())
    LOGGER.info(
        "Aligned %s rows for %s; rows with no valid AEF support: %s",
        len(aligned),
        asset.year,
        missing_rows,
    )
    return aligned


def align_one_year_rasterio_average(
    labels_for_year: pd.DataFrame,
    asset: AefYearAsset,
    align_config: AlignmentConfig,
) -> pd.DataFrame:
    """Align one year with a Rasterio/GDAL-averaged AEF 30 m grid."""
    LOGGER.info(
        "Aligning %s rows for %s from %s with %s",
        len(labels_for_year),
        asset.year,
        asset.preferred_read_path,
        align_config.method,
    )
    with rasterio.open(asset.preferred_read_path) as dataset:
        band_indexes = resolve_band_indexes(dataset, align_config.bands)
        target_grid = aef_average_target_grid(dataset, align_config.support_cells_per_side)
        x_values, y_values = transformed_label_points(
            labels_for_year,
            align_config.label_crs,
            dataset.crs,
        )
        target_rows, target_cols, target_mask = target_pixel_indices(
            target_grid,
            x_values,
            y_values,
        )
        with WarpedVRT(
            dataset,
            crs=dataset.crs,
            transform=target_grid.transform,
            width=target_grid.width,
            height=target_grid.height,
            resampling=Resampling.average,
            src_nodata=dataset.nodata,
            nodata=dataset.nodata,
            dtype="float32",
        ) as vrt:
            feature_matrix = read_target_pixel_values(
                vrt,
                band_indexes,
                target_rows,
                target_cols,
                target_mask,
                align_config.target_row_chunk_size,
            )
        expected_pixel_counts, valid_pixel_counts = source_pixel_counts_for_target_cells(
            dataset=dataset,
            band_index=band_indexes[0],
            target_rows=target_rows,
            target_cols=target_cols,
            target_mask=target_mask,
            cells_per_side=align_config.support_cells_per_side,
        )
    features = pd.DataFrame(feature_matrix, columns=list(align_config.bands))
    features["aef_x"] = x_values
    features["aef_y"] = y_values
    features["aef_expected_pixel_count"] = expected_pixel_counts
    features["aef_valid_pixel_count"] = valid_pixel_counts
    features["aef_missing_pixel_count"] = expected_pixel_counts - valid_pixel_counts
    features["aef_alignment_method"] = align_config.method
    features["aef_source_path"] = str(asset.preferred_read_path)
    features["aef_source_href"] = asset.source_href
    aligned = pd.concat(
        [labels_for_year.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    missing_rows = int((aligned["aef_valid_pixel_count"] == 0).sum())
    LOGGER.info(
        "Aligned %s rows for %s; rows with no valid AEF support: %s",
        len(aligned),
        asset.year,
        missing_rows,
    )
    return aligned


def transformed_label_points(
    labels: pd.DataFrame,
    source_crs: str,
    target_crs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform label station points into the raster CRS with GeoPandas."""
    if target_crs is None:
        msg = "AEF raster has no CRS"
        raise ValueError(msg)
    station_points = gpd.GeoDataFrame(
        labels[["kelpwatch_station_id"]].copy(),
        geometry=gpd.points_from_xy(labels["longitude"], labels["latitude"]),
        crs=source_crs,
    ).to_crs(target_crs)
    x_values = station_points.geometry.x.to_numpy(dtype=float)
    y_values = station_points.geometry.y.to_numpy(dtype=float)
    return x_values, y_values


def aef_average_target_grid(dataset: Any, cells_per_side: int) -> TargetGrid:
    """Build an AEF-aligned coarser target grid for Rasterio average resampling."""
    return TargetGrid(
        transform=dataset.transform * Affine.scale(cells_per_side, cells_per_side),
        width=math.ceil(int(dataset.width) / cells_per_side),
        height=math.ceil(int(dataset.height) / cells_per_side),
    )


def target_pixel_indices(
    target_grid: TargetGrid,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target-grid coordinates to row/column indices and an in-grid mask."""
    rows, cols = rowcol(target_grid.transform, x_values, y_values)
    row_array = np.asarray(rows, dtype=int)
    col_array = np.asarray(cols, dtype=int)
    target_mask = (
        (row_array >= 0)
        & (row_array < target_grid.height)
        & (col_array >= 0)
        & (col_array < target_grid.width)
    )
    return row_array, col_array, target_mask


def read_target_pixel_values(
    dataset: Any,
    band_indexes: tuple[int, ...],
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    target_mask: np.ndarray,
    target_row_chunk_size: int,
) -> np.ndarray:
    """Read one value per target-grid pixel from a Rasterio dataset or VRT."""
    values = np.full((target_rows.size, len(band_indexes)), np.nan, dtype=np.float32)
    if not np.any(target_mask):
        return values
    chunk_indices = target_chunk_indices(target_rows, target_mask, target_row_chunk_size)
    for indices in chunk_indices:
        window = target_window_for_indices(target_rows, target_cols, indices)
        window_rows = target_rows[indices] - int(window.row_off)
        window_cols = target_cols[indices] - int(window.col_off)
        for band_position, band_index in enumerate(band_indexes):
            band = dataset.read(band_index, window=window, masked=True).astype(np.float32)
            band_array = np.ma.filled(band, np.nan)
            values[indices, band_position] = band_array[window_rows, window_cols]
    return values


def target_chunk_indices(
    target_rows: np.ndarray,
    target_mask: np.ndarray,
    target_row_chunk_size: int,
) -> list[np.ndarray]:
    """Group in-grid target pixel indices into row chunks."""
    valid_indices = np.flatnonzero(target_mask)
    if valid_indices.size == 0:
        return []
    base_row = int(target_rows[valid_indices].min())
    chunk_ids = (target_rows[valid_indices] - base_row) // target_row_chunk_size
    chunks: list[np.ndarray] = []
    for chunk_id in np.unique(chunk_ids):
        chunks.append(valid_indices[chunk_ids == chunk_id])
    return chunks


def target_window_for_indices(
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    indices: np.ndarray,
) -> Window:
    """Build the minimal Rasterio window containing selected target pixels."""
    row_start = int(target_rows[indices].min())
    row_stop = int(target_rows[indices].max()) + 1
    col_start = int(target_cols[indices].min())
    col_stop = int(target_cols[indices].max()) + 1
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def source_pixel_counts_for_target_cells(
    *,
    dataset: Any,
    band_index: int,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    target_mask: np.ndarray,
    cells_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Count expected and valid source pixels for AEF-aligned target cells."""
    expected_counts = np.zeros(target_rows.size, dtype=int)
    valid_counts = np.zeros(target_rows.size, dtype=int)
    if not np.any(target_mask):
        return expected_counts, valid_counts
    source_rows, source_cols, source_mask = source_pixels_for_target_cells(
        dataset=dataset,
        target_rows=target_rows,
        target_cols=target_cols,
        target_mask=target_mask,
        cells_per_side=cells_per_side,
    )
    expected_counts = source_mask.sum(axis=1).astype(int)
    if not np.any(source_mask):
        return expected_counts, valid_counts
    source_read_window = support_window(
        SupportPixels(rows=source_rows, cols=source_cols, expected_mask=source_mask)
    )
    window_rows = source_rows - int(source_read_window.row_off)
    window_cols = source_cols - int(source_read_window.col_off)
    mask = dataset.read_masks(band_index, window=source_read_window)
    valid_source_mask = np.zeros(source_rows.shape, dtype=bool)
    valid_source_mask[source_mask] = mask[window_rows[source_mask], window_cols[source_mask]] > 0
    valid_counts = valid_source_mask.sum(axis=1).astype(int)
    return expected_counts, valid_counts


def source_pixels_for_target_cells(
    *,
    dataset: Any,
    target_rows: np.ndarray,
    target_cols: np.ndarray,
    target_mask: np.ndarray,
    cells_per_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map AEF-aligned target pixels to their covered source pixel blocks."""
    offsets = np.arange(cells_per_side, dtype=int)
    row_offsets, col_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    source_rows = target_rows[:, None] * cells_per_side + row_offsets.ravel()[None, :]
    source_cols = target_cols[:, None] * cells_per_side + col_offsets.ravel()[None, :]
    source_mask = (
        target_mask[:, None]
        & (source_rows >= 0)
        & (source_rows < int(dataset.height))
        & (source_cols >= 0)
        & (source_cols < int(dataset.width))
    )
    return source_rows, source_cols, source_mask


def support_pixel_indices(
    *,
    dataset: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    cells_per_side: int,
) -> SupportPixels:
    """Build station support pixel row/column arrays for a square centered window."""
    center_rows, center_cols = rowcol(dataset.transform, x_values, y_values)
    row_array = np.asarray(center_rows, dtype=int)
    col_array = np.asarray(center_cols, dtype=int)
    half_width = cells_per_side // 2
    offsets = np.arange(-half_width, half_width + 1, dtype=int)
    row_offsets, col_offsets = np.meshgrid(offsets, offsets, indexing="ij")
    support_rows = row_array[:, None] + row_offsets.ravel()[None, :]
    support_cols = col_array[:, None] + col_offsets.ravel()[None, :]
    expected_mask = (
        (support_rows >= 0)
        & (support_rows < int(dataset.height))
        & (support_cols >= 0)
        & (support_cols < int(dataset.width))
    )
    return SupportPixels(rows=support_rows, cols=support_cols, expected_mask=expected_mask)


def resolve_band_indexes(dataset: Any, bands: tuple[str, ...]) -> tuple[int, ...]:
    """Resolve configured band names to one-based Rasterio band indexes."""
    descriptions = tuple(description or "" for description in dataset.descriptions)
    description_to_index = {
        description: index + 1 for index, description in enumerate(descriptions) if description
    }
    if all(band in description_to_index for band in bands):
        return tuple(description_to_index[band] for band in bands)
    numeric_indexes = tuple(index_from_aef_band_name(band) for band in bands)
    if all(index <= int(dataset.count) for index in numeric_indexes):
        return numeric_indexes
    if len(bands) <= int(dataset.count):
        LOGGER.warning(
            "AEF raster band descriptions do not match requested names; using first %s bands",
            len(bands),
        )
        return tuple(range(1, len(bands) + 1))
    msg = f"raster has {dataset.count} bands but requested {len(bands)} bands"
    raise ValueError(msg)


def index_from_aef_band_name(name: str) -> int:
    """Convert an AEF band name like A00 into a one-based raster band index."""
    match = re.fullmatch(r"A(\d+)", name)
    if match is None:
        msg = f"unsupported AEF band name: {name}"
        raise ValueError(msg)
    return int(match.group(1)) + 1


def read_support_values(
    dataset: Any,
    band_indexes: tuple[int, ...],
    support: SupportPixels,
) -> np.ndarray:
    """Read raster values covering all support pixels for all requested bands."""
    values = np.full(
        (len(band_indexes), support.rows.shape[0], support.rows.shape[1]),
        np.nan,
        dtype=np.float32,
    )
    if not np.any(support.expected_mask):
        return values
    window = support_window(support)
    window_rows = support.rows - int(window.row_off)
    window_cols = support.cols - int(window.col_off)
    for band_position, band_index in enumerate(band_indexes):
        band = dataset.read(band_index, window=window, masked=True).astype(np.float32)
        band_array = np.ma.filled(band, np.nan)
        band_values = np.full(support.rows.shape, np.nan, dtype=np.float32)
        band_values[support.expected_mask] = band_array[
            window_rows[support.expected_mask],
            window_cols[support.expected_mask],
        ]
        values[band_position] = band_values
    return values


def support_window(support: SupportPixels) -> Window:
    """Build the minimal Rasterio window containing all in-raster support pixels."""
    valid_rows = support.rows[support.expected_mask]
    valid_cols = support.cols[support.expected_mask]
    row_start = int(valid_rows.min())
    row_stop = int(valid_rows.max()) + 1
    col_start = int(valid_cols.min())
    col_stop = int(valid_cols.max()) + 1
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def mean_features_from_support(
    values: np.ndarray,
    expected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average feature values over pixels valid for every requested band."""
    all_band_valid = np.all(np.isfinite(values), axis=0) & expected_mask
    valid_pixel_counts = all_band_valid.sum(axis=1).astype(int)
    masked_values = np.where(all_band_valid[None, :, :], values, np.nan)
    sums = np.nansum(masked_values, axis=2)
    means = np.full((values.shape[1], values.shape[0]), np.nan, dtype=np.float32)
    valid_rows = valid_pixel_counts > 0
    if np.any(valid_rows):
        means[valid_rows, :] = (sums[:, valid_rows] / valid_pixel_counts[valid_rows]).T
    return means, valid_pixel_counts


def should_write_fast_method_comparison(align_config: AlignmentConfig) -> bool:
    """Return whether this run should emit a fast method comparison summary."""
    return (
        align_config.fast
        and align_config.method == RASTERIO_AVERAGE_METHOD
        and align_config.comparison_table_path is not None
    )


def build_fast_method_comparison(
    labels: pd.DataFrame,
    assets: dict[int, AefYearAsset],
    candidate: pd.DataFrame,
    align_config: AlignmentConfig,
) -> list[dict[str, object]]:
    """Compare the fast Rasterio average method against station-centered means."""
    reference_config = AlignmentConfig(
        config_path=align_config.config_path,
        years=align_config.years,
        label_path=align_config.label_path,
        label_manifest_path=align_config.label_manifest_path,
        label_crs=align_config.label_crs,
        tile_manifest_path=align_config.tile_manifest_path,
        output_table_path=align_config.output_table_path,
        output_manifest_path=align_config.output_manifest_path,
        summary_table_path=align_config.summary_table_path,
        comparison_table_path=align_config.comparison_table_path,
        bands=align_config.bands,
        method=STATION_CENTERED_METHOD,
        support_cells_per_side=align_config.support_cells_per_side,
        target_row_chunk_size=align_config.target_row_chunk_size,
        fast=align_config.fast,
        max_stations=align_config.max_stations,
    )
    reference_frames = [
        align_one_year_station_centered(labels_for_year, assets[int(year)], reference_config)
        for year, labels_for_year in labels.groupby("year", sort=True)
    ]
    reference = pd.concat(reference_frames, ignore_index=True)
    return build_comparison_rows(candidate, reference, align_config.bands)


def build_comparison_rows(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    bands: tuple[str, ...],
) -> list[dict[str, object]]:
    """Build per-year feature-difference summaries for two alignment methods."""
    join_keys = ["year", "kelpwatch_station_id"]
    candidate_columns = join_keys + list(bands)
    reference_columns = join_keys + list(bands)
    merged = candidate[candidate_columns].merge(
        reference[reference_columns],
        on=join_keys,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for year, group in merged.groupby("year", sort=True):
        candidate_values = group[[f"{band}_candidate" for band in bands]].to_numpy(dtype=float)
        reference_values = group[[f"{band}_reference" for band in bands]].to_numpy(dtype=float)
        absolute_diff = np.abs(candidate_values - reference_values)
        finite_diff = absolute_diff[np.isfinite(absolute_diff)]
        row_max_diff = nanmax_by_row(absolute_diff)
        finite_row_max = row_max_diff[np.isfinite(row_max_diff)]
        rows.append(
            {
                "year": int(year),
                "candidate_method": RASTERIO_AVERAGE_METHOD,
                "reference_method": STATION_CENTERED_METHOD,
                "row_count": int(len(group)),
                "band_count": len(bands),
                "compared_value_count": int(finite_diff.size),
                "mean_abs_diff": finite_stat(finite_diff, np.nanmean),
                "median_abs_diff": finite_percentile(finite_diff, 50),
                "p95_abs_diff": finite_percentile(finite_diff, 95),
                "max_abs_diff": finite_stat(finite_diff, np.nanmax),
                "row_max_abs_diff_median": finite_percentile(finite_row_max, 50),
                "row_max_abs_diff_p95": finite_percentile(finite_row_max, 95),
            }
        )
    return rows


def nanmax_by_row(values: np.ndarray) -> np.ndarray:
    """Return row-wise maxima while preserving all-NaN rows as NaN."""
    result = np.full(values.shape[0], np.nan, dtype=float)
    valid_rows = np.any(np.isfinite(values), axis=1)
    if np.any(valid_rows):
        result[valid_rows] = np.nanmax(values[valid_rows], axis=1)
    return result


def finite_stat(values: np.ndarray, function: Any) -> float:
    """Return a statistic for finite values or NaN when no values are finite."""
    if values.size == 0:
        return float("nan")
    return float(function(values))


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a percentile for finite values or NaN when no values are finite."""
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def build_summary_rows(aligned: pd.DataFrame, bands: tuple[str, ...]) -> list[dict[str, object]]:
    """Build per-year summary rows for the aligned table."""
    rows: list[dict[str, object]] = []
    for year, group in aligned.groupby("year", sort=True):
        feature_missing = group[list(bands)].isna().any(axis=1)
        expected_counts = group["aef_expected_pixel_count"].to_numpy(dtype=float)
        valid_counts = group["aef_valid_pixel_count"].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "row_count": int(len(group)),
                "station_count": int(group["kelpwatch_station_id"].nunique()),
                "complete_feature_row_count": int((~feature_missing).sum()),
                "missing_feature_row_count": int(feature_missing.sum()),
                "expected_pixel_count_min": percentile_value(expected_counts, 0),
                "expected_pixel_count_median": percentile_value(expected_counts, 50),
                "expected_pixel_count_max": percentile_value(expected_counts, 100),
                "valid_pixel_count_min": percentile_value(valid_counts, 0),
                "valid_pixel_count_median": percentile_value(valid_counts, 50),
                "valid_pixel_count_max": percentile_value(valid_counts, 100),
                "kelp_valid_count": int(group["kelp_max_y"].notna().sum()),
                "kelp_gt0_count": sum_bool_column(group, "kelp_present_gt0_y"),
                "kelp_ge_1pct_count": sum_bool_column(group, "kelp_present_ge_1pct_y"),
                "kelp_ge_5pct_count": sum_bool_column(group, "kelp_present_ge_5pct_y"),
                "kelp_ge_10pct_count": sum_bool_column(group, "kelp_present_ge_10pct_y"),
            }
        )
    return rows


def percentile_value(values: np.ndarray, percentile: float) -> float:
    """Return a percentile value or NaN for an empty array."""
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def sum_bool_column(dataframe: pd.DataFrame, column: str) -> int:
    """Sum a boolean diagnostic column when it exists."""
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].fillna(False).sum())


def write_aligned_table(aligned: pd.DataFrame, output_path: Path) -> None:
    """Write the aligned feature/label table to parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(output_path, index=False)


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write aligned-table summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write fast alignment method-comparison rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_alignment_manifest(
    aligned: pd.DataFrame,
    align_config: AlignmentConfig,
    assets: dict[int, AefYearAsset],
) -> None:
    """Write the aligned-table manifest JSON."""
    payload = {
        "command": "align",
        "config_path": str(align_config.config_path),
        "fast": align_config.fast,
        "years": list(align_config.years),
        "max_stations": align_config.max_stations,
        "label_path": str(align_config.label_path),
        "label_manifest_path": str(align_config.label_manifest_path),
        "label_crs": align_config.label_crs,
        "tile_manifest_path": str(align_config.tile_manifest_path),
        "output_table": str(align_config.output_table_path),
        "summary_table": str(align_config.summary_table_path),
        "output_manifest": str(align_config.output_manifest_path),
        "comparison_table": (
            str(align_config.comparison_table_path)
            if align_config.comparison_table_path is not None
            else None
        ),
        "alignment_method": align_config.method,
        "support_cells_per_side": align_config.support_cells_per_side,
        "target_row_chunk_size": align_config.target_row_chunk_size,
        "bands": list(align_config.bands),
        "row_count": int(len(aligned)),
        "station_count": int(aligned["kelpwatch_station_id"].nunique()),
        "feature_missing_row_count": int(
            aligned[list(align_config.bands)].isna().any(axis=1).sum()
        ),
        "assets": {
            str(year): {
                "preferred_read_path": str(asset.preferred_read_path),
                "source_href": asset.source_href,
                "raster": asset.raster_metadata,
            }
            for year, asset in sorted(assets.items())
        },
        "schema": list(aligned.columns),
    }
    write_json(align_config.output_manifest_path, payload)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"expected JSON object at {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


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


def require_float_value(value: object, name: str) -> float:
    """Validate a numeric dynamic value as a finite positive float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"field must be numeric: {name}"
        raise ValueError(msg)
    parsed = float(value)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Validate an optional positive integer dynamic value."""
    if value is None:
        return default
    parsed = require_int_value(value, name)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed
