"""Build full-grid AEF/Kelpwatch alignment and sampled training artifacts."""

from __future__ import annotations

import csv
import json
import logging
import operator
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import rasterio  # type: ignore[import-untyped]
from rasterio.enums import Resampling  # type: ignore[import-untyped]
from rasterio.vrt import WarpedVRT  # type: ignore[import-untyped]
from rasterio.warp import transform as warp_transform  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

from kelp_aef.alignment.feature_label_table import (
    RASTERIO_AVERAGE_METHOD,
    AefYearAsset,
    TargetGrid,
    aef_average_target_grid,
    label_crs_from_manifest,
    load_aef_year_assets,
    parse_bands,
    resolve_band_indexes,
    source_pixel_counts_for_target_cells,
    support_cells_from_resolution,
    transformed_label_points,
)
from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.reporting_mask import (
    DEFAULT_PRIMARY_DOMAIN,
    MASK_DETAIL_COLUMNS,
    MASK_KEY_COLUMN,
    MASK_RETAIN_COLUMN,
    load_reporting_domain_mask,
    mask_columns_in_frame,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_ROW_CHUNK_SIZE = 128
DEFAULT_BACKGROUND_ROWS_PER_YEAR = 250_000
DEFAULT_RANDOM_SEED = 13
DEFAULT_SAMPLE_WEIGHT_COLUMN = "sample_weight"
DEFAULT_LABEL_CRS = "EPSG:4326"
ASSUMED_BACKGROUND = "assumed_background"
KELPWATCH_STATION = "kelpwatch_station"
AREA_COLUMNS = ("kelp_max_y", "kelp_fraction_y", "area_q1", "area_q2", "area_q3", "area_q4")
BOOLEAN_LABEL_COLUMNS = (
    "kelp_present_gt0_y",
    "kelp_present_ge_1pct_y",
    "kelp_present_ge_5pct_y",
    "kelp_present_ge_10pct_y",
)
LABEL_METADATA_COLUMNS = ("max_area_quarter", "valid_quarter_count", "nonzero_quarter_count")
FULL_GRID_SUMMARY_FIELDS = (
    "year",
    "label_source",
    "row_count",
    "sampled_row_count",
    "complete_feature_row_count",
    "missing_feature_row_count",
    "observed_canopy_area",
)
SAMPLE_SUMMARY_FIELDS = (
    "year",
    "label_source",
    "row_count",
    "sample_weight_min",
    "sample_weight_max",
)
MASKED_SAMPLE_SUMMARY_FIELDS = (
    "year",
    "label_source",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "row_count",
    "kelpwatch_observed_row_count",
    "kelpwatch_positive_row_count",
    "sample_weight_min",
    "sample_weight_max",
)


@dataclass(frozen=True)
class SampleDomainMaskConfig:
    """Resolved plausible-kelp mask settings for model-input sampling."""

    table_path: Path
    manifest_path: Path | None
    policy: str
    output_path: Path
    manifest_output_path: Path
    summary_path: Path
    fail_on_dropped_positive: bool


@dataclass(frozen=True)
class FullGridAlignmentConfig:
    """Resolved config values for full-grid alignment."""

    config_path: Path
    years: tuple[int, ...]
    label_path: Path
    label_manifest_path: Path
    label_crs: str
    tile_manifest_path: Path
    bands: tuple[str, ...]
    support_cells_per_side: int
    target_row_chunk_size: int
    full_grid_output_path: Path
    full_grid_manifest_path: Path
    full_grid_summary_path: Path
    sample_output_path: Path
    sample_manifest_path: Path
    sample_summary_path: Path
    background_rows_per_year: int
    include_all_kelpwatch_observed: bool
    random_seed: int
    sample_weight_column: str
    sample_domain_mask: SampleDomainMaskConfig | None
    fast: bool
    row_window: tuple[int, int] | None
    col_window: tuple[int, int] | None


@dataclass(frozen=True)
class MaskedSampleResult:
    """Metadata returned after writing a masked background sample sidecar."""

    sample: pd.DataFrame
    population_counts: dict[int, dict[str, int]]
    retained_counts: dict[str, int]
    dropped_counts: dict[str, int]
    dropped_observed_row_count: int
    dropped_positive_row_count: int


@dataclass(frozen=True)
class GridWindow:
    """Target-grid row and column bounds for one chunk."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int


@dataclass
class CountAccumulator:
    """Mutable counters for full-grid and sample summary rows."""

    full_counts: dict[tuple[int, str], int]
    full_complete_counts: dict[tuple[int, str], int]
    full_observed_area: dict[tuple[int, str], float]
    sample_counts: dict[tuple[int, str], int]
    duplicate_cell_counts: dict[int, int]


def align_full_grid(config_path: Path, *, fast: bool = False) -> int:
    """Build full-grid alignment and background-inclusive sample artifacts."""
    grid_config = load_full_grid_alignment_config(config_path, fast=fast)
    LOGGER.info("Building full-grid alignment for years %s", list(grid_config.years))
    labels = load_labels(grid_config)
    assets = load_aef_year_assets(grid_config.tile_manifest_path, grid_config.years)
    reset_output_path(grid_config.full_grid_output_path)
    reset_output_path(grid_config.sample_output_path)
    counters = CountAccumulator({}, {}, {}, {}, {})
    population_counts = full_grid_population_counts()

    for year in grid_config.years:
        asset = assets[year]
        labels_for_year = labels.loc[labels["year"] == year].copy()
        year_population = align_full_grid_year(
            labels_for_year=labels_for_year,
            asset=asset,
            grid_config=grid_config,
            counters=counters,
        )
        population_counts[int(year)] = year_population

    sample = finalize_sample_weights(
        sample_path=grid_config.sample_output_path,
        population_counts=population_counts,
        sample_weight_column=grid_config.sample_weight_column,
    )
    masked_sample_result = write_masked_sample_artifacts(sample, grid_config)
    write_full_grid_summary(counters, sample, grid_config)
    write_full_grid_manifest(
        counters,
        sample,
        population_counts,
        assets,
        grid_config,
        masked_sample_result,
    )
    validate_fast_sample(sample, grid_config)
    LOGGER.info("Wrote full-grid table: %s", grid_config.full_grid_output_path)
    LOGGER.info("Wrote background sample table: %s", grid_config.sample_output_path)
    if masked_sample_result is not None and grid_config.sample_domain_mask is not None:
        LOGGER.info(
            "Wrote masked background sample table: %s",
            grid_config.sample_domain_mask.output_path,
        )
    LOGGER.info("Wrote full-grid manifest: %s", grid_config.full_grid_manifest_path)
    return 0


def load_full_grid_alignment_config(config_path: Path, *, fast: bool) -> FullGridAlignmentConfig:
    """Load full-grid alignment settings from the workflow config."""
    config = load_yaml_config(config_path)
    years_config = require_mapping(config.get("years"), "years")
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    features = require_mapping(config.get("features"), "features")
    feature_paths = require_mapping(features.get("paths"), "features.paths")
    alignment = require_mapping(config.get("alignment"), "alignment")
    full_grid = require_mapping(alignment.get("full_grid"), "alignment.full_grid")
    background_sample = require_mapping(
        alignment.get("background_sample"), "alignment.background_sample"
    )
    selected_years = resolve_years(years_config, full_grid, fast)
    label_path = Path(
        require_string(label_paths.get("annual_labels"), "labels.paths.annual_labels")
    )
    label_manifest_path = Path(
        require_string(
            label_paths.get("annual_label_manifest"),
            "labels.paths.annual_label_manifest",
        )
    )
    fast_config = optional_mapping(full_grid.get("fast"), "alignment.full_grid.fast")
    full_output = Path(
        require_string(full_grid.get("output_table"), "alignment.full_grid.output_table")
    )
    full_manifest = Path(
        require_string(full_grid.get("output_manifest"), "alignment.full_grid.output_manifest")
    )
    full_summary = Path(
        require_string(full_grid.get("summary_table"), "alignment.full_grid.summary_table")
    )
    sample_output = Path(
        require_string(
            background_sample.get("output_table"),
            "alignment.background_sample.output_table",
        )
    )
    sample_manifest = Path(
        require_string(
            background_sample.get("output_manifest"),
            "alignment.background_sample.output_manifest",
        )
    )
    sample_summary = Path(
        require_string(
            background_sample.get("summary_table"),
            "alignment.background_sample.summary_table",
        )
    )
    sample_domain_mask = load_sample_domain_mask_config(
        config=config,
        background_sample=background_sample,
        sample_output=sample_output,
        sample_manifest=sample_manifest,
        sample_summary=sample_summary,
        fast=fast,
    )
    return FullGridAlignmentConfig(
        config_path=config_path,
        years=selected_years,
        label_path=label_path,
        label_manifest_path=label_manifest_path,
        label_crs=label_crs_from_manifest(label_manifest_path),
        tile_manifest_path=Path(
            require_string(feature_paths.get("tile_manifest"), "features.paths.tile_manifest")
        ),
        bands=parse_bands(features.get("bands")),
        support_cells_per_side=support_cells_from_resolution(
            label_resolution_m=require_float_value(
                labels.get("native_resolution_m"),
                "labels.native_resolution_m",
            ),
            feature_resolution_m=require_float_value(
                features.get("native_resolution_m"),
                "features.native_resolution_m",
            ),
        ),
        target_row_chunk_size=optional_positive_int(
            full_grid.get("target_row_chunk_size"),
            "alignment.full_grid.target_row_chunk_size",
            DEFAULT_TARGET_ROW_CHUNK_SIZE,
        ),
        full_grid_output_path=fast_path(full_output, fast, fast_config, "output_table"),
        full_grid_manifest_path=fast_path(full_manifest, fast, fast_config, "output_manifest"),
        full_grid_summary_path=fast_path(full_summary, fast, fast_config, "summary_table"),
        sample_output_path=suffix_path(sample_output, ".fast") if fast else sample_output,
        sample_manifest_path=suffix_path(sample_manifest, ".fast") if fast else sample_manifest,
        sample_summary_path=suffix_path(sample_summary, ".fast") if fast else sample_summary,
        background_rows_per_year=optional_positive_int(
            background_sample.get("background_rows_per_year"),
            "alignment.background_sample.background_rows_per_year",
            DEFAULT_BACKGROUND_ROWS_PER_YEAR,
        ),
        include_all_kelpwatch_observed=optional_bool(
            background_sample.get("include_all_kelpwatch_observed"),
            "alignment.background_sample.include_all_kelpwatch_observed",
            True,
        ),
        random_seed=optional_int(
            background_sample.get("random_seed"),
            "alignment.background_sample.random_seed",
            DEFAULT_RANDOM_SEED,
        ),
        sample_weight_column=str(
            background_sample.get("sample_weight_column", DEFAULT_SAMPLE_WEIGHT_COLUMN)
        ),
        sample_domain_mask=sample_domain_mask,
        fast=fast,
        row_window=window_from_config(
            fast_config.get("row_window"), "alignment.full_grid.fast.row_window"
        )
        if fast
        else None,
        col_window=window_from_config(
            fast_config.get("col_window"), "alignment.full_grid.fast.col_window"
        )
        if fast
        else None,
    )


def load_sample_domain_mask_config(
    *,
    config: dict[str, Any],
    background_sample: dict[str, Any],
    sample_output: Path,
    sample_manifest: Path,
    sample_summary: Path,
    fast: bool,
) -> SampleDomainMaskConfig | None:
    """Load optional model-input domain-mask settings from the workflow config."""
    mask_value = background_sample.get("domain_mask")
    if mask_value is None:
        return None
    mask_config = require_mapping(mask_value, "alignment.background_sample.domain_mask")
    reporting_mask = load_reporting_domain_mask(config)
    policy = str(
        mask_config.get(
            "policy",
            reporting_mask.primary_domain if reporting_mask is not None else DEFAULT_PRIMARY_DOMAIN,
        )
    )
    if policy in {"none", "unmasked"}:
        return None
    if policy != DEFAULT_PRIMARY_DOMAIN:
        msg = f"unsupported training/sampling domain-mask policy: {policy}"
        raise ValueError(msg)
    if reporting_mask is None:
        msg = "alignment.background_sample.domain_mask requires reports.domain_mask mask inputs"
        raise ValueError(msg)
    fast_config = optional_mapping(
        mask_config.get("fast"), "alignment.background_sample.domain_mask.fast"
    )
    output_value = mask_config.get("output_table")
    manifest_value = mask_config.get("output_manifest")
    summary_value = mask_config.get("summary_table")
    output_path = (
        Path(
            require_string(
                output_value,
                "alignment.background_sample.domain_mask.output_table",
            )
        )
        if output_value is not None
        else suffix_path(sample_output, ".masked")
    )
    manifest_output_path = (
        Path(
            require_string(
                manifest_value,
                "alignment.background_sample.domain_mask.output_manifest",
            )
        )
        if manifest_value is not None
        else suffix_path(sample_manifest, ".masked")
    )
    summary_path = (
        Path(
            require_string(
                summary_value,
                "alignment.background_sample.domain_mask.summary_table",
            )
        )
        if summary_value is not None
        else suffix_path(sample_summary, ".masked")
    )
    return SampleDomainMaskConfig(
        table_path=reporting_mask.table_path,
        manifest_path=reporting_mask.manifest_path,
        policy=policy,
        output_path=fast_path(output_path, fast, fast_config, "output_table"),
        manifest_output_path=fast_path(
            manifest_output_path,
            fast,
            fast_config,
            "output_manifest",
        ),
        summary_path=fast_path(summary_path, fast, fast_config, "summary_table"),
        fail_on_dropped_positive=optional_bool(
            mask_config.get("fail_on_dropped_positive"),
            "alignment.background_sample.domain_mask.fail_on_dropped_positive",
            True,
        ),
    )


def resolve_years(
    years_config: dict[str, Any], full_grid: dict[str, Any], fast: bool
) -> tuple[int, ...]:
    """Resolve full-grid years from smoke config or fast config."""
    smoke_years = years_config.get("smoke")
    if not isinstance(smoke_years, list) or not smoke_years:
        msg = "config field must be a non-empty list of years: years.smoke"
        raise ValueError(msg)
    configured_years = tuple(require_int_value(year, "years.smoke[]") for year in smoke_years)
    if not fast:
        return configured_years
    fast_config = optional_mapping(full_grid.get("fast"), "alignment.full_grid.fast")
    fast_years = fast_config.get("years")
    if isinstance(fast_years, list) and fast_years:
        return tuple(
            require_int_value(year, "alignment.full_grid.fast.years[]") for year in fast_years
        )
    return (configured_years[-1],)


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional mapping config value."""
    if value is None:
        return {}
    return require_mapping(value, name)


def fast_path(path: Path, fast: bool, fast_config: dict[str, Any], key: str) -> Path:
    """Resolve a full-grid fast output path."""
    if not fast:
        return path
    if key in fast_config:
        return Path(require_string(fast_config.get(key), f"alignment.full_grid.fast.{key}"))
    return suffix_path(path, ".fast")


def suffix_path(path: Path, suffix: str) -> Path:
    """Insert a suffix before a path's final extension."""
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def window_from_config(value: object, name: str) -> tuple[int, int] | None:
    """Parse a two-element half-open window from config."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        msg = f"config field must be a two-element list: {name}"
        raise ValueError(msg)
    start = require_int_value(value[0], f"{name}[0]")
    stop = require_int_value(value[1], f"{name}[1]")
    if stop <= start:
        msg = f"window stop must be greater than start: {name}"
        raise ValueError(msg)
    return start, stop


def load_labels(grid_config: FullGridAlignmentConfig) -> pd.DataFrame:
    """Load annual labels for the selected years."""
    if not grid_config.label_path.exists():
        msg = f"annual label parquet does not exist: {grid_config.label_path}"
        raise FileNotFoundError(msg)
    labels = pd.read_parquet(grid_config.label_path)
    selected = labels.loc[labels["year"].isin(grid_config.years)].copy()
    if selected.empty:
        msg = f"annual labels contain no selected years: {list(grid_config.years)}"
        raise ValueError(msg)
    return selected


def reset_output_path(path: Path) -> None:
    """Remove and recreate an output dataset directory or parent directory."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def full_grid_population_counts() -> dict[int, dict[str, int]]:
    """Return an empty population-count mapping."""
    return {}


def align_full_grid_year(
    *,
    labels_for_year: pd.DataFrame,
    asset: AefYearAsset,
    grid_config: FullGridAlignmentConfig,
    counters: CountAccumulator,
) -> dict[str, int]:
    """Align one year of AEF target-grid rows and sample training rows."""
    LOGGER.info(
        "Building full-grid alignment for %s from %s", asset.year, asset.preferred_read_path
    )
    part_index = 0
    with rasterio.open(asset.preferred_read_path) as dataset:
        band_indexes = resolve_band_indexes(dataset, grid_config.bands)
        target_grid = aef_average_target_grid(dataset, grid_config.support_cells_per_side)
        label_lookup = target_label_lookup(labels_for_year, dataset, target_grid, grid_config)
        windows = grid_windows(target_grid, grid_config)
        population = {
            KELPWATCH_STATION: int(len(label_lookup)),
            ASSUMED_BACKGROUND: int(sum(window_cell_count(window) for window in windows))
            - int(len(label_lookup)),
        }
        counters.duplicate_cell_counts[int(asset.year)] = int(
            (label_lookup["kelpwatch_station_count"] > 1).sum()
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
            for window in windows:
                chunk = full_grid_chunk(
                    dataset=dataset,
                    vrt=vrt,
                    band_indexes=band_indexes,
                    target_grid=target_grid,
                    window=window,
                    label_lookup=label_lookup,
                    asset=asset,
                    grid_config=grid_config,
                )
                update_full_grid_counters(counters, chunk, grid_config.bands)
                write_part(chunk, grid_config.full_grid_output_path, asset.year, part_index)
                sample = sample_chunk(chunk, population, grid_config)
                if not sample.empty:
                    update_sample_counters(counters, sample)
                    write_part(sample, grid_config.sample_output_path, asset.year, part_index)
                part_index += 1
                LOGGER.info(
                    "Wrote year %s chunk %s with %s rows and %s sampled rows",
                    asset.year,
                    part_index,
                    len(chunk),
                    len(sample),
                )
    return population


def target_label_lookup(
    labels_for_year: pd.DataFrame,
    dataset: Any,
    target_grid: TargetGrid,
    grid_config: FullGridAlignmentConfig,
) -> pd.DataFrame:
    """Map Kelpwatch station labels to AEF target-grid cell ids."""
    if labels_for_year.empty:
        return empty_label_lookup()
    x_values, y_values = transformed_label_points(
        labels_for_year, grid_config.label_crs, dataset.crs
    )
    from kelp_aef.alignment.feature_label_table import target_pixel_indices

    target_rows, target_cols, target_mask = target_pixel_indices(target_grid, x_values, y_values)
    labels = labels_for_year.loc[target_mask].copy()
    labels["aef_grid_row"] = target_rows[target_mask]
    labels["aef_grid_col"] = target_cols[target_mask]
    labels["aef_grid_cell_id"] = cell_ids(
        labels["aef_grid_row"].to_numpy(dtype=np.int64),
        labels["aef_grid_col"].to_numpy(dtype=np.int64),
        target_grid.width,
    )
    if grid_config.row_window is not None:
        labels = labels.loc[
            (labels["aef_grid_row"] >= grid_config.row_window[0])
            & (labels["aef_grid_row"] < grid_config.row_window[1])
        ]
    if grid_config.col_window is not None:
        labels = labels.loc[
            (labels["aef_grid_col"] >= grid_config.col_window[0])
            & (labels["aef_grid_col"] < grid_config.col_window[1])
        ]
    if labels.empty:
        return empty_label_lookup()
    return aggregate_labels_by_cell(labels)


def empty_label_lookup() -> pd.DataFrame:
    """Return an empty label lookup with stable columns."""
    columns = [
        "aef_grid_cell_id",
        "kelpwatch_station_id",
        "kelpwatch_station_count",
        *AREA_COLUMNS,
        *LABEL_METADATA_COLUMNS,
        *BOOLEAN_LABEL_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)


def aggregate_labels_by_cell(labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one or more Kelpwatch labels assigned to the same target cell."""
    rows: list[dict[str, object]] = []
    for cell_id, group in labels.groupby("aef_grid_cell_id", sort=False):
        station_ids = group["kelpwatch_station_id"].dropna().astype(int).to_list()
        row: dict[str, object] = {
            "aef_grid_cell_id": int(cell_id),
            "kelpwatch_station_id": station_ids[0] if len(station_ids) == 1 else pd.NA,
            "kelpwatch_station_count": len(station_ids),
        }
        for column in AREA_COLUMNS:
            row[column] = float(group[column].mean()) if column in group else 0.0
        for column in LABEL_METADATA_COLUMNS:
            row[column] = first_or_na(group[column]) if column in group else pd.NA
        for column in BOOLEAN_LABEL_COLUMNS:
            row[column] = bool(group[column].fillna(False).any()) if column in group else False
        rows.append(row)
    return pd.DataFrame(rows)


def first_or_na(series: pd.Series) -> object:
    """Return the first non-null series value or pandas NA."""
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def grid_windows(target_grid: TargetGrid, grid_config: FullGridAlignmentConfig) -> list[GridWindow]:
    """Build target-grid windows for full or fast alignment."""
    row_start = 0 if grid_config.row_window is None else grid_config.row_window[0]
    row_stop = target_grid.height if grid_config.row_window is None else grid_config.row_window[1]
    col_start = 0 if grid_config.col_window is None else grid_config.col_window[0]
    col_stop = target_grid.width if grid_config.col_window is None else grid_config.col_window[1]
    row_stop = min(row_stop, target_grid.height)
    col_stop = min(col_stop, target_grid.width)
    windows: list[GridWindow] = []
    for start in range(row_start, row_stop, grid_config.target_row_chunk_size):
        stop = min(start + grid_config.target_row_chunk_size, row_stop)
        windows.append(GridWindow(start, stop, col_start, col_stop))
    return windows


def window_cell_count(window: GridWindow) -> int:
    """Return the number of target-grid cells in a window."""
    return (window.row_stop - window.row_start) * (window.col_stop - window.col_start)


def full_grid_chunk(
    *,
    dataset: Any,
    vrt: Any,
    band_indexes: tuple[int, ...],
    target_grid: TargetGrid,
    window: GridWindow,
    label_lookup: pd.DataFrame,
    asset: AefYearAsset,
    grid_config: FullGridAlignmentConfig,
) -> pd.DataFrame:
    """Read one target-grid chunk and attach label/provenance columns."""
    row_values, col_values = row_col_vectors(window)
    raster_window = Window(
        col_off=window.col_start,
        row_off=window.row_start,
        width=window.col_stop - window.col_start,
        height=window.row_stop - window.row_start,
    )
    feature_values = read_vrt_features(vrt, band_indexes, raster_window)
    expected_counts, valid_counts = source_pixel_counts_for_target_cells(
        dataset=dataset,
        band_index=band_indexes[0],
        target_rows=row_values,
        target_cols=col_values,
        target_mask=np.ones(row_values.shape, dtype=bool),
        cells_per_side=grid_config.support_cells_per_side,
    )
    longitude, latitude = lon_lat_for_cells(dataset.crs, target_grid, row_values, col_values)
    frame = pd.DataFrame(feature_values, columns=list(grid_config.bands))
    frame.insert(0, "year", int(asset.year))
    frame.insert(1, "aef_grid_row", row_values.astype(np.int32))
    frame.insert(2, "aef_grid_col", col_values.astype(np.int32))
    frame.insert(3, "aef_grid_cell_id", cell_ids(row_values, col_values, target_grid.width))
    frame["longitude"] = longitude
    frame["latitude"] = latitude
    frame["aef_expected_pixel_count"] = expected_counts.astype(np.int16)
    frame["aef_valid_pixel_count"] = valid_counts.astype(np.int16)
    frame["aef_missing_pixel_count"] = (
        frame["aef_expected_pixel_count"] - frame["aef_valid_pixel_count"]
    ).astype(np.int16)
    frame["aef_alignment_method"] = RASTERIO_AVERAGE_METHOD
    frame["aef_source_path"] = str(asset.preferred_read_path)
    frame["aef_source_href"] = asset.source_href
    return attach_labels(frame, label_lookup)


def row_col_vectors(window: GridWindow) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened target-grid row and column vectors for a window."""
    rows = np.arange(window.row_start, window.row_stop, dtype=np.int64)
    cols = np.arange(window.col_start, window.col_stop, dtype=np.int64)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    return row_grid.ravel(), col_grid.ravel()


def read_vrt_features(vrt: Any, band_indexes: tuple[int, ...], window: Window) -> np.ndarray:
    """Read averaged VRT features for a target-grid window."""
    data = vrt.read(band_indexes, window=window, masked=True).astype(np.float32)
    filled = np.ma.filled(data, np.nan)
    return cast(np.ndarray, filled.reshape((len(band_indexes), -1)).T)


def lon_lat_for_cells(
    source_crs: object, target_grid: TargetGrid, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform target-grid cell centers to longitude and latitude arrays."""
    xs, ys = rasterio.transform.xy(target_grid.transform, rows, cols, offset="center")
    longitudes, latitudes = warp_transform(source_crs, DEFAULT_LABEL_CRS, xs, ys)
    return np.asarray(longitudes, dtype=np.float64), np.asarray(latitudes, dtype=np.float64)


def cell_ids(rows: np.ndarray, cols: np.ndarray, width: int) -> np.ndarray:
    """Build stable integer target-grid cell ids from row and column arrays."""
    return rows.astype(np.int64) * np.int64(width) + cols.astype(np.int64)


def attach_labels(frame: pd.DataFrame, label_lookup: pd.DataFrame) -> pd.DataFrame:
    """Attach Kelpwatch labels and assumed-background defaults to a full-grid chunk."""
    merged = frame.merge(label_lookup, on="aef_grid_cell_id", how="left", sort=False)
    observed_mask = merged["kelpwatch_station_count"].notna()
    merged["label_source"] = np.where(observed_mask, KELPWATCH_STATION, ASSUMED_BACKGROUND)
    merged["is_kelpwatch_observed"] = observed_mask.to_numpy(dtype=bool)
    merged["kelpwatch_station_count"] = merged["kelpwatch_station_count"].fillna(0).astype(np.int16)
    for column in AREA_COLUMNS:
        merged[column] = merged[column].fillna(0.0).astype(np.float32)
    for column in BOOLEAN_LABEL_COLUMNS:
        merged[column] = merged[column].fillna(False).astype(bool)
    for column in LABEL_METADATA_COLUMNS:
        if column not in merged:
            merged[column] = pd.NA
    return merged


def sample_chunk(
    chunk: pd.DataFrame,
    population: dict[str, int],
    grid_config: FullGridAlignmentConfig,
) -> pd.DataFrame:
    """Select deterministic background and observed rows for baseline training."""
    observed_mask = chunk["label_source"] == KELPWATCH_STATION
    background_mask = chunk["label_source"] == ASSUMED_BACKGROUND
    sample_mask = np.zeros(len(chunk), dtype=bool)
    if grid_config.include_all_kelpwatch_observed:
        sample_mask |= observed_mask.to_numpy(dtype=bool)
    background_count = max(population.get(ASSUMED_BACKGROUND, 0), 1)
    probability = min(1.0, grid_config.background_rows_per_year / background_count)
    if probability >= 1.0:
        sample_mask |= background_mask.to_numpy(dtype=bool)
    else:
        sample_mask |= background_mask.to_numpy(dtype=bool) & deterministic_sample_mask(
            chunk["aef_grid_cell_id"].to_numpy(dtype=np.int64),
            int(chunk["year"].iloc[0]),
            grid_config.random_seed,
            probability,
        )
    sample = chunk.loc[sample_mask].copy()
    sample[grid_config.sample_weight_column] = 1.0
    return sample


def deterministic_sample_mask(
    cell_id_values: np.ndarray, year: int, seed: int, probability: float
) -> np.ndarray:
    """Return a stable pseudo-random sample mask from cell ids and year."""
    modulus = np.uint64(2**32)
    hashed = (
        cell_id_values.astype(np.uint64) * np.uint64(1_103_515_245)
        + np.uint64(seed)
        + np.uint64(year) * np.uint64(2_654_435_761)
    ) % modulus
    return cast(np.ndarray, hashed.astype(np.float64) / float(modulus) < probability)


def update_full_grid_counters(
    counters: CountAccumulator, chunk: pd.DataFrame, bands: tuple[str, ...]
) -> None:
    """Update full-grid summary counters from one chunk."""
    feature_complete = ~chunk[list(bands)].isna().any(axis=1)
    for keys, group in chunk.groupby(["year", "label_source"], sort=False):
        year, label_source = cast(tuple[int, str], keys)
        key = (int(year), str(label_source))
        counters.full_counts[key] = counters.full_counts.get(key, 0) + len(group)
        counters.full_complete_counts[key] = counters.full_complete_counts.get(key, 0) + int(
            feature_complete.loc[group.index].sum()
        )
        counters.full_observed_area[key] = counters.full_observed_area.get(key, 0.0) + float(
            group["kelp_max_y"].sum()
        )


def update_sample_counters(counters: CountAccumulator, sample: pd.DataFrame) -> None:
    """Update sample summary counters from one sampled chunk."""
    for keys, group in sample.groupby(["year", "label_source"], sort=False):
        year, label_source = cast(tuple[int, str], keys)
        key = (int(year), str(label_source))
        counters.sample_counts[key] = counters.sample_counts.get(key, 0) + len(group)


def write_part(dataframe: pd.DataFrame, output_path: Path, year: int, part_index: int) -> None:
    """Write one Parquet part to a dataset directory."""
    output_path.mkdir(parents=True, exist_ok=True)
    part_path = output_path / f"part-year={year}-chunk={part_index:05d}.parquet"
    dataframe.to_parquet(part_path, index=False)


def finalize_sample_weights(
    *,
    sample_path: Path,
    population_counts: dict[int, dict[str, int]],
    sample_weight_column: str,
) -> pd.DataFrame:
    """Rewrite the sampled table with exact expansion weights."""
    sample = pd.read_parquet(sample_path)
    if sample.empty:
        msg = f"background sample is empty: {sample_path}"
        raise ValueError(msg)
    sample[sample_weight_column] = 1.0
    for keys, group in sample.groupby(["year", "label_source"], sort=False):
        year, label_source = cast(tuple[int, str], keys)
        if label_source != ASSUMED_BACKGROUND:
            continue
        population = population_counts[int(year)][ASSUMED_BACKGROUND]
        weight = population / max(len(group), 1)
        sample.loc[group.index, sample_weight_column] = float(weight)
    reset_output_path(sample_path)
    write_part(sample, sample_path, year=0, part_index=0)
    return sample


def write_masked_sample_artifacts(
    sample: pd.DataFrame, grid_config: FullGridAlignmentConfig
) -> MaskedSampleResult | None:
    """Write a retained-domain sidecar sample and its audit artifacts."""
    mask_config = grid_config.sample_domain_mask
    if mask_config is None:
        return None
    LOGGER.info("Applying %s mask to background sample", mask_config.policy)
    joined = join_sample_to_domain_mask(sample, mask_config)
    retained_mask = joined[MASK_RETAIN_COLUMN].astype(bool)
    dropped = joined.loc[~retained_mask]
    dropped_positive = dropped.loc[
        (dropped["label_source"] == KELPWATCH_STATION) & (dropped["kelp_max_y"] > 0)
    ]
    if mask_config.fail_on_dropped_positive and not dropped_positive.empty:
        msg = f"domain mask dropped Kelpwatch-positive training rows: {len(dropped_positive)} rows"
        raise ValueError(msg)
    population_counts = masked_full_grid_population_counts(
        grid_config.full_grid_output_path,
        mask_config,
    )
    masked = joined.loc[retained_mask].copy()
    if masked.empty:
        msg = f"domain mask retained no sample rows: {mask_config.table_path}"
        raise ValueError(msg)
    masked = recompute_sample_weights(
        masked,
        population_counts,
        grid_config.sample_weight_column,
    )
    summary_frame = joined.copy()
    summary_frame.loc[masked.index, grid_config.sample_weight_column] = masked[
        grid_config.sample_weight_column
    ]
    summary_rows = masked_sample_summary_rows(
        summary_frame,
        grid_config.sample_weight_column,
    )
    reset_output_path(mask_config.output_path)
    write_part(masked, mask_config.output_path, year=0, part_index=0)
    write_csv(summary_rows, mask_config.summary_path, MASKED_SAMPLE_SUMMARY_FIELDS)
    result = MaskedSampleResult(
        sample=masked,
        population_counts=population_counts,
        retained_counts=count_rows_by_year_label(masked),
        dropped_counts=count_rows_by_year_label(dropped),
        dropped_observed_row_count=observed_row_count(dropped),
        dropped_positive_row_count=len(dropped_positive),
    )
    write_masked_sample_manifest(
        result=result,
        source_sample=sample,
        summary_rows=summary_rows,
        grid_config=grid_config,
        mask_config=mask_config,
    )
    LOGGER.info(
        "Retained %s of %s sample rows after domain mask",
        len(masked),
        len(joined),
    )
    return result


def join_sample_to_domain_mask(
    sample: pd.DataFrame, mask_config: SampleDomainMaskConfig
) -> pd.DataFrame:
    """Join sample rows to static plausible-kelp mask metadata."""
    if MASK_KEY_COLUMN not in sample.columns:
        msg = f"domain mask requires sample column: {MASK_KEY_COLUMN}"
        raise ValueError(msg)
    clean = sample.drop(columns=mask_columns_in_frame(sample), errors="ignore")
    joined = clean.join(read_sample_mask_lookup(mask_config), on=MASK_KEY_COLUMN, how="left")
    missing = joined[MASK_RETAIN_COLUMN].isna()
    if bool(missing.any()):
        msg = f"domain mask is missing rows for {int(missing.sum())} sample cells"
        raise ValueError(msg)
    return cast(pd.DataFrame, joined)


def read_sample_mask_lookup(mask_config: SampleDomainMaskConfig) -> pd.DataFrame:
    """Read mask metadata indexed by target-grid cell id for sample joins."""
    available_names = set(pl.scan_parquet(str(mask_config.table_path)).collect_schema().names())
    selected_columns = [column for column in MASK_DETAIL_COLUMNS if column in available_names]
    mask = pd.read_parquet(mask_config.table_path, columns=selected_columns)
    missing = [column for column in (MASK_KEY_COLUMN, MASK_RETAIN_COLUMN) if column not in mask]
    if missing:
        msg = f"domain mask table is missing required columns: {missing}"
        raise ValueError(msg)
    columns = [column for column in MASK_DETAIL_COLUMNS if column in mask.columns]
    indexed = mask[columns].set_index(MASK_KEY_COLUMN)
    if not indexed.index.is_unique:
        msg = f"domain mask table has duplicate cell ids: {mask_config.table_path}"
        raise ValueError(msg)
    return cast(pd.DataFrame, indexed)


def masked_full_grid_population_counts(
    full_grid_path: Path,
    mask_config: SampleDomainMaskConfig,
) -> dict[int, dict[str, int]]:
    """Count retained full-grid cells by year and label source for weighting."""
    full_grid = pl.scan_parquet(str(full_grid_path)).select(
        ["year", "label_source", MASK_KEY_COLUMN]
    )
    mask = pl.scan_parquet(str(mask_config.table_path)).select(
        [MASK_KEY_COLUMN, MASK_RETAIN_COLUMN]
    )
    missing = (
        full_grid.join(mask.select(MASK_KEY_COLUMN), on=MASK_KEY_COLUMN, how="anti")
        .select(pl.len().alias("row_count"))
        .collect()
    )
    missing_count = int(missing["row_count"][0])
    if missing_count:
        msg = f"domain mask is missing rows for {missing_count} full-grid cells"
        raise ValueError(msg)
    counts_frame = (
        full_grid.join(
            mask.filter(pl.col(MASK_RETAIN_COLUMN)).select(MASK_KEY_COLUMN),
            on=MASK_KEY_COLUMN,
            how="inner",
        )
        .group_by(["year", "label_source"])
        .agg(pl.len().alias("row_count"))
        .collect()
    )
    counts: dict[int, dict[str, int]] = {}
    for row in counts_frame.to_dicts():
        year = int(cast(Any, row["year"]))
        label_source = str(row["label_source"])
        counts.setdefault(year, {})[label_source] = int(cast(Any, row["row_count"]))
    return counts


def recompute_sample_weights(
    sample: pd.DataFrame,
    population_counts: dict[int, dict[str, int]],
    sample_weight_column: str,
) -> pd.DataFrame:
    """Recompute sample expansion weights against the retained mask domain."""
    weighted = sample.copy()
    weighted[sample_weight_column] = 1.0
    for keys, group in weighted.groupby(["year", "label_source"], sort=False):
        year, label_source = cast(tuple[int, str], keys)
        if label_source != ASSUMED_BACKGROUND:
            continue
        population = population_counts.get(int(year), {}).get(ASSUMED_BACKGROUND, 0)
        if population <= 0:
            msg = f"no retained assumed-background population for year {int(year)}"
            raise ValueError(msg)
        weighted.loc[group.index, sample_weight_column] = float(population / max(len(group), 1))
    return weighted


def masked_sample_summary_rows(
    dataframe: pd.DataFrame,
    sample_weight_column: str,
) -> list[dict[str, object]]:
    """Summarize retained and dropped sample rows by year, source, and mask reason."""
    rows: list[dict[str, object]] = []
    group_columns = ["year", "label_source", MASK_RETAIN_COLUMN, "domain_mask_reason"]
    for keys, group in dataframe.groupby(group_columns, dropna=False, sort=True):
        year, label_source, retained, reason = cast(tuple[int, str, bool, object], keys)
        weights = group[sample_weight_column].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "label_source": str(label_source),
                "is_plausible_kelp_domain": bool(retained),
                "domain_mask_reason": "" if pd.isna(reason) else str(reason),
                "row_count": int(len(group)),
                "kelpwatch_observed_row_count": observed_row_count(group),
                "kelpwatch_positive_row_count": positive_observed_row_count(group),
                "sample_weight_min": float(np.nanmin(weights)),
                "sample_weight_max": float(np.nanmax(weights)),
            }
        )
    return rows


def count_rows_by_year_label(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count rows by stable year and label-source manifest keys."""
    counts: dict[str, int] = {}
    for keys, group in dataframe.groupby(["year", "label_source"], sort=True):
        year, label_source = cast(tuple[int, str], keys)
        counts[f"{int(year)}:{label_source}"] = int(len(group))
    return counts


def observed_row_count(dataframe: pd.DataFrame) -> int:
    """Count Kelpwatch-observed rows in a sample frame."""
    if "is_kelpwatch_observed" in dataframe.columns:
        return int(dataframe["is_kelpwatch_observed"].fillna(False).sum())
    return int((dataframe["label_source"] == KELPWATCH_STATION).sum())


def positive_observed_row_count(dataframe: pd.DataFrame) -> int:
    """Count Kelpwatch-observed rows with positive annual max canopy."""
    observed = (
        dataframe["is_kelpwatch_observed"].fillna(False).astype(bool)
        if "is_kelpwatch_observed" in dataframe.columns
        else dataframe["label_source"] == KELPWATCH_STATION
    )
    return int((observed & (dataframe["kelp_max_y"] > 0)).sum())


def write_full_grid_summary(
    counters: CountAccumulator, sample: pd.DataFrame, grid_config: FullGridAlignmentConfig
) -> None:
    """Write full-grid and sample summary CSV files."""
    full_rows = []
    for key in sorted(counters.full_counts):
        row_count = counters.full_counts[key]
        full_rows.append(
            {
                "year": key[0],
                "label_source": key[1],
                "row_count": row_count,
                "sampled_row_count": counters.sample_counts.get(key, 0),
                "complete_feature_row_count": counters.full_complete_counts.get(key, 0),
                "missing_feature_row_count": row_count - counters.full_complete_counts.get(key, 0),
                "observed_canopy_area": counters.full_observed_area.get(key, 0.0),
            }
        )
    write_csv(full_rows, grid_config.full_grid_summary_path, FULL_GRID_SUMMARY_FIELDS)
    sample_rows = []
    for key, group in sample.groupby(["year", "label_source"], sort=True):
        year, label_source = cast(tuple[int, str], key)
        weights = group[grid_config.sample_weight_column].to_numpy(dtype=float)
        sample_rows.append(
            {
                "year": int(year),
                "label_source": str(label_source),
                "row_count": int(len(group)),
                "sample_weight_min": float(np.nanmin(weights)),
                "sample_weight_max": float(np.nanmax(weights)),
            }
        )
    write_csv(sample_rows, grid_config.sample_summary_path, SAMPLE_SUMMARY_FIELDS)


def write_csv(
    rows: list[dict[str, object]], output_path: Path, fieldnames: tuple[str, ...]
) -> None:
    """Write dictionary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_full_grid_manifest(
    counters: CountAccumulator,
    sample: pd.DataFrame,
    population_counts: dict[int, dict[str, int]],
    assets: dict[int, AefYearAsset],
    grid_config: FullGridAlignmentConfig,
    masked_sample_result: MaskedSampleResult | None,
) -> None:
    """Write manifests for full-grid and sampled training artifacts."""
    full_payload = {
        "command": "align-full-grid",
        "config_path": str(grid_config.config_path),
        "fast": grid_config.fast,
        "years": list(grid_config.years),
        "full_grid_output": str(grid_config.full_grid_output_path),
        "sample_output": str(grid_config.sample_output_path),
        "full_grid_summary": str(grid_config.full_grid_summary_path),
        "sample_summary": str(grid_config.sample_summary_path),
        "bands": list(grid_config.bands),
        "alignment_method": RASTERIO_AVERAGE_METHOD,
        "target_row_chunk_size": grid_config.target_row_chunk_size,
        "label_source_counts": {
            f"{year}:{label_source}": count
            for (year, label_source), count in sorted(counters.full_counts.items())
        },
        "sample_counts": {
            f"{year}:{label_source}": count
            for (year, label_source), count in sorted(counters.sample_counts.items())
        },
        "population_counts": population_counts,
        "duplicate_cell_counts": counters.duplicate_cell_counts,
        "assets": {
            str(year): {
                "preferred_read_path": str(asset.preferred_read_path),
                "source_href": asset.source_href,
                "raster": asset.raster_metadata,
            }
            for year, asset in sorted(assets.items())
        },
    }
    if masked_sample_result is not None and grid_config.sample_domain_mask is not None:
        full_payload["masked_sample_output"] = str(grid_config.sample_domain_mask.output_path)
        full_payload["masked_sample_counts"] = masked_sample_result.retained_counts
        full_payload["masked_population_counts"] = masked_sample_result.population_counts
    sample_payload = {
        "command": "align-full-grid",
        "artifact": "background_sample",
        "sample_output": str(grid_config.sample_output_path),
        "sample_row_count": int(len(sample)),
        "sample_weight_column": grid_config.sample_weight_column,
        "background_rows_per_year": grid_config.background_rows_per_year,
        "include_all_kelpwatch_observed": grid_config.include_all_kelpwatch_observed,
        "random_seed": grid_config.random_seed,
        "schema": list(sample.columns),
    }
    write_json(grid_config.full_grid_manifest_path, full_payload)
    write_json(grid_config.sample_manifest_path, sample_payload)


def write_masked_sample_manifest(
    *,
    result: MaskedSampleResult,
    source_sample: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    grid_config: FullGridAlignmentConfig,
    mask_config: SampleDomainMaskConfig,
) -> None:
    """Write a JSON manifest for the masked model-input sample sidecar."""
    payload = {
        "command": "align-full-grid",
        "artifact": "background_sample_domain_mask",
        "config_path": str(grid_config.config_path),
        "fast": grid_config.fast,
        "domain_policy": mask_config.policy,
        "mask_table": str(mask_config.table_path),
        "mask_manifest": str(mask_config.manifest_path) if mask_config.manifest_path else None,
        "source_sample_output": str(grid_config.sample_output_path),
        "masked_sample_output": str(mask_config.output_path),
        "masked_sample_summary": str(mask_config.summary_path),
        "source_sample_row_count": int(len(source_sample)),
        "masked_sample_row_count": int(len(result.sample)),
        "dropped_sample_row_count": int(len(source_sample) - len(result.sample)),
        "retained_counts": result.retained_counts,
        "dropped_counts": result.dropped_counts,
        "dropped_observed_row_count": result.dropped_observed_row_count,
        "dropped_positive_row_count": result.dropped_positive_row_count,
        "population_counts": result.population_counts,
        "sample_weight_column": grid_config.sample_weight_column,
        "background_rows_per_year": grid_config.background_rows_per_year,
        "include_all_kelpwatch_observed": grid_config.include_all_kelpwatch_observed,
        "random_seed": grid_config.random_seed,
        "fail_on_dropped_positive": mask_config.fail_on_dropped_positive,
        "summary_row_count": len(summary_rows),
        "schema": list(result.sample.columns),
    }
    write_json(mask_config.manifest_output_path, payload)


def validate_fast_sample(sample: pd.DataFrame, grid_config: FullGridAlignmentConfig) -> None:
    """Validate that a fast sample exercises observed positives, zeros, and background."""
    if not grid_config.fast:
        return
    observed = sample.loc[sample["label_source"] == KELPWATCH_STATION]
    background = sample.loc[sample["label_source"] == ASSUMED_BACKGROUND]
    has_positive = bool((observed["kelp_max_y"] > 0).any()) if not observed.empty else False
    has_zero = bool((observed["kelp_max_y"] == 0).any()) if not observed.empty else False
    if not has_positive or not has_zero or background.empty:
        msg = (
            "fast full-grid sample must include observed positive, observed zero, "
            "and assumed-background rows"
        )
        raise ValueError(msg)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk."""
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
    """Validate a positive numeric dynamic value."""
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


def optional_int(value: object, name: str, default: int) -> int:
    """Validate an optional integer dynamic value."""
    if value is None:
        return default
    return require_int_value(value, name)


def optional_bool(value: object, name: str, default: bool) -> bool:
    """Validate an optional boolean dynamic value."""
    if value is None:
        return default
    if not isinstance(value, bool):
        msg = f"field must be a boolean: {name}"
        raise ValueError(msg)
    return value
