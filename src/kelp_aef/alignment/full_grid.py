"""Build full-grid AEF/Kelpwatch alignment and sampled training artifacts."""

from __future__ import annotations

import csv
import json
import logging
import math
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
POST_HOC_MASKED_SAMPLE_POLICY = "post_hoc_masked_background_sample"
CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY = "crm_stratified_mask_first_sample"
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
CRM_STRATIFIED_SAMPLE_SUMMARY_FIELDS = (
    "year",
    "label_source",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "depth_bin",
    "population_row_count",
    "sampled_row_count",
    "dropped_row_count",
    "configured_fraction",
    "configured_min_rows_per_year",
    "configured_max_rows_per_year",
    "effective_sample_fraction",
    "kelpwatch_observed_row_count",
    "kelpwatch_positive_row_count",
    "sample_weight_min",
    "sample_weight_max",
)
CRM_STRATIFIED_REQUIRED_MASK_COLUMNS = (
    MASK_KEY_COLUMN,
    MASK_RETAIN_COLUMN,
    "domain_mask_reason",
    "depth_bin",
)


@dataclass(frozen=True)
class SampleDomainMaskConfig:
    """Resolved plausible-kelp mask settings for model-input sampling."""

    table_path: Path
    manifest_path: Path | None
    policy: str
    sampling_policy: str
    output_path: Path
    manifest_output_path: Path
    summary_path: Path
    quotas: tuple[CrmStratifiedQuota, ...]
    default_fraction: float
    default_min_rows_per_year: int
    random_seed: int
    fail_on_dropped_positive: bool


@dataclass(frozen=True)
class CrmStratifiedQuota:
    """One configured assumed-background sampling quota rule."""

    domain_mask_reason: str | None
    depth_bin: str | None
    fraction: float
    min_rows_per_year: int
    max_rows_per_year: int | None


@dataclass(frozen=True)
class CrmStratifiedSampleConfig:
    """Resolved settings for CRM-stratified background sidecar sampling."""

    table_path: Path
    manifest_path: Path | None
    policy: str
    output_path: Path
    manifest_output_path: Path
    summary_path: Path
    quotas: tuple[CrmStratifiedQuota, ...]
    default_fraction: float
    default_min_rows_per_year: int
    random_seed: int


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
    crm_stratified_sample: CrmStratifiedSampleConfig | None
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
class CrmStratifiedSelection:
    """Selected CRM-stratified sample rows and audit summaries."""

    selected_metadata: pd.DataFrame
    summary_rows: list[dict[str, object]]
    population_row_count: int
    sampled_row_count: int
    retained_counts: dict[str, int]
    population_counts: dict[str, int]
    dropped_counts: dict[str, int]


@dataclass(frozen=True)
class CrmStratifiedSampleResult:
    """Metadata returned after writing the CRM-stratified sample sidecar."""

    sample: pd.DataFrame
    selection: CrmStratifiedSelection


@dataclass(frozen=True)
class MaskDroppedFullGridCounts:
    """Dropped full-grid counts from applying the retained-domain mask first."""

    row_count: int
    observed_row_count: int
    positive_observed_row_count: int
    counts_by_year_label: dict[str, int]
    counts_by_stratum: dict[str, int]


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
    crm_stratified_sample_result = write_crm_stratified_sample_artifacts(grid_config)
    write_full_grid_summary(counters, sample, grid_config)
    write_full_grid_manifest(
        counters,
        sample,
        population_counts,
        assets,
        grid_config,
        masked_sample_result,
        crm_stratified_sample_result,
    )
    validate_fast_sample(sample, grid_config)
    LOGGER.info("Wrote full-grid table: %s", grid_config.full_grid_output_path)
    LOGGER.info("Wrote background sample table: %s", grid_config.sample_output_path)
    if masked_sample_result is not None and grid_config.sample_domain_mask is not None:
        LOGGER.info(
            "Wrote masked background sample table: %s",
            grid_config.sample_domain_mask.output_path,
        )
    if crm_stratified_sample_result is not None and grid_config.crm_stratified_sample is not None:
        LOGGER.info(
            "Wrote CRM-stratified background sample table: %s",
            grid_config.crm_stratified_sample.output_path,
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
    crm_stratified_sample = load_crm_stratified_sample_config(
        config=config,
        background_sample=background_sample,
        sample_output=sample_output,
        sample_manifest=sample_manifest,
        sample_summary=sample_summary,
        random_seed=optional_int(
            background_sample.get("random_seed"),
            "alignment.background_sample.random_seed",
            DEFAULT_RANDOM_SEED,
        ),
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
            background_sample.get("legacy_background_rows_per_year")
            or background_sample.get("background_rows_per_year"),
            (
                "alignment.background_sample.legacy_background_rows_per_year "
                "or alignment.background_sample.background_rows_per_year"
            ),
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
        crm_stratified_sample=crm_stratified_sample,
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
    sampling_policy = str(mask_config.get("sampling_policy", POST_HOC_MASKED_SAMPLE_POLICY))
    if sampling_policy not in {
        POST_HOC_MASKED_SAMPLE_POLICY,
        CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY,
    }:
        msg = f"unsupported masked sample policy: {sampling_policy}"
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
        sampling_policy=sampling_policy,
        output_path=fast_path(output_path, fast, fast_config, "output_table"),
        manifest_output_path=fast_path(
            manifest_output_path,
            fast,
            fast_config,
            "output_manifest",
        ),
        summary_path=fast_path(summary_path, fast, fast_config, "summary_table"),
        quotas=read_crm_stratified_quotas(
            mask_config,
            "alignment.background_sample.domain_mask.strata",
        )
        if sampling_policy == CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY
        else (),
        default_fraction=optional_probability(
            mask_config.get("default_fraction"),
            "alignment.background_sample.domain_mask.default_fraction",
            0.0,
        ),
        default_min_rows_per_year=optional_nonnegative_int(
            mask_config.get("default_min_rows_per_year"),
            "alignment.background_sample.domain_mask.default_min_rows_per_year",
            0,
        ),
        random_seed=optional_int(
            mask_config.get("random_seed"),
            "alignment.background_sample.domain_mask.random_seed",
            DEFAULT_RANDOM_SEED,
        ),
        fail_on_dropped_positive=optional_bool(
            mask_config.get("fail_on_dropped_positive"),
            "alignment.background_sample.domain_mask.fail_on_dropped_positive",
            True,
        ),
    )


def load_crm_stratified_sample_config(
    *,
    config: dict[str, Any],
    background_sample: dict[str, Any],
    sample_output: Path,
    sample_manifest: Path,
    sample_summary: Path,
    random_seed: int,
    fast: bool,
) -> CrmStratifiedSampleConfig | None:
    """Load optional CRM-stratified model-input sidecar settings."""
    sidecar_value = background_sample.get("crm_stratified")
    if sidecar_value is None:
        return None
    sidecar = require_mapping(sidecar_value, "alignment.background_sample.crm_stratified")
    if not optional_bool(
        sidecar.get("enabled"), "alignment.background_sample.crm_stratified.enabled", True
    ):
        return None
    reporting_mask = load_reporting_domain_mask(config)
    policy = str(
        sidecar.get(
            "policy",
            reporting_mask.primary_domain if reporting_mask is not None else DEFAULT_PRIMARY_DOMAIN,
        )
    )
    if policy != DEFAULT_PRIMARY_DOMAIN:
        msg = f"unsupported CRM-stratified sampling policy: {policy}"
        raise ValueError(msg)
    if reporting_mask is None:
        msg = "alignment.background_sample.crm_stratified requires reports.domain_mask mask inputs"
        raise ValueError(msg)
    fast_config = optional_mapping(
        sidecar.get("fast"), "alignment.background_sample.crm_stratified.fast"
    )
    output_path = crm_stratified_path(
        sidecar,
        "output_table",
        suffix_path(sample_output, ".crm_stratified.masked"),
    )
    manifest_path = crm_stratified_path(
        sidecar,
        "output_manifest",
        suffix_path(sample_manifest, ".crm_stratified.masked"),
    )
    summary_path = crm_stratified_path(
        sidecar,
        "summary_table",
        suffix_path(sample_summary, ".crm_stratified.masked"),
    )
    return CrmStratifiedSampleConfig(
        table_path=reporting_mask.table_path,
        manifest_path=reporting_mask.manifest_path,
        policy=policy,
        output_path=fast_path(output_path, fast, fast_config, "output_table"),
        manifest_output_path=fast_path(manifest_path, fast, fast_config, "output_manifest"),
        summary_path=fast_path(summary_path, fast, fast_config, "summary_table"),
        quotas=read_crm_stratified_quotas(
            sidecar,
            "alignment.background_sample.crm_stratified.strata",
        ),
        default_fraction=optional_probability(
            sidecar.get("default_fraction"),
            "alignment.background_sample.crm_stratified.default_fraction",
            0.02,
        ),
        default_min_rows_per_year=optional_nonnegative_int(
            sidecar.get("default_min_rows_per_year"),
            "alignment.background_sample.crm_stratified.default_min_rows_per_year",
            0,
        ),
        random_seed=optional_int(
            sidecar.get("random_seed"),
            "alignment.background_sample.crm_stratified.random_seed",
            random_seed,
        ),
    )


def crm_stratified_path(config: dict[str, Any], key: str, default: Path) -> Path:
    """Read a CRM-stratified sidecar path from config with a default."""
    value = config.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"alignment.background_sample.crm_stratified.{key}"))


def read_crm_stratified_quotas(
    config: dict[str, Any],
    config_name: str,
) -> tuple[CrmStratifiedQuota, ...]:
    """Read ordered CRM stratum quota rules from config."""
    values = config.get("strata")
    if not isinstance(values, list) or not values:
        msg = f"{config_name} must be a non-empty list"
        raise ValueError(msg)
    quotas = []
    for index, value in enumerate(values):
        item_name = f"{config_name}[{index}]"
        item = require_mapping(value, item_name)
        quotas.append(
            CrmStratifiedQuota(
                domain_mask_reason=optional_string(item.get("domain_mask_reason")),
                depth_bin=optional_string(item.get("depth_bin")),
                fraction=optional_probability(item.get("fraction"), f"{item_name}.fraction", 0.0),
                min_rows_per_year=optional_nonnegative_int(
                    item.get("min_rows_per_year"),
                    f"{item_name}.min_rows_per_year",
                    0,
                ),
                max_rows_per_year=optional_positive_int_or_none(
                    item.get("max_rows_per_year"),
                    f"{item_name}.max_rows_per_year",
                ),
            )
        )
    return tuple(quotas)


def optional_string(value: object) -> str | None:
    """Read an optional string value from dynamic config."""
    if value is None:
        return None
    return str(value)


def optional_probability(value: object, name: str, default: float) -> float:
    """Read a probability value from dynamic config."""
    if value is None:
        return default
    if isinstance(value, bool):
        msg = f"field must be numeric, not boolean: {name}"
        raise ValueError(msg)
    parsed = float(cast(Any, value))
    if not 0.0 <= parsed <= 1.0:
        msg = f"field must be between 0 and 1: {name}"
        raise ValueError(msg)
    return parsed


def optional_nonnegative_int(value: object, name: str, default: int) -> int:
    """Validate an optional non-negative integer dynamic value."""
    if value is None:
        return default
    parsed = require_int_value(value, name)
    if parsed < 0:
        msg = f"field must be non-negative: {name}"
        raise ValueError(msg)
    return parsed


def optional_positive_int_or_none(value: object, name: str) -> int | None:
    """Validate an optional positive integer dynamic value."""
    if value is None:
        return None
    parsed = require_int_value(value, name)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


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
    if mask_config.sampling_policy == CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY:
        return write_mask_first_crm_stratified_sample_artifacts(grid_config, mask_config)
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


def write_mask_first_crm_stratified_sample_artifacts(
    grid_config: FullGridAlignmentConfig,
    mask_config: SampleDomainMaskConfig,
) -> MaskedSampleResult:
    """Write the default mask-first CRM-stratified sample and audit manifest."""
    LOGGER.info(
        "Building default %s sample from retained full-grid rows",
        CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY,
    )
    sample_config = crm_stratified_config_from_mask_config(mask_config)
    result = build_crm_stratified_sample_result(grid_config, sample_config)
    if result.selection.selected_metadata.empty:
        msg = "mask-first CRM-stratified sampling selected no rows"
        raise ValueError(msg)
    if result.sample.empty:
        msg = "mask-first CRM-stratified selected keys produced no sample rows"
        raise ValueError(msg)
    mask_drops = mask_dropped_full_grid_counts(
        full_grid_path=grid_config.full_grid_output_path,
        sample_config=sample_config,
    )
    if mask_config.fail_on_dropped_positive and mask_drops.positive_observed_row_count:
        msg = (
            "domain mask dropped Kelpwatch-positive full-grid rows: "
            f"{mask_drops.positive_observed_row_count} rows"
        )
        raise ValueError(msg)
    reset_output_path(mask_config.output_path)
    write_part(result.sample, mask_config.output_path, year=0, part_index=0)
    write_csv(
        result.selection.summary_rows,
        mask_config.summary_path,
        CRM_STRATIFIED_SAMPLE_SUMMARY_FIELDS,
    )
    write_mask_first_crm_stratified_sample_manifest(
        result=result,
        grid_config=grid_config,
        sample_config=sample_config,
        mask_drops=mask_drops,
    )
    LOGGER.info(
        "Selected %s default mask-first rows from %s retained population rows",
        result.selection.sampled_row_count,
        result.selection.population_row_count,
    )
    return MaskedSampleResult(
        sample=result.sample,
        population_counts=year_label_counts_from_crm_counts(result.selection.population_counts),
        retained_counts=year_label_flat_counts_from_crm_counts(result.selection.retained_counts),
        dropped_counts=mask_drops.counts_by_year_label,
        dropped_observed_row_count=mask_drops.observed_row_count,
        dropped_positive_row_count=mask_drops.positive_observed_row_count,
    )


def crm_stratified_config_from_mask_config(
    mask_config: SampleDomainMaskConfig,
) -> CrmStratifiedSampleConfig:
    """Adapt default masked-sample settings to the CRM-stratified sampler."""
    return CrmStratifiedSampleConfig(
        table_path=mask_config.table_path,
        manifest_path=mask_config.manifest_path,
        policy=mask_config.policy,
        output_path=mask_config.output_path,
        manifest_output_path=mask_config.manifest_output_path,
        summary_path=mask_config.summary_path,
        quotas=mask_config.quotas,
        default_fraction=mask_config.default_fraction,
        default_min_rows_per_year=mask_config.default_min_rows_per_year,
        random_seed=mask_config.random_seed,
    )


def mask_dropped_full_grid_counts(
    *,
    full_grid_path: Path,
    sample_config: CrmStratifiedSampleConfig,
) -> MaskDroppedFullGridCounts:
    """Count full-grid rows dropped by the retained-domain mask."""
    validate_crm_stratified_mask_schema(sample_config)
    full_grid = pl.scan_parquet(str(full_grid_path)).select(
        [
            "year",
            "label_source",
            MASK_KEY_COLUMN,
            "kelp_max_y",
            "is_kelpwatch_observed",
        ]
    )
    mask = pl.scan_parquet(str(sample_config.table_path)).select(
        list(CRM_STRATIFIED_REQUIRED_MASK_COLUMNS)
    )
    missing_count = crm_stratified_missing_mask_count(full_grid, mask)
    if missing_count:
        msg = f"domain mask is missing rows for {missing_count} full-grid cells"
        raise ValueError(msg)
    dropped = full_grid.join(mask, on=MASK_KEY_COLUMN, how="inner").filter(
        ~pl.col(MASK_RETAIN_COLUMN)
    )
    row_count = int(dropped.select(pl.len().alias("row_count")).collect()["row_count"][0])
    observed_row_count = int(
        dropped.filter(pl.col("is_kelpwatch_observed"))
        .select(pl.len().alias("row_count"))
        .collect()["row_count"][0]
    )
    positive_observed_row_count = int(
        dropped.filter(pl.col("is_kelpwatch_observed") & (pl.col("kelp_max_y") > 0))
        .select(pl.len().alias("row_count"))
        .collect()["row_count"][0]
    )
    return MaskDroppedFullGridCounts(
        row_count=row_count,
        observed_row_count=observed_row_count,
        positive_observed_row_count=positive_observed_row_count,
        counts_by_year_label=year_label_counts_from_polars(dropped),
        counts_by_stratum=stratum_counts_from_polars(dropped),
    )


def year_label_counts_from_polars(frame: Any) -> dict[str, int]:
    """Count lazy frame rows by year and label source."""
    counts = (
        frame.group_by(["year", "label_source"])
        .agg(pl.len().alias("row_count"))
        .collect()
        .to_dicts()
    )
    return {
        f"{int(cast(Any, row['year']))}:{row['label_source']}": int(cast(Any, row["row_count"]))
        for row in counts
    }


def stratum_counts_from_polars(frame: Any) -> dict[str, int]:
    """Count lazy frame rows by year, label source, mask reason, and depth bin."""
    counts = (
        frame.group_by(["year", "label_source", "domain_mask_reason", "depth_bin"])
        .agg(pl.len().alias("row_count"))
        .collect()
        .to_dicts()
    )
    return {
        (
            f"{int(cast(Any, row['year']))}:{row['label_source']}:"
            f"{row['domain_mask_reason']}:{row['depth_bin']}"
        ): int(cast(Any, row["row_count"]))
        for row in counts
    }


def year_label_counts_from_crm_counts(
    counts: dict[str, int],
) -> dict[int, dict[str, int]]:
    """Collapse CRM-stratified manifest counts to nested year/label counts."""
    nested: dict[int, dict[str, int]] = {}
    for key, count in counts.items():
        year_text, label_source, *_ = key.split(":")
        year = int(year_text)
        nested.setdefault(year, {})[label_source] = nested.setdefault(year, {}).get(
            label_source, 0
        ) + int(count)
    return nested


def year_label_flat_counts_from_crm_counts(counts: dict[str, int]) -> dict[str, int]:
    """Collapse CRM-stratified manifest counts to stable year/label keys."""
    nested = year_label_counts_from_crm_counts(counts)
    return {
        f"{year}:{label_source}": count
        for year, label_counts in sorted(nested.items())
        for label_source, count in sorted(label_counts.items())
    }


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


def write_crm_stratified_sample_artifacts(
    grid_config: FullGridAlignmentConfig,
) -> CrmStratifiedSampleResult | None:
    """Write the optional CRM-stratified masked sample sidecar."""
    sample_config = grid_config.crm_stratified_sample
    if sample_config is None:
        return None
    LOGGER.info("Building CRM-stratified background sample from retained full-grid rows")
    result = build_crm_stratified_sample_result(grid_config, sample_config)
    reset_output_path(sample_config.output_path)
    write_part(result.sample, sample_config.output_path, year=0, part_index=0)
    write_csv(
        result.selection.summary_rows,
        sample_config.summary_path,
        CRM_STRATIFIED_SAMPLE_SUMMARY_FIELDS,
    )
    write_crm_stratified_sample_manifest(
        result=result,
        grid_config=grid_config,
        sample_config=sample_config,
    )
    LOGGER.info(
        "Selected %s CRM-stratified rows from %s retained population rows",
        result.selection.sampled_row_count,
        result.selection.population_row_count,
    )
    return result


def build_crm_stratified_sample_result(
    grid_config: FullGridAlignmentConfig,
    sample_config: CrmStratifiedSampleConfig,
) -> CrmStratifiedSampleResult:
    """Build a CRM-stratified sample result from retained full-grid rows."""
    population = read_crm_stratified_population_frame(
        full_grid_path=grid_config.full_grid_output_path,
        sample_config=sample_config,
    )
    selection = select_crm_stratified_sample_metadata(population, sample_config)
    if selection.selected_metadata.empty:
        msg = "CRM-stratified sampling selected no rows"
        raise ValueError(msg)
    sample = crm_stratified_sample_frame(
        full_grid_path=grid_config.full_grid_output_path,
        sample_config=sample_config,
        selected_metadata=selection.selected_metadata,
        sample_weight_column=grid_config.sample_weight_column,
    )
    if sample.empty:
        msg = "CRM-stratified selected keys produced no sample rows"
        raise ValueError(msg)
    return CrmStratifiedSampleResult(sample=sample, selection=selection)


def read_crm_stratified_population_frame(
    *,
    full_grid_path: Path,
    sample_config: CrmStratifiedSampleConfig,
) -> pd.DataFrame:
    """Read retained full-grid identifiers and mask strata for sampling."""
    validate_crm_stratified_mask_schema(sample_config)
    full_grid = pl.scan_parquet(str(full_grid_path)).select(
        [
            "year",
            "label_source",
            MASK_KEY_COLUMN,
            "kelp_max_y",
            "is_kelpwatch_observed",
        ]
    )
    mask = pl.scan_parquet(str(sample_config.table_path)).select(
        list(CRM_STRATIFIED_REQUIRED_MASK_COLUMNS)
    )
    missing_count = crm_stratified_missing_mask_count(full_grid, mask)
    if missing_count:
        msg = f"domain mask is missing rows for {missing_count} full-grid cells"
        raise ValueError(msg)
    retained = (
        full_grid.join(mask, on=MASK_KEY_COLUMN, how="inner")
        .filter(pl.col(MASK_RETAIN_COLUMN))
        .collect()
        .to_pandas()
    )
    if retained.empty:
        msg = f"CRM-stratified sampling retained no full-grid rows: {sample_config.table_path}"
        raise ValueError(msg)
    retained["domain_mask_reason"] = retained["domain_mask_reason"].map(normalize_stratum_value)
    retained["depth_bin"] = retained["depth_bin"].map(normalize_stratum_value)
    return cast(pd.DataFrame, retained)


def validate_crm_stratified_mask_schema(sample_config: CrmStratifiedSampleConfig) -> None:
    """Validate that the mask table contains CRM stratum columns."""
    available = set(pl.scan_parquet(str(sample_config.table_path)).collect_schema().names())
    missing = [column for column in CRM_STRATIFIED_REQUIRED_MASK_COLUMNS if column not in available]
    if missing:
        msg = f"CRM-stratified sampling mask is missing required columns: {missing}"
        raise ValueError(msg)


def crm_stratified_missing_mask_count(full_grid: Any, mask: Any) -> int:
    """Count full-grid rows whose target cell is absent from the mask table."""
    missing = (
        full_grid.join(mask.select(MASK_KEY_COLUMN), on=MASK_KEY_COLUMN, how="anti")
        .select(pl.len().alias("row_count"))
        .collect()
    )
    return int(missing["row_count"][0])


def select_crm_stratified_sample_metadata(
    population: pd.DataFrame,
    sample_config: CrmStratifiedSampleConfig,
) -> CrmStratifiedSelection:
    """Select all observed rows and quota-ranked assumed-background rows."""
    observed = population.loc[population["label_source"] == KELPWATCH_STATION].copy()
    background = population.loc[population["label_source"] == ASSUMED_BACKGROUND].copy()
    selected_parts = [observed]
    if not background.empty:
        selected_parts.extend(select_background_stratum_rows(background, sample_config).values())
    selected = pd.concat(selected_parts, ignore_index=False).sort_index()
    summary_rows = crm_stratified_summary_rows(population, selected, sample_config)
    return CrmStratifiedSelection(
        selected_metadata=selected,
        summary_rows=summary_rows,
        population_row_count=int(len(population)),
        sampled_row_count=int(len(selected)),
        retained_counts=crm_stratified_counts(selected),
        population_counts=crm_stratified_counts(population),
        dropped_counts=crm_stratified_dropped_counts(population, selected),
    )


def select_background_stratum_rows(
    background: pd.DataFrame,
    sample_config: CrmStratifiedSampleConfig,
) -> dict[tuple[int, str, str], pd.DataFrame]:
    """Select deterministic background rows for each year/reason/depth stratum."""
    selected: dict[tuple[int, str, str], pd.DataFrame] = {}
    group_columns = ["year", "domain_mask_reason", "depth_bin"]
    for keys, group in background.groupby(group_columns, sort=True, dropna=False):
        year, reason, depth_bin = cast(tuple[int, str, str], keys)
        quota = quota_for_stratum(str(reason), str(depth_bin), sample_config)
        row_quota = row_quota_for_population(len(group), quota)
        if row_quota <= 0:
            continue
        scores = deterministic_sample_scores(
            group[MASK_KEY_COLUMN].to_numpy(dtype=np.int64),
            int(year),
            sample_config.random_seed,
        )
        cell_ids = group[MASK_KEY_COLUMN].to_numpy(dtype=np.int64)
        order = np.lexsort((cell_ids, scores))
        selected[(int(year), str(reason), str(depth_bin))] = group.iloc[order[:row_quota]].copy()
    return selected


def row_quota_for_population(population_count: int, quota: CrmStratifiedQuota) -> int:
    """Return a capped row quota for one stratum population."""
    if population_count <= 0:
        return 0
    fraction_count = int(math.ceil(population_count * quota.fraction))
    requested = max(fraction_count, quota.min_rows_per_year)
    if quota.max_rows_per_year is not None:
        requested = min(requested, quota.max_rows_per_year)
    return min(population_count, requested)


def quota_for_stratum(
    domain_mask_reason: str,
    depth_bin: str,
    sample_config: CrmStratifiedSampleConfig,
) -> CrmStratifiedQuota:
    """Return the first matching quota rule for a CRM stratum."""
    for quota in sample_config.quotas:
        reason_match = quota.domain_mask_reason in {None, domain_mask_reason}
        depth_match = quota.depth_bin in {None, depth_bin}
        if reason_match and depth_match:
            return quota
    return CrmStratifiedQuota(
        domain_mask_reason=domain_mask_reason,
        depth_bin=depth_bin,
        fraction=sample_config.default_fraction,
        min_rows_per_year=sample_config.default_min_rows_per_year,
        max_rows_per_year=None,
    )


def deterministic_sample_scores(cell_id_values: np.ndarray, year: int, seed: int) -> np.ndarray:
    """Return stable pseudo-random scores from cell ids and year."""
    modulus = np.uint64(2**32)
    hashed = (
        cell_id_values.astype(np.uint64) * np.uint64(1_103_515_245)
        + np.uint64(seed)
        + np.uint64(year) * np.uint64(2_654_435_761)
    ) % modulus
    return cast(np.ndarray, hashed.astype(np.float64) / float(modulus))


def crm_stratified_sample_frame(
    *,
    full_grid_path: Path,
    sample_config: CrmStratifiedSampleConfig,
    selected_metadata: pd.DataFrame,
    sample_weight_column: str,
) -> pd.DataFrame:
    """Read selected full-grid rows, join mask metadata, and attach weights."""
    selected_keys = selected_metadata[["year", MASK_KEY_COLUMN]].drop_duplicates()
    full_grid = pl.scan_parquet(str(full_grid_path))
    selected = pl.LazyFrame(selected_keys)
    mask = pl.scan_parquet(str(sample_config.table_path)).select(list(MASK_DETAIL_COLUMNS))
    sample = (
        full_grid.join(selected, on=["year", MASK_KEY_COLUMN], how="inner")
        .join(mask, on=MASK_KEY_COLUMN, how="left")
        .collect()
        .to_pandas()
    )
    if sample[MASK_RETAIN_COLUMN].isna().any():
        missing_count = int(sample[MASK_RETAIN_COLUMN].isna().sum())
        msg = f"domain mask is missing rows for {missing_count} selected sample cells"
        raise ValueError(msg)
    sample["domain_mask_reason"] = sample["domain_mask_reason"].map(normalize_stratum_value)
    sample["depth_bin"] = sample["depth_bin"].map(normalize_stratum_value)
    return assign_crm_stratified_sample_weights(
        sample,
        selected_metadata,
        sample_weight_column,
    )


def assign_crm_stratified_sample_weights(
    sample: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    sample_weight_column: str,
) -> pd.DataFrame:
    """Assign expansion weights by retained CRM stratum."""
    weighted = sample.copy()
    weighted[sample_weight_column] = 1.0
    background = selected_metadata.loc[selected_metadata["label_source"] == ASSUMED_BACKGROUND]
    if background.empty:
        return weighted
    for keys, group in background.groupby(
        ["year", "domain_mask_reason", "depth_bin"], sort=False, dropna=False
    ):
        year, reason, depth_bin = cast(tuple[int, str, str], keys)
        population_count = int(group["stratum_population_count"].iloc[0])
        sample_count = len(group)
        sample_mask = (
            (weighted["year"].astype(int) == int(year))
            & (weighted["label_source"] == ASSUMED_BACKGROUND)
            & (weighted["domain_mask_reason"] == str(reason))
            & (weighted["depth_bin"] == str(depth_bin))
        )
        weighted.loc[sample_mask, sample_weight_column] = float(population_count / sample_count)
    return weighted


def crm_stratified_summary_rows(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    sample_config: CrmStratifiedSampleConfig,
) -> list[dict[str, object]]:
    """Build CRM-stratified population, sample, and weight summary rows."""
    selected_counts = crm_stratified_group_sizes(selected)
    rows: list[dict[str, object]] = []
    group_columns = [
        "year",
        "label_source",
        MASK_RETAIN_COLUMN,
        "domain_mask_reason",
        "depth_bin",
    ]
    for keys, group in population.groupby(group_columns, sort=True, dropna=False):
        year, label_source, retained, reason, depth_bin = cast(
            tuple[int, str, bool, str, str], keys
        )
        key = (int(year), str(label_source), bool(retained), str(reason), str(depth_bin))
        sampled_count = selected_counts.get(key, 0)
        quota = quota_for_stratum(str(reason), str(depth_bin), sample_config)
        sample_weight = (
            1.0
            if label_source == KELPWATCH_STATION or sampled_count == 0
            else len(group) / sampled_count
        )
        rows.append(
            {
                "year": int(year),
                "label_source": str(label_source),
                "is_plausible_kelp_domain": bool(retained),
                "domain_mask_reason": str(reason),
                "depth_bin": str(depth_bin),
                "population_row_count": int(len(group)),
                "sampled_row_count": int(sampled_count),
                "dropped_row_count": int(len(group) - sampled_count),
                "configured_fraction": 1.0 if label_source == KELPWATCH_STATION else quota.fraction,
                "configured_min_rows_per_year": 0
                if label_source == KELPWATCH_STATION
                else quota.min_rows_per_year,
                "configured_max_rows_per_year": ""
                if label_source == KELPWATCH_STATION or quota.max_rows_per_year is None
                else quota.max_rows_per_year,
                "effective_sample_fraction": len(group) and sampled_count / len(group),
                "kelpwatch_observed_row_count": observed_row_count(group),
                "kelpwatch_positive_row_count": positive_observed_row_count(group),
                "sample_weight_min": sample_weight if sampled_count else math.nan,
                "sample_weight_max": sample_weight if sampled_count else math.nan,
            }
        )
    attach_stratum_population_counts(selected, population)
    return rows


def crm_stratified_group_sizes(
    dataframe: pd.DataFrame,
) -> dict[tuple[int, str, bool, str, str], int]:
    """Count CRM-stratified rows by year, source, retain flag, reason, and depth."""
    counts: dict[tuple[int, str, bool, str, str], int] = {}
    if dataframe.empty:
        return counts
    group_columns = [
        "year",
        "label_source",
        MASK_RETAIN_COLUMN,
        "domain_mask_reason",
        "depth_bin",
    ]
    for keys, group in dataframe.groupby(group_columns, sort=True, dropna=False):
        year, label_source, retained, reason, depth_bin = cast(
            tuple[int, str, bool, str, str], keys
        )
        counts[(int(year), str(label_source), bool(retained), str(reason), str(depth_bin))] = int(
            len(group)
        )
    return counts


def attach_stratum_population_counts(selected: pd.DataFrame, population: pd.DataFrame) -> None:
    """Attach background stratum population counts to selected metadata in place."""
    if selected.empty:
        selected["stratum_population_count"] = pd.Series(dtype="int64")
        return
    counts = (
        population.loc[population["label_source"] == ASSUMED_BACKGROUND]
        .groupby(["year", "domain_mask_reason", "depth_bin"], sort=False)
        .size()
        .rename("stratum_population_count")
        .reset_index()
    )
    if counts.empty:
        selected["stratum_population_count"] = 0
        return
    merged = selected.merge(
        counts,
        on=["year", "domain_mask_reason", "depth_bin"],
        how="left",
        validate="many_to_one",
    )
    selected["stratum_population_count"] = (
        merged["stratum_population_count"].fillna(0).to_numpy(dtype=np.int64)
    )


def crm_stratified_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count rows by stable CRM-stratified manifest key."""
    counts: dict[str, int] = {}
    if dataframe.empty:
        return counts
    for keys, group in dataframe.groupby(
        ["year", "label_source", "domain_mask_reason", "depth_bin"],
        sort=True,
        dropna=False,
    ):
        year, label_source, reason, depth_bin = cast(tuple[int, str, str, str], keys)
        key = f"{int(year)}:{label_source}:{reason}:{depth_bin}"
        counts[key] = int(len(group))
    return counts


def crm_stratified_dropped_counts(
    population: pd.DataFrame, selected: pd.DataFrame
) -> dict[str, int]:
    """Count population rows not selected by stable CRM-stratified manifest key."""
    population_counts = crm_stratified_counts(population)
    selected_counts = crm_stratified_counts(selected)
    return {
        key: count - selected_counts.get(key, 0)
        for key, count in population_counts.items()
        if count - selected_counts.get(key, 0) > 0
    }


def normalize_stratum_value(value: object) -> str:
    """Normalize missing or dynamic stratum values to stable strings."""
    if pd.isna(value):
        return ""
    return str(value)


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
    crm_stratified_sample_result: CrmStratifiedSampleResult | None,
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
        full_payload["masked_sample_policy"] = grid_config.sample_domain_mask.sampling_policy
        full_payload["masked_sample_counts"] = masked_sample_result.retained_counts
        full_payload["masked_population_counts"] = masked_sample_result.population_counts
    if crm_stratified_sample_result is not None and grid_config.crm_stratified_sample is not None:
        full_payload["crm_stratified_sample_output"] = str(
            grid_config.crm_stratified_sample.output_path
        )
        full_payload["crm_stratified_sample_counts"] = (
            crm_stratified_sample_result.selection.retained_counts
        )
        full_payload["crm_stratified_population_counts"] = (
            crm_stratified_sample_result.selection.population_counts
        )
    sample_payload = {
        "command": "align-full-grid",
        "artifact": "background_sample",
        "sample_output": str(grid_config.sample_output_path),
        "sample_row_count": int(len(sample)),
        "sample_weight_column": grid_config.sample_weight_column,
        "legacy_background_rows_per_year": grid_config.background_rows_per_year,
        "background_rows_per_year_controls_default_masked_workflow": False
        if grid_config.sample_domain_mask is not None
        and grid_config.sample_domain_mask.sampling_policy
        == CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY
        else True,
        "include_all_kelpwatch_observed": grid_config.include_all_kelpwatch_observed,
        "random_seed": grid_config.random_seed,
        "schema": list(sample.columns),
    }
    write_json(grid_config.full_grid_manifest_path, full_payload)
    write_json(grid_config.sample_manifest_path, sample_payload)


def write_crm_stratified_sample_manifest(
    *,
    result: CrmStratifiedSampleResult,
    grid_config: FullGridAlignmentConfig,
    sample_config: CrmStratifiedSampleConfig,
) -> None:
    """Write a JSON manifest for the CRM-stratified sample sidecar."""
    payload = {
        "command": "align-full-grid",
        "artifact": "background_sample_crm_stratified_domain_mask",
        "config_path": str(grid_config.config_path),
        "fast": grid_config.fast,
        "domain_policy": sample_config.policy,
        "mask_table": str(sample_config.table_path),
        "mask_manifest": str(sample_config.manifest_path) if sample_config.manifest_path else None,
        "source_full_grid_output": str(grid_config.full_grid_output_path),
        "crm_stratified_sample_output": str(sample_config.output_path),
        "crm_stratified_sample_summary": str(sample_config.summary_path),
        "population_row_count": result.selection.population_row_count,
        "sample_row_count": result.selection.sampled_row_count,
        "retained_counts": result.selection.retained_counts,
        "population_counts": result.selection.population_counts,
        "dropped_counts": result.selection.dropped_counts,
        "sample_weight_column": grid_config.sample_weight_column,
        "sampling_policy": {
            "quota_type": "per_year_crm_stratum_fraction_with_min_max_caps",
            "strata": [
                {
                    "domain_mask_reason": quota.domain_mask_reason,
                    "depth_bin": quota.depth_bin,
                    "fraction": quota.fraction,
                    "min_rows_per_year": quota.min_rows_per_year,
                    "max_rows_per_year": quota.max_rows_per_year,
                }
                for quota in sample_config.quotas
            ],
            "default_fraction": sample_config.default_fraction,
            "default_min_rows_per_year": sample_config.default_min_rows_per_year,
            "deterministic_key": ["aef_grid_cell_id", "year", "random_seed"],
            "random_seed": sample_config.random_seed,
            "all_retained_kelpwatch_rows_kept": True,
            "sample_weight_policy": (
                "assumed-background weights are retained stratum population divided "
                "by sampled stratum rows; kelpwatch rows keep weight 1.0"
            ),
        },
        "summary_row_count": len(result.selection.summary_rows),
        "schema": list(result.sample.columns),
    }
    write_json(sample_config.manifest_output_path, payload)


def write_mask_first_crm_stratified_sample_manifest(
    *,
    result: CrmStratifiedSampleResult,
    grid_config: FullGridAlignmentConfig,
    sample_config: CrmStratifiedSampleConfig,
    mask_drops: MaskDroppedFullGridCounts,
) -> None:
    """Write the default mask-first CRM-stratified sample manifest."""
    payload = {
        "command": "align-full-grid",
        "artifact": "background_sample_domain_mask",
        "config_path": str(grid_config.config_path),
        "fast": grid_config.fast,
        "domain_policy": sample_config.policy,
        "masked_sample_policy": CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY,
        "mask_first": True,
        "mask_table": str(sample_config.table_path),
        "mask_manifest": str(sample_config.manifest_path) if sample_config.manifest_path else None,
        "source_full_grid_output": str(grid_config.full_grid_output_path),
        "masked_sample_output": str(sample_config.output_path),
        "masked_sample_summary": str(sample_config.summary_path),
        "population_row_count": result.selection.population_row_count,
        "sample_row_count": result.selection.sampled_row_count,
        "retained_domain_population_counts": result.selection.population_counts,
        "sampled_retained_counts": result.selection.retained_counts,
        "retained_domain_quota_dropped_counts": result.selection.dropped_counts,
        "mask_dropped_row_count": mask_drops.row_count,
        "mask_dropped_counts": mask_drops.counts_by_year_label,
        "mask_dropped_stratum_counts": mask_drops.counts_by_stratum,
        "mask_dropped_observed_row_count": mask_drops.observed_row_count,
        "mask_dropped_positive_row_count": mask_drops.positive_observed_row_count,
        "sample_weight_column": grid_config.sample_weight_column,
        "legacy_background_rows_per_year": grid_config.background_rows_per_year,
        "background_rows_per_year_controls_default_masked_workflow": False,
        "include_all_kelpwatch_observed": grid_config.include_all_kelpwatch_observed,
        "fail_on_dropped_positive": grid_config.sample_domain_mask.fail_on_dropped_positive
        if grid_config.sample_domain_mask is not None
        else True,
        "sampling_policy": {
            "name": CRM_STRATIFIED_MASK_FIRST_SAMPLE_POLICY,
            "quota_type": "per_year_crm_stratum_fraction_with_min_max_caps",
            "sampling_population": "retained plausible-kelp domain full-grid rows",
            "strata": [
                {
                    "domain_mask_reason": quota.domain_mask_reason,
                    "depth_bin": quota.depth_bin,
                    "fraction": quota.fraction,
                    "min_rows_per_year": quota.min_rows_per_year,
                    "max_rows_per_year": quota.max_rows_per_year,
                }
                for quota in sample_config.quotas
            ],
            "default_fraction": sample_config.default_fraction,
            "default_min_rows_per_year": sample_config.default_min_rows_per_year,
            "deterministic_key": ["aef_grid_cell_id", "year", "random_seed"],
            "random_seed": sample_config.random_seed,
            "all_retained_kelpwatch_rows_kept": True,
            "sample_weight_policy": (
                "assumed-background weights are retained stratum population divided "
                "by sampled stratum rows; kelpwatch rows keep weight 1.0"
            ),
            "model_feature_policy": (
                "CRM depth, elevation, depth_bin, and domain_mask_reason are sampling "
                "and diagnostics context only, not model predictors"
            ),
        },
        "summary_row_count": len(result.selection.summary_rows),
        "schema": list(result.sample.columns),
    }
    write_json(sample_config.manifest_output_path, payload)


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
        "masked_sample_policy": mask_config.sampling_policy,
        "mask_first": False,
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
        "legacy_background_rows_per_year": grid_config.background_rows_per_year,
        "background_rows_per_year_controls_default_masked_workflow": True,
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
