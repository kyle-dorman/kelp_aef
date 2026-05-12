"""Build a conservative plausible-kelp domain mask from aligned CRM support."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.domain.crm_alignment import (
    fast_path,
    load_json_object,
    optional_float,
    optional_mapping,
    optional_path_field,
    optional_positive_int,
    path_field,
    reset_output_path,
    schema_for_manifest,
    window_from_config,
    write_csv,
    write_json,
)

LOGGER = logging.getLogger(__name__)

MASK_VERSION = "plausible_kelp_domain_mask_v1"
RULE_SET = "crm_depth_elevation_v1"
MISSING_CRM_POLICY = "retain_for_qa"
DEFAULT_MAX_DEPTH_M = 100.0
DEFAULT_DEFINITE_LAND_ELEVATION_M = 5.0
DEFAULT_AMBIGUOUS_COAST_BAND_M = (-5.0, 5.0)
DEFAULT_NEARSHORE_SHALLOW_DEPTH_M = 40.0
DEFAULT_INTERMEDIATE_DEPTH_M = 50.0
DEFAULT_ROW_CHUNK_SIZE = 250_000

QA_MISSING_CRM = "qa_missing_crm"
RETAINED_AMBIGUOUS_COAST = "retained_ambiguous_coast"
DROPPED_DEFINITE_LAND = "dropped_definite_land"
DROPPED_DEEP_WATER = "dropped_deep_water"
RETAINED_DEPTH_0_100M = "retained_depth_0_100m"
MASK_MISSING = "mask_missing"

REASON_ORDER = (
    QA_MISSING_CRM,
    RETAINED_AMBIGUOUS_COAST,
    DROPPED_DEFINITE_LAND,
    DROPPED_DEEP_WATER,
    RETAINED_DEPTH_0_100M,
)
REASON_PLAUSIBLE = {
    QA_MISSING_CRM: True,
    RETAINED_AMBIGUOUS_COAST: True,
    DROPPED_DEFINITE_LAND: False,
    DROPPED_DEEP_WATER: False,
    RETAINED_DEPTH_0_100M: True,
    MASK_MISSING: False,
}
DEPTH_BIN_ORDER = (
    QA_MISSING_CRM,
    "land_positive",
    "ambiguous_coast",
    "0_40m",
    "40_50m",
    "50_100m",
    "100m_plus",
)
CRM_INPUT_COLUMNS = (
    "aef_grid_row",
    "aef_grid_col",
    "aef_grid_cell_id",
    "longitude",
    "latitude",
    "crm_elevation_m",
    "crm_depth_m",
    "crm_source_product_id",
    "crm_vertical_datum",
    "crm_value_status",
    "cudem_value_status",
    "usgs_3dep_value_status",
)
MASK_OUTPUT_COLUMNS = (
    "aef_grid_row",
    "aef_grid_col",
    "aef_grid_cell_id",
    "longitude",
    "latitude",
    "crm_elevation_m",
    "crm_depth_m",
    "crm_source_product_id",
    "crm_vertical_datum",
    "crm_value_status",
    "cudem_value_status",
    "usgs_3dep_value_status",
    "is_plausible_kelp_domain",
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "domain_mask_depth_threshold_m",
    "domain_mask_rule_set",
    "depth_bin",
    "elevation_bin",
)
COVERAGE_FIELDS = (
    "summary_level",
    "domain_mask_reason",
    "crm_source_product_id",
    "is_plausible_kelp_domain",
    "cell_count",
    "cell_fraction",
)
RETENTION_FIELDS = (
    "year",
    "label_source",
    "domain_mask_reason",
    "positive_cell_year_rows",
    "positive_unique_cells",
    "retained_positive_cell_year_rows",
    "dropped_positive_cell_year_rows",
    "retained_positive_fraction",
)
DEPTH_BIN_FIELDS = (
    "depth_bin",
    "is_plausible_kelp_domain",
    "cell_count",
    "cell_fraction",
    "kelpwatch_positive_cell_year_rows",
    "kelpwatch_positive_unique_cells",
)


@dataclass(frozen=True)
class DomainMaskThresholds:
    """Thresholds that define the first CRM-only plausible-kelp mask."""

    max_depth_m: float
    definite_land_elevation_m: float
    ambiguous_coast_elevation_min_m: float
    ambiguous_coast_elevation_max_m: float
    nearshore_shallow_depth_m: float
    intermediate_depth_m: float


@dataclass(frozen=True)
class DomainMaskConfig:
    """Resolved config for the plausible-kelp domain-mask command."""

    config_path: Path
    region_name: str
    years: tuple[int, ...]
    aligned_crm_table: Path
    aligned_crm_manifest: Path
    full_grid_table: Path
    full_grid_manifest: Path | None
    cusp_source_manifest: Path | None
    output_table: Path
    output_manifest: Path
    coverage_summary_table: Path
    kelpwatch_retention_table: Path
    depth_bin_summary_table: Path
    visual_qa_figure: Path | None
    thresholds: DomainMaskThresholds
    row_chunk_size: int
    fast: bool
    fast_years: tuple[int, ...]
    row_window: tuple[int, int] | None
    col_window: tuple[int, int] | None


@dataclass
class DomainMaskAccumulator:
    """Running counters for the mask table and coverage summaries."""

    total_cells: int = 0
    retained_cells: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)
    reason_product_counts: Counter[tuple[str, str, bool]] = field(default_factory=Counter)
    depth_bin_counts: Counter[tuple[str, bool]] = field(default_factory=Counter)
    crm_status_counts: Counter[str] = field(default_factory=Counter)


def build_domain_mask(config_path: Path, *, fast: bool = False) -> int:
    """Build the configured plausible-kelp domain mask and QA artifacts."""
    mask_config = load_domain_mask_config(config_path, fast=fast)
    LOGGER.info(
        "Building plausible-kelp domain mask for %s with %s mode",
        mask_config.region_name,
        "fast" if fast else "full",
    )
    LOGGER.info("Output overwrite policy: replace existing mask artifacts.")
    accumulator = write_domain_mask_table(mask_config)
    if accumulator.total_cells == 0:
        msg = "aligned CRM input produced no target-grid cells for the configured mask"
        raise ValueError(msg)

    mask_frame = load_mask_qa_frame(mask_config.output_table)
    positive_rows = load_kelpwatch_positive_rows(mask_config)
    retention_rows = build_kelpwatch_retention_rows(positive_rows, mask_frame)
    depth_bin_rows = build_depth_bin_summary_rows(mask_frame, positive_rows)
    write_csv(
        mask_config.coverage_summary_table,
        COVERAGE_FIELDS,
        coverage_summary_rows(accumulator),
    )
    write_csv(mask_config.kelpwatch_retention_table, RETENTION_FIELDS, retention_rows)
    write_csv(mask_config.depth_bin_summary_table, DEPTH_BIN_FIELDS, depth_bin_rows)
    if mask_config.visual_qa_figure is not None:
        write_visual_qa_figure(mask_frame, positive_rows, mask_config)
    manifest = build_mask_manifest(
        mask_config=mask_config,
        accumulator=accumulator,
        retention_rows=retention_rows,
        depth_bin_rows=depth_bin_rows,
        schema=mask_arrow_schema(),
    )
    write_json(mask_config.output_manifest, manifest)
    LOGGER.info("Wrote plausible-kelp mask table: %s", mask_config.output_table)
    LOGGER.info("Wrote plausible-kelp mask manifest: %s", mask_config.output_manifest)
    return 0


def load_domain_mask_config(config_path: Path, *, fast: bool) -> DomainMaskConfig:
    """Load plausible-kelp mask settings from the workflow config."""
    config = load_yaml_config(config_path)
    domain = require_mapping(config.get("domain"), "domain")
    mask_config = require_mapping(domain.get("plausible_kelp_mask"), "domain.plausible_kelp_mask")
    crm_alignment = require_mapping(
        require_mapping(domain.get("noaa_crm"), "domain.noaa_crm").get("alignment"),
        "domain.noaa_crm.alignment",
    )
    alignment = require_mapping(config.get("alignment"), "alignment")
    full_grid = require_mapping(alignment.get("full_grid"), "alignment.full_grid")
    fast_grid = optional_mapping(full_grid.get("fast"), "alignment.full_grid.fast")
    fast_outputs = optional_mapping(mask_config.get("fast"), "domain.plausible_kelp_mask.fast")
    region = require_mapping(config.get("region"), "region")
    return DomainMaskConfig(
        config_path=config_path,
        region_name=require_string(region.get("name"), "region.name"),
        years=load_years(config),
        aligned_crm_table=fast_path(
            path_field(
                mask_config, "aligned_crm_table", "domain.plausible_kelp_mask.aligned_crm_table"
            ),
            fast,
            fast_outputs,
            "aligned_crm_table",
        ),
        aligned_crm_manifest=fast_path(
            path_field(
                mask_config,
                "aligned_crm_manifest",
                "domain.plausible_kelp_mask.aligned_crm_manifest",
            ),
            fast,
            fast_outputs,
            "aligned_crm_manifest",
        ),
        full_grid_table=path_field(
            mask_config,
            "full_grid_table",
            "domain.plausible_kelp_mask.full_grid_table",
        ),
        full_grid_manifest=optional_path_field(
            mask_config,
            "full_grid_manifest",
            "domain.plausible_kelp_mask.full_grid_manifest",
        ),
        cusp_source_manifest=optional_path_field(
            mask_config,
            "cusp_source_manifest",
            "domain.plausible_kelp_mask.cusp_source_manifest",
        )
        or optional_path_field(
            crm_alignment,
            "cusp_source_manifest",
            "domain.noaa_crm.alignment.cusp_source_manifest",
        ),
        output_table=fast_path(
            path_field(mask_config, "output_table", "domain.plausible_kelp_mask.output_table"),
            fast,
            fast_outputs,
            "output_table",
        ),
        output_manifest=fast_path(
            path_field(
                mask_config,
                "output_manifest",
                "domain.plausible_kelp_mask.output_manifest",
            ),
            fast,
            fast_outputs,
            "output_manifest",
        ),
        coverage_summary_table=fast_path(
            path_field(
                mask_config,
                "coverage_summary_table",
                "domain.plausible_kelp_mask.coverage_summary_table",
            ),
            fast,
            fast_outputs,
            "coverage_summary_table",
        ),
        kelpwatch_retention_table=fast_path(
            path_field(
                mask_config,
                "kelpwatch_retention_table",
                "domain.plausible_kelp_mask.kelpwatch_retention_table",
            ),
            fast,
            fast_outputs,
            "kelpwatch_retention_table",
        ),
        depth_bin_summary_table=fast_path(
            path_field(
                mask_config,
                "depth_bin_summary_table",
                "domain.plausible_kelp_mask.depth_bin_summary_table",
            ),
            fast,
            fast_outputs,
            "depth_bin_summary_table",
        ),
        visual_qa_figure=optional_fast_path(mask_config, fast_outputs, fast, "visual_qa_figure"),
        thresholds=load_domain_mask_thresholds(mask_config),
        row_chunk_size=optional_positive_int(
            mask_config.get("row_chunk_size"),
            "domain.plausible_kelp_mask.row_chunk_size",
            DEFAULT_ROW_CHUNK_SIZE,
        ),
        fast=fast,
        fast_years=load_fast_years(fast_grid),
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
    )


def load_years(config: dict[str, Any]) -> tuple[int, ...]:
    """Read configured smoke years for retention QA filtering."""
    years_config = require_mapping(config.get("years"), "years")
    years = years_config.get("smoke")
    if not isinstance(years, list) or not years:
        msg = "config field must be a non-empty list of years: years.smoke"
        raise ValueError(msg)
    return tuple(require_int(year, "years.smoke[]") for year in years)


def load_fast_years(fast_grid: dict[str, Any]) -> tuple[int, ...]:
    """Read configured full-grid fast years."""
    years = fast_grid.get("years")
    if not isinstance(years, list) or not years:
        return ()
    return tuple(require_int(year, "alignment.full_grid.fast.years[]") for year in years)


def require_int(value: object, name: str) -> int:
    """Validate a dynamic config value as an integer."""
    if not isinstance(value, int):
        msg = f"config field must be an integer: {name}"
        raise ValueError(msg)
    return value


def optional_fast_path(
    mask_config: dict[str, Any],
    fast_outputs: dict[str, Any],
    fast: bool,
    key: str,
) -> Path | None:
    """Resolve an optional output path and its fast override."""
    configured = optional_path_field(mask_config, key, f"domain.plausible_kelp_mask.{key}")
    if configured is None:
        return None
    return fast_path(configured, fast, fast_outputs, key)


def load_domain_mask_thresholds(mask_config: dict[str, Any]) -> DomainMaskThresholds:
    """Load the mask threshold block with conservative defaults."""
    ambiguous_band = load_ambiguous_coast_band(mask_config.get("ambiguous_coast_elevation_band_m"))
    return DomainMaskThresholds(
        max_depth_m=optional_float(
            mask_config.get("max_depth_m"),
            "domain.plausible_kelp_mask.max_depth_m",
            DEFAULT_MAX_DEPTH_M,
        ),
        definite_land_elevation_m=optional_float(
            mask_config.get("definite_land_elevation_m"),
            "domain.plausible_kelp_mask.definite_land_elevation_m",
            DEFAULT_DEFINITE_LAND_ELEVATION_M,
        ),
        ambiguous_coast_elevation_min_m=ambiguous_band[0],
        ambiguous_coast_elevation_max_m=ambiguous_band[1],
        nearshore_shallow_depth_m=optional_float(
            mask_config.get("nearshore_shallow_depth_m"),
            "domain.plausible_kelp_mask.nearshore_shallow_depth_m",
            DEFAULT_NEARSHORE_SHALLOW_DEPTH_M,
        ),
        intermediate_depth_m=optional_float(
            mask_config.get("intermediate_depth_m"),
            "domain.plausible_kelp_mask.intermediate_depth_m",
            DEFAULT_INTERMEDIATE_DEPTH_M,
        ),
    )


def load_ambiguous_coast_band(value: object) -> tuple[float, float]:
    """Load and validate the ambiguous-coast elevation band."""
    if value is None:
        return DEFAULT_AMBIGUOUS_COAST_BAND_M
    if not isinstance(value, list) or len(value) != 2:
        msg = (
            "config field must be a two-element list: "
            "domain.plausible_kelp_mask.ambiguous_coast_elevation_band_m"
        )
        raise ValueError(msg)
    lower = require_number(
        value[0], "domain.plausible_kelp_mask.ambiguous_coast_elevation_band_m[0]"
    )
    upper = require_number(
        value[1], "domain.plausible_kelp_mask.ambiguous_coast_elevation_band_m[1]"
    )
    if upper < lower:
        msg = "ambiguous coast elevation band upper bound must be >= lower bound"
        raise ValueError(msg)
    return lower, upper


def require_number(value: object, name: str) -> float:
    """Validate a dynamic config value as numeric."""
    if not isinstance(value, int | float):
        msg = f"config field must be a number: {name}"
        raise ValueError(msg)
    return float(value)


def write_domain_mask_table(mask_config: DomainMaskConfig) -> DomainMaskAccumulator:
    """Classify the aligned CRM support table and write the mask Parquet file."""
    if not mask_config.aligned_crm_table.exists():
        msg = f"aligned CRM table does not exist: {mask_config.aligned_crm_table}"
        raise FileNotFoundError(msg)
    reset_output_path(mask_config.output_table)
    accumulator = DomainMaskAccumulator()
    schema = mask_arrow_schema()
    with pq.ParquetWriter(mask_config.output_table, schema) as writer:  # type: ignore[no-untyped-call]
        for frame in iter_crm_input_chunks(mask_config):
            classified = classify_domain_mask_frame(frame, mask_config.thresholds)
            update_mask_accumulator(accumulator, classified)
            table = pa.Table.from_pandas(classified, schema=schema, preserve_index=False)
            writer.write_table(table)
            LOGGER.info(
                "Classified %s target cells for plausible-kelp mask",
                accumulator.total_cells,
            )
    return accumulator


def iter_crm_input_chunks(mask_config: DomainMaskConfig) -> Iterator[pd.DataFrame]:
    """Yield aligned CRM input chunks filtered to the configured fast window."""
    dataset = pds.dataset(mask_config.aligned_crm_table)  # type: ignore[no-untyped-call]
    filter_expression = fast_window_filter(mask_config)
    scanner = dataset.scanner(
        columns=list(CRM_INPUT_COLUMNS),
        filter=filter_expression,
        batch_size=mask_config.row_chunk_size,
    )
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def fast_window_filter(mask_config: DomainMaskConfig) -> Any | None:
    """Build a PyArrow dataset filter for fast row/column windows."""
    filter_expression = None
    if mask_config.row_window is not None:
        row_filter = (dataset_field("aef_grid_row") >= mask_config.row_window[0]) & (
            dataset_field("aef_grid_row") < mask_config.row_window[1]
        )
        filter_expression = (
            row_filter if filter_expression is None else filter_expression & row_filter
        )
    if mask_config.col_window is not None:
        col_filter = (dataset_field("aef_grid_col") >= mask_config.col_window[0]) & (
            dataset_field("aef_grid_col") < mask_config.col_window[1]
        )
        filter_expression = (
            col_filter if filter_expression is None else filter_expression & col_filter
        )
    return filter_expression


def dataset_field(name: str) -> Any:
    """Return a PyArrow dataset field expression with local typing suppression."""
    return pds.field(name)  # type: ignore[attr-defined,no-untyped-call]


def classify_domain_mask_frame(
    frame: pd.DataFrame,
    thresholds: DomainMaskThresholds,
) -> pd.DataFrame:
    """Apply explicit CRM-depth/elevation mask rules to one input frame."""
    result = frame[list(CRM_INPUT_COLUMNS)].copy()
    count = len(result)
    elevation = result["crm_elevation_m"].to_numpy(dtype=np.float64)
    depth = result["crm_depth_m"].to_numpy(dtype=np.float64)
    status = result["crm_value_status"].fillna("").astype(str).to_numpy()
    valid = (status == "valid") & np.isfinite(elevation) & np.isfinite(depth)
    ambiguous = (
        valid
        & (elevation >= thresholds.ambiguous_coast_elevation_min_m)
        & (elevation <= thresholds.ambiguous_coast_elevation_max_m)
    )
    definite_land = valid & ~ambiguous & (elevation > thresholds.definite_land_elevation_m)
    deep_water = valid & ~ambiguous & ~definite_land & (depth > thresholds.max_depth_m)
    retained_depth = valid & ~ambiguous & ~definite_land & ~deep_water

    reason = np.full(count, QA_MISSING_CRM, dtype=object)
    detail = np.full(count, "CRM value missing or not valid; retained for QA", dtype=object)
    plausible = np.ones(count, dtype=bool)
    reason[ambiguous] = RETAINED_AMBIGUOUS_COAST
    detail[ambiguous] = (
        "CRM elevation within configured ambiguous coast band; retained for first-pass caution"
    )
    reason[definite_land] = DROPPED_DEFINITE_LAND
    detail[definite_land] = "CRM elevation exceeds configured definite-land threshold"
    plausible[definite_land] = False
    reason[deep_water] = DROPPED_DEEP_WATER
    detail[deep_water] = "CRM depth exceeds configured broad maximum depth threshold"
    plausible[deep_water] = False
    reason[retained_depth] = RETAINED_DEPTH_0_100M
    detail[retained_depth] = "CRM-valid cell within configured broad depth threshold"
    result["is_plausible_kelp_domain"] = plausible
    result["domain_mask_reason"] = reason
    result["domain_mask_detail"] = detail
    result["domain_mask_version"] = MASK_VERSION
    result["domain_mask_depth_threshold_m"] = np.float32(thresholds.max_depth_m)
    result["domain_mask_rule_set"] = RULE_SET
    result["depth_bin"] = classify_depth_bins(elevation, depth, valid, ambiguous, thresholds)
    result["elevation_bin"] = classify_elevation_bins(elevation, valid, ambiguous, thresholds)
    return result[list(MASK_OUTPUT_COLUMNS)]


def classify_depth_bins(
    elevation: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    ambiguous: np.ndarray,
    thresholds: DomainMaskThresholds,
) -> np.ndarray:
    """Classify values into depth/elevation bins used for QA summaries."""
    bins = np.full(len(elevation), QA_MISSING_CRM, dtype=object)
    land = valid & ~ambiguous & (elevation > thresholds.definite_land_elevation_m)
    bins[land] = "land_positive"
    bins[ambiguous] = "ambiguous_coast"
    ocean = valid & ~ambiguous & ~land
    bins[ocean & (depth <= thresholds.nearshore_shallow_depth_m)] = "0_40m"
    bins[
        ocean
        & (depth > thresholds.nearshore_shallow_depth_m)
        & (depth <= thresholds.intermediate_depth_m)
    ] = "40_50m"
    bins[ocean & (depth > thresholds.intermediate_depth_m) & (depth <= thresholds.max_depth_m)] = (
        "50_100m"
    )
    bins[ocean & (depth > thresholds.max_depth_m)] = "100m_plus"
    return bins


def classify_elevation_bins(
    elevation: np.ndarray,
    valid: np.ndarray,
    ambiguous: np.ndarray,
    thresholds: DomainMaskThresholds,
) -> np.ndarray:
    """Classify CRM elevations into broad QA categories."""
    bins = np.full(len(elevation), QA_MISSING_CRM, dtype=object)
    bins[ambiguous] = "ambiguous_coast"
    bins[valid & ~ambiguous & (elevation > thresholds.definite_land_elevation_m)] = "definite_land"
    bins[valid & ~ambiguous & (elevation <= thresholds.ambiguous_coast_elevation_min_m)] = (
        "subtidal_ocean"
    )
    bins[
        valid
        & ~ambiguous
        & (elevation < thresholds.definite_land_elevation_m)
        & (elevation > thresholds.ambiguous_coast_elevation_max_m)
    ] = "low_positive_elevation"
    return bins


def mask_arrow_schema() -> pa.Schema:
    """Return the fixed Parquet schema for the domain mask table."""
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
            pa.field("crm_vertical_datum", pa.string()),
            pa.field("crm_value_status", pa.string()),
            pa.field("cudem_value_status", pa.string()),
            pa.field("usgs_3dep_value_status", pa.string()),
            pa.field("is_plausible_kelp_domain", pa.bool_()),
            pa.field("domain_mask_reason", pa.string()),
            pa.field("domain_mask_detail", pa.string()),
            pa.field("domain_mask_version", pa.string()),
            pa.field("domain_mask_depth_threshold_m", pa.float32()),
            pa.field("domain_mask_rule_set", pa.string()),
            pa.field("depth_bin", pa.string()),
            pa.field("elevation_bin", pa.string()),
        ]
    )


def update_mask_accumulator(
    accumulator: DomainMaskAccumulator,
    frame: pd.DataFrame,
) -> None:
    """Update coverage counters from one classified mask frame."""
    accumulator.total_cells += len(frame)
    retained = frame["is_plausible_kelp_domain"].to_numpy(dtype=bool)
    accumulator.retained_cells += int(np.sum(retained))
    accumulator.reason_counts.update(frame["domain_mask_reason"].astype(str).to_list())
    accumulator.crm_status_counts.update(frame["crm_value_status"].astype(str).to_list())
    for row in frame[
        ["domain_mask_reason", "crm_source_product_id", "is_plausible_kelp_domain"]
    ].itertuples(index=False):
        reason = str(row.domain_mask_reason)
        product = (
            "missing_product"
            if pd.isna(row.crm_source_product_id)
            else str(row.crm_source_product_id)
        )
        accumulator.reason_product_counts[
            (reason, product, bool(row.is_plausible_kelp_domain))
        ] += 1
    for row in frame[["depth_bin", "is_plausible_kelp_domain"]].itertuples(index=False):
        accumulator.depth_bin_counts[(str(row.depth_bin), bool(row.is_plausible_kelp_domain))] += 1


def coverage_summary_rows(accumulator: DomainMaskAccumulator) -> list[dict[str, object]]:
    """Build retained/dropped coverage rows by reason and source product."""
    total = max(accumulator.total_cells, 1)
    dropped_cells = accumulator.total_cells - accumulator.retained_cells
    rows: list[dict[str, object]] = [
        coverage_row("overall", "all", "all", True, accumulator.retained_cells, total),
        coverage_row("overall", "all", "all", False, dropped_cells, total),
    ]
    for reason in sorted(accumulator.reason_counts, key=reason_sort_key):
        plausible = REASON_PLAUSIBLE.get(reason, False)
        rows.append(
            coverage_row(
                "reason",
                reason,
                "all",
                plausible,
                accumulator.reason_counts[reason],
                total,
            )
        )
    for (reason, product, plausible), count in sorted(accumulator.reason_product_counts.items()):
        rows.append(coverage_row("reason_by_product", reason, product, plausible, count, total))
    return rows


def reason_sort_key(reason: str) -> tuple[int, str]:
    """Return stable sort keys with configured reason precedence first."""
    if reason in REASON_ORDER:
        return REASON_ORDER.index(reason), reason
    return len(REASON_ORDER), reason


def coverage_row(
    summary_level: str,
    reason: str,
    product: str,
    plausible: bool,
    count: int,
    total: int,
) -> dict[str, object]:
    """Build one coverage summary row."""
    return {
        "summary_level": summary_level,
        "domain_mask_reason": reason,
        "crm_source_product_id": product,
        "is_plausible_kelp_domain": plausible,
        "cell_count": int(count),
        "cell_fraction": float(count / total),
    }


def load_mask_qa_frame(path: Path) -> pd.DataFrame:
    """Load the completed mask table columns needed for downstream QA."""
    columns = [
        "aef_grid_row",
        "aef_grid_col",
        "aef_grid_cell_id",
        "longitude",
        "latitude",
        "crm_elevation_m",
        "crm_depth_m",
        "is_plausible_kelp_domain",
        "domain_mask_reason",
        "depth_bin",
    ]
    return pd.read_parquet(path, columns=columns)


def load_kelpwatch_positive_rows(mask_config: DomainMaskConfig) -> pd.DataFrame:
    """Load positive Kelpwatch full-grid rows for retention QA."""
    if not mask_config.full_grid_table.exists():
        msg = f"full-grid table does not exist: {mask_config.full_grid_table}"
        raise FileNotFoundError(msg)
    dataset = pds.dataset(mask_config.full_grid_table)  # type: ignore[no-untyped-call]
    names = set(dataset.schema.names)
    columns = [column for column in full_grid_qa_columns(names) if column in names]
    filter_expression = positive_full_grid_filter(mask_config, names)
    table = dataset.to_table(columns=columns, filter=filter_expression)
    frame = table.to_pandas()
    if frame.empty:
        return empty_positive_frame()
    return frame


def full_grid_qa_columns(names: set[str]) -> list[str]:
    """Return available full-grid columns needed for Kelpwatch-positive QA."""
    columns = ["year", "aef_grid_cell_id", "aef_grid_row", "aef_grid_col", "label_source"]
    for optional_column in ("kelp_max_y", "kelp_fraction_y", "kelp_present_gt0_y"):
        if optional_column in names:
            columns.append(optional_column)
    return columns


def positive_full_grid_filter(mask_config: DomainMaskConfig, names: set[str]) -> Any:
    """Build a dataset filter for positive Kelpwatch rows and optional fast scope."""
    filter_expression = positive_label_expression(names)
    year_scope = (
        mask_config.fast_years if mask_config.fast and mask_config.fast_years else mask_config.years
    )
    if year_scope:
        filter_expression = filter_expression & year_filter_expression(year_scope)
    if mask_config.fast and mask_config.row_window is not None and "aef_grid_row" in names:
        filter_expression = (
            filter_expression
            & (dataset_field("aef_grid_row") >= mask_config.row_window[0])
            & (dataset_field("aef_grid_row") < mask_config.row_window[1])
        )
    if mask_config.fast and mask_config.col_window is not None and "aef_grid_col" in names:
        filter_expression = (
            filter_expression
            & (dataset_field("aef_grid_col") >= mask_config.col_window[0])
            & (dataset_field("aef_grid_col") < mask_config.col_window[1])
        )
    return filter_expression


def positive_label_expression(names: set[str]) -> Any:
    """Return the best available expression for Kelpwatch-positive rows."""
    if "kelp_present_gt0_y" in names:
        return dataset_field("kelp_present_gt0_y") == True  # noqa: E712
    if "kelp_max_y" in names:
        return dataset_field("kelp_max_y") > 0
    if "kelp_fraction_y" in names:
        return dataset_field("kelp_fraction_y") > 0
    msg = "full-grid table must contain kelp_present_gt0_y, kelp_max_y, or kelp_fraction_y"
    raise ValueError(msg)


def year_filter_expression(years: tuple[int, ...]) -> Any:
    """Build an OR expression for configured full-grid years."""
    expression = dataset_field("year") == years[0]
    for year in years[1:]:
        expression = expression | (dataset_field("year") == year)
    return expression


def empty_positive_frame() -> pd.DataFrame:
    """Return an empty positive-row frame with the expected core columns."""
    return pd.DataFrame(
        {
            "year": pd.Series(dtype="int64"),
            "aef_grid_cell_id": pd.Series(dtype="int64"),
            "label_source": pd.Series(dtype="object"),
        }
    )


def merge_positive_rows(positive_rows: pd.DataFrame, mask_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach mask status and depth bins to positive Kelpwatch rows."""
    mask_lookup = mask_frame[
        [
            "aef_grid_cell_id",
            "aef_grid_row",
            "aef_grid_col",
            "is_plausible_kelp_domain",
            "domain_mask_reason",
            "depth_bin",
        ]
    ].copy()
    merged = positive_rows.merge(
        mask_lookup, on="aef_grid_cell_id", how="left", suffixes=("", "_mask")
    )
    missing = merged["is_plausible_kelp_domain"].isna()
    if bool(missing.any()):
        merged.loc[missing, "is_plausible_kelp_domain"] = False
        merged.loc[missing, "domain_mask_reason"] = MASK_MISSING
        merged.loc[missing, "depth_bin"] = MASK_MISSING
    merged["is_plausible_kelp_domain"] = merged["is_plausible_kelp_domain"].astype(bool)
    return merged


def build_kelpwatch_retention_rows(
    positive_rows: pd.DataFrame,
    mask_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Summarize retained and dropped Kelpwatch-positive rows by year and reason."""
    if positive_rows.empty:
        return []
    merged = merge_positive_rows(positive_rows, mask_frame)
    rows: list[dict[str, object]] = []
    for (year, label_source), group in merged.groupby(["year", "label_source"], dropna=False):
        rows.append(retention_row(year, label_source, "all", group))
        for reason, reason_group in group.groupby("domain_mask_reason", dropna=False):
            rows.append(retention_row(year, label_source, str(reason), reason_group))
    return rows


def retention_row(
    year: object,
    label_source: object,
    reason: str,
    group: pd.DataFrame,
) -> dict[str, object]:
    """Build one Kelpwatch-positive retention summary row."""
    retained = group["is_plausible_kelp_domain"].to_numpy(dtype=bool)
    total = int(len(group))
    retained_count = int(np.sum(retained))
    return {
        "year": year,
        "label_source": label_source,
        "domain_mask_reason": reason,
        "positive_cell_year_rows": total,
        "positive_unique_cells": int(group["aef_grid_cell_id"].nunique()),
        "retained_positive_cell_year_rows": retained_count,
        "dropped_positive_cell_year_rows": total - retained_count,
        "retained_positive_fraction": float(retained_count / total) if total else None,
    }


def build_depth_bin_summary_rows(
    mask_frame: pd.DataFrame,
    positive_rows: pd.DataFrame,
) -> list[dict[str, object]]:
    """Summarize mask coverage and positive Kelpwatch rows by depth bin."""
    total_cells = max(len(mask_frame), 1)
    positive_counts = positive_counts_by_depth_bin(positive_rows, mask_frame)
    rows: list[dict[str, object]] = []
    grouped = mask_frame.groupby(["depth_bin", "is_plausible_kelp_domain"], dropna=False)
    for (depth_bin, plausible), group in grouped:
        key = (str(depth_bin), bool(plausible))
        positive_row_count, positive_unique_count = positive_counts.get(key, (0, 0))
        rows.append(
            {
                "depth_bin": str(depth_bin),
                "is_plausible_kelp_domain": bool(plausible),
                "cell_count": int(len(group)),
                "cell_fraction": float(len(group) / total_cells),
                "kelpwatch_positive_cell_year_rows": positive_row_count,
                "kelpwatch_positive_unique_cells": positive_unique_count,
            }
        )
    return sorted(rows, key=lambda row: depth_bin_sort_key(str(row["depth_bin"])))


def positive_counts_by_depth_bin(
    positive_rows: pd.DataFrame,
    mask_frame: pd.DataFrame,
) -> dict[tuple[str, bool], tuple[int, int]]:
    """Count positive Kelpwatch rows and cells by mask depth bin."""
    if positive_rows.empty:
        return {}
    merged = merge_positive_rows(positive_rows, mask_frame)
    counts: dict[tuple[str, bool], tuple[int, int]] = {}
    for (depth_bin, plausible), group in merged.groupby(
        ["depth_bin", "is_plausible_kelp_domain"], dropna=False
    ):
        counts[(str(depth_bin), bool(plausible))] = (
            int(len(group)),
            int(group["aef_grid_cell_id"].nunique()),
        )
    return counts


def depth_bin_sort_key(depth_bin: str) -> tuple[int, str]:
    """Return stable sort keys with configured depth bins first."""
    if depth_bin in DEPTH_BIN_ORDER:
        return DEPTH_BIN_ORDER.index(depth_bin), depth_bin
    return len(DEPTH_BIN_ORDER), depth_bin


def build_mask_manifest(
    *,
    mask_config: DomainMaskConfig,
    accumulator: DomainMaskAccumulator,
    retention_rows: list[dict[str, object]],
    depth_bin_rows: list[dict[str, object]],
    schema: pa.Schema,
) -> dict[str, Any]:
    """Build the JSON manifest for the plausible-kelp domain mask."""
    retained_positive_rows = sum(
        int(cast(Any, row["retained_positive_cell_year_rows"]))
        for row in retention_rows
        if row["domain_mask_reason"] == "all"
    )
    total_positive_rows = sum(
        int(cast(Any, row["positive_cell_year_rows"]))
        for row in retention_rows
        if row["domain_mask_reason"] == "all"
    )
    return {
        "command": "build-domain-mask",
        "config_path": str(mask_config.config_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "fast": mask_config.fast,
        "overwrite_policy": "replace_existing_outputs",
        "region_name": mask_config.region_name,
        "mask_version": MASK_VERSION,
        "rule_set": RULE_SET,
        "reason_precedence": list(REASON_ORDER),
        "missing_crm_policy": MISSING_CRM_POLICY,
        "thresholds": thresholds_manifest(mask_config.thresholds),
        "fast_scope": {
            "years": list(mask_config.fast_years) if mask_config.fast else [],
            "row_window": mask_config.row_window,
            "col_window": mask_config.col_window,
        },
        "inputs": {
            "aligned_crm_table": str(mask_config.aligned_crm_table),
            "aligned_crm_manifest": str(mask_config.aligned_crm_manifest),
            "full_grid_table": str(mask_config.full_grid_table),
            "full_grid_manifest": str(mask_config.full_grid_manifest)
            if mask_config.full_grid_manifest is not None
            else None,
            "cusp_source_manifest": str(mask_config.cusp_source_manifest)
            if mask_config.cusp_source_manifest is not None
            else None,
        },
        "input_manifest_summaries": input_manifest_summaries(mask_config),
        "row_counts": {
            "mask_cells": accumulator.total_cells,
            "retained_cells": accumulator.retained_cells,
            "dropped_cells": accumulator.total_cells - accumulator.retained_cells,
            "kelpwatch_positive_cell_year_rows": total_positive_rows,
            "retained_kelpwatch_positive_cell_year_rows": retained_positive_rows,
        },
        "reason_counts": dict(sorted(accumulator.reason_counts.items())),
        "crm_value_status_counts": dict(sorted(accumulator.crm_status_counts.items())),
        "depth_bin_rows": depth_bin_rows,
        "outputs": {
            "table": str(mask_config.output_table),
            "manifest": str(mask_config.output_manifest),
            "coverage_summary_table": str(mask_config.coverage_summary_table),
            "kelpwatch_retention_table": str(mask_config.kelpwatch_retention_table),
            "depth_bin_summary_table": str(mask_config.depth_bin_summary_table),
            "visual_qa_figure": str(mask_config.visual_qa_figure)
            if mask_config.visual_qa_figure is not None
            else None,
        },
        "output_schema": schema_for_manifest(schema),
        "downstream_scope": {
            "applied_to_predictions": False,
            "applied_to_training": False,
            "deferred_tasks": ["P1-13", "P1-14"],
        },
    }


def thresholds_manifest(thresholds: DomainMaskThresholds) -> dict[str, float | list[float]]:
    """Convert threshold settings to manifest-friendly values."""
    return {
        "max_depth_m": thresholds.max_depth_m,
        "definite_land_elevation_m": thresholds.definite_land_elevation_m,
        "ambiguous_coast_elevation_band_m": [
            thresholds.ambiguous_coast_elevation_min_m,
            thresholds.ambiguous_coast_elevation_max_m,
        ],
        "nearshore_shallow_depth_m": thresholds.nearshore_shallow_depth_m,
        "intermediate_depth_m": thresholds.intermediate_depth_m,
    }


def input_manifest_summaries(mask_config: DomainMaskConfig) -> dict[str, Any]:
    """Load compact provenance summaries from the configured input manifests."""
    return {
        "aligned_crm": optional_manifest_summary(mask_config.aligned_crm_manifest),
        "full_grid": optional_manifest_summary(mask_config.full_grid_manifest),
        "noaa_cusp": optional_manifest_summary(mask_config.cusp_source_manifest),
    }


def optional_manifest_summary(path: Path | None) -> dict[str, Any] | None:
    """Return a compact manifest summary when a manifest path exists."""
    if path is None or not path.exists():
        return None
    manifest = load_json_object(path, f"manifest {path}")
    summary: dict[str, Any] = {}
    for key in (
        "command",
        "created_at",
        "fast",
        "record_count",
        "coverage_counts",
        "target_grid",
        "qa_source_status",
        "source",
    ):
        if key in manifest:
            summary[key] = manifest[key]
    return summary


def write_visual_qa_figure(
    mask_frame: pd.DataFrame,
    positive_rows: pd.DataFrame,
    mask_config: DomainMaskConfig,
) -> None:
    """Write a three-panel visual QA figure for the domain mask."""
    if mask_config.visual_qa_figure is None:
        return
    mask_config.visual_qa_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    plot_reason_grid(cast(Any, axes[0]), mask_frame)
    plot_elevation_grid(cast(Any, axes[1]), mask_frame)
    plot_positive_overlay(cast(Any, axes[2]), mask_frame, positive_rows)
    fig.suptitle("Plausible Kelp Domain Mask QA", fontsize=14)
    fig.savefig(mask_config.visual_qa_figure, dpi=180)
    plt.close(fig)
    LOGGER.info("Wrote plausible-kelp mask visual QA figure: %s", mask_config.visual_qa_figure)


def plot_reason_grid(axis: Any, mask_frame: pd.DataFrame) -> None:
    """Draw the categorical mask reason raster."""
    reason_labels = [
        "unfilled",
        RETAINED_DEPTH_0_100M,
        RETAINED_AMBIGUOUS_COAST,
        DROPPED_DEEP_WATER,
        DROPPED_DEFINITE_LAND,
        QA_MISSING_CRM,
    ]
    reason_values = {
        RETAINED_DEPTH_0_100M: 1,
        RETAINED_AMBIGUOUS_COAST: 2,
        DROPPED_DEEP_WATER: 3,
        DROPPED_DEFINITE_LAND: 4,
        QA_MISSING_CRM: 5,
    }
    grid, extent = grid_from_values(
        mask_frame,
        mask_frame["domain_mask_reason"].map(reason_values).fillna(0).to_numpy(dtype=np.int16),
        fill_value=0,
    )
    cmap = ListedColormap(["#f2f2f2", "#2a9d8f", "#f4a261", "#264653", "#8d99ae", "#d62828"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    image = axis.imshow(grid, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    colorbar = plt.colorbar(image, ax=axis, ticks=np.arange(0, 6), shrink=0.78)
    colorbar.ax.set_yticklabels(reason_labels)
    axis.set_title("Mask reason")
    format_grid_axis(axis, extent)


def plot_elevation_grid(axis: Any, mask_frame: pd.DataFrame) -> None:
    """Draw clipped CRM elevation for shoreline/depth context."""
    elevation = mask_frame["crm_elevation_m"].to_numpy(dtype=np.float32)
    clipped = np.clip(elevation, -120.0, 30.0)
    grid, extent = grid_from_values(mask_frame, clipped, fill_value=np.nan)
    image = axis.imshow(grid, origin="upper", cmap="terrain", interpolation="nearest")
    colorbar = plt.colorbar(image, ax=axis, shrink=0.78)
    colorbar.set_label("CRM elevation m, clipped -120 to 30")
    axis.set_title("CRM elevation context")
    format_grid_axis(axis, extent)


def plot_positive_overlay(
    axis: Any,
    mask_frame: pd.DataFrame,
    positive_rows: pd.DataFrame,
) -> None:
    """Draw Kelpwatch-positive cells over a retained/dropped background."""
    background_values = mask_frame["is_plausible_kelp_domain"].to_numpy(dtype=np.int16)
    grid, extent = grid_from_values(mask_frame, background_values, fill_value=0)
    cmap = ListedColormap(["#d9d9d9", "#edf8f5"])
    axis.imshow(grid, origin="upper", cmap=cmap, interpolation="nearest")
    if not positive_rows.empty:
        merged = merge_positive_rows(positive_rows, mask_frame).drop_duplicates("aef_grid_cell_id")
        min_row, _, min_col, _ = extent
        retained = merged["is_plausible_kelp_domain"].to_numpy(dtype=bool)
        axis.scatter(
            merged.loc[retained, "aef_grid_col_mask"] - min_col,
            merged.loc[retained, "aef_grid_row_mask"] - min_row,
            s=3,
            c="#007f5f",
            alpha=0.5,
            linewidths=0,
            label="retained positive",
        )
        axis.scatter(
            merged.loc[~retained, "aef_grid_col_mask"] - min_col,
            merged.loc[~retained, "aef_grid_row_mask"] - min_row,
            s=6,
            c="#d00000",
            alpha=0.75,
            linewidths=0,
            label="dropped positive",
        )
        axis.legend(loc="lower right", frameon=True, fontsize=8)
    axis.set_title("Kelpwatch-positive overlay")
    format_grid_axis(axis, extent)


def grid_from_values(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    fill_value: float | int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rasterize target-grid values into a dense row/column array."""
    min_row = int(frame["aef_grid_row"].min())
    max_row = int(frame["aef_grid_row"].max())
    min_col = int(frame["aef_grid_col"].min())
    max_col = int(frame["aef_grid_col"].max())
    shape = (max_row - min_row + 1, max_col - min_col + 1)
    grid = np.full(shape, fill_value, dtype=values.dtype)
    rows = frame["aef_grid_row"].to_numpy(dtype=np.int64) - min_row
    cols = frame["aef_grid_col"].to_numpy(dtype=np.int64) - min_col
    grid[rows, cols] = values
    return grid, (min_row, max_row, min_col, max_col)


def format_grid_axis(axis: Any, extent: tuple[int, int, int, int]) -> None:
    """Apply common row/column labels to a visual QA axis."""
    min_row, max_row, min_col, max_col = extent
    axis.set_xlabel(f"AEF grid column offset from {min_col}")
    axis.set_ylabel(f"AEF grid row offset from {min_row}")
    axis.set_aspect("equal")
    axis.text(
        0.01,
        0.01,
        f"rows {min_row}-{max_row}, cols {min_col}-{max_col}",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
    )
