"""Shared helpers for applying the plausible-kelp domain mask in reports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import polars as pl

from kelp_aef.config import require_mapping, require_string

DEFAULT_MASK_STATUS = "plausible_kelp_domain"
DEFAULT_EVALUATION_SCOPE = "full_grid_masked"
DEFAULT_PRIMARY_DOMAIN = "plausible_kelp_domain"
UNMASKED_STATUS = "unmasked"
UNMASKED_EVALUATION_SCOPE = "full_grid_prediction"
MASK_KEY_COLUMN = "aef_grid_cell_id"
MASK_RETAIN_COLUMN = "is_plausible_kelp_domain"
MASK_DETAIL_COLUMNS = (
    MASK_KEY_COLUMN,
    MASK_RETAIN_COLUMN,
    "domain_mask_reason",
    "domain_mask_detail",
    "domain_mask_version",
    "crm_elevation_m",
    "crm_depth_m",
    "depth_bin",
    "elevation_bin",
)


@dataclass(frozen=True)
class ReportingDomainMask:
    """Resolved plausible-kelp mask settings for reporting-only filtering."""

    table_path: Path
    manifest_path: Path | None
    mask_status: str
    evaluation_scope: str
    primary_domain: str
    off_domain_audit_path: Path | None


def load_reporting_domain_mask(config: dict[str, Any]) -> ReportingDomainMask | None:
    """Load optional report-domain mask settings from a workflow config."""
    reports_value = config.get("reports")
    if reports_value is None:
        return None
    reports = require_mapping(reports_value, "reports")
    mask_value = reports.get("domain_mask")
    if mask_value is None:
        return None
    mask = require_mapping(mask_value, "reports.domain_mask")
    return ReportingDomainMask(
        table_path=Path(require_string(mask.get("mask_table"), "reports.domain_mask.mask_table")),
        manifest_path=optional_path(mask.get("mask_manifest")),
        mask_status=str(mask.get("mask_status", DEFAULT_MASK_STATUS)),
        evaluation_scope=str(mask.get("evaluation_scope", DEFAULT_EVALUATION_SCOPE)),
        primary_domain=str(mask.get("primary_full_grid_domain", DEFAULT_PRIMARY_DOMAIN)),
        off_domain_audit_path=optional_path(mask.get("off_domain_audit_table")),
    )


def optional_path(value: object) -> Path | None:
    """Return an optional filesystem path from a dynamic config value."""
    if value is None:
        return None
    return Path(str(value))


def mask_status(mask_config: ReportingDomainMask | None) -> str:
    """Return the report mask-status label for a full-grid row."""
    if mask_config is None:
        return UNMASKED_STATUS
    return mask_config.mask_status


def evaluation_scope(mask_config: ReportingDomainMask | None) -> str:
    """Return the report evaluation-scope label for a full-grid row."""
    if mask_config is None:
        return UNMASKED_EVALUATION_SCOPE
    return mask_config.evaluation_scope


def masked_output_path(
    outputs: dict[str, Any],
    *,
    unmasked_key: str,
    masked_key: str,
    default: Path,
    mask_config: ReportingDomainMask | None,
) -> Path:
    """Read a masked sidecar output path when reporting mask settings are active."""
    key = (
        masked_key
        if mask_config is not None and outputs.get(masked_key) is not None
        else unmasked_key
    )
    value = outputs.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"reports.outputs.{key}"))


def apply_reporting_domain_mask(
    dataframe: pd.DataFrame,
    mask_config: ReportingDomainMask | None,
    *,
    retain: bool = True,
) -> pd.DataFrame:
    """Join a prediction frame to the mask and keep retained or dropped cells."""
    if mask_config is None:
        return dataframe.copy()
    if MASK_KEY_COLUMN not in dataframe.columns:
        msg = f"reporting domain mask requires prediction column: {MASK_KEY_COLUMN}"
        raise ValueError(msg)
    clean = dataframe.drop(columns=mask_columns_in_frame(dataframe), errors="ignore")
    merged = clean.join(read_mask_lookup(mask_config), on=MASK_KEY_COLUMN, how="left")
    missing = merged[MASK_RETAIN_COLUMN].isna()
    if bool(missing.any()):
        msg = f"reporting domain mask is missing rows for {int(missing.sum())} prediction cells"
        raise ValueError(msg)
    retain_mask = merged[MASK_RETAIN_COLUMN].astype(bool)
    selected = retain_mask if retain else ~retain_mask
    return cast(pd.DataFrame, merged.loc[selected].copy())


@lru_cache(maxsize=8)
def read_mask_frame(mask_config: ReportingDomainMask) -> pd.DataFrame:
    """Read the configured mask columns needed for reporting joins."""
    available_names = set(pl.scan_parquet(str(mask_config.table_path)).collect_schema().names())
    columns = [column for column in MASK_DETAIL_COLUMNS if column in available_names]
    mask = pd.read_parquet(mask_config.table_path, columns=columns)
    missing = [column for column in (MASK_KEY_COLUMN, MASK_RETAIN_COLUMN) if column not in mask]
    if missing:
        msg = f"domain mask table is missing required columns: {missing}"
        raise ValueError(msg)
    columns = [column for column in MASK_DETAIL_COLUMNS if column in mask.columns]
    return cast(pd.DataFrame, mask[columns].copy())


@lru_cache(maxsize=8)
def read_mask_lookup(mask_config: ReportingDomainMask) -> pd.DataFrame:
    """Read mask metadata indexed by target-grid cell id for repeated joins."""
    mask = read_mask_frame(mask_config).set_index(MASK_KEY_COLUMN)
    if not mask.index.is_unique:
        msg = f"domain mask table has duplicate cell ids: {mask_config.table_path}"
        raise ValueError(msg)
    return cast(pd.DataFrame, mask)


def mask_columns_in_frame(dataframe: pd.DataFrame) -> list[str]:
    """Return mask metadata columns already present on a prediction frame."""
    return [
        column
        for column in MASK_DETAIL_COLUMNS
        if column != MASK_KEY_COLUMN and column in dataframe
    ]


def filter_polars_to_reporting_domain(
    dataframe: pl.LazyFrame,
    mask_config: ReportingDomainMask | None,
) -> pl.LazyFrame:
    """Filter a lazy full-grid frame to retained plausible-kelp cells."""
    if mask_config is None:
        return dataframe
    schema_names = set(dataframe.collect_schema().names())
    if MASK_KEY_COLUMN not in schema_names:
        msg = f"reporting domain mask requires inference column: {MASK_KEY_COLUMN}"
        raise ValueError(msg)
    mask = (
        pl.scan_parquet(str(mask_config.table_path))
        .select([MASK_KEY_COLUMN, MASK_RETAIN_COLUMN])
        .filter(pl.col(MASK_RETAIN_COLUMN))
        .select(MASK_KEY_COLUMN)
    )
    return dataframe.join(mask, on=MASK_KEY_COLUMN, how="inner")
