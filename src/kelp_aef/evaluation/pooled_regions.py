"""Build pooled region samples and summarize training-regime comparisons."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.evaluation.baselines import parse_bands
from kelp_aef.evaluation.hurdle import (
    TRAINING_REGIME_COMPARISON_FIELDS,
    path_metadata,
    training_regime_model_family,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMN = "kelp_fraction_y"
DEFAULT_TARGET_AREA_COLUMN = "kelp_max_y"
DEFAULT_SOURCE_REGION_COLUMN = "source_region"
DEFAULT_POSITIVE_THRESHOLD_FRACTION = 0.10
SUMMARY_FIELDS = (
    "source_region",
    "split",
    "year",
    "label_source",
    "row_count",
    "kelpwatch_observed_count",
    "positive_ge_10pct_count",
    "positive_ge_10pct_rate",
    "observed_canopy_area",
    "sample_weight_sum",
)


@dataclass(frozen=True)
class PooledRegionInput:
    """One configured source-region sample for pooled fitting."""

    name: str
    config_path: Path | None
    input_table_path: Path


@dataclass(frozen=True)
class PooledRegionSampleConfig:
    """Resolved paths and settings for the pooled sample builder."""

    config_path: Path
    source_region_column: str
    regions: tuple[PooledRegionInput, ...]
    output_table_path: Path
    split_manifest_path: Path
    manifest_path: Path
    summary_table_path: Path
    feature_columns: tuple[str, ...]
    target_column: str
    target_area_column: str
    positive_threshold_fraction: float
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]


@dataclass(frozen=True)
class ComparisonInput:
    """One comparison CSV to include in the combined cross-regime table."""

    name: str
    path: Path
    training_regime: str
    model_origin_region: str
    evaluation_region: str


@dataclass(frozen=True)
class TrainingRegimeComparisonConfig:
    """Resolved paths and inputs for the combined comparison writer."""

    config_path: Path
    inputs: tuple[ComparisonInput, ...]
    model_comparison_path: Path
    primary_summary_path: Path
    manifest_path: Path
    primary_split: str
    primary_year: int
    primary_mask_status: str
    primary_evaluation_scope: str
    primary_label_source: str


def build_pooled_region_sample(config_path: Path) -> int:
    """Build a pooled Monterey+Big Sur model-input sample from configured inputs."""
    pooled_config = load_pooled_region_sample_config(config_path)
    frames = [read_region_sample(region, pooled_config) for region in pooled_config.regions]
    pooled = pd.concat(frames, ignore_index=True, sort=False)
    split_manifest = pooled_split_manifest(pooled, pooled_config)
    summary_rows = pooled_summary_rows(pooled, split_manifest, pooled_config)
    write_parquet(pooled, pooled_config.output_table_path)
    write_parquet(split_manifest, pooled_config.split_manifest_path)
    write_csv_rows(summary_rows, pooled_config.summary_table_path, SUMMARY_FIELDS)
    write_pooled_manifest(pooled, split_manifest, summary_rows, pooled_config)
    LOGGER.info("Wrote pooled sample table: %s", pooled_config.output_table_path)
    LOGGER.info("Wrote pooled split manifest: %s", pooled_config.split_manifest_path)
    return 0


def write_training_regime_comparison(config_path: Path) -> int:
    """Write the combined Monterey/Big Sur training-regime comparison table."""
    comparison_config = load_training_regime_comparison_config(config_path)
    rows: list[dict[str, object]] = []
    for input_config in comparison_config.inputs:
        input_rows = read_comparison_input(input_config)
        rows.extend(input_rows)
        LOGGER.info("Read %s comparison rows from %s", len(input_rows), input_config.path)
    primary_rows = primary_comparison_rows(rows, comparison_config)
    write_csv_rows(rows, comparison_config.model_comparison_path, TRAINING_REGIME_COMPARISON_FIELDS)
    write_csv_rows(
        primary_rows,
        comparison_config.primary_summary_path,
        TRAINING_REGIME_COMPARISON_FIELDS,
    )
    write_training_regime_comparison_manifest(rows, primary_rows, comparison_config)
    LOGGER.info(
        "Wrote combined training-regime comparison: %s",
        comparison_config.model_comparison_path,
    )
    LOGGER.info("Wrote combined primary summary: %s", comparison_config.primary_summary_path)
    return 0


def load_pooled_region_sample_config(config_path: Path) -> PooledRegionSampleConfig:
    """Load pooled sample input and output paths from a workflow config."""
    config = load_yaml_config(config_path)
    pooled = require_mapping(config.get("pooled_region_sample"), "pooled_region_sample")
    regions = require_mapping(pooled.get("regions"), "pooled_region_sample.regions")
    features = require_mapping(config.get("features"), "features")
    splits = require_mapping(config.get("splits"), "splits")
    models = require_mapping(config.get("models"), "models")
    baselines = require_mapping(models.get("baselines"), "models.baselines")
    source_region_column = str(pooled.get("source_region_column", DEFAULT_SOURCE_REGION_COLUMN))
    return PooledRegionSampleConfig(
        config_path=config_path,
        source_region_column=source_region_column,
        regions=tuple(
            pooled_region_input(name, value, config_path)
            for name, value in sorted(regions.items(), key=lambda item: str(item[0]))
        ),
        output_table_path=config_path_value(pooled, "output_table", config_path),
        split_manifest_path=config_path_value(pooled, "split_manifest", config_path),
        manifest_path=config_path_value(pooled, "manifest", config_path),
        summary_table_path=config_path_value(pooled, "summary_table", config_path),
        feature_columns=parse_bands(features.get("bands")),
        target_column=str(baselines.get("target", DEFAULT_TARGET_COLUMN)),
        target_area_column=str(pooled.get("target_area_column", DEFAULT_TARGET_AREA_COLUMN)),
        positive_threshold_fraction=float(
            pooled.get("positive_threshold_fraction", DEFAULT_POSITIVE_THRESHOLD_FRACTION)
        ),
        train_years=read_year_list(splits, "train_years"),
        validation_years=read_year_list(splits, "validation_years"),
        test_years=read_year_list(splits, "test_years"),
    )


def pooled_region_input(
    name: object,
    value: object,
    config_path: Path,
) -> PooledRegionInput:
    """Load one pooled source-region input entry."""
    region_name = str(name)
    region = require_mapping(value, f"pooled_region_sample.regions.{region_name}")
    config_value = region.get("config")
    return PooledRegionInput(
        name=region_name,
        config_path=(
            resolve_config_path(config_value, config_path) if config_value is not None else None
        ),
        input_table_path=config_path_value(region, "input_table", config_path),
    )


def read_region_sample(
    region: PooledRegionInput,
    pooled_config: PooledRegionSampleConfig,
) -> pd.DataFrame:
    """Read one region sample and attach its source-region provenance."""
    frame = pd.read_parquet(region.input_table_path)
    validate_region_sample(frame, region, pooled_config)
    output = frame.copy()
    output[pooled_config.source_region_column] = region.name
    return output


def validate_region_sample(
    dataframe: pd.DataFrame,
    region: PooledRegionInput,
    pooled_config: PooledRegionSampleConfig,
) -> None:
    """Validate that one region sample has the pooled model columns."""
    required = [
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        pooled_config.target_column,
        pooled_config.target_area_column,
        *pooled_config.feature_columns,
    ]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        msg = f"pooled input {region.name} is missing columns: {missing}"
        raise ValueError(msg)


def pooled_split_manifest(
    pooled: pd.DataFrame,
    pooled_config: PooledRegionSampleConfig,
) -> pd.DataFrame:
    """Build the canonical pooled split manifest across all source regions."""
    columns = split_manifest_columns(pooled, pooled_config)
    manifest = pooled.loc[:, columns].copy()
    manifest["split"] = assign_splits(manifest["year"], pooled_config)
    feature_complete = pooled.loc[:, list(pooled_config.feature_columns)].notna().all(axis=1)
    target_complete = pooled[pooled_config.target_column].notna()
    manifest["has_complete_features"] = feature_complete.to_numpy(dtype=bool)
    manifest["has_target"] = target_complete.to_numpy(dtype=bool)
    manifest["used_for_training_eval"] = manifest["has_complete_features"] & manifest["has_target"]
    manifest["drop_reason"] = pooled_drop_reasons(manifest)
    return manifest


def split_manifest_columns(
    pooled: pd.DataFrame,
    pooled_config: PooledRegionSampleConfig,
) -> list[str]:
    """Return stable identity, target, and audit columns for the split manifest."""
    columns = [
        pooled_config.source_region_column,
        "year",
        "kelpwatch_station_id",
        "longitude",
        "latitude",
        pooled_config.target_column,
    ]
    optional = (
        "aef_grid_cell_id",
        "aef_grid_row",
        "aef_grid_col",
        "label_source",
        "is_kelpwatch_observed",
        "kelpwatch_station_count",
        "sample_weight",
        "is_plausible_kelp_domain",
        "domain_mask_reason",
        "domain_mask_detail",
        "domain_mask_version",
        "crm_elevation_m",
        "crm_depth_m",
        "depth_bin",
        "elevation_bin",
    )
    for column in optional:
        if column in pooled.columns and column not in columns:
            columns.append(column)
    return columns


def assign_splits(years: pd.Series, pooled_config: PooledRegionSampleConfig) -> pd.Series:
    """Assign year-based train, validation, and test labels."""
    split_by_year = {
        **{year: "train" for year in pooled_config.train_years},
        **{year: "validation" for year in pooled_config.validation_years},
        **{year: "test" for year in pooled_config.test_years},
    }
    split = years.astype(int).map(split_by_year)
    if split.isna().any():
        missing = sorted(set(int(year) for year in years.loc[split.isna()].unique()))
        msg = f"pooled sample rows contain years not assigned to a split: {missing}"
        raise ValueError(msg)
    return split.astype(str)


def pooled_drop_reasons(manifest: pd.DataFrame) -> pd.Series:
    """Build drop reasons for the pooled split manifest."""
    reasons = pd.Series(index=manifest.index, data="", dtype="object")
    reasons.loc[~manifest["has_complete_features"]] = "missing_features"
    missing_target = ~manifest["has_target"]
    reasons.loc[missing_target & (reasons != "")] += ";missing_target"
    reasons.loc[missing_target & (reasons == "")] = "missing_target"
    return reasons


def pooled_summary_rows(
    pooled: pd.DataFrame,
    split_manifest: pd.DataFrame,
    pooled_config: PooledRegionSampleConfig,
) -> list[dict[str, object]]:
    """Summarize the pooled sample by source region, split, year, and label source."""
    frame = pooled.copy()
    frame["split"] = split_manifest["split"].to_numpy(dtype=object)
    if "label_source" not in frame.columns:
        frame["label_source"] = "unknown"
    rows: list[dict[str, object]] = []
    group_columns = [pooled_config.source_region_column, "split", "year", "label_source"]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        source_region, split, year, label_source = cast(tuple[object, object, object, object], keys)
        rows.append(
            pooled_summary_row(
                group,
                source_region,
                split,
                year,
                label_source,
                pooled_config,
            )
        )
    all_group_columns = [pooled_config.source_region_column, "split", "year"]
    for keys, group in frame.groupby(all_group_columns, sort=True, dropna=False):
        source_region, split, year = cast(tuple[object, object, object], keys)
        rows.append(pooled_summary_row(group, source_region, split, year, "all", pooled_config))
    return rows


def pooled_summary_row(
    group: pd.DataFrame,
    source_region: object,
    split: object,
    year: object,
    label_source: object,
    pooled_config: PooledRegionSampleConfig,
) -> dict[str, object]:
    """Build one pooled sample summary row."""
    target = group[pooled_config.target_column].astype(float)
    positive = target >= pooled_config.positive_threshold_fraction
    observed = (
        group["is_kelpwatch_observed"].fillna(False).astype(bool)
        if "is_kelpwatch_observed" in group.columns
        else pd.Series(False, index=group.index)
    )
    sample_weight_sum = (
        float(group["sample_weight"].sum()) if "sample_weight" in group.columns else np.nan
    )
    return {
        "source_region": source_region,
        "split": split,
        "year": year,
        "label_source": label_source,
        "row_count": int(len(group)),
        "kelpwatch_observed_count": int(observed.sum()),
        "positive_ge_10pct_count": int(positive.sum()),
        "positive_ge_10pct_rate": safe_ratio(float(positive.sum()), float(len(group))),
        "observed_canopy_area": float(group[pooled_config.target_area_column].sum()),
        "sample_weight_sum": sample_weight_sum,
    }


def write_pooled_manifest(
    pooled: pd.DataFrame,
    split_manifest: pd.DataFrame,
    summary_rows: list[dict[str, object]],
    pooled_config: PooledRegionSampleConfig,
) -> None:
    """Write a JSON manifest for pooled sample construction."""
    payload = {
        "command": "build-pooled-region-sample",
        "config_path": str(pooled_config.config_path),
        "source_region_column": pooled_config.source_region_column,
        "target_column": pooled_config.target_column,
        "target_area_column": pooled_config.target_area_column,
        "positive_threshold_fraction": pooled_config.positive_threshold_fraction,
        "feature_columns": list(pooled_config.feature_columns),
        "split_years": {
            "train": list(pooled_config.train_years),
            "validation": list(pooled_config.validation_years),
            "test": list(pooled_config.test_years),
        },
        "inputs": [
            {
                "source_region": region.name,
                "config_path": str(region.config_path) if region.config_path is not None else None,
                "input_table": path_metadata(region.input_table_path),
            }
            for region in pooled_config.regions
        ],
        "outputs": {
            "output_table": path_metadata(pooled_config.output_table_path),
            "split_manifest": path_metadata(pooled_config.split_manifest_path),
            "summary_table": path_metadata(pooled_config.summary_table_path),
            "manifest": str(pooled_config.manifest_path),
        },
        "row_counts": {
            "pooled_rows": int(len(pooled)),
            "split_manifest_rows": int(len(split_manifest)),
            "summary_rows": len(summary_rows),
            "by_source_region": {
                str(key): int(value)
                for key, value in pooled[pooled_config.source_region_column].value_counts().items()
            },
        },
        "region_metadata_used_as_predictor": False,
        "test_rows_used_for_training_calibration_or_threshold_selection": False,
    }
    write_json(pooled_config.manifest_path, payload)


def load_training_regime_comparison_config(
    config_path: Path,
) -> TrainingRegimeComparisonConfig:
    """Load combined training-regime comparison inputs from config."""
    config = load_yaml_config(config_path)
    comparison = require_mapping(
        config.get("training_regime_comparison"),
        "training_regime_comparison",
    )
    inputs = require_mapping(
        comparison.get("inputs"),
        "training_regime_comparison.inputs",
    )
    return TrainingRegimeComparisonConfig(
        config_path=config_path,
        inputs=tuple(
            comparison_input(name, value, config_path)
            for name, value in sorted(inputs.items(), key=lambda item: str(item[0]))
        ),
        model_comparison_path=config_path_value(comparison, "model_comparison", config_path),
        primary_summary_path=config_path_value(comparison, "primary_summary", config_path),
        manifest_path=config_path_value(comparison, "manifest", config_path),
        primary_split=str(comparison.get("primary_split", "test")),
        primary_year=int(comparison.get("primary_year", 2022)),
        primary_mask_status=str(comparison.get("primary_mask_status", "plausible_kelp_domain")),
        primary_evaluation_scope=str(
            comparison.get("primary_evaluation_scope", "full_grid_masked")
        ),
        primary_label_source=str(comparison.get("primary_label_source", "all")),
    )


def comparison_input(name: object, value: object, config_path: Path) -> ComparisonInput:
    """Load one comparison input table and canonical provenance labels."""
    input_name = str(name)
    entry = require_mapping(value, f"training_regime_comparison.inputs.{input_name}")
    return ComparisonInput(
        name=input_name,
        path=config_path_value(entry, "path", config_path),
        training_regime=require_string(
            entry.get("training_regime"),
            f"training_regime_comparison.inputs.{input_name}.training_regime",
        ),
        model_origin_region=require_string(
            entry.get("model_origin_region"),
            f"training_regime_comparison.inputs.{input_name}.model_origin_region",
        ),
        evaluation_region=require_string(
            entry.get("evaluation_region"),
            f"training_regime_comparison.inputs.{input_name}.evaluation_region",
        ),
    )


def read_comparison_input(input_config: ComparisonInput) -> list[dict[str, object]]:
    """Read and normalize one comparison input CSV."""
    if not input_config.path.exists():
        msg = f"training-regime comparison input does not exist: {input_config.path}"
        raise FileNotFoundError(msg)
    frame = pd.read_csv(input_config.path)
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        row_dict = cast(dict[str, object], row)
        rows.append(normalized_comparison_row(row_dict, input_config))
    return rows


def normalized_comparison_row(
    row: dict[str, object],
    input_config: ComparisonInput,
) -> dict[str, object]:
    """Convert one raw row into the canonical cross-regime comparison schema."""
    model_name = str(row.get("model_name", ""))
    return {
        "training_regime": input_config.training_regime,
        "model_origin_region": input_config.model_origin_region,
        "evaluation_region": input_config.evaluation_region,
        "model_name": model_name,
        "model_family": row.get("model_family") or training_regime_model_family(model_name),
        "composition_policy": row.get("composition_policy", ""),
        "split": row.get("split", ""),
        "year": row.get("year", ""),
        "mask_status": row.get("mask_status", ""),
        "evaluation_scope": row.get("evaluation_scope", ""),
        "label_source": row.get("label_source", ""),
        "row_count": row.get("row_count", ""),
        "mae": row.get("mae", np.nan),
        "rmse": row.get("rmse", np.nan),
        "r2": row.get("r2", np.nan),
        "f1_ge_10pct": row.get("f1_ge_10pct", np.nan),
        "observed_canopy_area": row.get("observed_canopy_area", np.nan),
        "predicted_canopy_area": row.get("predicted_canopy_area", np.nan),
        "area_pct_bias": row.get("area_pct_bias", np.nan),
        "source_table": str(input_config.path),
    }


def primary_comparison_rows(
    rows: list[dict[str, object]],
    comparison_config: TrainingRegimeComparisonConfig,
) -> list[dict[str, object]]:
    """Return the held-out retained-domain rows for the combined primary table."""
    output = []
    for row in rows:
        if (
            str(row.get("split")) == comparison_config.primary_split
            and str(row.get("year")) == str(comparison_config.primary_year)
            and str(row.get("mask_status")) == comparison_config.primary_mask_status
            and str(row.get("evaluation_scope")) == comparison_config.primary_evaluation_scope
            and str(row.get("label_source")) == comparison_config.primary_label_source
        ):
            output.append(row)
    return output


def write_training_regime_comparison_manifest(
    rows: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
    comparison_config: TrainingRegimeComparisonConfig,
) -> None:
    """Write a manifest for the combined cross-regime comparison outputs."""
    payload = {
        "command": "compare-training-regimes",
        "config_path": str(comparison_config.config_path),
        "canonical_training_regimes": [
            "monterey_only",
            "big_sur_only",
            "pooled_monterey_big_sur",
        ],
        "canonical_model_origin_regions": ["monterey", "big_sur", "monterey_big_sur"],
        "canonical_evaluation_regions": ["monterey", "big_sur"],
        "primary_filters": {
            "split": comparison_config.primary_split,
            "year": comparison_config.primary_year,
            "mask_status": comparison_config.primary_mask_status,
            "evaluation_scope": comparison_config.primary_evaluation_scope,
            "label_source": comparison_config.primary_label_source,
        },
        "inputs": [
            {
                "name": input_config.name,
                "path": path_metadata(input_config.path),
                "training_regime": input_config.training_regime,
                "model_origin_region": input_config.model_origin_region,
                "evaluation_region": input_config.evaluation_region,
            }
            for input_config in comparison_config.inputs
        ],
        "outputs": {
            "model_comparison": path_metadata(comparison_config.model_comparison_path),
            "primary_summary": path_metadata(comparison_config.primary_summary_path),
            "manifest": str(comparison_config.manifest_path),
        },
        "row_counts": {
            "comparison_rows": len(rows),
            "primary_summary_rows": len(primary_rows),
        },
        "raw_transfer_rows_normalized": True,
        "test_rows_used_for_training_calibration_or_threshold_selection": False,
    }
    write_json(comparison_config.manifest_path, payload)


def read_year_list(config: dict[str, Any], key: str) -> tuple[int, ...]:
    """Read a non-empty year list from a config mapping."""
    values = config.get(key)
    if not isinstance(values, list) or not values:
        msg = f"config field must be a non-empty list of years: splits.{key}"
        raise ValueError(msg)
    return tuple(int(value) for value in values)


def config_path_value(config: dict[str, Any], key: str, config_path: Path) -> Path:
    """Read a path config value, resolving relative paths against the config file."""
    value = require_string(config.get(key), key)
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent / path


def resolve_config_path(value: object, config_path: Path) -> Path:
    """Resolve an optional nested config path."""
    text = require_string(value, "config")
    path = Path(text)
    if path.is_absolute():
        return path
    return config_path.parent / path


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or NaN when the denominator is zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def write_parquet(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a Parquet file after ensuring its parent exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def write_csv_rows(
    rows: list[dict[str, object]],
    output_path: Path,
    fields: tuple[str, ...],
) -> None:
    """Write dictionaries to CSV with a stable header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON object with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
