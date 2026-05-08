"""Map baseline predictions, residuals, and area-bias summaries."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import html
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
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.dataset as ds
from shapely.geometry import MultiPolygon, Polygon

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.evaluation.baselines import (
    percent_bias,
    r2_score,
    root_mean_squared_error,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "ridge_regression"
DEFAULT_MAP_SPLIT = "test"
DEFAULT_MAP_YEAR = 2022
DEFAULT_LATITUDE_BAND_COUNT = 12
DEFAULT_TOP_RESIDUAL_COUNT = 50
DEFAULT_INTERACTIVE_MAX_ROWS = 50_000
REQUIRED_PREDICTION_COLUMNS = (
    "year",
    "split",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_fraction_y",
    "kelp_max_y",
    "model_name",
    "pred_kelp_fraction_y",
    "pred_kelp_fraction_y_clipped",
    "pred_kelp_max_y",
    "residual_kelp_fraction_y",
    "residual_kelp_max_y",
)
OPTIONAL_PREDICTION_COLUMNS = (
    "aef_grid_cell_id",
    "aef_grid_row",
    "aef_grid_col",
    "label_source",
    "is_kelpwatch_observed",
    "kelpwatch_station_count",
)
AREA_BIAS_YEAR_FIELDS = (
    "model_name",
    "split",
    "year",
    "row_count",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mae_fraction",
    "rmse_fraction",
    "r2_fraction",
)
AREA_BIAS_LATITUDE_FIELDS = (
    "model_name",
    "split",
    "year",
    "latitude_band",
    "latitude_min",
    "latitude_max",
    "row_count",
    "observed_canopy_area",
    "predicted_canopy_area",
    "area_bias",
    "area_pct_bias",
    "mae_fraction",
    "rmse_fraction",
)
TOP_RESIDUAL_FIELDS = (
    "residual_type",
    "rank",
    "model_name",
    "split",
    "year",
    "kelpwatch_station_id",
    "longitude",
    "latitude",
    "kelp_max_y",
    "pred_kelp_max_y",
    "residual_kelp_max_y",
    "abs_residual_kelp_max_y",
)


@dataclass(frozen=True)
class ResidualMapConfig:
    """Resolved config values for residual maps and area-bias QA."""

    config_path: Path
    predictions_path: Path
    metrics_path: Path
    footprint_path: Path
    model_name: str
    map_split: str
    map_year: int
    latitude_band_count: int
    top_residual_count: int
    static_map_path: Path
    scatter_path: Path
    interactive_html_path: Path
    area_bias_by_year_path: Path
    area_bias_by_latitude_path: Path
    top_residuals_path: Path
    manifest_path: Path


@dataclass
class AreaAccumulator:
    """Streaming sums for one model/split/year area-bias row."""

    row_count: int = 0
    metric_count: int = 0
    observed_canopy_area: float = 0.0
    predicted_canopy_area: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    observed_fraction_sum: float = 0.0
    observed_fraction_squared_sum: float = 0.0


def map_residuals(config_path: Path) -> int:
    """Write prediction maps, residual figures, and area-bias QA tables."""
    map_config = load_residual_map_config(config_path)
    LOGGER.info("Loading baseline predictions: %s", map_config.predictions_path)
    map_rows = read_selected_prediction_rows(map_config)
    validate_predictions(map_rows)
    if map_rows.empty:
        msg = f"no predictions found for model: {map_config.model_name}"
        raise ValueError(msg)
    footprint = read_footprint(map_config.footprint_path)
    area_by_year = area_bias_by_year_from_predictions(map_config)
    area_by_latitude = area_bias_by_latitude_band(map_rows, map_config.latitude_band_count)
    top_residuals = top_residual_stations(map_rows, map_config.top_residual_count)

    write_static_map(map_rows, footprint, map_config.static_map_path, map_config)
    write_observed_vs_predicted(map_rows, map_config.scatter_path, map_config)
    write_interactive_html(map_rows, footprint, map_config.interactive_html_path, map_config)
    write_csv(area_by_year, map_config.area_bias_by_year_path, AREA_BIAS_YEAR_FIELDS)
    write_csv(area_by_latitude, map_config.area_bias_by_latitude_path, AREA_BIAS_LATITUDE_FIELDS)
    write_csv(top_residuals, map_config.top_residuals_path, TOP_RESIDUAL_FIELDS)
    write_manifest(
        map_config=map_config,
        map_rows=map_rows,
        area_by_year=area_by_year,
        area_by_latitude=area_by_latitude,
        top_residuals=top_residuals,
    )
    LOGGER.info("Wrote residual static map: %s", map_config.static_map_path)
    LOGGER.info("Wrote observed-vs-predicted figure: %s", map_config.scatter_path)
    LOGGER.info("Wrote residual interactive HTML: %s", map_config.interactive_html_path)
    LOGGER.info("Wrote area bias by year: %s", map_config.area_bias_by_year_path)
    LOGGER.info("Wrote area bias by latitude band: %s", map_config.area_bias_by_latitude_path)
    LOGGER.info("Wrote top residual stations: %s", map_config.top_residuals_path)
    LOGGER.info("Wrote map residuals manifest: %s", map_config.manifest_path)
    return 0


def read_selected_prediction_rows(map_config: ResidualMapConfig) -> pd.DataFrame:
    """Read prediction rows for the configured model, split, and year."""
    columns = prediction_columns(map_config.predictions_path)
    if not map_config.predictions_path.is_dir():
        predictions = pd.read_parquet(map_config.predictions_path, columns=columns)
        model_predictions = predictions.loc[
            predictions["model_name"] == map_config.model_name
        ].copy()
        return selected_map_rows(model_predictions, map_config)
    dataset = ds.dataset(map_config.predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    available_columns = set(dataset.schema.names)
    selected_columns = [column for column in columns if column in available_columns]
    expression = (
        (dataset_field("model_name") == map_config.model_name)
        & (dataset_field("split") == map_config.map_split)
        & (dataset_field("year") == map_config.map_year)
    )
    table = dataset.to_table(columns=selected_columns, filter=expression)
    selected = table.to_pandas()
    LOGGER.info(
        "Selected %s map rows for model=%s split=%s year=%s",
        len(selected),
        map_config.model_name,
        map_config.map_split,
        map_config.map_year,
    )
    return cast(pd.DataFrame, selected)


def prediction_columns(path: Path) -> list[str]:
    """Return required and optional prediction columns available at a path."""
    columns = [*REQUIRED_PREDICTION_COLUMNS, *OPTIONAL_PREDICTION_COLUMNS]
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    available_columns = set(dataset.schema.names)
    return [column for column in columns if column in available_columns]


def dataset_field(name: str) -> Any:
    """Return a PyArrow dataset field expression with a typed wrapper."""
    return cast(Any, ds).field(name)


def load_residual_map_config(config_path: Path) -> ResidualMapConfig:
    """Load residual map settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    models = require_mapping(config.get("models"), "models")
    baselines = require_mapping(models.get("baselines"), "models.baselines")
    reports = require_mapping(config.get("reports"), "reports")
    outputs = require_mapping(reports.get("outputs"), "reports.outputs")
    map_settings = optional_mapping(reports.get("map_residuals"), "reports.map_residuals")
    figures_dir = Path(require_string(reports.get("figures_dir"), "reports.figures_dir"))
    tables_dir = Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))
    return ResidualMapConfig(
        config_path=config_path,
        predictions_path=Path(
            require_string(baselines.get("predictions"), "models.baselines.predictions")
        ),
        metrics_path=Path(require_string(baselines.get("metrics"), "models.baselines.metrics")),
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        model_name=str(map_settings.get("model_name", DEFAULT_MODEL_NAME)),
        map_split=str(map_settings.get("split", DEFAULT_MAP_SPLIT)),
        map_year=optional_int(
            map_settings.get("year"),
            "reports.map_residuals.year",
            DEFAULT_MAP_YEAR,
        ),
        latitude_band_count=optional_positive_int(
            map_settings.get("latitude_band_count"),
            "reports.map_residuals.latitude_band_count",
            DEFAULT_LATITUDE_BAND_COUNT,
        ),
        top_residual_count=optional_positive_int(
            map_settings.get("top_residual_count"),
            "reports.map_residuals.top_residual_count",
            DEFAULT_TOP_RESIDUAL_COUNT,
        ),
        static_map_path=output_path(
            outputs,
            "ridge_observed_predicted_residual_map",
            figures_dir / "ridge_2022_observed_predicted_residual.png",
        ),
        scatter_path=output_path(
            outputs,
            "ridge_observed_vs_predicted",
            figures_dir / "ridge_observed_vs_predicted.png",
        ),
        interactive_html_path=output_path(
            outputs,
            "ridge_residual_interactive",
            figures_dir / "ridge_2022_residual_interactive.html",
        ),
        area_bias_by_year_path=output_path(
            outputs,
            "area_bias_by_year",
            tables_dir / "area_bias_by_year.csv",
        ),
        area_bias_by_latitude_path=output_path(
            outputs,
            "area_bias_by_latitude_band",
            tables_dir / "area_bias_by_latitude_band.csv",
        ),
        top_residuals_path=output_path(
            outputs,
            "top_residual_stations",
            tables_dir / "top_residual_stations.csv",
        ),
        manifest_path=output_path(
            outputs,
            "map_residuals_manifest",
            Path(require_string(config.get("data_root"), "data_root"))
            / "interim/map_residuals_manifest.json",
        ),
    )


def optional_mapping(value: object, name: str) -> dict[str, Any]:
    """Return an optional config mapping, treating missing values as empty."""
    if value is None:
        return {}
    return require_mapping(value, name)


def output_path(outputs: dict[str, Any], key: str, default: Path) -> Path:
    """Read an optional report output path from config."""
    value = outputs.get(key)
    if value is None:
        return default
    return Path(require_string(value, f"reports.outputs.{key}"))


def validate_predictions(dataframe: pd.DataFrame) -> None:
    """Validate that prediction rows contain the columns needed for mapping."""
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in dataframe.columns]
    if missing:
        msg = f"prediction table is missing required columns: {missing}"
        raise ValueError(msg)


def selected_map_rows(dataframe: pd.DataFrame, map_config: ResidualMapConfig) -> pd.DataFrame:
    """Select one model/split/year for spatial map QA."""
    selected = dataframe.loc[
        (dataframe["split"] == map_config.map_split) & (dataframe["year"] == map_config.map_year)
    ].copy()
    if selected.empty:
        msg = (
            "no prediction rows found for "
            f"model={map_config.model_name}, "
            f"split={map_config.map_split}, "
            f"year={map_config.map_year}"
        )
        raise ValueError(msg)
    LOGGER.info(
        "Selected %s map rows for model=%s split=%s year=%s",
        len(selected),
        map_config.model_name,
        map_config.map_split,
        map_config.map_year,
    )
    return selected


def read_footprint(path: Path) -> Polygon | MultiPolygon | None:
    """Read the configured footprint geometry for optional plot outlines."""
    if not path.exists():
        LOGGER.warning("Footprint path does not exist; maps will omit outline: %s", path)
        return None
    dataframe = gpd.read_file(path)
    if dataframe.empty:
        return None
    geometry = dataframe.to_crs("EPSG:4326").geometry.iloc[0]
    if isinstance(geometry, Polygon | MultiPolygon):
        return geometry
    return None


def area_bias_by_year(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Build area-bias summary rows grouped by model, split, and year."""
    rows: list[dict[str, object]] = []
    for keys, group in dataframe.groupby(["model_name", "split", "year"], sort=True):
        model_name, split, year = cast(tuple[str, str, int], keys)
        rows.append(
            area_metric_row(
                group,
                model_name=str(model_name),
                split=str(split),
                year=int(year),
            )
        )
    return rows


def area_bias_by_year_from_predictions(
    map_config: ResidualMapConfig,
) -> list[dict[str, object]]:
    """Build year-level area-bias rows, streaming directory datasets when needed."""
    if not map_config.predictions_path.is_dir():
        predictions = pd.read_parquet(
            map_config.predictions_path,
            columns=prediction_columns(map_config.predictions_path),
        )
        model_predictions = predictions.loc[
            predictions["model_name"] == map_config.model_name
        ].copy()
        return area_bias_by_year(model_predictions)
    dataset = ds.dataset(map_config.predictions_path, format="parquet")  # type: ignore[no-untyped-call]
    columns = [
        "model_name",
        "split",
        "year",
        "kelp_fraction_y",
        "pred_kelp_fraction_y_clipped",
        "kelp_max_y",
        "pred_kelp_max_y",
    ]
    available_columns = set(dataset.schema.names)
    missing = [column for column in columns if column not in available_columns]
    if missing:
        msg = f"prediction dataset is missing required area-bias columns: {missing}"
        raise ValueError(msg)
    expression = dataset_field("model_name") == map_config.model_name
    accumulators: dict[tuple[str, str, int], AreaAccumulator] = {}
    for batch in dataset.to_batches(columns=columns, filter=expression, batch_size=100_000):
        batch_frame = batch.to_pandas()
        update_area_accumulators(accumulators, batch_frame)
    rows = [
        area_metric_row_from_accumulator(model_name, split, year, accumulator)
        for (model_name, split, year), accumulator in sorted(accumulators.items())
    ]
    LOGGER.info("Built %s streamed area-bias-by-year rows", len(rows))
    return rows


def update_area_accumulators(
    accumulators: dict[tuple[str, str, int], AreaAccumulator],
    dataframe: pd.DataFrame,
) -> None:
    """Update streaming area-bias sums from one prediction batch."""
    for keys, group in dataframe.groupby(["model_name", "split", "year"], sort=True):
        model_name, split, year = cast(tuple[str, str, int], keys)
        key = (str(model_name), str(split), int(year))
        accumulator = accumulators.setdefault(key, AreaAccumulator())
        observed_fraction = group["kelp_fraction_y"].to_numpy(dtype=float)
        predicted_fraction = group["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
        observed_area = group["kelp_max_y"].to_numpy(dtype=float)
        predicted_area = group["pred_kelp_max_y"].to_numpy(dtype=float)
        finite_mask = np.isfinite(observed_fraction) & np.isfinite(predicted_fraction)
        residual = observed_fraction[finite_mask] - predicted_fraction[finite_mask]
        accumulator.row_count += int(len(group))
        accumulator.metric_count += int(np.count_nonzero(finite_mask))
        accumulator.observed_canopy_area += float(np.nansum(observed_area))
        accumulator.predicted_canopy_area += float(np.nansum(predicted_area))
        accumulator.absolute_error_sum += float(np.nansum(np.abs(residual)))
        accumulator.squared_error_sum += float(np.nansum(residual**2))
        accumulator.observed_fraction_sum += float(np.nansum(observed_fraction[finite_mask]))
        accumulator.observed_fraction_squared_sum += float(
            np.nansum(observed_fraction[finite_mask] ** 2)
        )


def area_metric_row_from_accumulator(
    model_name: str,
    split: str,
    year: int,
    accumulator: AreaAccumulator,
) -> dict[str, object]:
    """Build one area-bias row from streaming accumulator state."""
    return {
        "model_name": model_name,
        "split": split,
        "year": year,
        "row_count": accumulator.row_count,
        "observed_canopy_area": accumulator.observed_canopy_area,
        "predicted_canopy_area": accumulator.predicted_canopy_area,
        "area_bias": accumulator.predicted_canopy_area - accumulator.observed_canopy_area,
        "area_pct_bias": percent_bias(
            accumulator.predicted_canopy_area,
            accumulator.observed_canopy_area,
        ),
        "mae_fraction": safe_divide(
            accumulator.absolute_error_sum,
            accumulator.metric_count,
        ),
        "rmse_fraction": math.sqrt(
            safe_divide(accumulator.squared_error_sum, accumulator.metric_count)
        ),
        "r2_fraction": r2_from_accumulator(accumulator),
    }


def safe_divide(numerator: float, denominator: int) -> float:
    """Divide two values, returning NaN for zero denominators."""
    if denominator == 0:
        return math.nan
    return numerator / denominator


def r2_from_accumulator(accumulator: AreaAccumulator) -> float:
    """Compute R2 from streaming sufficient statistics."""
    if accumulator.metric_count == 0:
        return math.nan
    total_sum_squares = accumulator.observed_fraction_squared_sum - (
        accumulator.observed_fraction_sum**2 / accumulator.metric_count
    )
    if total_sum_squares == 0:
        return math.nan
    return 1.0 - accumulator.squared_error_sum / total_sum_squares


def area_bias_by_latitude_band(
    dataframe: pd.DataFrame, latitude_band_count: int
) -> list[dict[str, object]]:
    """Build area-bias summaries within deterministic latitude bands."""
    banded = dataframe.copy()
    banded["latitude_band_index"] = latitude_band_indices(
        banded["latitude"].to_numpy(dtype=float),
        latitude_band_count,
    )
    rows: list[dict[str, object]] = []
    for keys, group in banded.groupby(
        ["model_name", "split", "year", "latitude_band_index"], sort=True
    ):
        model_name, split, year, band_index = cast(tuple[str, str, int, int], keys)
        row = area_metric_row(group, model_name=str(model_name), split=str(split), year=int(year))
        latitude_min = float(group["latitude"].min())
        latitude_max = float(group["latitude"].max())
        row.update(
            {
                "latitude_band": f"{int(band_index):02d}",
                "latitude_min": latitude_min,
                "latitude_max": latitude_max,
            }
        )
        rows.append(row)
    return [{field: row.get(field, "") for field in AREA_BIAS_LATITUDE_FIELDS} for row in rows]


def latitude_band_indices(latitudes: np.ndarray, band_count: int) -> np.ndarray:
    """Assign latitudes to equal-width deterministic band indices."""
    minimum = float(np.nanmin(latitudes))
    maximum = float(np.nanmax(latitudes))
    if maximum == minimum:
        return np.zeros(latitudes.shape, dtype=int)
    scaled = (latitudes - minimum) / (maximum - minimum)
    indices = np.floor(scaled * band_count).astype(int)
    return cast(np.ndarray, np.clip(indices, 0, band_count - 1))


def area_metric_row(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    year: int,
) -> dict[str, object]:
    """Build one area-bias and regression-metric summary row."""
    observed_fraction = dataframe["kelp_fraction_y"].to_numpy(dtype=float)
    predicted_fraction = dataframe["pred_kelp_fraction_y_clipped"].to_numpy(dtype=float)
    observed_area = float(np.nansum(dataframe["kelp_max_y"].to_numpy(dtype=float)))
    predicted_area = float(np.nansum(dataframe["pred_kelp_max_y"].to_numpy(dtype=float)))
    return {
        "model_name": model_name,
        "split": split,
        "year": year,
        "row_count": int(len(dataframe)),
        "observed_canopy_area": observed_area,
        "predicted_canopy_area": predicted_area,
        "area_bias": predicted_area - observed_area,
        "area_pct_bias": percent_bias(predicted_area, observed_area),
        "mae_fraction": mean_absolute_error(observed_fraction, predicted_fraction),
        "rmse_fraction": root_mean_squared_error(observed_fraction, predicted_fraction),
        "r2_fraction": r2_score(observed_fraction, predicted_fraction),
    }


def mean_absolute_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.nanmean(np.abs(observed - predicted)))


def top_residual_stations(dataframe: pd.DataFrame, top_count: int) -> list[dict[str, object]]:
    """Select largest underprediction and overprediction residual stations."""
    underpredicted = residual_rows(
        dataframe.loc[dataframe["residual_kelp_max_y"] > 0],
        residual_type="underprediction",
        ascending=False,
        top_count=top_count,
    )
    overpredicted = residual_rows(
        dataframe.loc[dataframe["residual_kelp_max_y"] < 0],
        residual_type="overprediction",
        ascending=True,
        top_count=top_count,
    )
    return underpredicted + overpredicted


def residual_rows(
    dataframe: pd.DataFrame,
    *,
    residual_type: str,
    ascending: bool,
    top_count: int,
) -> list[dict[str, object]]:
    """Build ranked residual rows for one residual sign."""
    sorted_rows = dataframe.sort_values("residual_kelp_max_y", ascending=ascending).head(top_count)
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(sorted_rows.to_dict("records"), start=1):
        rows.append(
            {
                "residual_type": residual_type,
                "rank": rank,
                "model_name": row["model_name"],
                "split": row["split"],
                "year": int(row["year"]),
                "kelpwatch_station_id": nullable_int(row.get("kelpwatch_station_id")),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
                "kelp_max_y": float(row["kelp_max_y"]),
                "pred_kelp_max_y": float(row["pred_kelp_max_y"]),
                "residual_kelp_max_y": float(row["residual_kelp_max_y"]),
                "abs_residual_kelp_max_y": abs(float(row["residual_kelp_max_y"])),
            }
        )
    return rows


def nullable_int(value: object) -> int | None:
    """Convert a nullable numeric value to int or None."""
    if pd.isna(value):
        return None
    if not isinstance(value, int | float | np.integer | np.floating):
        return None
    return int(value)


def write_static_map(
    dataframe: pd.DataFrame,
    footprint: Polygon | MultiPolygon | None,
    output_path: Path,
    map_config: ResidualMapConfig,
) -> None:
    """Write a three-panel observed, predicted, and residual static map."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    observed = dataframe["kelp_max_y"].to_numpy(dtype=float)
    predicted = dataframe["pred_kelp_max_y"].to_numpy(dtype=float)
    residual = dataframe["residual_kelp_max_y"].to_numpy(dtype=float)
    canopy_vmax = safe_percentile(np.concatenate([observed, predicted]), 99)
    residual_limit = max(abs(safe_percentile(np.abs(residual), 99)), 1.0)
    canopy_norm = Normalize(vmin=0.0, vmax=canopy_vmax)
    residual_norm = TwoSlopeNorm(vcenter=0.0, vmin=-residual_limit, vmax=residual_limit)
    plot_point_map(
        axes[0],
        dataframe,
        observed,
        title=f"Observed Kelpwatch {map_config.map_year}",
        cmap="viridis",
        norm=canopy_norm,
        footprint=footprint,
    )
    plot_point_map(
        axes[1],
        dataframe,
        predicted,
        title=f"Ridge predicted {map_config.map_year}",
        cmap="viridis",
        norm=canopy_norm,
        footprint=footprint,
    )
    plot_point_map(
        axes[2],
        dataframe,
        residual,
        title="Residual observed - predicted",
        cmap="RdBu_r",
        norm=residual_norm,
        footprint=footprint,
    )
    fig.suptitle(f"{map_config.model_name} | {map_config.map_split} | {map_config.map_year}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_point_map(
    axis: Any,
    dataframe: pd.DataFrame,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    norm: Normalize | TwoSlopeNorm,
    footprint: Polygon | MultiPolygon | None,
) -> None:
    """Draw one lon/lat point map panel."""
    if {"aef_grid_row", "aef_grid_col"}.issubset(dataframe.columns):
        plot_grid_map(
            axis,
            dataframe,
            values,
            title=title,
            cmap=cmap,
            norm=norm,
            footprint=footprint,
        )
        return
    scatter = axis.scatter(
        dataframe["longitude"],
        dataframe["latitude"],
        c=values,
        s=4,
        linewidths=0,
        alpha=0.85,
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    if footprint is not None:
        plot_geometry_outline(axis, footprint)
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    set_point_bounds(axis, dataframe)
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_aspect("equal", adjustable="box")
    plt.colorbar(scatter, ax=axis, shrink=0.8)


def plot_grid_map(
    axis: Any,
    dataframe: pd.DataFrame,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    norm: Normalize | TwoSlopeNorm,
    footprint: Polygon | MultiPolygon | None,
) -> None:
    """Draw one rasterized target-grid map panel from row/column ids."""
    rows = dataframe["aef_grid_row"].to_numpy(dtype=int)
    cols = dataframe["aef_grid_col"].to_numpy(dtype=int)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    image = np.full((row_max - row_min + 1, col_max - col_min + 1), np.nan, dtype=float)
    image[rows - row_min, cols - col_min] = values
    extent = grid_extent(dataframe)
    artist = axis.imshow(
        image,
        origin="upper",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    if footprint is not None:
        plot_geometry_outline(axis, footprint)
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_aspect("equal", adjustable="box")
    plt.colorbar(artist, ax=axis, shrink=0.8)


def grid_extent(dataframe: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return approximate lon/lat extent for a regular mapped grid."""
    longitudes = dataframe["longitude"].to_numpy(dtype=float)
    latitudes = dataframe["latitude"].to_numpy(dtype=float)
    return (
        float(np.nanmin(longitudes)),
        float(np.nanmax(longitudes)),
        float(np.nanmin(latitudes)),
        float(np.nanmax(latitudes)),
    )


def set_point_bounds(axis: Any, dataframe: pd.DataFrame) -> None:
    """Set map axes from point bounds so broad footprints do not dominate."""
    longitudes = dataframe["longitude"].to_numpy(dtype=float)
    latitudes = dataframe["latitude"].to_numpy(dtype=float)
    longitude_min = float(np.nanmin(longitudes))
    longitude_max = float(np.nanmax(longitudes))
    latitude_min = float(np.nanmin(latitudes))
    latitude_max = float(np.nanmax(latitudes))
    longitude_pad = (longitude_max - longitude_min) * 0.05 or 0.01
    latitude_pad = (latitude_max - latitude_min) * 0.05 or 0.01
    axis.set_xlim(longitude_min - longitude_pad, longitude_max + longitude_pad)
    axis.set_ylim(latitude_min - latitude_pad, latitude_max + latitude_pad)


def plot_geometry_outline(axis: Any, geometry: Polygon | MultiPolygon) -> None:
    """Draw a footprint polygon or multipolygon outline."""
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    for polygon in polygons:
        x_values, y_values = polygon.exterior.xy
        axis.plot(x_values, y_values, color="black", linewidth=0.8)


def write_observed_vs_predicted(
    dataframe: pd.DataFrame,
    output_path: Path,
    map_config: ResidualMapConfig,
) -> None:
    """Write observed-vs-predicted hexbin QA panels by split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    vmax = safe_percentile(
        dataframe[["kelp_max_y", "pred_kelp_max_y"]].to_numpy(dtype=float).ravel(),
        99,
    )
    for axis, split in zip(axes, ("train", "validation", "test"), strict=True):
        split_rows = dataframe.loc[dataframe["split"] == split]
        if split_rows.empty:
            axis.set_axis_off()
            continue
        axis.hexbin(
            split_rows["kelp_max_y"],
            split_rows["pred_kelp_max_y"],
            gridsize=45,
            extent=(0.0, vmax, 0.0, vmax),
            mincnt=1,
            cmap="magma",
        )
        axis.plot([0, vmax], [0, vmax], color="white", linewidth=1.0)
        axis.plot([0, vmax], [0, vmax], color="black", linewidth=0.4)
        axis.set_title(split)
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
    fig.suptitle(f"Observed vs predicted canopy area | {map_config.model_name}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_interactive_html(
    dataframe: pd.DataFrame,
    footprint: Polygon | MultiPolygon | None,
    output_path: Path,
    map_config: ResidualMapConfig,
) -> None:
    """Write a self-contained side-by-side HTML explorer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_interactive_html(dataframe, footprint, map_config))


def build_interactive_html(
    dataframe: pd.DataFrame,
    footprint: Polygon | MultiPolygon | None,
    map_config: ResidualMapConfig,
) -> str:
    """Build the HTML document for linked observed/predicted/residual exploration."""
    records = interactive_records(dataframe)
    footprint_records = footprint_coordinate_records(footprint)
    payload = {
        "records": records,
        "footprint": footprint_records,
        "meta": {
            "model": map_config.model_name,
            "split": map_config.map_split,
            "year": map_config.map_year,
            "residual_sign": "observed minus predicted",
        },
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Kelp AEF Residual Explorer</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f7f4; color: #1f2933; }}
header {{ padding: 14px 18px; border-bottom: 1px solid #d8d8d2; background: white; }}
h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
.subhead {{ margin-top: 4px; font-size: 13px; color: #52606d; }}
.layout {{ display: grid; grid-template-columns: 1fr 300px; gap: 0; min-height: calc(100vh - 62px); }}
.panels {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 10px; padding: 12px; }}
.panel {{ background: white; border: 1px solid #d8d8d2; border-radius: 6px; overflow: hidden; }}
.panel-title {{ padding: 8px 10px; font-size: 13px; font-weight: 650; border-bottom: 1px solid #e6e6df; }}
canvas {{ display: block; width: 100%; height: 660px; cursor: crosshair; }}
aside {{ background: white; border-left: 1px solid #d8d8d2; padding: 14px; }}
.metric {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; padding: 7px 0; border-bottom: 1px solid #ecece6; font-size: 13px; }}
.metric span:first-child {{ color: #52606d; }}
.metric span:last-child {{ font-variant-numeric: tabular-nums; text-align: right; }}
.legend {{ margin-top: 18px; font-size: 12px; color: #52606d; line-height: 1.45; }}
</style>
</head>
<body>
<header>
<h1>Kelpwatch labels, ridge predictions, and residuals</h1>
<div class="subhead">{html.escape(map_config.model_name)} | {html.escape(map_config.map_split)} | {map_config.map_year} | residual = observed - predicted</div>
</header>
<div class="layout">
<main class="panels">
<section class="panel"><div class="panel-title">Observed canopy area</div><canvas id="observed"></canvas></section>
<section class="panel"><div class="panel-title">Predicted canopy area</div><canvas id="predicted"></canvas></section>
<section class="panel"><div class="panel-title">Residual</div><canvas id="residual"></canvas></section>
</main>
<aside>
<h2 style="font-size:15px;margin:0 0 10px;">Selected station</h2>
<div id="details"><div class="metric"><span>Station</span><span>-</span></div></div>
<div class="legend">
Observed and predicted panels share a canopy color scale. Residual colors are blue for overprediction and red for underprediction.
</div>
</aside>
</div>
<script>
const PAYLOAD = {json.dumps(payload, separators=(",", ":"))};
const DATA = PAYLOAD.records;
const FOOTPRINT = PAYLOAD.footprint;
const PANELS = [
  {{ id: "observed", field: "observed", mode: "canopy" }},
  {{ id: "predicted", field: "predicted", mode: "canopy" }},
  {{ id: "residual", field: "residual", mode: "residual" }}
];
let selectedIndex = -1;
const bounds = computeBounds(DATA);
const canopyMax = percentile(DATA.flatMap(d => [d.observed, d.predicted]), 0.99);
const residualMax = Math.max(1, percentile(DATA.map(d => Math.abs(d.residual)), 0.99));
function computeBounds(data) {{
  const xs = data.map(d => d.lon), ys = data.map(d => d.lat);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX) * 0.05 || 0.01, padY = (maxY - minY) * 0.05 || 0.01;
  return {{ minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }};
}}
function percentile(values, q) {{
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 1;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))] || 1;
}}
function resizeCanvas(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
}}
function project(d, canvas) {{
  const x = (d.lon - bounds.minX) / (bounds.maxX - bounds.minX) * canvas.width;
  const y = canvas.height - (d.lat - bounds.minY) / (bounds.maxY - bounds.minY) * canvas.height;
  return [x, y];
}}
function colorCanopy(value) {{
  const t = Math.max(0, Math.min(1, value / canopyMax));
  const r = Math.round(252 - 216 * t);
  const g = Math.round(251 - 95 * t);
  const b = Math.round(244 - 205 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function colorResidual(value) {{
  const t = Math.max(-1, Math.min(1, value / residualMax));
  if (t < 0) {{
    const a = -t;
    return `rgb(${{Math.round(245 - 178*a)}},${{Math.round(247 - 100*a)}},${{Math.round(250 - 33*a)}})`;
  }}
  return `rgb(${{Math.round(245 - 65*t)}},${{Math.round(247 - 204*t)}},${{Math.round(250 - 180*t)}})`;
}}
function drawFootprint(ctx, canvas) {{
  ctx.save();
  ctx.strokeStyle = "rgba(20,20,20,0.75)";
  ctx.lineWidth = Math.max(1, canvas.width / 900);
  for (const ring of FOOTPRINT) {{
    ctx.beginPath();
    ring.forEach((coord, idx) => {{
      const d = {{ lon: coord[0], lat: coord[1] }};
      const [x, y] = project(d, canvas);
      if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
  ctx.restore();
}}
function drawPanel(panel) {{
  const canvas = document.getElementById(panel.id);
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawFootprint(ctx, canvas);
  const radius = Math.max(1.3, canvas.width / 520);
  DATA.forEach((d, idx) => {{
    const [x, y] = project(d, canvas);
    ctx.fillStyle = panel.mode === "canopy" ? colorCanopy(d[panel.field]) : colorResidual(d[panel.field]);
    ctx.globalAlpha = 0.82;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (idx === selectedIndex) {{
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#111";
      ctx.lineWidth = Math.max(2, canvas.width / 420);
      ctx.stroke();
    }}
  }});
  ctx.globalAlpha = 1;
}}
function redraw() {{ PANELS.forEach(drawPanel); }}
function nearestPoint(event, canvas) {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
  const mx = (event.clientX - rect.left) * scaleX, my = (event.clientY - rect.top) * scaleY;
  let best = -1, bestDist = Infinity;
  DATA.forEach((d, idx) => {{
    const [x, y] = project(d, canvas);
    const dist = (x - mx) ** 2 + (y - my) ** 2;
    if (dist < bestDist) {{ bestDist = dist; best = idx; }}
  }});
  return bestDist < (canvas.width * 0.018) ** 2 ? best : -1;
}}
function setDetails(d) {{
  const rows = [
    ["Station", d.id],
    ["Observed area", d.observed.toFixed(1)],
    ["Predicted area", d.predicted.toFixed(1)],
    ["Residual", d.residual.toFixed(1)],
    ["Observed frac", d.observed_fraction.toFixed(3)],
    ["Predicted frac", d.predicted_fraction.toFixed(3)],
    ["Longitude", d.lon.toFixed(6)],
    ["Latitude", d.lat.toFixed(6)]
  ];
  document.getElementById("details").innerHTML = rows.map(r => `<div class="metric"><span>${{r[0]}}</span><span>${{r[1]}}</span></div>`).join("");
}}
PANELS.forEach(panel => {{
  const canvas = document.getElementById(panel.id);
  canvas.addEventListener("mousemove", event => {{
    const idx = nearestPoint(event, canvas);
    if (idx !== selectedIndex) {{
      selectedIndex = idx;
      if (idx >= 0) setDetails(DATA[idx]);
      redraw();
    }}
  }});
}});
window.addEventListener("resize", redraw);
redraw();
</script>
</body>
</html>
"""


def interactive_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convert selected prediction rows into compact HTML JSON records."""
    if len(dataframe) > DEFAULT_INTERACTIVE_MAX_ROWS:
        dataframe = dataframe.sample(DEFAULT_INTERACTIVE_MAX_ROWS, random_state=13)
    records: list[dict[str, object]] = []
    for row in dataframe.to_dict("records"):
        records.append(
            {
                "id": row_identifier(row),
                "lon": float(row["longitude"]),
                "lat": float(row["latitude"]),
                "observed": float(row["kelp_max_y"]),
                "predicted": float(row["pred_kelp_max_y"]),
                "residual": float(row["residual_kelp_max_y"]),
                "observed_fraction": float(row["kelp_fraction_y"]),
                "predicted_fraction": float(row["pred_kelp_fraction_y_clipped"]),
            }
        )
    return records


def row_identifier(row: dict[str, object]) -> str:
    """Return a display identifier for station or full-grid prediction rows."""
    station_id = row.get("kelpwatch_station_id")
    if not pd.isna(station_id):
        return str(int(cast(float, station_id)))
    cell_id = row.get("aef_grid_cell_id")
    if cell_id is not None and not pd.isna(cell_id):
        return f"cell {int(cast(float, cell_id))}"
    return "background"


def footprint_coordinate_records(
    footprint: Polygon | MultiPolygon | None,
) -> list[list[list[float]]]:
    """Convert footprint geometry outlines to JSON-friendly coordinate rings."""
    if footprint is None:
        return []
    polygons = list(footprint.geoms) if isinstance(footprint, MultiPolygon) else [footprint]
    return [
        [[float(x), float(y)] for x, y in polygon.exterior.coords]
        for polygon in polygons
        if not polygon.is_empty
    ]


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    """Return a positive percentile value for plotting scales."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    value = float(np.nanpercentile(finite, percentile))
    return max(value, 1.0)


def write_csv(rows: list[dict[str, object]], output_path: Path, fields: tuple[str, ...]) -> None:
    """Write rows to CSV with a stable field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    *,
    map_config: ResidualMapConfig,
    map_rows: pd.DataFrame,
    area_by_year: list[dict[str, object]],
    area_by_latitude: list[dict[str, object]],
    top_residuals: list[dict[str, object]],
) -> None:
    """Write a JSON manifest describing map residual QA outputs."""
    payload = {
        "command": "map-residuals",
        "config_path": str(map_config.config_path),
        "predictions": str(map_config.predictions_path),
        "metrics": str(map_config.metrics_path),
        "footprint": str(map_config.footprint_path),
        "model_name": map_config.model_name,
        "split": map_config.map_split,
        "year": map_config.map_year,
        "map_row_count": int(len(map_rows)),
        "residual_sign": "observed minus predicted",
        "latitude_band_count": map_config.latitude_band_count,
        "top_residual_count_per_sign": map_config.top_residual_count,
        "outputs": {
            "static_map": str(map_config.static_map_path),
            "scatter": str(map_config.scatter_path),
            "interactive_html": str(map_config.interactive_html_path),
            "area_bias_by_year": str(map_config.area_bias_by_year_path),
            "area_bias_by_latitude_band": str(map_config.area_bias_by_latitude_path),
            "top_residuals": str(map_config.top_residuals_path),
            "manifest": str(map_config.manifest_path),
        },
        "row_counts": {
            "area_bias_by_year": len(area_by_year),
            "area_bias_by_latitude_band": len(area_by_latitude),
            "top_residuals": len(top_residuals),
        },
    }
    write_json(map_config.manifest_path, payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def optional_int(value: object, name: str, default: int) -> int:
    """Validate an optional integer config value."""
    if value is None:
        return default
    return require_int_value(value, name)


def optional_positive_int(value: object, name: str, default: int) -> int:
    """Validate an optional positive integer config value."""
    if value is None:
        return default
    parsed = require_int_value(value, name)
    if parsed <= 0:
        msg = f"field must be positive: {name}"
        raise ValueError(msg)
    return parsed


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like dynamic value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))
