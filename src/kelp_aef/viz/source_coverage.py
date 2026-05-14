"""Early source-coverage visual QA for region smoke configs."""

from __future__ import annotations

import csv
import html
import json
import logging
import operator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, SupportsIndex, cast

import geopandas as gpd  # type: ignore[import-untyped]
import matplotlib
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.regions import region_display_name, region_output_slug

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

LOGGER = logging.getLogger(__name__)

SUMMARY_FIELDS = (
    "region",
    "source",
    "artifact",
    "status",
    "selected_years",
    "missing_years",
    "record_count",
    "local_file_count",
    "missing_local_file_count",
    "positive_count",
    "valid_count",
    "bounds",
    "notes",
    "path",
)

SOURCE_COLORS = {
    "aef": "#2563eb",
    "kelpwatch": "#16a34a",
    "noaa_crm": "#7c3aed",
    "noaa_cudem": "#0891b2",
    "noaa_cusp": "#dc2626",
    "usgs_3dep": "#d97706",
}

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceCoverageConfig:
    """Resolved config values needed by the source-coverage QA step."""

    config_path: Path
    region_name: str
    region_slug: str
    region_display_name: str
    years: tuple[int, ...]
    footprint_path: Path
    footprint_crs: str
    footprint_bbox: Bounds | None
    source_stac_item_id: str | None
    source_crs: str | None
    source_tile_uri: str | None
    aef_catalog_summary: Path
    aef_tile_manifest: Path
    kelpwatch_source_manifest: Path
    kelpwatch_qa_table: Path
    noaa_crm_query_manifest: Path
    noaa_crm_source_manifest: Path
    noaa_cudem_query_manifest: Path
    noaa_cudem_tile_manifest: Path
    noaa_cusp_query_manifest: Path
    noaa_cusp_source_manifest: Path
    usgs_3dep_query_manifest: Path
    usgs_3dep_source_manifest: Path
    summary_table: Path
    coverage_manifest: Path
    figure_path: Path
    html_path: Path


@dataclass(frozen=True)
class SourceLayer:
    """One source bounds layer for source-coverage maps."""

    source: str
    label: str
    bounds: Bounds
    status: str


def load_source_coverage_config(config_path: Path) -> SourceCoverageConfig:
    """Load source-coverage QA settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    years_config = require_mapping(config.get("years"), "years")
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    features = require_mapping(config.get("features"), "features")
    feature_paths = require_mapping(features.get("paths"), "features.paths")
    domain = require_mapping(config.get("domain"), "domain")
    reports = require_mapping(config.get("reports"), "reports")

    region_name = require_string(region.get("name"), "region.name")
    region_slug = region_output_slug(region_name)
    raw_years = years_config.get("smoke")
    if not isinstance(raw_years, list):
        msg = "config field must be a list of years: years.smoke"
        raise ValueError(msg)
    years = tuple(require_int_value(year, "years.smoke[]") for year in raw_years)

    source_manifest = Path(
        require_string(label_paths.get("source_manifest"), "labels.paths.source_manifest")
    )
    figures_dir = Path(require_string(reports.get("figures_dir"), "reports.figures_dir"))
    tables_dir = Path(require_string(reports.get("tables_dir"), "reports.tables_dir"))

    noaa_crm = require_mapping(domain.get("noaa_crm"), "domain.noaa_crm")
    noaa_cudem = require_mapping(domain.get("noaa_cudem"), "domain.noaa_cudem")
    noaa_cusp = require_mapping(domain.get("noaa_cusp"), "domain.noaa_cusp")
    usgs_3dep = require_mapping(domain.get("usgs_3dep"), "domain.usgs_3dep")

    return SourceCoverageConfig(
        config_path=config_path,
        region_name=region_name,
        region_slug=region_slug,
        region_display_name=region_display_name(region_name),
        years=years,
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        footprint_crs=str(region.get("crs", "EPSG:4326")),
        footprint_bbox=optional_bounds(geometry.get("bbox")),
        source_stac_item_id=optional_string(geometry.get("source_stac_item_id")),
        source_crs=optional_string(geometry.get("source_crs")),
        source_tile_uri=optional_string(geometry.get("source_tile_uri")),
        aef_catalog_summary=Path(
            require_string(
                feature_paths.get("catalog_query_summary"),
                "features.paths.catalog_query_summary",
            )
        ),
        aef_tile_manifest=Path(
            require_string(feature_paths.get("tile_manifest"), "features.paths.tile_manifest")
        ),
        kelpwatch_source_manifest=source_manifest,
        kelpwatch_qa_table=tables_dir / f"kelpwatch_{region_slug}_source_qa.csv",
        noaa_crm_query_manifest=Path(
            require_string(noaa_crm.get("query_manifest"), "domain.noaa_crm.query_manifest")
        ),
        noaa_crm_source_manifest=Path(
            require_string(noaa_crm.get("source_manifest"), "domain.noaa_crm.source_manifest")
        ),
        noaa_cudem_query_manifest=Path(
            require_string(
                noaa_cudem.get("query_manifest"),
                "domain.noaa_cudem.query_manifest",
            )
        ),
        noaa_cudem_tile_manifest=Path(
            require_string(noaa_cudem.get("tile_manifest"), "domain.noaa_cudem.tile_manifest")
        ),
        noaa_cusp_query_manifest=Path(
            require_string(noaa_cusp.get("query_manifest"), "domain.noaa_cusp.query_manifest")
        ),
        noaa_cusp_source_manifest=Path(
            require_string(noaa_cusp.get("source_manifest"), "domain.noaa_cusp.source_manifest")
        ),
        usgs_3dep_query_manifest=Path(
            require_string(usgs_3dep.get("query_manifest"), "domain.usgs_3dep.query_manifest")
        ),
        usgs_3dep_source_manifest=Path(
            require_string(usgs_3dep.get("source_manifest"), "domain.usgs_3dep.source_manifest")
        ),
        summary_table=tables_dir / f"{region_slug}_source_coverage_summary.csv",
        coverage_manifest=source_manifest.parent / f"{region_slug}_source_coverage_manifest.json",
        figure_path=figures_dir / f"{region_slug}_source_coverage_qa.png",
        html_path=figures_dir / f"{region_slug}_source_coverage_interactive_qa.html",
    )


def visualize_source_coverage(config_path: Path) -> int:
    """Run the early source-coverage QA step."""
    qa_config = load_source_coverage_config(config_path)
    footprint, footprint_created = ensure_footprint(qa_config)
    summary_rows, source_layers = build_summary_rows(qa_config, footprint, footprint_created)

    write_summary_csv(summary_rows, qa_config.summary_table)
    write_static_figure(
        qa_config=qa_config,
        footprint=footprint,
        source_layers=source_layers,
        summary_rows=summary_rows,
    )
    write_interactive_html(
        qa_config=qa_config,
        footprint=footprint,
        source_layers=source_layers,
        summary_rows=summary_rows,
    )
    write_manifest(
        qa_config=qa_config,
        footprint=footprint,
        footprint_created=footprint_created,
        source_layers=source_layers,
        summary_rows=summary_rows,
    )

    LOGGER.info("Wrote source coverage summary: %s", qa_config.summary_table)
    LOGGER.info("Wrote source coverage figure: %s", qa_config.figure_path)
    LOGGER.info("Wrote source coverage interactive QA: %s", qa_config.html_path)
    LOGGER.info("Wrote source coverage manifest: %s", qa_config.coverage_manifest)
    return 0


def ensure_footprint(qa_config: SourceCoverageConfig) -> tuple[BaseGeometry, bool]:
    """Read the configured footprint, creating it from the config bbox if needed."""
    if not qa_config.footprint_path.exists():
        if qa_config.footprint_bbox is None:
            msg = (
                f"region footprint is missing and no bbox is configured: "
                f"{qa_config.footprint_path}"
            )
            raise FileNotFoundError(msg)
        write_bbox_footprint(qa_config)
        created = True
    else:
        created = False

    dataframe = gpd.read_file(qa_config.footprint_path).to_crs("EPSG:4326")
    if dataframe.empty:
        msg = f"region footprint contains no features: {qa_config.footprint_path}"
        raise ValueError(msg)
    return cast(BaseGeometry, dataframe.geometry.iloc[0]), created


def write_bbox_footprint(qa_config: SourceCoverageConfig) -> None:
    """Materialize a one-feature GeoJSON footprint from the configured bbox."""
    if qa_config.footprint_bbox is None:
        msg = "cannot write a bbox footprint without configured bounds"
        raise ValueError(msg)
    footprint = box(*qa_config.footprint_bbox)
    payload = {
        "type": "FeatureCollection",
        "name": f"{qa_config.region_slug}_aef_footprint",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "region_name": qa_config.region_name,
                    "source_stac_item_id": qa_config.source_stac_item_id,
                    "source_crs": qa_config.source_crs,
                    "source_tile_uri": qa_config.source_tile_uri,
                    "provenance": "configs region.geometry.bbox",
                },
                "geometry": mapping(footprint),
            }
        ],
    }
    qa_config.footprint_path.parent.mkdir(parents=True, exist_ok=True)
    qa_config.footprint_path.write_text(json.dumps(payload, indent=2) + "\n")


def build_summary_rows(
    qa_config: SourceCoverageConfig,
    footprint: BaseGeometry,
    footprint_created: bool,
) -> tuple[list[dict[str, object]], list[SourceLayer]]:
    """Build coverage summary rows and map layers from current artifacts."""
    rows = [
        summary_row(
            qa_config=qa_config,
            source="footprint",
            artifact="configured_region_footprint",
            status="created" if footprint_created else "exists",
            bounds=bounds_from_geometry(footprint),
            notes="Footprint created from config bbox." if footprint_created else "",
            path=qa_config.footprint_path,
        )
    ]
    layers = [
        SourceLayer(
            source="footprint",
            label="Footprint",
            bounds=bounds_from_geometry(footprint),
            status="created" if footprint_created else "exists",
        )
    ]

    aef_catalog_row = aef_catalog_summary_row(qa_config)
    rows.append(aef_catalog_row)
    aef_tile_row, aef_layers = aef_tile_manifest_row(qa_config)
    rows.append(aef_tile_row)
    layers.extend(aef_layers)

    rows.append(kelpwatch_source_row(qa_config))
    rows.append(kelpwatch_positive_row(qa_config))

    for query_path, source_name, selected_key in (
        (qa_config.noaa_crm_query_manifest, "noaa_crm", "selected_products"),
        (qa_config.noaa_cudem_query_manifest, "noaa_cudem", "selected_tiles"),
        (qa_config.noaa_cusp_query_manifest, "noaa_cusp", "selected_artifacts"),
        (qa_config.usgs_3dep_query_manifest, "usgs_3dep", "selected_artifacts"),
    ):
        row, query_layers = query_manifest_row(qa_config, source_name, query_path, selected_key)
        rows.append(row)
        layers.extend(query_layers)

    for source_path, source_name, artifact in (
        (qa_config.noaa_crm_source_manifest, "noaa_crm", "source_manifest"),
        (qa_config.noaa_cudem_tile_manifest, "noaa_cudem", "tile_manifest"),
        (qa_config.noaa_cusp_source_manifest, "noaa_cusp", "source_manifest"),
        (qa_config.usgs_3dep_source_manifest, "usgs_3dep", "source_manifest"),
    ):
        row, source_layers = source_manifest_row(qa_config, source_name, artifact, source_path)
        rows.append(row)
        layers.extend(source_layers)

    return rows, layers


def aef_catalog_summary_row(qa_config: SourceCoverageConfig) -> dict[str, object]:
    """Summarize the configured AEF catalog query summary artifact."""
    data = load_json_object_if_exists(qa_config.aef_catalog_summary)
    if data is None:
        return missing_row(qa_config, "aef", "catalog_query_summary", qa_config.aef_catalog_summary)
    selected_assets = list_from_mapping(data, "selected_assets")
    selected_years = sorted(
        int(asset["year"])
        for asset in selected_assets
        if isinstance(asset, dict) and isinstance(asset.get("year"), int)
    )
    missing_years = missing_year_values(qa_config.years, selected_years)
    status = "complete" if not missing_years else "incomplete"
    return summary_row(
        qa_config=qa_config,
        source="aef",
        artifact="catalog_query_summary",
        status=status,
        selected_years=selected_years,
        missing_years=missing_years,
        record_count=len(selected_assets),
        bounds=optional_bounds(data.get("footprint_bounds")),
        notes=f"raw_spatial_candidate_count={data.get('raw_spatial_candidate_count')}",
        path=qa_config.aef_catalog_summary,
    )


def aef_tile_manifest_row(
    qa_config: SourceCoverageConfig,
) -> tuple[dict[str, object], list[SourceLayer]]:
    """Summarize the configured AEF tile manifest and local files."""
    data = load_json_object_if_exists(qa_config.aef_tile_manifest)
    if data is None:
        return (
            missing_row(qa_config, "aef", "tile_manifest", qa_config.aef_tile_manifest),
            [],
        )
    records = list_from_mapping(data, "records")
    selected_years = sorted(
        int(record["year"])
        for record in records
        if isinstance(record, dict) and isinstance(record.get("year"), int)
    )
    missing_years = missing_year_values(qa_config.years, selected_years)
    local_count = count_existing_local_files(records)
    missing_local_count = max(len(records) - local_count, 0)
    status = "complete" if not missing_years and missing_local_count == 0 else "incomplete"
    layers = layers_from_records("aef", records)
    return (
        summary_row(
            qa_config=qa_config,
            source="aef",
            artifact="tile_manifest",
            status=status,
            selected_years=selected_years,
            missing_years=missing_years,
            record_count=len(records),
            local_file_count=local_count,
            missing_local_file_count=missing_local_count,
            notes=validation_status_note(records),
            path=qa_config.aef_tile_manifest,
        ),
        layers,
    )


def kelpwatch_source_row(qa_config: SourceCoverageConfig) -> dict[str, object]:
    """Summarize the configured Kelpwatch source manifest."""
    data = load_json_object_if_exists(qa_config.kelpwatch_source_manifest)
    if data is None:
        return missing_row(
            qa_config,
            "kelpwatch",
            "source_manifest",
            qa_config.kelpwatch_source_manifest,
        )
    transfer = require_mapping(data.get("transfer"), "kelpwatch_source_manifest.transfer")
    local_path = optional_path(transfer.get("local_path"))
    local_count = 1 if local_path is not None and local_path.exists() else 0
    missing_local_count = 0 if local_count == 1 else 1
    status = "complete" if local_count == 1 else "missing_local_file"
    years = [int(year) for year in data.get("years", []) if isinstance(year, int)]
    return summary_row(
        qa_config=qa_config,
        source="kelpwatch",
        artifact="source_manifest",
        status=status,
        selected_years=years,
        missing_years=missing_year_values(qa_config.years, years),
        record_count=1,
        local_file_count=local_count,
        missing_local_file_count=missing_local_count,
        bounds=source_bounds(data),
        notes=f"transfer_status={transfer.get('status')}",
        path=qa_config.kelpwatch_source_manifest,
    )


def kelpwatch_positive_row(qa_config: SourceCoverageConfig) -> dict[str, object]:
    """Summarize Kelpwatch positive source coverage inside the footprint."""
    if not qa_config.kelpwatch_qa_table.exists():
        return missing_row(
            qa_config,
            "kelpwatch",
            "positive_source_qa",
            qa_config.kelpwatch_qa_table,
        )
    with qa_config.kelpwatch_qa_table.open(newline="") as file:
        rows = list(csv.DictReader(file))
    years = sorted(int(row["year"]) for row in rows if row.get("year", "").isdigit())
    positive_count = sum(int(float(row.get("nonzero_count", 0) or 0)) for row in rows)
    valid_count = sum(int(float(row.get("valid_count", 0) or 0)) for row in rows)
    missing_years = missing_year_values(qa_config.years, years)
    status = "complete" if not missing_years else "incomplete"
    return summary_row(
        qa_config=qa_config,
        source="kelpwatch",
        artifact="positive_source_qa",
        status=status,
        selected_years=years,
        missing_years=missing_years,
        record_count=len(rows),
        positive_count=positive_count,
        valid_count=valid_count,
        notes="annual-max nonzero cells inside configured footprint",
        path=qa_config.kelpwatch_qa_table,
    )


def query_manifest_row(
    qa_config: SourceCoverageConfig,
    source_name: str,
    path: Path,
    selected_key: str,
) -> tuple[dict[str, object], list[SourceLayer]]:
    """Summarize a source query manifest."""
    data = load_json_object_if_exists(path)
    if data is None:
        return missing_row(qa_config, source_name, "query_manifest", path), []
    records = list_from_mapping(data, selected_key)
    status = str(data.get("query_status", "selected" if records else "no_records"))
    if bool(data.get("dry_run")) and not records:
        status = "planned"
    elif not records:
        status = "no_records"
    return (
        summary_row(
            qa_config=qa_config,
            source=source_name,
            artifact="query_manifest",
            status=status,
            record_count=len(records),
            bounds=source_bounds(data),
            notes=f"dry_run={data.get('dry_run', False)}",
            path=path,
        ),
        layers_from_records(source_name, records),
    )


def source_manifest_row(
    qa_config: SourceCoverageConfig,
    source_name: str,
    artifact: str,
    path: Path,
) -> tuple[dict[str, object], list[SourceLayer]]:
    """Summarize a source download or registration manifest."""
    data = load_json_object_if_exists(path)
    if data is None:
        return missing_row(qa_config, source_name, artifact, path), []
    records = list_from_mapping(data, "records")
    local_count = count_existing_local_files(records)
    missing_local_count = max(len(records) - local_count, 0)
    if bool(data.get("dry_run")):
        status = "planned"
    elif records and missing_local_count == 0:
        status = "complete"
    elif records:
        status = "missing_local_files"
    else:
        status = "no_records"
    return (
        summary_row(
            qa_config=qa_config,
            source=source_name,
            artifact=artifact,
            status=status,
            record_count=len(records),
            local_file_count=local_count,
            missing_local_file_count=missing_local_count,
            bounds=source_bounds(data),
            notes=validation_status_note(records),
            path=path,
        ),
        layers_from_records(source_name, records),
    )


def missing_row(
    qa_config: SourceCoverageConfig, source: str, artifact: str, path: Path
) -> dict[str, object]:
    """Build a summary row for a missing artifact."""
    return summary_row(
        qa_config=qa_config,
        source=source,
        artifact=artifact,
        status="missing",
        notes="artifact missing",
        path=path,
    )


def summary_row(
    *,
    qa_config: SourceCoverageConfig,
    source: str,
    artifact: str,
    status: str,
    selected_years: list[int] | None = None,
    missing_years: list[int] | None = None,
    record_count: int | None = None,
    local_file_count: int | None = None,
    missing_local_file_count: int | None = None,
    positive_count: int | None = None,
    valid_count: int | None = None,
    bounds: Bounds | None = None,
    notes: str = "",
    path: Path | None = None,
) -> dict[str, object]:
    """Build one CSV-ready source coverage summary row."""
    return {
        "region": qa_config.region_slug,
        "source": source,
        "artifact": artifact,
        "status": status,
        "selected_years": join_ints(selected_years or []),
        "missing_years": join_ints(missing_years or []),
        "record_count": record_count if record_count is not None else "",
        "local_file_count": local_file_count if local_file_count is not None else "",
        "missing_local_file_count": (
            missing_local_file_count if missing_local_file_count is not None else ""
        ),
        "positive_count": positive_count if positive_count is not None else "",
        "valid_count": valid_count if valid_count is not None else "",
        "bounds": bounds_to_string(bounds),
        "notes": notes,
        "path": str(path) if path is not None else "",
    }


def layers_from_records(source_name: str, records: list[Any]) -> list[SourceLayer]:
    """Build map layers from manifest records that expose WGS84-ish bounds."""
    layers: list[SourceLayer] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        bounds = record_bounds(record)
        if bounds is None:
            continue
        label = record_label(source_name, record, index)
        status = record_status(record)
        layers.append(SourceLayer(source=source_name, label=label, bounds=bounds, status=status))
    return layers


def record_bounds(record: dict[str, Any]) -> Bounds | None:
    """Extract display bounds from a manifest record."""
    for key in ("catalog_bounds", "bounds"):
        bounds = optional_bounds(record.get(key))
        if bounds is not None:
            return bounds
    coverage = record.get("coverage_check")
    if isinstance(coverage, dict):
        for key in ("intersection_bounds", "recommended_subset_bounds", "product_bounds"):
            bounds = optional_bounds(coverage.get(key))
            if bounds is not None:
                return bounds
    raster = record.get("raster")
    if isinstance(raster, dict):
        return optional_bounds(raster.get("bounds"))
    return None


def source_bounds(data: dict[str, Any]) -> Bounds | None:
    """Extract a single representative bounds value from a manifest."""
    region = data.get("region")
    if isinstance(region, dict):
        bounds = optional_bounds(region.get("bounds"))
        if bounds is not None:
            return bounds
    source = data.get("source")
    if isinstance(source, dict):
        bounds = optional_bounds(source.get("bounds"))
        if bounds is not None:
            return bounds
    return optional_bounds(data.get("bounds"))


def record_label(source_name: str, record: dict[str, Any], index: int) -> str:
    """Return a compact layer label for a manifest record."""
    for key in ("year", "product_id", "tile_id", "artifact_id", "source_id", "title"):
        value = record.get(key)
        if value not in (None, ""):
            return f"{source_name}: {value}"
    return f"{source_name}: {index}"


def record_status(record: dict[str, Any]) -> str:
    """Return a compact status label from a manifest record."""
    transfer = record.get("transfer")
    if isinstance(transfer, dict) and transfer.get("status") is not None:
        return str(transfer["status"])
    if record.get("validation_status") is not None:
        return str(record["validation_status"])
    raster = record.get("raster")
    if isinstance(raster, dict) and raster.get("validation_status") is not None:
        return str(raster["validation_status"])
    vector = record.get("vector")
    if isinstance(vector, dict) and vector.get("validation_status") is not None:
        return str(vector["validation_status"])
    return "selected"


def count_existing_local_files(records: list[Any]) -> int:
    """Count manifest records whose declared local path exists."""
    count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        local_path = local_path_from_record(record)
        if local_path is not None and local_path.exists():
            count += 1
    return count


def local_path_from_record(record: dict[str, Any]) -> Path | None:
    """Read a local file path from a source or transfer manifest record."""
    for key in ("local_tiff_path", "local_path"):
        path = optional_path(record.get(key))
        if path is not None:
            return path
    transfer = record.get("transfer")
    if isinstance(transfer, dict):
        return optional_path(transfer.get("local_path"))
    return None


def validation_status_note(records: list[Any]) -> str:
    """Summarize validation or transfer statuses in a compact note."""
    statuses: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        status = record_status(record)
        statuses[status] = statuses.get(status, 0) + 1
    return "; ".join(f"{key}={value}" for key, value in sorted(statuses.items()))


def write_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write the source coverage summary table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_static_figure(
    *,
    qa_config: SourceCoverageConfig,
    footprint: BaseGeometry,
    source_layers: list[SourceLayer],
    summary_rows: list[dict[str, object]],
) -> None:
    """Write a static source-coverage QA figure."""
    qa_config.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (map_axis, table_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    draw_source_map(map_axis, footprint, source_layers, qa_config)
    draw_status_panel(table_axis, summary_rows, qa_config)
    fig.tight_layout()
    fig.savefig(qa_config.figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_source_map(
    axis: Any,
    footprint: BaseGeometry,
    source_layers: list[SourceLayer],
    qa_config: SourceCoverageConfig,
) -> None:
    """Draw footprint and source bounds onto a matplotlib axis."""
    footprint_bounds = bounds_from_geometry(footprint)
    for layer in source_layers:
        if layer.source == "footprint":
            continue
        color = SOURCE_COLORS.get(layer.source, "#64748b")
        draw_bounds_rect(axis, layer.bounds, color=color, label=layer.source)
    draw_bounds_rect(axis, footprint_bounds, color="#111827", label="footprint", linewidth=2.0)
    west, south, east, north = footprint_bounds
    pad_x = max((east - west) * 0.12, 0.01)
    pad_y = max((north - south) * 0.12, 0.01)
    axis.set_xlim(west - pad_x, east + pad_x)
    axis.set_ylim(south - pad_y, north + pad_y)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(f"{qa_config.region_display_name} Source Coverage")
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    axis.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    axis.grid(True, alpha=0.2)


def draw_bounds_rect(
    axis: Any,
    bounds: Bounds,
    *,
    color: str,
    label: str,
    linewidth: float = 1.2,
) -> None:
    """Draw one source bounds rectangle on a matplotlib axis."""
    west, south, east, north = bounds
    axis.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=label,
            alpha=0.9,
        )
    )


def draw_status_panel(
    axis: Any,
    summary_rows: list[dict[str, object]],
    qa_config: SourceCoverageConfig,
) -> None:
    """Draw a compact text status panel in the static QA figure."""
    axis.axis("off")
    lines = [f"{qa_config.region_display_name} early source QA", ""]
    for row in summary_rows:
        source = str(row["source"])
        artifact = str(row["artifact"])
        status = str(row["status"])
        records = str(row["record_count"])
        missing_years = str(row["missing_years"])
        suffix = f", records {records}" if records else ""
        if missing_years:
            suffix = f"{suffix}, missing years {missing_years}"
        lines.append(f"{source} / {artifact}: {status}{suffix}")
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        wrap=True,
    )


def write_interactive_html(
    *,
    qa_config: SourceCoverageConfig,
    footprint: BaseGeometry,
    source_layers: list[SourceLayer],
    summary_rows: list[dict[str, object]],
) -> None:
    """Write a lightweight HTML source-coverage QA viewer."""
    qa_config.html_path.parent.mkdir(parents=True, exist_ok=True)
    qa_config.html_path.write_text(
        build_interactive_html(
            qa_config=qa_config,
            footprint=footprint,
            source_layers=source_layers,
            summary_rows=summary_rows,
        )
    )


def build_interactive_html(
    *,
    qa_config: SourceCoverageConfig,
    footprint: BaseGeometry,
    source_layers: list[SourceLayer],
    summary_rows: list[dict[str, object]],
) -> str:
    """Build a self-contained HTML source-coverage viewer."""
    viewport = SvgViewport.from_bounds(bounds_from_geometry(footprint))
    controls = build_layer_controls(source_layers)
    svg = build_svg(viewport, source_layers)
    table = build_summary_table(summary_rows)
    title = html.escape(f"{qa_config.region_display_name} Source Coverage QA")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
    .layout {{ display: grid; grid-template-columns: minmax(360px, 1fr) 420px; gap: 20px; }}
    svg {{ width: 100%; max-height: 620px; border: 1px solid #cbd5e1; background: #f8fafc; }}
    label {{ display: inline-block; margin: 0 12px 10px 0; font-size: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 5px 6px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    .source-layer.hidden {{ display: none; }}
    .meta {{ color: #475569; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Bounds are drawn in WGS84 and clipped visually to the configured footprint.</p>
  <div>{controls}</div>
  <div class="layout">
    <div>{svg}</div>
    <div>{table}</div>
  </div>
  <script>
    document.querySelectorAll('input[data-source]').forEach((input) => {{
      input.addEventListener('change', () => {{
        document.querySelectorAll(`.source-${{input.dataset.source}}`).forEach((node) => {{
          node.classList.toggle('hidden', !input.checked);
        }});
      }});
    }});
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class SvgViewport:
    """Coordinate converter for the self-contained source-coverage SVG."""

    west: float
    south: float
    east: float
    north: float
    width: int = 840
    height: int = 560

    @classmethod
    def from_bounds(cls, bounds: Bounds) -> SvgViewport:
        """Create a padded SVG viewport from WGS84 bounds."""
        west, south, east, north = bounds
        pad_x = max((east - west) * 0.12, 0.01)
        pad_y = max((north - south) * 0.12, 0.01)
        return cls(west=west - pad_x, south=south - pad_y, east=east + pad_x, north=north + pad_y)

    def xy(self, lon: float, lat: float) -> tuple[float, float]:
        """Project longitude and latitude into SVG coordinates."""
        width = max(self.east - self.west, 1e-12)
        height = max(self.north - self.south, 1e-12)
        x = (lon - self.west) / width * self.width
        y = self.height - ((lat - self.south) / height * self.height)
        return x, y


def build_layer_controls(source_layers: list[SourceLayer]) -> str:
    """Build checkbox controls for the HTML source layers."""
    sources = sorted({layer.source for layer in source_layers if layer.source != "footprint"})
    controls = []
    for source in sources:
        color = SOURCE_COLORS.get(source, "#64748b")
        controls.append(
            f'<label><input type="checkbox" data-source="{html.escape(source)}" checked /> '
            f'<span style="color:{color}">{html.escape(source)}</span></label>'
        )
    return "\n".join(controls)


def build_svg(viewport: SvgViewport, source_layers: list[SourceLayer]) -> str:
    """Build the source-coverage bounds SVG."""
    rects = []
    for layer in source_layers:
        color = (
            "#111827"
            if layer.source == "footprint"
            else SOURCE_COLORS.get(layer.source, "#64748b")
        )
        width = "2.8" if layer.source == "footprint" else "1.8"
        rects.append(svg_rect(viewport, layer, color=color, stroke_width=width))
    return (
        f'<svg viewBox="0 0 {viewport.width} {viewport.height}" '
        'role="img" aria-label="Source coverage bounds">'
        + "\n".join(rects)
        + "</svg>"
    )


def svg_rect(
    viewport: SvgViewport,
    layer: SourceLayer,
    *,
    color: str,
    stroke_width: str,
) -> str:
    """Build one SVG rectangle for a source bounds layer."""
    west, south, east, north = layer.bounds
    x0, y0 = viewport.xy(west, north)
    x1, y1 = viewport.xy(east, south)
    source_class = f"source-layer source-{html.escape(layer.source)}"
    label = html.escape(f"{layer.label} ({layer.status})")
    return (
        f'<rect class="{source_class}" x="{x0:.2f}" y="{y0:.2f}" '
        f'width="{max(x1 - x0, 0.1):.2f}" height="{max(y1 - y0, 0.1):.2f}" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}">'
        f"<title>{label}</title></rect>"
    )


def build_summary_table(summary_rows: list[dict[str, object]]) -> str:
    """Build the HTML summary table."""
    head = "<tr><th>Source</th><th>Artifact</th><th>Status</th><th>Records</th></tr>"
    body_rows = []
    for row in summary_rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['source']))}</td>"
            f"<td>{html.escape(str(row['artifact']))}</td>"
            f"<td>{html.escape(str(row['status']))}</td>"
            f"<td>{html.escape(str(row['record_count']))}</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(body_rows)}</table>"


def write_manifest(
    *,
    qa_config: SourceCoverageConfig,
    footprint: BaseGeometry,
    footprint_created: bool,
    source_layers: list[SourceLayer],
    summary_rows: list[dict[str, object]],
) -> None:
    """Write the source-coverage manifest."""
    qa_config.coverage_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": "visualize-source-coverage",
        "config_path": str(qa_config.config_path),
        "created_at": datetime.now(UTC).isoformat(),
        "region": {
            "name": qa_config.region_name,
            "slug": qa_config.region_slug,
            "display_name": qa_config.region_display_name,
            "footprint_path": str(qa_config.footprint_path),
            "footprint_created": footprint_created,
            "footprint_bounds": list(bounds_from_geometry(footprint)),
        },
        "years": list(qa_config.years),
        "outputs": {
            "summary_table": str(qa_config.summary_table),
            "figure": str(qa_config.figure_path),
            "html": str(qa_config.html_path),
            "manifest": str(qa_config.coverage_manifest),
        },
        "source_layer_count": len(source_layers),
        "summary_rows": summary_rows,
    }
    qa_config.coverage_manifest.write_text(json.dumps(payload, indent=2) + "\n")


def load_json_object_if_exists(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if the path exists."""
    if not path.exists():
        return None
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"expected JSON object at {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def list_from_mapping(mapping: dict[str, Any], key: str) -> list[Any]:
    """Return a list field from a dynamic mapping, or an empty list."""
    value = mapping.get(key)
    return value if isinstance(value, list) else []


def optional_string(value: object) -> str | None:
    """Validate an optional string value."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"expected optional string, got {type(value).__name__}"
        raise ValueError(msg)
    return value


def optional_path(value: object) -> Path | None:
    """Convert an optional string value to a path."""
    if value in (None, ""):
        return None
    return Path(require_string(value, "path value"))


def optional_bounds(value: object) -> Bounds | None:
    """Parse optional west, south, east, north bounds."""
    if value is None:
        return None
    if isinstance(value, list | tuple) and len(value) == 4:
        west, south, east, north = value
        return (
            float(cast(Any, west)),
            float(cast(Any, south)),
            float(cast(Any, east)),
            float(cast(Any, north)),
        )
    if isinstance(value, dict):
        west = value.get("west", value.get("left"))
        south = value.get("south", value.get("bottom"))
        east = value.get("east", value.get("right"))
        north = value.get("north", value.get("top"))
        if None not in (west, south, east, north):
            return (
                float(cast(Any, west)),
                float(cast(Any, south)),
                float(cast(Any, east)),
                float(cast(Any, north)),
            )
    return None


def bounds_from_geometry(geometry: BaseGeometry) -> Bounds:
    """Return WGS84-ish bounds from a shapely geometry."""
    west, south, east, north = geometry.bounds
    return (float(west), float(south), float(east), float(north))


def bounds_to_string(bounds: Bounds | None) -> str:
    """Format optional bounds for a compact CSV cell."""
    if bounds is None:
        return ""
    return ",".join(f"{value:.8g}" for value in bounds)


def missing_year_values(expected_years: tuple[int, ...], selected_years: list[int]) -> list[int]:
    """Return expected years not found in a selected year list."""
    selected = set(selected_years)
    return [year for year in expected_years if year not in selected]


def join_ints(values: list[int]) -> str:
    """Join integer values for compact CSV output."""
    return "|".join(str(value) for value in values)


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))
