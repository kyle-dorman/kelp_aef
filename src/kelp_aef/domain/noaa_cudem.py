"""Query and download NOAA CUDEM tiles for the configured domain."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import geopandas as gpd  # type: ignore[import-untyped]
import rasterio  # type: ignore[import-untyped]
import requests

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
URL_FIELD_CANDIDATES = (
    "URL",
    "url",
    "Url",
    "DOWNLOAD",
    "download",
    "FILE_URL",
    "file_url",
)


@dataclass(frozen=True)
class NoaaCudemConfig:
    """Resolved config values for the NOAA CUDEM tile workflow."""

    config_path: Path
    region_name: str
    region_geometry_path: Path
    source_name: str
    source_role: str
    product_id: str
    metadata_url: str
    bulk_url: str
    tile_index_url: str
    url_list_url: str | None
    tile_index_path: Path
    local_tile_root: Path
    query_manifest: Path
    tile_manifest: Path
    horizontal_crs: str
    vertical_datum: str
    units: str
    resolution: str
    grid_spacing_arc_seconds: float
    elevation_sign_convention: str
    metadata_summary: Path | None


@dataclass(frozen=True)
class RemoteInfo:
    """Remote availability and size metadata from a HEAD request."""

    exists: bool | None
    status_code: int | None
    content_length_bytes: int | None
    final_url: str | None
    error: str | None


@dataclass(frozen=True)
class TransferResult:
    """Result of ensuring one NOAA CUDEM tile exists locally or is planned."""

    status: str
    source_uri: str
    local_path: Path
    file_size_bytes: int | None
    remote_info: RemoteInfo | None
    error: str | None


def load_noaa_cudem_config(config_path: Path) -> NoaaCudemConfig:
    """Load NOAA CUDEM settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    domain = require_mapping(config.get("domain"), "domain")
    cudem = require_mapping(domain.get("noaa_cudem"), "domain.noaa_cudem")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    return NoaaCudemConfig(
        config_path=config_path,
        region_name=require_string(region.get("name"), "region.name"),
        region_geometry_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        source_name=require_string(cudem.get("source_name"), "domain.noaa_cudem.source_name"),
        source_role=require_string(cudem.get("source_role"), "domain.noaa_cudem.source_role"),
        product_id=require_string(cudem.get("product_id"), "domain.noaa_cudem.product_id"),
        metadata_url=require_string(cudem.get("metadata_url"), "domain.noaa_cudem.metadata_url"),
        bulk_url=require_string(cudem.get("bulk_url"), "domain.noaa_cudem.bulk_url"),
        tile_index_url=require_string(
            cudem.get("tile_index_url"), "domain.noaa_cudem.tile_index_url"
        ),
        url_list_url=optional_string(cudem.get("url_list_url"), "domain.noaa_cudem.url_list_url"),
        tile_index_path=Path(
            require_string(cudem.get("tile_index_path"), "domain.noaa_cudem.tile_index_path")
        ),
        local_tile_root=Path(
            require_string(cudem.get("local_tile_root"), "domain.noaa_cudem.local_tile_root")
        ),
        query_manifest=Path(
            require_string(cudem.get("query_manifest"), "domain.noaa_cudem.query_manifest")
        ),
        tile_manifest=Path(
            require_string(cudem.get("tile_manifest"), "domain.noaa_cudem.tile_manifest")
        ),
        horizontal_crs=require_string(
            cudem.get("horizontal_crs"), "domain.noaa_cudem.horizontal_crs"
        ),
        vertical_datum=require_string(
            cudem.get("vertical_datum"), "domain.noaa_cudem.vertical_datum"
        ),
        units=require_string(cudem.get("units"), "domain.noaa_cudem.units"),
        resolution=require_string(cudem.get("resolution"), "domain.noaa_cudem.resolution"),
        grid_spacing_arc_seconds=require_float(
            cudem.get("grid_spacing_arc_seconds"),
            "domain.noaa_cudem.grid_spacing_arc_seconds",
        ),
        elevation_sign_convention=require_string(
            cudem.get("elevation_sign_convention"),
            "domain.noaa_cudem.elevation_sign_convention",
        ),
        metadata_summary=optional_path(report_outputs.get("metadata_summary")),
    )


def optional_string(value: object, name: str) -> str | None:
    """Validate an optional string config value."""
    if value is None:
        return None
    return require_string(value, name)


def optional_path(value: object) -> Path | None:
    """Convert an optional config string into a path."""
    if value is None:
        return None
    return Path(require_string(value, "optional path"))


def require_float(value: object, name: str) -> float:
    """Validate a dynamic config value as a floating-point number."""
    if not isinstance(value, int | float):
        msg = f"config field must be a number: {name}"
        raise ValueError(msg)
    return float(value)


def query_noaa_cudem(
    config_path: Path,
    *,
    dry_run: bool = False,
    manifest_output: Path | None = None,
    tile_index_path: Path | None = None,
    download_index: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    force: bool = False,
) -> int:
    """Query the CUDEM tile index for tiles intersecting the configured geometry."""
    cudem_config = load_noaa_cudem_config(config_path)
    manifest_path = manifest_output or cudem_config.query_manifest
    resolved_index_path = tile_index_path or cudem_config.tile_index_path
    LOGGER.info("Querying NOAA CUDEM tile index for %s", cudem_config.region_name)

    region = region_for_query(cudem_config.region_geometry_path)
    index_status = ensure_tile_index_available(
        cudem_config=cudem_config,
        tile_index_path=resolved_index_path,
        dry_run=dry_run,
        download_index=download_index,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
    )
    selected_tiles: list[dict[str, Any]] = []
    query_status = "planned_no_index_read"
    if resolved_index_path.exists():
        selected_tiles = selected_tiles_from_index(cudem_config, resolved_index_path, region)
        query_status = "selected_tiles"
    elif not dry_run:
        msg = (
            f"NOAA CUDEM tile index is missing: {resolved_index_path}. "
            "Run with --download-index after review, or pass --tile-index-path."
        )
        raise FileNotFoundError(msg)

    manifest = build_query_manifest(
        cudem_config=cudem_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        tile_index_path=resolved_index_path,
        index_status=index_status,
        query_status=query_status,
        region=region,
        selected_tiles=selected_tiles,
    )
    write_manifest_if_allowed(manifest_path, manifest, dry_run)
    LOGGER.info("NOAA CUDEM query selected %s tiles", len(selected_tiles))
    return 0


def write_manifest_if_allowed(path: Path, manifest: dict[str, Any], dry_run: bool) -> None:
    """Write a manifest for real runs or explicit dry-run output paths."""
    if dry_run and str(path) == manifest["configured_manifest_path"]:
        LOGGER.info("Dry run complete; use --manifest-output to save the plan.")
        return
    write_json(path, manifest)
    LOGGER.info("Wrote manifest: %s", path)


def region_for_query(geometry_path: Path) -> gpd.GeoDataFrame:
    """Read the configured query geometry in WGS84."""
    if not geometry_path.exists():
        msg = f"region geometry does not exist: {geometry_path}"
        raise FileNotFoundError(msg)
    return gpd.read_file(geometry_path).to_crs("EPSG:4326")


def ensure_tile_index_available(
    *,
    cudem_config: NoaaCudemConfig,
    tile_index_path: Path,
    dry_run: bool,
    download_index: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
) -> dict[str, Any]:
    """Ensure the tile index exists locally, or return a dry-run plan."""
    if tile_index_path.exists() and not force:
        return {
            "status": "skipped_existing",
            "source_uri": cudem_config.tile_index_url,
            "local_path": str(tile_index_path),
            "file_size_bytes": tile_index_path.stat().st_size,
        }
    if dry_run or not download_index:
        return {
            "status": "dry_run" if dry_run else "missing",
            "source_uri": cudem_config.tile_index_url,
            "local_path": str(tile_index_path),
            "file_size_bytes": local_file_size(tile_index_path),
            "download_required": not tile_index_path.exists(),
        }
    with requests.Session() as session:
        download_file(
            session=session,
            source_uri=cudem_config.tile_index_url,
            local_path=tile_index_path,
            timeout_seconds=timeout_seconds,
            chunk_size_bytes=chunk_size_bytes,
        )
    return {
        "status": "downloaded",
        "source_uri": cudem_config.tile_index_url,
        "local_path": str(tile_index_path),
        "file_size_bytes": tile_index_path.stat().st_size,
    }


def selected_tiles_from_index(
    cudem_config: NoaaCudemConfig, tile_index_path: Path, region: gpd.GeoDataFrame
) -> list[dict[str, Any]]:
    """Select CUDEM tiles from a local tile index by geometry intersection."""
    index = gpd.read_file(tile_index_path).to_crs("EPSG:4326")
    region_union = region.geometry.union_all()
    selected = index[index.geometry.intersects(region_union)]
    rows = cast(list[dict[str, Any]], selected.to_dict("records"))
    return [
        tile_record_from_index_row(cudem_config, row, geometry)
        for row, geometry in zip(rows, selected.geometry, strict=True)
    ]


def tile_record_from_index_row(
    cudem_config: NoaaCudemConfig, row: dict[str, Any], geometry: Any
) -> dict[str, Any]:
    """Build one selected-tile manifest record from a tile-index row."""
    source_uri = source_uri_from_index_row(row)
    return {
        "tile_id": tile_id_from_row(row, source_uri),
        "source_uri": source_uri,
        "local_path": str(local_tile_path_for_source_uri(source_uri, cudem_config.local_tile_root)),
        "bounds": bounds_dict(coerce_bounds(geometry.bounds)),
        "horizontal_crs": cudem_config.horizontal_crs,
        "vertical_datum": cudem_config.vertical_datum,
        "units": cudem_config.units,
        "resolution": cudem_config.resolution,
        "grid_spacing_arc_seconds": cudem_config.grid_spacing_arc_seconds,
        "elevation_sign_convention": cudem_config.elevation_sign_convention,
    }


def source_uri_from_index_row(row: dict[str, Any]) -> str:
    """Return the source download URL from a CUDEM tile-index row."""
    for field in URL_FIELD_CANDIDATES:
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    msg = f"tile-index row is missing a source URL field; tried {URL_FIELD_CANDIDATES}"
    raise ValueError(msg)


def tile_id_from_row(row: dict[str, Any], source_uri: str) -> str:
    """Return a stable tile id from a row field or source basename."""
    for field in ("tile_id", "TILE_ID", "name", "Name", "id", "ID"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return Path(urlparse(source_uri).path).stem


def local_tile_path_for_source_uri(source_uri: str, local_tile_root: Path) -> Path:
    """Map a remote CUDEM tile URL to a local tile path."""
    basename = Path(urlparse(source_uri).path).name
    if not basename:
        msg = f"source URI does not contain a file name: {source_uri}"
        raise ValueError(msg)
    return local_tile_root / basename


def bounds_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    """Convert bounds tuple into a named JSON object."""
    west, south, east, north = bounds
    return {"west": west, "south": south, "east": east, "north": north}


def coerce_bounds(values: Any) -> tuple[float, float, float, float]:
    """Convert a four-value bounds object into a typed bounds tuple."""
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 4:
        msg = f"bounds must contain four values, got {len(bounds)}"
        raise ValueError(msg)
    west, south, east, north = bounds
    return west, south, east, north


def build_query_manifest(
    *,
    cudem_config: NoaaCudemConfig,
    manifest_path: Path,
    dry_run: bool,
    tile_index_path: Path,
    index_status: dict[str, Any],
    query_status: str,
    region: gpd.GeoDataFrame,
    selected_tiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CUDEM tile-query manifest."""
    return {
        "command": "query-noaa-cudem",
        "config_path": str(cudem_config.config_path),
        "configured_manifest_path": str(cudem_config.query_manifest),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "query_status": query_status,
        "source": source_metadata(cudem_config),
        "region": {
            "name": cudem_config.region_name,
            "geometry_path": str(cudem_config.region_geometry_path),
            "bounds": bounds_dict(coerce_bounds(region.total_bounds)),
        },
        "tile_index": index_status,
        "selected_tile_count": len(selected_tiles),
        "selected_tiles": selected_tiles,
    }


def source_metadata(cudem_config: NoaaCudemConfig) -> dict[str, Any]:
    """Return common NOAA CUDEM source metadata for manifests."""
    return {
        "source_name": cudem_config.source_name,
        "source_role": cudem_config.source_role,
        "product_id": cudem_config.product_id,
        "metadata_url": cudem_config.metadata_url,
        "bulk_url": cudem_config.bulk_url,
        "tile_index_url": cudem_config.tile_index_url,
        "url_list_url": cudem_config.url_list_url,
        "horizontal_crs": cudem_config.horizontal_crs,
        "vertical_datum": cudem_config.vertical_datum,
        "units": cudem_config.units,
        "resolution": cudem_config.resolution,
        "grid_spacing_arc_seconds": cudem_config.grid_spacing_arc_seconds,
        "elevation_sign_convention": cudem_config.elevation_sign_convention,
        "license_access_notes": (
            "NOAA/NCEI public data; produced by NOAA NCEI and not subject "
            "to copyright protection within the United States."
        ),
    }


def download_noaa_cudem(
    config_path: Path,
    *,
    dry_run: bool = False,
    skip_remote_checks: bool = False,
    manifest_output: Path | None = None,
    query_manifest: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    force: bool = False,
) -> int:
    """Download or plan CUDEM tile downloads from a tile-query manifest."""
    cudem_config = load_noaa_cudem_config(config_path)
    source_query_manifest = query_manifest or cudem_config.query_manifest
    manifest_path = manifest_output or cudem_config.tile_manifest
    query = load_query_manifest(source_query_manifest)
    selected_tiles = cast(list[dict[str, Any]], query.get("selected_tiles", []))
    LOGGER.info("Planning NOAA CUDEM download for %s selected tiles", len(selected_tiles))

    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        for tile in selected_tiles:
            records.append(
                process_tile_record(
                    tile=tile,
                    session=session,
                    dry_run=dry_run,
                    skip_remote_checks=skip_remote_checks,
                    timeout_seconds=timeout_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                    force=force,
                )
            )

    manifest = build_download_manifest(
        cudem_config=cudem_config,
        manifest_path=manifest_path,
        query_manifest=source_query_manifest,
        dry_run=dry_run,
        skip_remote_checks=skip_remote_checks,
        records=records,
    )
    if dry_run and manifest_output is None:
        LOGGER.info("Dry run complete; use --manifest-output to save the plan.")
    else:
        write_json(manifest_path, manifest)
        LOGGER.info("Wrote NOAA CUDEM tile manifest: %s", manifest_path)

    if not dry_run and cudem_config.metadata_summary is not None and manifest_output is None:
        update_metadata_summary(cudem_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", cudem_config.metadata_summary)

    return 0


def load_query_manifest(path: Path) -> dict[str, Any]:
    """Load a NOAA CUDEM tile-query manifest."""
    if not path.exists():
        msg = f"NOAA CUDEM query manifest does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        msg = f"query manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def process_tile_record(
    *,
    tile: dict[str, Any],
    session: requests.Session,
    dry_run: bool,
    skip_remote_checks: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
) -> dict[str, Any]:
    """Download or plan one selected CUDEM tile and return its manifest record."""
    source_uri = require_string(tile.get("source_uri"), "selected_tiles[].source_uri")
    local_path = Path(require_string(tile.get("local_path"), "selected_tiles[].local_path"))
    remote_info = (
        None if skip_remote_checks else head_remote_asset(session, source_uri, timeout_seconds)
    )
    transfer = ensure_tile_available(
        source_uri=source_uri,
        local_path=local_path,
        session=session,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
        remote_info=remote_info,
    )
    raster_metadata = raster_metadata_for_manifest(local_path, dry_run)
    return {
        "tile_id": tile.get("tile_id"),
        "source_uri": source_uri,
        "local_path": str(local_path),
        "bounds": tile.get("bounds"),
        "transfer": transfer_for_manifest(transfer),
        "raster": raster_metadata,
    }


def head_remote_asset(
    session: requests.Session,
    source_uri: str,
    timeout_seconds: float,
) -> RemoteInfo:
    """Check remote availability without downloading the source body."""
    try:
        response = session.head(source_uri, allow_redirects=True, timeout=timeout_seconds)
    except requests.RequestException as exc:
        LOGGER.warning("NOAA CUDEM remote check failed for %s: %s", source_uri, exc)
        return RemoteInfo(
            exists=None,
            status_code=None,
            content_length_bytes=None,
            final_url=None,
            error=str(exc),
        )
    content_length = response.headers.get("Content-Length")
    return RemoteInfo(
        exists=remote_exists_from_status_code(response.status_code),
        status_code=response.status_code,
        content_length_bytes=int(content_length) if content_length else None,
        final_url=response.url,
        error=None,
    )


def remote_exists_from_status_code(status_code: int) -> bool | None:
    """Convert an HTTP status code into a conservative existence flag."""
    if 200 <= status_code < 400:
        return True
    if status_code == 404:
        return False
    return None


def ensure_tile_available(
    *,
    source_uri: str,
    local_path: Path,
    session: requests.Session,
    dry_run: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
    remote_info: RemoteInfo | None,
) -> TransferResult:
    """Ensure one configured NOAA CUDEM tile is local or record the plan."""
    if dry_run:
        return TransferResult(
            status="dry_run",
            source_uri=source_uri,
            local_path=local_path,
            file_size_bytes=local_file_size(local_path),
            remote_info=remote_info,
            error=None,
        )
    if local_path.exists() and not force:
        return TransferResult(
            status="skipped_existing",
            source_uri=source_uri,
            local_path=local_path,
            file_size_bytes=local_path.stat().st_size,
            remote_info=remote_info,
            error=None,
        )

    download_file(
        session=session,
        source_uri=source_uri,
        local_path=local_path,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
    )
    return TransferResult(
        status="downloaded",
        source_uri=source_uri,
        local_path=local_path,
        file_size_bytes=local_path.stat().st_size,
        remote_info=remote_info,
        error=None,
    )


def local_file_size(path: Path) -> int | None:
    """Return local file size if the file exists."""
    return path.stat().st_size if path.exists() else None


def download_file(
    *,
    session: requests.Session,
    source_uri: str,
    local_path: Path,
    timeout_seconds: float,
    chunk_size_bytes: int,
) -> None:
    """Download a NOAA CUDEM file via streaming HTTP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(local_path.suffix + ".part")
    with session.get(source_uri, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    file.write(chunk)
    temporary_path.replace(local_path)


def raster_metadata_for_manifest(path: Path, dry_run: bool) -> dict[str, Any]:
    """Inspect local raster metadata for the tile manifest."""
    if dry_run and not path.exists():
        return {"validation_status": "not_checked_dry_run"}
    if not path.exists():
        return {"validation_status": "missing"}
    try:
        with rasterio.open(path) as dataset:
            return {
                "validation_status": "valid",
                "driver": dataset.driver,
                "crs": str(dataset.crs) if dataset.crs is not None else None,
                "bounds": bounds_dict(coerce_bounds(dataset.bounds)),
                "shape": {"height": dataset.height, "width": dataset.width},
                "band_count": dataset.count,
                "dtypes": list(dataset.dtypes),
                "nodata": dataset.nodata,
            }
    except Exception as exc:  # noqa: BLE001
        return {"validation_status": "invalid", "error": str(exc)}


def build_download_manifest(
    *,
    cudem_config: NoaaCudemConfig,
    manifest_path: Path,
    query_manifest: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CUDEM tile-download manifest."""
    return {
        "command": "download-noaa-cudem",
        "config_path": str(cudem_config.config_path),
        "manifest_path": str(manifest_path),
        "query_manifest": str(query_manifest),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "source": source_metadata(cudem_config),
        "record_count": len(records),
        "records": records,
    }


def transfer_for_manifest(transfer: TransferResult) -> dict[str, Any]:
    """Convert a transfer result into a JSON manifest object."""
    return {
        "status": transfer.status,
        "source_uri": transfer.source_uri,
        "local_path": str(transfer.local_path),
        "file_size_bytes": transfer.file_size_bytes,
        "remote": remote_info_for_manifest(transfer.remote_info),
        "error": transfer.error,
    }


def remote_info_for_manifest(remote_info: RemoteInfo | None) -> dict[str, Any] | None:
    """Convert remote HEAD metadata into a JSON manifest object."""
    if remote_info is None:
        return None
    return {
        "exists": remote_info.exists,
        "status_code": remote_info.status_code,
        "content_length_bytes": remote_info.content_length_bytes,
        "final_url": remote_info.final_url,
        "error": remote_info.error,
    }


def update_metadata_summary(summary_path: Path, manifest: dict[str, Any]) -> None:
    """Merge the NOAA CUDEM tile manifest pointer into the metadata summary."""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary["noaa_cudem"] = {
        "manifest_path": manifest["manifest_path"],
        "query_manifest": manifest["query_manifest"],
        "source": manifest["source"],
        "record_count": manifest["record_count"],
    }
    write_json(summary_path, summary)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON payload, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
