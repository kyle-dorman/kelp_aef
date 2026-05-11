"""Query and download NOAA CUSP shoreline data for the configured domain."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import geopandas as gpd  # type: ignore[import-untyped]
import requests
from shapely.geometry import box

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class NoaaCuspConfig:
    """Resolved config values for the NOAA CUSP shoreline workflow."""

    config_path: Path
    region_name: str
    region_geometry_path: Path
    source_name: str
    source_role: str
    product_id: str
    metadata_url: str
    source_page_url: str
    viewer_url: str
    package_region: str
    source_uri: str
    local_source_root: Path
    query_manifest: Path
    source_manifest: Path
    horizontal_crs: str
    data_format: str
    shoreline_reference: str
    scale: str
    geometry_type: str
    package_bounds: tuple[float, float, float, float]
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
    """Result of ensuring one NOAA CUSP source package exists locally or is planned."""

    status: str
    source_uri: str
    local_path: Path
    file_size_bytes: int | None
    remote_info: RemoteInfo | None
    error: str | None


def load_noaa_cusp_config(config_path: Path) -> NoaaCuspConfig:
    """Load NOAA CUSP settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    domain = require_mapping(config.get("domain"), "domain")
    cusp = require_mapping(domain.get("noaa_cusp"), "domain.noaa_cusp")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    return NoaaCuspConfig(
        config_path=config_path,
        region_name=require_string(region.get("name"), "region.name"),
        region_geometry_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        source_name=require_string(cusp.get("source_name"), "domain.noaa_cusp.source_name"),
        source_role=require_string(cusp.get("source_role"), "domain.noaa_cusp.source_role"),
        product_id=require_string(cusp.get("product_id"), "domain.noaa_cusp.product_id"),
        metadata_url=require_string(cusp.get("metadata_url"), "domain.noaa_cusp.metadata_url"),
        source_page_url=require_string(
            cusp.get("source_page_url"), "domain.noaa_cusp.source_page_url"
        ),
        viewer_url=require_string(cusp.get("viewer_url"), "domain.noaa_cusp.viewer_url"),
        package_region=require_string(
            cusp.get("package_region"), "domain.noaa_cusp.package_region"
        ),
        source_uri=require_string(cusp.get("source_uri"), "domain.noaa_cusp.source_uri"),
        local_source_root=Path(
            require_string(cusp.get("local_source_root"), "domain.noaa_cusp.local_source_root")
        ),
        query_manifest=Path(
            require_string(cusp.get("query_manifest"), "domain.noaa_cusp.query_manifest")
        ),
        source_manifest=Path(
            require_string(cusp.get("source_manifest"), "domain.noaa_cusp.source_manifest")
        ),
        horizontal_crs=require_string(
            cusp.get("horizontal_crs"), "domain.noaa_cusp.horizontal_crs"
        ),
        data_format=require_string(cusp.get("data_format"), "domain.noaa_cusp.data_format"),
        shoreline_reference=require_string(
            cusp.get("shoreline_reference"), "domain.noaa_cusp.shoreline_reference"
        ),
        scale=require_string(cusp.get("scale"), "domain.noaa_cusp.scale"),
        geometry_type=require_string(cusp.get("geometry_type"), "domain.noaa_cusp.geometry_type"),
        package_bounds=require_bounds(
            cusp.get("package_bounds"), "domain.noaa_cusp.package_bounds"
        ),
        metadata_summary=optional_path(report_outputs.get("metadata_summary")),
    )


def optional_path(value: object) -> Path | None:
    """Convert an optional config string into a path."""
    if value is None:
        return None
    return Path(require_string(value, "optional path"))


def require_bounds(value: object, name: str) -> tuple[float, float, float, float]:
    """Validate a bounds mapping as a west, south, east, north tuple."""
    mapping = require_mapping(value, name)
    return (
        require_number(mapping.get("west"), f"{name}.west"),
        require_number(mapping.get("south"), f"{name}.south"),
        require_number(mapping.get("east"), f"{name}.east"),
        require_number(mapping.get("north"), f"{name}.north"),
    )


def require_number(value: object, name: str) -> float:
    """Validate a dynamic config value as a number."""
    if not isinstance(value, int | float):
        msg = f"config field must be a number: {name}"
        raise ValueError(msg)
    return float(value)


def query_noaa_cusp(
    config_path: Path,
    *,
    dry_run: bool = False,
    manifest_output: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    skip_remote_checks: bool = False,
) -> int:
    """Query the configured NOAA CUSP source package for Monterey coverage."""
    cusp_config = load_noaa_cusp_config(config_path)
    manifest_path = manifest_output or cusp_config.query_manifest
    LOGGER.info("Querying NOAA CUSP source package for %s", cusp_config.region_name)

    region = region_for_query(cusp_config.region_geometry_path)
    coverage = coverage_check(cusp_config.package_bounds, region)
    selected_artifacts = []
    if coverage["intersects_region"]:
        selected_artifacts.append(selected_artifact_record(cusp_config, coverage))
    remote = None
    if selected_artifacts and not dry_run and not skip_remote_checks:
        with requests.Session() as session:
            remote = head_remote_asset(session, cusp_config.source_uri, timeout_seconds)

    manifest = build_query_manifest(
        cusp_config=cusp_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        region=region,
        coverage=coverage,
        remote=remote,
        selected_artifacts=selected_artifacts,
    )
    write_manifest_if_allowed(manifest_path, manifest, dry_run)
    LOGGER.info("NOAA CUSP query selected %s source packages", len(selected_artifacts))
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


def coverage_check(
    package_bounds: tuple[float, float, float, float],
    region: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Check whether the configured CUSP package bounds intersect the region."""
    package_geometry = box(*package_bounds)
    region_union = region.geometry.union_all()
    intersects_region = bool(package_geometry.intersects(region_union))
    return {
        "intersects_region": intersects_region,
        "selection_method": "configured_package_bounds_intersection",
        "package_bounds": bounds_dict(package_bounds),
        "region_bounds": bounds_dict(coerce_bounds(region.total_bounds)),
    }


def selected_artifact_record(
    cusp_config: NoaaCuspConfig,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Build one selected CUSP source-package manifest record."""
    local_path = local_source_path_for_source_uri(
        cusp_config.source_uri, cusp_config.local_source_root
    )
    return {
        "artifact_id": f"noaa_cusp_{cusp_config.package_region.lower()}",
        "package_region": cusp_config.package_region,
        "source_uri": cusp_config.source_uri,
        "local_path": str(local_path),
        "bounds": coverage["package_bounds"],
        "horizontal_crs": cusp_config.horizontal_crs,
        "data_format": cusp_config.data_format,
        "shoreline_reference": cusp_config.shoreline_reference,
        "scale": cusp_config.scale,
        "geometry_type": cusp_config.geometry_type,
        "coverage_check": coverage,
    }


def local_source_path_for_source_uri(source_uri: str, local_source_root: Path) -> Path:
    """Map a remote CUSP package URL to a local source-package path."""
    basename = Path(urlparse(source_uri).path).name
    if not basename:
        msg = f"source URI does not contain a file name: {source_uri}"
        raise ValueError(msg)
    return local_source_root / basename


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
    cusp_config: NoaaCuspConfig,
    manifest_path: Path,
    dry_run: bool,
    region: gpd.GeoDataFrame,
    coverage: dict[str, Any],
    remote: RemoteInfo | None,
    selected_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CUSP source-query manifest."""
    return {
        "command": "query-noaa-cusp",
        "config_path": str(cusp_config.config_path),
        "configured_manifest_path": str(cusp_config.query_manifest),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "query_status": "selected_sources" if selected_artifacts else "no_package_coverage",
        "source": source_metadata(cusp_config),
        "region": {
            "name": cusp_config.region_name,
            "geometry_path": str(cusp_config.region_geometry_path),
            "bounds": bounds_dict(coerce_bounds(region.total_bounds)),
        },
        "coverage_check": coverage,
        "remote": remote_info_for_manifest(remote),
        "selected_artifact_count": len(selected_artifacts),
        "selected_artifacts": selected_artifacts,
    }


def source_metadata(cusp_config: NoaaCuspConfig) -> dict[str, Any]:
    """Return common NOAA CUSP source metadata for manifests."""
    return {
        "source_name": cusp_config.source_name,
        "source_role": cusp_config.source_role,
        "product_id": cusp_config.product_id,
        "metadata_url": cusp_config.metadata_url,
        "source_page_url": cusp_config.source_page_url,
        "viewer_url": cusp_config.viewer_url,
        "package_region": cusp_config.package_region,
        "source_uri": cusp_config.source_uri,
        "horizontal_crs": cusp_config.horizontal_crs,
        "data_format": cusp_config.data_format,
        "shoreline_reference": cusp_config.shoreline_reference,
        "scale": cusp_config.scale,
        "geometry_type": cusp_config.geometry_type,
        "license_access_notes": (
            "NOAA public data; access constraints are none, with NOAA credit requested "
            "and standard NOAA no-warranty/use-at-own-risk notices."
        ),
    }


def download_noaa_cusp(
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
    """Download or plan CUSP source-package downloads from a query manifest."""
    cusp_config = load_noaa_cusp_config(config_path)
    source_query_manifest = query_manifest or cusp_config.query_manifest
    manifest_path = manifest_output or cusp_config.source_manifest
    query = load_query_manifest(source_query_manifest)
    selected_artifacts = cast(list[dict[str, Any]], query.get("selected_artifacts", []))
    LOGGER.info("Planning NOAA CUSP download for %s selected sources", len(selected_artifacts))

    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        for artifact in selected_artifacts:
            records.append(
                process_artifact_record(
                    artifact=artifact,
                    session=session,
                    dry_run=dry_run,
                    skip_remote_checks=skip_remote_checks,
                    timeout_seconds=timeout_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                    force=force,
                )
            )

    manifest = build_download_manifest(
        cusp_config=cusp_config,
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
        LOGGER.info("Wrote NOAA CUSP source manifest: %s", manifest_path)

    if not dry_run and cusp_config.metadata_summary is not None and manifest_output is None:
        update_metadata_summary(cusp_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", cusp_config.metadata_summary)

    return 0


def load_query_manifest(path: Path) -> dict[str, Any]:
    """Load a NOAA CUSP source-query manifest."""
    if not path.exists():
        msg = f"NOAA CUSP query manifest does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        msg = f"query manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def process_artifact_record(
    *,
    artifact: dict[str, Any],
    session: requests.Session,
    dry_run: bool,
    skip_remote_checks: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
) -> dict[str, Any]:
    """Download or plan one selected CUSP source package and return its record."""
    source_uri = require_string(artifact.get("source_uri"), "selected_artifacts[].source_uri")
    local_path = Path(require_string(artifact.get("local_path"), "selected_artifacts[].local_path"))
    remote_info = None
    if not dry_run and not skip_remote_checks:
        remote_info = head_remote_asset(session, source_uri, timeout_seconds)
    transfer = ensure_source_available(
        source_uri=source_uri,
        local_path=local_path,
        session=session,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
        remote_info=remote_info,
    )
    vector_metadata = vector_metadata_for_manifest(local_path, dry_run)
    return {
        "artifact_id": artifact.get("artifact_id"),
        "package_region": artifact.get("package_region"),
        "source_uri": source_uri,
        "local_path": str(local_path),
        "bounds": artifact.get("bounds"),
        "coverage_check": artifact.get("coverage_check"),
        "transfer": transfer_for_manifest(transfer),
        "vector": vector_metadata,
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
        LOGGER.warning("NOAA CUSP remote check failed for %s: %s", source_uri, exc)
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


def ensure_source_available(
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
    """Ensure one configured NOAA CUSP package is local or record the plan."""
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
    """Download a NOAA CUSP source package via streaming HTTP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_suffix(local_path.suffix + ".part")
    with session.get(source_uri, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    file.write(chunk)
    temporary_path.replace(local_path)


def vector_metadata_for_manifest(path: Path, dry_run: bool) -> dict[str, Any]:
    """Inspect local vector metadata for the source manifest."""
    if dry_run and not path.exists():
        return {"validation_status": "not_checked_dry_run"}
    if not path.exists():
        return {"validation_status": "missing"}
    try:
        dataset_uri = vector_dataset_uri(path)
        frame = gpd.read_file(dataset_uri)
        frame_wgs84 = frame.to_crs("EPSG:4326") if frame.crs is not None else frame
        geometry_types = sorted(str(value) for value in frame.geometry.geom_type.dropna().unique())
        return {
            "validation_status": "valid",
            "driver": "ESRI Shapefile ZIP" if path.suffix.lower() == ".zip" else None,
            "crs": str(frame.crs) if frame.crs is not None else None,
            "bounds": bounds_dict(coerce_bounds(frame_wgs84.total_bounds)),
            "geometry_types": geometry_types,
            "feature_count": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
        }
    except Exception as exc:  # noqa: BLE001
        return {"validation_status": "invalid", "error": str(exc)}


def vector_dataset_uri(path: Path) -> str:
    """Return a GeoPandas-readable URI for a local vector source package."""
    if path.suffix.lower() == ".zip":
        return f"zip://{path}"
    return str(path)


def build_download_manifest(
    *,
    cusp_config: NoaaCuspConfig,
    manifest_path: Path,
    query_manifest: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CUSP source-download manifest."""
    return {
        "command": "download-noaa-cusp",
        "config_path": str(cusp_config.config_path),
        "manifest_path": str(manifest_path),
        "query_manifest": str(query_manifest),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "source": source_metadata(cusp_config),
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
    """Merge the NOAA CUSP source manifest pointer into the metadata summary."""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary["noaa_cusp"] = {
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
