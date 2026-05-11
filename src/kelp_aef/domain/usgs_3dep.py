"""Query and download USGS 3DEP DEM tiles for the configured domain."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlencode, urlparse

import geopandas as gpd  # type: ignore[import-untyped]
import rasterio  # type: ignore[import-untyped]
import requests

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
DEFAULT_TNM_MAX_PRODUCTS = 100
PRODUCT_SOURCE_URI_FIELDS = ("downloadURL", "downloadUrl", "download_url", "url")
PRODUCT_ID_FIELDS = ("sourceId", "sourceID", "id", "title")
PRODUCT_TILE_PATTERN = re.compile(r"(USGS_13_[ns]\d{2}[ew]\d{3})(?:_\d{8})?\.tif$", re.I)


@dataclass(frozen=True)
class Usgs3depConfig:
    """Resolved config values for the USGS 3DEP DEM workflow."""

    config_path: Path
    region_name: str
    region_geometry_path: Path
    source_name: str
    source_role: str
    product_id: str
    metadata_url: str
    data_catalog_url: str
    tnm_api_url: str
    dataset_name: str
    product_extent: str
    product_format: str
    selection_policy: str
    local_source_root: Path
    query_manifest: Path
    source_manifest: Path
    horizontal_crs: str
    vertical_datum: str
    units: str
    resolution: str
    grid_spacing_arc_seconds: float
    elevation_sign_convention: str
    data_format: str
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
    """Result of ensuring one USGS 3DEP source raster exists locally or is planned."""

    status: str
    source_uri: str
    local_path: Path
    file_size_bytes: int | None
    remote_info: RemoteInfo | None
    error: str | None


def load_usgs_3dep_config(config_path: Path) -> Usgs3depConfig:
    """Load USGS 3DEP settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    domain = require_mapping(config.get("domain"), "domain")
    usgs_3dep = require_mapping(domain.get("usgs_3dep"), "domain.usgs_3dep")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    return Usgs3depConfig(
        config_path=config_path,
        region_name=require_string(region.get("name"), "region.name"),
        region_geometry_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        source_name=require_string(usgs_3dep.get("source_name"), "domain.usgs_3dep.source_name"),
        source_role=require_string(usgs_3dep.get("source_role"), "domain.usgs_3dep.source_role"),
        product_id=require_string(usgs_3dep.get("product_id"), "domain.usgs_3dep.product_id"),
        metadata_url=require_string(usgs_3dep.get("metadata_url"), "domain.usgs_3dep.metadata_url"),
        data_catalog_url=require_string(
            usgs_3dep.get("data_catalog_url"), "domain.usgs_3dep.data_catalog_url"
        ),
        tnm_api_url=require_string(usgs_3dep.get("tnm_api_url"), "domain.usgs_3dep.tnm_api_url"),
        dataset_name=require_string(usgs_3dep.get("dataset_name"), "domain.usgs_3dep.dataset_name"),
        product_extent=require_string(
            usgs_3dep.get("product_extent"), "domain.usgs_3dep.product_extent"
        ),
        product_format=require_string(
            usgs_3dep.get("product_format"), "domain.usgs_3dep.product_format"
        ),
        selection_policy=require_string(
            usgs_3dep.get("selection_policy"), "domain.usgs_3dep.selection_policy"
        ),
        local_source_root=Path(
            require_string(usgs_3dep.get("local_source_root"), "domain.usgs_3dep.local_source_root")
        ),
        query_manifest=Path(
            require_string(usgs_3dep.get("query_manifest"), "domain.usgs_3dep.query_manifest")
        ),
        source_manifest=Path(
            require_string(usgs_3dep.get("source_manifest"), "domain.usgs_3dep.source_manifest")
        ),
        horizontal_crs=require_string(
            usgs_3dep.get("horizontal_crs"), "domain.usgs_3dep.horizontal_crs"
        ),
        vertical_datum=require_string(
            usgs_3dep.get("vertical_datum"), "domain.usgs_3dep.vertical_datum"
        ),
        units=require_string(usgs_3dep.get("units"), "domain.usgs_3dep.units"),
        resolution=require_string(usgs_3dep.get("resolution"), "domain.usgs_3dep.resolution"),
        grid_spacing_arc_seconds=require_float(
            usgs_3dep.get("grid_spacing_arc_seconds"),
            "domain.usgs_3dep.grid_spacing_arc_seconds",
        ),
        elevation_sign_convention=require_string(
            usgs_3dep.get("elevation_sign_convention"),
            "domain.usgs_3dep.elevation_sign_convention",
        ),
        data_format=require_string(usgs_3dep.get("data_format"), "domain.usgs_3dep.data_format"),
        metadata_summary=optional_path(report_outputs.get("metadata_summary")),
    )


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


def query_usgs_3dep(
    config_path: Path,
    *,
    dry_run: bool = False,
    manifest_output: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_products: int = DEFAULT_TNM_MAX_PRODUCTS,
) -> int:
    """Query TNMAccess for 3DEP DEM products intersecting the configured geometry."""
    usgs_config = load_usgs_3dep_config(config_path)
    manifest_path = manifest_output or usgs_config.query_manifest
    LOGGER.info("Querying USGS 3DEP products for %s", usgs_config.region_name)

    region = region_for_query(usgs_config.region_geometry_path)
    region_bounds = coerce_bounds(region.total_bounds)
    request = build_tnm_api_request(usgs_config, region_bounds, max_products=max_products)
    api = (
        api_status_for_dry_run(request)
        if dry_run
        else fetch_tnm_api_status(request, timeout_seconds)
    )
    selected_artifacts = []
    if not dry_run:
        selected_products = select_latest_product_per_tile(
            cast(list[dict[str, Any]], api.get("products", []))
        )
        selected_artifacts = [
            selected_artifact_record(usgs_config, product, region_bounds)
            for product in selected_products
        ]

    manifest = build_query_manifest(
        usgs_config=usgs_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        region=region,
        request=request,
        api=api,
        selected_artifacts=selected_artifacts,
    )
    write_manifest_if_allowed(manifest_path, manifest, dry_run)
    LOGGER.info("USGS 3DEP query selected %s source rasters", len(selected_artifacts))
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


def build_tnm_api_request(
    usgs_config: Usgs3depConfig,
    region_bounds: tuple[float, float, float, float],
    *,
    max_products: int = DEFAULT_TNM_MAX_PRODUCTS,
) -> dict[str, Any]:
    """Build the TNMAccess product-query request for the region bounds."""
    params = {
        "datasets": usgs_config.dataset_name,
        "bbox": bbox_parameter(region_bounds),
        "prodExtents": usgs_config.product_extent,
        "prodFormats": usgs_config.product_format,
        "outputFormat": "JSON",
        "max": str(max_products),
    }
    return {
        "endpoint": usgs_config.tnm_api_url,
        "params": params,
        "url": f"{usgs_config.tnm_api_url}?{urlencode(params)}",
        "selection_method": "tnm_access_bbox_intersection",
    }


def bbox_parameter(bounds: tuple[float, float, float, float]) -> str:
    """Convert bounds into a TNMAccess bbox query string."""
    west, south, east, north = bounds
    return f"{west},{south},{east},{north}"


def api_status_for_dry_run(request: dict[str, Any]) -> dict[str, Any]:
    """Build the dry-run API status without sending an HTTP request."""
    return {
        "status": "not_requested_dry_run",
        "request_url": request["url"],
        "product_count": None,
        "total": None,
        "products": [],
        "error": None,
    }


def fetch_tnm_api_status(request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """Fetch TNMAccess product metadata for the planned request."""
    try:
        with requests.Session() as session:
            response = session.get(
                str(request["endpoint"]),
                params=cast(dict[str, str], request["params"]),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
    except requests.RequestException as exc:
        LOGGER.warning("USGS 3DEP TNMAccess query failed for %s: %s", request["url"], exc)
        return {
            "status": "request_failed",
            "request_url": request["url"],
            "product_count": None,
            "products": [],
            "error": str(exc),
        }

    products = product_items_from_api_payload(payload)
    total = total_product_count(payload)
    return {
        "status": "queried",
        "request_url": response.url,
        "product_count": len(products),
        "total": total,
        "products": products,
        "error": None,
    }


def product_items_from_api_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract product item objects from a TNMAccess JSON response."""
    if not isinstance(payload, dict):
        msg = "TNMAccess response must be a JSON object"
        raise ValueError(msg)
    items = payload.get("items")
    if items is None:
        items = payload.get("products", [])
    if not isinstance(items, list):
        msg = "TNMAccess response item collection must be a JSON array"
        raise ValueError(msg)
    return [cast(dict[str, Any], item) for item in items if isinstance(item, dict)]


def total_product_count(payload: Any) -> int | None:
    """Return the total product count from a TNMAccess response if present."""
    if not isinstance(payload, dict):
        return None
    value = payload.get("total")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def select_latest_product_per_tile(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only the latest TNMAccess product for each 1 x 1 degree DEM tile."""
    selected: dict[str, dict[str, Any]] = {}
    for product in products:
        source_uri = source_uri_from_product(product)
        tile_key = tile_key_from_source_uri(source_uri) or artifact_id_from_product(
            product, source_uri
        )
        previous = selected.get(tile_key)
        if previous is None or product_sort_key(product) > product_sort_key(previous):
            selected[tile_key] = product
    return [selected[key] for key in sorted(selected)]


def product_sort_key(product: dict[str, Any]) -> tuple[str, str, str]:
    """Return a stable recency key for TNMAccess product selection."""
    publication_date = optional_product_string(product, "publicationDate") or ""
    last_updated = optional_product_string(product, "lastUpdated") or ""
    date_created = optional_product_string(product, "dateCreated") or ""
    return publication_date, last_updated, date_created


def selected_artifact_record(
    usgs_config: Usgs3depConfig,
    product: dict[str, Any],
    region_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Build one selected 3DEP source-raster manifest record from a TNM product."""
    source_uri = source_uri_from_product(product)
    local_path = local_source_path_for_source_uri(source_uri, usgs_config.local_source_root)
    return {
        "artifact_id": artifact_id_from_product(product, source_uri),
        "title": optional_product_string(product, "title"),
        "source_id": optional_product_string(product, "sourceId"),
        "source_uri": source_uri,
        "local_path": str(local_path),
        "source_role": usgs_config.source_role,
        "data_format": usgs_config.data_format,
        "bounds": product_bounds_for_manifest(product),
        "coverage_check": {
            "selection_method": "tnm_access_bbox_intersection",
            "selection_policy": usgs_config.selection_policy,
            "intersects_region": True,
            "region_bounds": bounds_dict(region_bounds),
        },
        "horizontal_crs": usgs_config.horizontal_crs,
        "vertical_datum": usgs_config.vertical_datum,
        "units": usgs_config.units,
        "resolution": usgs_config.resolution,
        "grid_spacing_arc_seconds": usgs_config.grid_spacing_arc_seconds,
        "elevation_sign_convention": usgs_config.elevation_sign_convention,
        "file_size_bytes": product_file_size(product),
        "publication_date": optional_product_string(product, "publicationDate"),
        "last_updated": optional_product_string(product, "lastUpdated"),
    }


def source_uri_from_product(product: dict[str, Any]) -> str:
    """Return the source download URL from a TNMAccess product item."""
    for field in PRODUCT_SOURCE_URI_FIELDS:
        value = product.get(field)
        if isinstance(value, str) and value:
            return value
    urls = product.get("urls")
    if isinstance(urls, dict):
        for value in urls.values():
            if isinstance(value, str) and value:
                return value
    msg = f"TNMAccess product is missing a source URL field; tried {PRODUCT_SOURCE_URI_FIELDS}"
    raise ValueError(msg)


def artifact_id_from_product(product: dict[str, Any], source_uri: str) -> str:
    """Return a stable artifact id from product metadata or source basename."""
    tile_key = tile_key_from_source_uri(source_uri)
    if tile_key is not None:
        return tile_key
    for field in PRODUCT_ID_FIELDS:
        value = product.get(field)
        if isinstance(value, str) and value:
            return value
    return Path(urlparse(source_uri).path).stem


def tile_key_from_source_uri(source_uri: str) -> str | None:
    """Return the 1 x 1 degree USGS 3DEP tile key from a source URI."""
    match = PRODUCT_TILE_PATTERN.search(Path(urlparse(source_uri).path).name)
    return match.group(1) if match else None


def optional_product_string(product: dict[str, Any], field: str) -> str | None:
    """Return a product string field when present."""
    value = product.get(field)
    return value if isinstance(value, str) and value else None


def product_file_size(product: dict[str, Any]) -> int | None:
    """Return a TNMAccess product file size when available."""
    for field in ("sizeInBytes", "size", "filesize"):
        value = product.get(field)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def product_bounds_for_manifest(product: dict[str, Any]) -> dict[str, float] | None:
    """Return product bounds from known TNMAccess bbox shapes when available."""
    for field in ("boundingBox", "bbox"):
        bounds = product.get(field)
        parsed = parse_product_bounds(bounds)
        if parsed is not None:
            return bounds_dict(parsed)
    return None


def parse_product_bounds(value: Any) -> tuple[float, float, float, float] | None:
    """Parse a product bounds object into west, south, east, north order."""
    if isinstance(value, list | tuple) and len(value) == 4:
        return coerce_bounds(value)
    if not isinstance(value, dict):
        return None
    west = first_number(value, ("west", "minX", "xmin", "minLon", "left"))
    south = first_number(value, ("south", "minY", "ymin", "minLat", "bottom"))
    east = first_number(value, ("east", "maxX", "xmax", "maxLon", "right"))
    north = first_number(value, ("north", "maxY", "ymax", "maxLat", "top"))
    if None in (west, south, east, north):
        return None
    return cast(tuple[float, float, float, float], (west, south, east, north))


def first_number(values: dict[str, Any], fields: tuple[str, ...]) -> float | None:
    """Return the first numeric value from a mapping field list."""
    for field in fields:
        value = values.get(field)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def local_source_path_for_source_uri(source_uri: str, local_source_root: Path) -> Path:
    """Map a remote 3DEP raster URL to a local source-raster path."""
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
    usgs_config: Usgs3depConfig,
    manifest_path: Path,
    dry_run: bool,
    region: gpd.GeoDataFrame,
    request: dict[str, Any],
    api: dict[str, Any],
    selected_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full USGS 3DEP source-query manifest."""
    return {
        "command": "query-usgs-3dep",
        "config_path": str(usgs_config.config_path),
        "configured_manifest_path": str(usgs_config.query_manifest),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "query_status": query_status(api, selected_artifacts, dry_run),
        "source": source_metadata(usgs_config),
        "region": {
            "name": usgs_config.region_name,
            "geometry_path": str(usgs_config.region_geometry_path),
            "bounds": bounds_dict(coerce_bounds(region.total_bounds)),
        },
        "tnm_request": request,
        "tnm_api": {
            "status": api["status"],
            "request_url": api["request_url"],
            "product_count": api["product_count"],
            "total": api.get("total"),
            "error": api["error"],
        },
        "selection_policy": usgs_config.selection_policy,
        "selected_artifact_count": len(selected_artifacts),
        "selected_artifacts": selected_artifacts,
    }


def query_status(
    api: dict[str, Any], selected_artifacts: list[dict[str, Any]], dry_run: bool
) -> str:
    """Return the manifest-level query status."""
    if dry_run:
        return "planned_no_api_request"
    if api["status"] != "queried":
        return "api_request_failed"
    if selected_artifacts:
        return "selected_sources"
    return "no_sources"


def source_metadata(usgs_config: Usgs3depConfig) -> dict[str, Any]:
    """Return common USGS 3DEP source metadata for manifests."""
    return {
        "source_name": usgs_config.source_name,
        "source_role": usgs_config.source_role,
        "product_id": usgs_config.product_id,
        "metadata_url": usgs_config.metadata_url,
        "data_catalog_url": usgs_config.data_catalog_url,
        "tnm_api_url": usgs_config.tnm_api_url,
        "dataset_name": usgs_config.dataset_name,
        "product_extent": usgs_config.product_extent,
        "product_format": usgs_config.product_format,
        "selection_policy": usgs_config.selection_policy,
        "horizontal_crs": usgs_config.horizontal_crs,
        "vertical_datum": usgs_config.vertical_datum,
        "units": usgs_config.units,
        "resolution": usgs_config.resolution,
        "grid_spacing_arc_seconds": usgs_config.grid_spacing_arc_seconds,
        "elevation_sign_convention": usgs_config.elevation_sign_convention,
        "data_format": usgs_config.data_format,
        "license_access_notes": (
            "USGS 3DEP data are public domain; this workflow treats 3DEP as a "
            "land-side fallback and not an offshore bathymetry source."
        ),
    }


def download_usgs_3dep(
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
    """Download or plan 3DEP source-raster downloads from a query manifest."""
    usgs_config = load_usgs_3dep_config(config_path)
    source_query_manifest = query_manifest or usgs_config.query_manifest
    manifest_path = manifest_output or usgs_config.source_manifest
    query = load_query_manifest(source_query_manifest)
    selected_artifacts = cast(list[dict[str, Any]], query.get("selected_artifacts", []))
    LOGGER.info("Planning USGS 3DEP download for %s selected sources", len(selected_artifacts))

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
        usgs_config=usgs_config,
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
        LOGGER.info("Wrote USGS 3DEP source manifest: %s", manifest_path)

    if not dry_run and usgs_config.metadata_summary is not None and manifest_output is None:
        update_metadata_summary(usgs_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", usgs_config.metadata_summary)

    return 0


def load_query_manifest(path: Path) -> dict[str, Any]:
    """Load a USGS 3DEP source-query manifest."""
    if not path.exists():
        msg = f"USGS 3DEP query manifest does not exist: {path}"
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
    """Download or plan one selected 3DEP source raster and return its record."""
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
    raster_metadata = raster_metadata_for_manifest(local_path, dry_run)
    return {
        "artifact_id": artifact.get("artifact_id"),
        "title": artifact.get("title"),
        "source_id": artifact.get("source_id"),
        "source_uri": source_uri,
        "local_path": str(local_path),
        "source_role": artifact.get("source_role"),
        "bounds": artifact.get("bounds"),
        "coverage_check": artifact.get("coverage_check"),
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
        LOGGER.warning("USGS 3DEP remote check failed for %s: %s", source_uri, exc)
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
    """Ensure one configured USGS 3DEP raster is local or record the plan."""
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
    """Download a USGS 3DEP raster via streaming HTTP."""
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
    """Inspect local raster metadata for the source manifest."""
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
    usgs_config: Usgs3depConfig,
    manifest_path: Path,
    query_manifest: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full USGS 3DEP source-download manifest."""
    return {
        "command": "download-usgs-3dep",
        "config_path": str(usgs_config.config_path),
        "manifest_path": str(manifest_path),
        "query_manifest": str(query_manifest),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "source": source_metadata(usgs_config),
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
    """Merge the USGS 3DEP source manifest pointer into the metadata summary."""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary["usgs_3dep"] = {
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
