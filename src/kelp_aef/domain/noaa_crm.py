"""Query and download NOAA CRM topo-bathy sources for the configured domain."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import geopandas as gpd  # type: ignore[import-untyped]
import rasterio  # type: ignore[import-untyped]
import requests
from shapely.geometry import box

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120.0
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
DEFAULT_QUERY_PADDING_DEGREES = 0.0
DEFAULT_DOWNLOAD_MAX_ATTEMPTS = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class NoaaCrmProduct:
    """Resolved config values for one NOAA CRM product in the registry."""

    product_id: str
    product_name: str
    product_version: str | None
    source_role: str
    metadata_url: str
    thredds_catalog_url: str | None
    source_uri: str
    opendap_url: str | None
    local_filename: str | None
    data_variable: str
    bounds: tuple[float, float, float, float]
    horizontal_crs: str
    vertical_datum: str
    units: str
    resolution: str
    grid_spacing_arc_seconds: float
    elevation_sign_convention: str
    data_format: str


@dataclass(frozen=True)
class NoaaCrmConfig:
    """Resolved config values for the NOAA CRM source workflow."""

    config_path: Path
    region_name: str
    region_geometry_path: Path
    source_name: str
    source_role: str
    source_page_url: str
    local_source_root: Path
    query_manifest: Path
    source_manifest: Path
    download_mode: str
    query_padding_degrees: float
    products: tuple[NoaaCrmProduct, ...]
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
    """Result of ensuring one NOAA CRM source file exists locally or is planned."""

    status: str
    source_uri: str
    local_path: Path
    file_size_bytes: int | None
    remote_info: RemoteInfo | None
    partial_file_size_bytes: int | None
    attempt_count: int | None
    error: str | None


def load_noaa_crm_config(config_path: Path) -> NoaaCrmConfig:
    """Load NOAA CRM settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    domain = require_mapping(config.get("domain"), "domain")
    crm = require_mapping(domain.get("noaa_crm"), "domain.noaa_crm")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    return NoaaCrmConfig(
        config_path=config_path,
        region_name=require_string(region.get("name"), "region.name"),
        region_geometry_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        source_name=require_string(crm.get("source_name"), "domain.noaa_crm.source_name"),
        source_role=require_string(crm.get("source_role"), "domain.noaa_crm.source_role"),
        source_page_url=require_string(
            crm.get("source_page_url"), "domain.noaa_crm.source_page_url"
        ),
        local_source_root=Path(
            require_string(crm.get("local_source_root"), "domain.noaa_crm.local_source_root")
        ),
        query_manifest=Path(
            require_string(crm.get("query_manifest"), "domain.noaa_crm.query_manifest")
        ),
        source_manifest=Path(
            require_string(crm.get("source_manifest"), "domain.noaa_crm.source_manifest")
        ),
        download_mode=require_string(crm.get("download_mode"), "domain.noaa_crm.download_mode"),
        query_padding_degrees=optional_float(
            crm.get("query_padding_degrees"),
            "domain.noaa_crm.query_padding_degrees",
            DEFAULT_QUERY_PADDING_DEGREES,
        ),
        products=load_product_registry(crm.get("products"), "domain.noaa_crm.products"),
        metadata_summary=optional_path(report_outputs.get("metadata_summary")),
    )


def load_product_registry(value: object, name: str) -> tuple[NoaaCrmProduct, ...]:
    """Validate the configured CRM product registry."""
    if not isinstance(value, list):
        msg = f"config field must be a list: {name}"
        raise ValueError(msg)
    products = tuple(load_product(product, f"{name}[]") for product in value)
    if not products:
        msg = f"config field must contain at least one product: {name}"
        raise ValueError(msg)
    return products


def load_product(value: object, name: str) -> NoaaCrmProduct:
    """Validate one configured CRM product."""
    product = require_mapping(value, name)
    return NoaaCrmProduct(
        product_id=require_string(product.get("product_id"), f"{name}.product_id"),
        product_name=require_string(product.get("product_name"), f"{name}.product_name"),
        product_version=optional_string(product.get("product_version"), f"{name}.product_version"),
        source_role=require_string(product.get("source_role"), f"{name}.source_role"),
        metadata_url=require_string(product.get("metadata_url"), f"{name}.metadata_url"),
        thredds_catalog_url=optional_string(
            product.get("thredds_catalog_url"), f"{name}.thredds_catalog_url"
        ),
        source_uri=require_string(product.get("source_uri"), f"{name}.source_uri"),
        opendap_url=optional_string(product.get("opendap_url"), f"{name}.opendap_url"),
        local_filename=optional_string(product.get("local_filename"), f"{name}.local_filename"),
        data_variable=require_string(product.get("data_variable"), f"{name}.data_variable"),
        bounds=require_bounds(product.get("bounds"), f"{name}.bounds"),
        horizontal_crs=require_string(product.get("horizontal_crs"), f"{name}.horizontal_crs"),
        vertical_datum=require_string(product.get("vertical_datum"), f"{name}.vertical_datum"),
        units=require_string(product.get("units"), f"{name}.units"),
        resolution=require_string(product.get("resolution"), f"{name}.resolution"),
        grid_spacing_arc_seconds=require_float(
            product.get("grid_spacing_arc_seconds"),
            f"{name}.grid_spacing_arc_seconds",
        ),
        elevation_sign_convention=require_string(
            product.get("elevation_sign_convention"),
            f"{name}.elevation_sign_convention",
        ),
        data_format=require_string(product.get("data_format"), f"{name}.data_format"),
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


def require_bounds(value: object, name: str) -> tuple[float, float, float, float]:
    """Validate a bounds mapping as a west, south, east, north tuple."""
    mapping = require_mapping(value, name)
    return (
        require_float(mapping.get("west"), f"{name}.west"),
        require_float(mapping.get("south"), f"{name}.south"),
        require_float(mapping.get("east"), f"{name}.east"),
        require_float(mapping.get("north"), f"{name}.north"),
    )


def require_float(value: object, name: str) -> float:
    """Validate a dynamic config value as a floating-point number."""
    if not isinstance(value, int | float):
        msg = f"config field must be a number: {name}"
        raise ValueError(msg)
    return float(value)


def optional_float(value: object, name: str, default: float) -> float:
    """Validate an optional dynamic config value as a floating-point number."""
    if value is None:
        return default
    return require_float(value, name)


def query_noaa_crm(
    config_path: Path,
    *,
    dry_run: bool = False,
    manifest_output: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    skip_remote_checks: bool = False,
) -> int:
    """Query NOAA CRM products that intersect the configured target-grid footprint."""
    crm_config = load_noaa_crm_config(config_path)
    manifest_path = manifest_output or crm_config.query_manifest
    LOGGER.info("Querying NOAA CRM product registry for %s", crm_config.region_name)

    region = region_for_query(crm_config.region_geometry_path)
    selected_products: list[dict[str, Any]] = []
    skipped_products: list[dict[str, Any]] = []

    with requests.Session() as session:
        for product in crm_config.products:
            remote_metadata = product_service_metadata(
                session=session,
                product=product,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
                skip_remote_checks=skip_remote_checks,
            )
            product_bounds = service_bounds_or_configured_bounds(product, remote_metadata)
            coverage = coverage_check(
                product_bounds=product_bounds,
                configured_product_bounds=product.bounds,
                region=region,
                padding_degrees=crm_config.query_padding_degrees,
            )
            if coverage["intersects_region"]:
                selected_products.append(
                    selected_product_record(
                        crm_config=crm_config,
                        product=product,
                        coverage=coverage,
                        remote_metadata=remote_metadata,
                    )
                )
            else:
                skipped_products.append(
                    skipped_product_record(
                        product=product,
                        coverage=coverage,
                        remote_metadata=remote_metadata,
                    )
                )

    manifest = build_query_manifest(
        crm_config=crm_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        skip_remote_checks=skip_remote_checks,
        region=region,
        selected_products=selected_products,
        skipped_products=skipped_products,
    )
    write_manifest_if_allowed(manifest_path, manifest, dry_run)
    LOGGER.info("NOAA CRM query selected %s products", len(selected_products))
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


def product_service_metadata(
    *,
    session: requests.Session,
    product: NoaaCrmProduct,
    dry_run: bool,
    timeout_seconds: float,
    skip_remote_checks: bool,
) -> dict[str, Any]:
    """Fetch lightweight THREDDS/OPeNDAP metadata for a product when enabled."""
    if dry_run or skip_remote_checks or product.opendap_url is None:
        return {"status": "not_checked", "metadata_uri": metadata_uri_for_product(product)}

    metadata_uri = metadata_uri_for_product(product)
    if metadata_uri is None:
        return {"status": "not_configured", "metadata_uri": None}

    try:
        response = session.get(metadata_uri, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("NOAA CRM metadata check failed for %s: %s", product.product_id, exc)
        return {"status": "error", "metadata_uri": metadata_uri, "error": str(exc)}

    parsed = parse_opendap_das(
        response.text,
        data_variable=product.data_variable,
        configured_vertical_datum=product.vertical_datum,
    )
    parsed["status"] = "fetched"
    parsed["metadata_uri"] = metadata_uri
    return parsed


def metadata_uri_for_product(product: NoaaCrmProduct) -> str | None:
    """Return the OPeNDAP DAS URI for a configured CRM product."""
    if product.opendap_url is None:
        return None
    return f"{product.opendap_url}.das"


def parse_opendap_das(
    das_text: str,
    *,
    data_variable: str,
    configured_vertical_datum: str,
) -> dict[str, Any]:
    """Parse lightweight bounds and vertical metadata from an OPeNDAP DAS response."""
    lon_range = actual_range_from_das(das_text, "lon")
    lat_range = actual_range_from_das(das_text, "lat")
    data_range = actual_range_from_das(das_text, data_variable)
    vertical_datum = quoted_attribute_from_das(das_text, "vert_crs_name")
    vertical_epsg = quoted_attribute_from_das(das_text, "vert_crs_epsg")
    parsed: dict[str, Any] = {
        "bounds": None,
        "data_actual_range": range_dict(data_range),
        "vertical_datum": vertical_datum or configured_vertical_datum,
        "vertical_epsg": vertical_epsg,
    }
    if lon_range is not None and lat_range is not None:
        parsed["bounds"] = bounds_dict((lon_range[0], lat_range[0], lon_range[1], lat_range[1]))
    return parsed


def actual_range_from_das(das_text: str, variable_name: str) -> tuple[float, float] | None:
    """Extract one variable actual_range tuple from an OPeNDAP DAS response."""
    pattern = rf"{re.escape(variable_name)}\s*\{{.*?actual_range\s+([^;]+);"
    match = re.search(pattern, das_text, flags=re.DOTALL)
    if match is None:
        return None
    values = [part.strip() for part in match.group(1).split(",")]
    if len(values) != 2:
        return None
    return float(values[0]), float(values[1])


def quoted_attribute_from_das(das_text: str, attribute_name: str) -> str | None:
    """Extract one quoted string attribute from an OPeNDAP DAS response."""
    pattern = rf'{re.escape(attribute_name)}\s+"([^"]+)"'
    match = re.search(pattern, das_text)
    return match.group(1) if match else None


def service_bounds_or_configured_bounds(
    product: NoaaCrmProduct,
    remote_metadata: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return service-discovered product bounds or the configured fallback bounds."""
    bounds = remote_metadata.get("bounds")
    if isinstance(bounds, dict):
        return bounds_tuple(bounds)
    return product.bounds


def coverage_check(
    *,
    product_bounds: tuple[float, float, float, float],
    configured_product_bounds: tuple[float, float, float, float],
    region: gpd.GeoDataFrame,
    padding_degrees: float,
) -> dict[str, Any]:
    """Check whether one CRM product intersects the configured region."""
    product_geometry = box(*product_bounds)
    region_union = region.geometry.union_all()
    intersection = product_geometry.intersection(region_union)
    intersects_region = not intersection.is_empty
    intersection_bounds = (
        bounds_dict(coerce_bounds(intersection.bounds)) if intersects_region else None
    )
    subset_bounds = (
        padded_subset_bounds(intersection.bounds, product_bounds, padding_degrees)
        if intersects_region
        else None
    )
    return {
        "intersects_region": intersects_region,
        "selection_method": "configured_region_geometry_intersection",
        "configured_product_bounds": bounds_dict(configured_product_bounds),
        "product_bounds": bounds_dict(product_bounds),
        "region_bounds": bounds_dict(coerce_bounds(region.total_bounds)),
        "intersection_bounds": intersection_bounds,
        "recommended_subset_bounds": subset_bounds,
        "query_padding_degrees": padding_degrees,
    }


def padded_subset_bounds(
    intersection_bounds: tuple[float, float, float, float],
    product_bounds: tuple[float, float, float, float],
    padding_degrees: float,
) -> dict[str, float]:
    """Pad intersection bounds and clip the result to product bounds."""
    west, south, east, north = coerce_bounds(intersection_bounds)
    product_west, product_south, product_east, product_north = product_bounds
    padded = (
        max(product_west, west - padding_degrees),
        max(product_south, south - padding_degrees),
        min(product_east, east + padding_degrees),
        min(product_north, north + padding_degrees),
    )
    return bounds_dict(padded)


def selected_product_record(
    *,
    crm_config: NoaaCrmConfig,
    product: NoaaCrmProduct,
    coverage: dict[str, Any],
    remote_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one selected CRM product manifest record."""
    local_path = local_source_path_for_product(product, crm_config.local_source_root)
    return {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "product_version": product.product_version,
        "source_role": product.source_role,
        "metadata_url": product.metadata_url,
        "thredds_catalog_url": product.thredds_catalog_url,
        "source_uri": product.source_uri,
        "opendap_url": product.opendap_url,
        "local_path": str(local_path),
        "download_mode": crm_config.download_mode,
        "data_variable": product.data_variable,
        "bounds": coverage["product_bounds"],
        "coverage_check": coverage,
        "remote_metadata": remote_metadata,
        "horizontal_crs": product.horizontal_crs,
        "vertical_datum": remote_metadata.get("vertical_datum") or product.vertical_datum,
        "vertical_epsg": remote_metadata.get("vertical_epsg"),
        "units": product.units,
        "resolution": product.resolution,
        "grid_spacing_arc_seconds": product.grid_spacing_arc_seconds,
        "elevation_sign_convention": product.elevation_sign_convention,
        "data_format": product.data_format,
    }


def skipped_product_record(
    *,
    product: NoaaCrmProduct,
    coverage: dict[str, Any],
    remote_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one skipped CRM product manifest record."""
    return {
        "product_id": product.product_id,
        "product_name": product.product_name,
        "skip_reason": "does_not_intersect_region",
        "source_uri": product.source_uri,
        "bounds": coverage["product_bounds"],
        "coverage_check": coverage,
        "remote_metadata": remote_metadata,
    }


def local_source_path_for_product(product: NoaaCrmProduct, local_source_root: Path) -> Path:
    """Map a CRM product to its configured local source path."""
    if product.local_filename:
        return local_source_root / product.local_filename
    return local_source_path_for_source_uri(product.source_uri, local_source_root)


def local_source_path_for_source_uri(source_uri: str, local_source_root: Path) -> Path:
    """Map a remote CRM source URL to a local source path."""
    basename = Path(urlparse(source_uri).path).name
    if not basename:
        msg = f"source URI does not contain a file name: {source_uri}"
        raise ValueError(msg)
    return local_source_root / basename


def build_query_manifest(
    *,
    crm_config: NoaaCrmConfig,
    manifest_path: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    region: gpd.GeoDataFrame,
    selected_products: list[dict[str, Any]],
    skipped_products: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CRM source-query manifest."""
    return {
        "command": "query-noaa-crm",
        "config_path": str(crm_config.config_path),
        "configured_manifest_path": str(crm_config.query_manifest),
        "manifest_path": str(manifest_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "query_status": "selected_sources" if selected_products else "no_product_coverage",
        "source": source_metadata(crm_config),
        "region": {
            "name": crm_config.region_name,
            "geometry_path": str(crm_config.region_geometry_path),
            "bounds": bounds_dict(coerce_bounds(region.total_bounds)),
        },
        "selected_product_count": len(selected_products),
        "selected_products": selected_products,
        "skipped_product_count": len(skipped_products),
        "skipped_products": skipped_products,
        "coverage_note": (
            "California coastal coverage is expected from the Southern California v2 "
            "and CRM Volume 7 pairing; this manifest selects only products that "
            "intersect the configured footprint."
        ),
    }


def source_metadata(crm_config: NoaaCrmConfig) -> dict[str, Any]:
    """Return common NOAA CRM source metadata for manifests."""
    return {
        "source_name": crm_config.source_name,
        "source_role": crm_config.source_role,
        "source_page_url": crm_config.source_page_url,
        "download_mode": crm_config.download_mode,
        "query_padding_degrees": crm_config.query_padding_degrees,
        "product_count": len(crm_config.products),
        "license_access_notes": (
            "NOAA/NCEI public data; use with NOAA attribution and standard NOAA "
            "no-warranty/use-at-own-risk notices."
        ),
    }


def download_noaa_crm(
    config_path: Path,
    *,
    dry_run: bool = False,
    skip_remote_checks: bool = False,
    manifest_output: Path | None = None,
    query_manifest: Path | None = None,
    timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    force: bool = False,
    max_attempts: int = DEFAULT_DOWNLOAD_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> int:
    """Download or plan NOAA CRM source downloads from a query manifest."""
    crm_config = load_noaa_crm_config(config_path)
    source_query_manifest = query_manifest or crm_config.query_manifest
    manifest_path = manifest_output or crm_config.source_manifest
    query = load_query_manifest(source_query_manifest)
    selected_products = cast(list[dict[str, Any]], query.get("selected_products", []))
    LOGGER.info("Planning NOAA CRM download for %s selected products", len(selected_products))

    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        for product in selected_products:
            records.append(
                process_product_record(
                    product=product,
                    session=session,
                    dry_run=dry_run,
                    skip_remote_checks=skip_remote_checks,
                    timeout_seconds=timeout_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                    force=force,
                    max_attempts=max_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
            )

    manifest = build_download_manifest(
        crm_config=crm_config,
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
        LOGGER.info("Wrote NOAA CRM source manifest: %s", manifest_path)

    if not dry_run and crm_config.metadata_summary is not None and manifest_output is None:
        update_metadata_summary(crm_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", crm_config.metadata_summary)

    return 0


def load_query_manifest(path: Path) -> dict[str, Any]:
    """Load a NOAA CRM source-query manifest."""
    if not path.exists():
        msg = f"NOAA CRM query manifest does not exist: {path}"
        raise FileNotFoundError(msg)
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        msg = f"query manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def process_product_record(
    *,
    product: dict[str, Any],
    session: requests.Session,
    dry_run: bool,
    skip_remote_checks: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    """Download or plan one selected CRM product and return its manifest record."""
    source_uri = require_string(product.get("source_uri"), "selected_products[].source_uri")
    local_path = Path(require_string(product.get("local_path"), "selected_products[].local_path"))
    remote_info = (
        None
        if dry_run or skip_remote_checks
        else head_remote_asset(session, source_uri, timeout_seconds)
    )
    transfer = ensure_source_available(
        source_uri=source_uri,
        local_path=local_path,
        session=session,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
        remote_info=remote_info,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    raster_metadata = raster_metadata_for_manifest(local_path, dry_run)
    return {
        "product_id": product.get("product_id"),
        "product_name": product.get("product_name"),
        "source_role": product.get("source_role"),
        "source_uri": source_uri,
        "opendap_url": product.get("opendap_url"),
        "local_path": str(local_path),
        "download_mode": product.get("download_mode"),
        "coverage_check": product.get("coverage_check"),
        "horizontal_crs": product.get("horizontal_crs"),
        "vertical_datum": product.get("vertical_datum"),
        "vertical_epsg": product.get("vertical_epsg"),
        "units": product.get("units"),
        "resolution": product.get("resolution"),
        "grid_spacing_arc_seconds": product.get("grid_spacing_arc_seconds"),
        "elevation_sign_convention": product.get("elevation_sign_convention"),
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
        LOGGER.warning("NOAA CRM remote check failed for %s: %s", source_uri, exc)
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
    max_attempts: int,
    retry_backoff_seconds: float,
) -> TransferResult:
    """Ensure one configured NOAA CRM source is local or record the plan."""
    if dry_run:
        return TransferResult(
            status="dry_run",
            source_uri=source_uri,
            local_path=local_path,
            file_size_bytes=local_file_size(local_path),
            remote_info=remote_info,
            partial_file_size_bytes=local_file_size(partial_path_for(local_path)),
            attempt_count=None,
            error=None,
        )
    if local_path.exists() and not force:
        return TransferResult(
            status="skipped_existing",
            source_uri=source_uri,
            local_path=local_path,
            file_size_bytes=local_path.stat().st_size,
            remote_info=remote_info,
            partial_file_size_bytes=local_file_size(partial_path_for(local_path)),
            attempt_count=0,
            error=None,
        )

    attempt_count = download_file(
        session=session,
        source_uri=source_uri,
        local_path=local_path,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return TransferResult(
        status="downloaded",
        source_uri=source_uri,
        local_path=local_path,
        file_size_bytes=local_path.stat().st_size,
        remote_info=remote_info,
        partial_file_size_bytes=local_file_size(partial_path_for(local_path)),
        attempt_count=attempt_count,
        error=None,
    )


def local_file_size(path: Path) -> int | None:
    """Return local file size if the file exists."""
    return path.stat().st_size if path.exists() else None


def partial_path_for(local_path: Path) -> Path:
    """Return the temporary partial-download path for a local source file."""
    return local_path.with_suffix(local_path.suffix + ".part")


def download_file(
    *,
    session: requests.Session,
    source_uri: str,
    local_path: Path,
    timeout_seconds: float,
    chunk_size_bytes: int,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> int:
    """Download a NOAA CRM source file via streaming HTTP with resume and retry."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = partial_path_for(local_path)
    last_error: Exception | None = None

    for attempt_number in range(1, max_attempts + 1):
        resume_from = local_file_size(temporary_path) or 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
        try:
            LOGGER.info(
                "Downloading %s to %s, attempt %s/%s%s",
                source_uri,
                local_path,
                attempt_number,
                max_attempts,
                f", resuming from {resume_from} bytes" if resume_from else "",
            )
            stream_response_to_file(
                session=session,
                source_uri=source_uri,
                temporary_path=temporary_path,
                headers=headers,
                timeout_seconds=timeout_seconds,
                chunk_size_bytes=chunk_size_bytes,
                resume_from=resume_from,
            )
            temporary_path.replace(local_path)
            return attempt_number
        except requests.RequestException as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc

        if attempt_number < max_attempts:
            sleep_seconds = retry_backoff_seconds * attempt_number
            LOGGER.warning(
                "NOAA CRM download failed for %s on attempt %s/%s: %s. Retrying in %.1f seconds.",
                source_uri,
                attempt_number,
                max_attempts,
                last_error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    msg = f"NOAA CRM download failed after {max_attempts} attempts: {source_uri}"
    if last_error is None:
        raise RuntimeError(msg)
    raise RuntimeError(msg) from last_error


def stream_response_to_file(
    *,
    session: requests.Session,
    source_uri: str,
    temporary_path: Path,
    headers: dict[str, str] | None,
    timeout_seconds: float,
    chunk_size_bytes: int,
    resume_from: int,
) -> None:
    """Stream one HTTP response into the temporary file, appending on 206 responses."""
    with session.get(
        source_uri,
        stream=True,
        timeout=timeout_seconds,
        headers=headers,
    ) as response:
        if resume_from and response.status_code == 416:
            LOGGER.warning("Remote rejected resume range for %s; restarting download.", source_uri)
            temporary_path.unlink(missing_ok=True)
            raise OSError("remote rejected resume range")
        response.raise_for_status()
        append_mode = resume_from > 0 and response.status_code == 206
        if resume_from > 0 and response.status_code == 200:
            LOGGER.warning("Remote ignored resume range for %s; restarting download.", source_uri)
        mode = "ab" if append_mode else "wb"
        expected_total_size = expected_total_size_from_response(response)
        with temporary_path.open(mode) as file:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    file.write(chunk)
        validate_temporary_download_size(temporary_path, expected_total_size)


def expected_total_size_from_response(response: requests.Response) -> int | None:
    """Return the expected final file size from Content-Range or Content-Length."""
    content_range = response.headers.get("Content-Range")
    if content_range is not None:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length is not None and response.status_code == 200:
        return int(content_length)
    return None


def validate_temporary_download_size(path: Path, expected_total_size: int | None) -> None:
    """Validate a temporary download against the expected final file size."""
    if expected_total_size is None:
        return
    actual_size = path.stat().st_size
    if actual_size < expected_total_size:
        msg = (
            f"incomplete download for {path}: "
            f"{actual_size} bytes < expected {expected_total_size} bytes"
        )
        raise OSError(msg)


def raster_metadata_for_manifest(path: Path, dry_run: bool) -> dict[str, Any]:
    """Inspect local raster metadata for the CRM source manifest."""
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
    crm_config: NoaaCrmConfig,
    manifest_path: Path,
    query_manifest: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full NOAA CRM source-download manifest."""
    return {
        "command": "download-noaa-crm",
        "config_path": str(crm_config.config_path),
        "manifest_path": str(manifest_path),
        "query_manifest": str(query_manifest),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "source": source_metadata(crm_config),
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
        "partial_file_size_bytes": transfer.partial_file_size_bytes,
        "attempt_count": transfer.attempt_count,
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
    """Merge the NOAA CRM source manifest pointer into the metadata summary."""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary["noaa_crm"] = {
        "manifest_path": manifest["manifest_path"],
        "query_manifest": manifest["query_manifest"],
        "source": manifest["source"],
        "record_count": manifest["record_count"],
    }
    write_json(summary_path, summary)


def bounds_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    """Convert bounds tuple into a named JSON object."""
    west, south, east, north = bounds
    return {"west": west, "south": south, "east": east, "north": north}


def bounds_tuple(bounds: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert a named JSON bounds object into a tuple."""
    return (
        require_float(bounds.get("west"), "bounds.west"),
        require_float(bounds.get("south"), "bounds.south"),
        require_float(bounds.get("east"), "bounds.east"),
        require_float(bounds.get("north"), "bounds.north"),
    )


def coerce_bounds(values: Any) -> tuple[float, float, float, float]:
    """Convert a four-value bounds object into a typed bounds tuple."""
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 4:
        msg = f"bounds must contain four values, got {len(bounds)}"
        raise ValueError(msg)
    west, south, east, north = bounds
    return west, south, east, north


def range_dict(values: tuple[float, float] | None) -> dict[str, float] | None:
    """Convert an optional two-value range into a named JSON object."""
    if values is None:
        return None
    return {"min": values[0], "max": values[1]}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON payload, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
