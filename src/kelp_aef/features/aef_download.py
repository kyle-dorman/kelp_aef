"""Download selected AlphaEarth/AEF tiles and write a local tile manifest."""

from __future__ import annotations

import json
import logging
import operator
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, SupportsIndex, cast
from urllib.parse import urlparse

import geopandas as gpd  # type: ignore[import-untyped]
import rasterio  # type: ignore[import-untyped]
import requests
from shapely.geometry.base import BaseGeometry

from kelp_aef.config import load_yaml_config, require_mapping, require_string
from kelp_aef.features.aef_catalog import vrt_href_for_tiff

LOGGER = logging.getLogger(__name__)

SOURCE_COOPERATIVE_BASE_URL = "https://data.source.coop"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
SOURCE_COLLECTION_MARKER = "/v1/annual"
VRT_SOURCE_ELEMENT_NAMES = frozenset({"SourceDataset", "SourceFilename"})


@dataclass(frozen=True)
class AefDownloadConfig:
    """Resolved config values needed by the AEF download step."""

    config_path: Path
    years: tuple[int, ...]
    grid: str
    s3_prefix: str
    local_mirror_root: Path
    catalog_query: Path
    catalog_query_summary: Path
    tile_manifest: Path
    metadata_summary: Path


@dataclass(frozen=True)
class AefCatalogAsset:
    """A selected AEF catalog asset with source and local mirror paths."""

    year: int
    grid: str
    source_tiff_href: str
    source_tiff_url: str
    local_tiff_path: Path
    source_vrt_href: str | None
    source_vrt_url: str | None
    local_vrt_path: Path | None
    catalog_bounds: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class RemoteAssetInfo:
    """Remote availability and size metadata from a HEAD request."""

    exists: bool | None
    status_code: int | None
    content_length_bytes: int | None
    final_url: str | None
    error: str | None


@dataclass(frozen=True)
class AssetTransferResult:
    """Result of ensuring one local file exists or is planned."""

    kind: str
    source_href: str | None
    source_url: str | None
    local_path: Path | None
    status: str
    remote_info: RemoteAssetInfo | None
    file_size_bytes: int | None
    error: str | None


@dataclass(frozen=True)
class LocalVrtResult:
    """Result of preparing a local-only VRT that points at the downloaded TIFF."""

    source_vrt_path: Path | None
    local_path: Path | None
    status: str
    file_size_bytes: int | None
    error: str | None


def load_aef_download_config(config_path: Path) -> AefDownloadConfig:
    """Load AEF download settings from the workflow config."""
    config = load_yaml_config(config_path)
    years_config = require_mapping(config.get("years"), "years")
    features = require_mapping(config.get("features"), "features")
    paths = require_mapping(features.get("paths"), "features.paths")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    raw_years = years_config.get("smoke")
    if not isinstance(raw_years, list):
        msg = "config field must be a list of years: years.smoke"
        raise ValueError(msg)
    years = tuple(require_int_value(year, "years.smoke[]") for year in raw_years)

    return AefDownloadConfig(
        config_path=config_path,
        years=years,
        grid=require_string(features.get("utm_grid"), "features.utm_grid"),
        s3_prefix=require_string(features.get("s3_prefix"), "features.s3_prefix"),
        local_mirror_root=Path(
            require_string(paths.get("local_mirror_root"), "features.paths.local_mirror_root")
        ),
        catalog_query=Path(
            require_string(paths.get("catalog_query"), "features.paths.catalog_query")
        ),
        catalog_query_summary=Path(
            require_string(
                paths.get("catalog_query_summary"), "features.paths.catalog_query_summary"
            )
        ),
        tile_manifest=Path(
            require_string(paths.get("tile_manifest"), "features.paths.tile_manifest")
        ),
        metadata_summary=Path(
            require_string(
                report_outputs.get("metadata_summary"), "reports.outputs.metadata_summary"
            )
        ),
    )


def download_aef(
    config_path: Path,
    *,
    dry_run: bool = False,
    skip_remote_checks: bool = False,
    manifest_output: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    force: bool = False,
) -> int:
    """Run the AEF tile download step."""
    download_config = load_aef_download_config(config_path)
    manifest_path = manifest_output or download_config.tile_manifest
    assets = load_catalog_assets(download_config)
    LOGGER.info("Loaded %s selected AEF catalog assets", len(assets))

    records: list[dict[str, Any]] = []
    with requests.Session() as session:
        for asset in assets:
            records.append(
                process_catalog_asset(
                    asset=asset,
                    session=session,
                    dry_run=dry_run,
                    skip_remote_checks=skip_remote_checks,
                    timeout_seconds=timeout_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                    force=force,
                )
            )

    manifest = build_manifest(
        download_config=download_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        skip_remote_checks=skip_remote_checks,
        records=records,
    )

    if dry_run and manifest_output is None:
        LOGGER.info(
            "Dry run complete; no manifest written. Use --manifest-output to save the plan."
        )
    else:
        write_json(manifest_path, manifest)
        LOGGER.info("Wrote AEF tile manifest: %s", manifest_path)

    if not dry_run:
        update_metadata_summary(download_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", download_config.metadata_summary)

    return 0


def load_catalog_assets(download_config: AefDownloadConfig) -> list[AefCatalogAsset]:
    """Read selected AEF assets from the configured GeoParquet catalog artifact."""
    if not download_config.catalog_query.exists():
        msg = f"catalog query artifact does not exist: {download_config.catalog_query}"
        raise FileNotFoundError(msg)

    dataframe = gpd.read_parquet(download_config.catalog_query)
    rows = cast(list[dict[str, Any]], dataframe.to_dict("records"))
    if not rows:
        msg = f"catalog query artifact contains no selected assets: {download_config.catalog_query}"
        raise ValueError(msg)

    assets = [catalog_asset_from_row(row, download_config) for row in rows]
    validate_catalog_asset_years(assets, download_config.years)
    return assets


def catalog_asset_from_row(
    row: dict[str, Any], download_config: AefDownloadConfig
) -> AefCatalogAsset:
    """Build one download asset from a selected catalog row."""
    source_tiff_href = selected_tiff_href(row)
    source_vrt_href = selected_vrt_href(row, source_tiff_href)
    source_tiff_key = source_key_for_href(source_tiff_href)
    parsed_year, parsed_grid = year_grid_from_source_key(source_tiff_key)
    year = optional_int_value(row.get("query_year"), "query_year") or parsed_year
    grid = parsed_grid or download_config.grid

    return AefCatalogAsset(
        year=year,
        grid=grid,
        source_tiff_href=source_tiff_href,
        source_tiff_url=download_url_for_href(source_tiff_href),
        local_tiff_path=local_path_for_href(
            source_tiff_href, download_config.local_mirror_root, download_config.s3_prefix
        ),
        source_vrt_href=source_vrt_href,
        source_vrt_url=download_url_for_href(source_vrt_href) if source_vrt_href else None,
        local_vrt_path=(
            local_path_for_href(
                source_vrt_href, download_config.local_mirror_root, download_config.s3_prefix
            )
            if source_vrt_href
            else None
        ),
        catalog_bounds=catalog_bounds_from_row(row),
    )


def validate_catalog_asset_years(assets: list[AefCatalogAsset], years: tuple[int, ...]) -> None:
    """Validate that the selected catalog rows cover each configured smoke year once."""
    counts_by_year = {year: 0 for year in years}
    unexpected_years: list[int] = []
    for asset in assets:
        if asset.year in counts_by_year:
            counts_by_year[asset.year] += 1
        else:
            unexpected_years.append(asset.year)

    missing_years = [year for year, count in counts_by_year.items() if count == 0]
    duplicate_years = [year for year, count in counts_by_year.items() if count > 1]
    if unexpected_years or missing_years or duplicate_years:
        msg = (
            "catalog query selected assets do not match configured years; "
            f"missing={missing_years}, duplicate={duplicate_years}, unexpected={unexpected_years}"
        )
        raise ValueError(msg)


def process_catalog_asset(
    *,
    asset: AefCatalogAsset,
    session: requests.Session,
    dry_run: bool,
    skip_remote_checks: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
) -> dict[str, Any]:
    """Download or plan one selected AEF asset and return its manifest record."""
    LOGGER.info("Processing AEF %s asset for %s", asset.grid, asset.year)
    tiff_remote = (
        None
        if skip_remote_checks
        else head_remote_asset(session, asset.source_tiff_url, timeout_seconds)
    )
    vrt_remote = (
        None
        if skip_remote_checks or asset.source_vrt_url is None
        else head_remote_asset(session, asset.source_vrt_url, timeout_seconds)
    )

    tiff_transfer = ensure_asset_available(
        kind="tiff",
        source_href=asset.source_tiff_href,
        source_url=asset.source_tiff_url,
        local_path=asset.local_tiff_path,
        remote_info=tiff_remote,
        session=session,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
        required=True,
    )
    vrt_transfer = ensure_optional_vrt_available(
        asset=asset,
        remote_info=vrt_remote,
        session=session,
        dry_run=dry_run,
        skip_remote_checks=skip_remote_checks,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
    )
    local_vrt = ensure_local_read_vrt_available(
        asset=asset,
        tiff_transfer=tiff_transfer,
        vrt_transfer=vrt_transfer,
        dry_run=dry_run,
    )
    preferred_read_path = choose_preferred_read_path(tiff_transfer, local_vrt)
    raster_metadata = raster_metadata_for_manifest(preferred_read_path, dry_run)

    return build_asset_manifest_record(
        asset=asset,
        tiff_transfer=tiff_transfer,
        vrt_transfer=vrt_transfer,
        local_vrt=local_vrt,
        preferred_read_path=preferred_read_path,
        raster_metadata=raster_metadata,
    )


def ensure_optional_vrt_available(
    *,
    asset: AefCatalogAsset,
    remote_info: RemoteAssetInfo | None,
    session: requests.Session,
    dry_run: bool,
    skip_remote_checks: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
) -> AssetTransferResult:
    """Download, plan, or record absence of the optional matching VRT asset."""
    if (
        asset.source_vrt_href is None
        or asset.source_vrt_url is None
        or asset.local_vrt_path is None
    ):
        return empty_transfer_result("vrt", "not_configured")
    if skip_remote_checks:
        return AssetTransferResult(
            kind="vrt",
            source_href=asset.source_vrt_href,
            source_url=asset.source_vrt_url,
            local_path=asset.local_vrt_path,
            status="not_checked",
            remote_info=None,
            file_size_bytes=file_size(asset.local_vrt_path),
            error=None,
        )
    if remote_info is not None and remote_info.exists is False:
        return AssetTransferResult(
            kind="vrt",
            source_href=asset.source_vrt_href,
            source_url=asset.source_vrt_url,
            local_path=asset.local_vrt_path,
            status="remote_missing",
            remote_info=remote_info,
            file_size_bytes=file_size(asset.local_vrt_path),
            error=None,
        )
    if remote_info is not None and remote_info.exists is None:
        return AssetTransferResult(
            kind="vrt",
            source_href=asset.source_vrt_href,
            source_url=asset.source_vrt_url,
            local_path=asset.local_vrt_path,
            status="remote_unknown",
            remote_info=remote_info,
            file_size_bytes=file_size(asset.local_vrt_path),
            error=remote_info.error,
        )

    return ensure_asset_available(
        kind="vrt",
        source_href=asset.source_vrt_href,
        source_url=asset.source_vrt_url,
        local_path=asset.local_vrt_path,
        remote_info=remote_info,
        session=session,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
        chunk_size_bytes=chunk_size_bytes,
        force=force,
        required=False,
    )


def ensure_asset_available(
    *,
    kind: str,
    source_href: str,
    source_url: str,
    local_path: Path,
    remote_info: RemoteAssetInfo | None,
    session: requests.Session,
    dry_run: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
    force: bool,
    required: bool,
) -> AssetTransferResult:
    """Ensure a source asset is present locally unless this is a dry run."""
    expected_size = remote_info.content_length_bytes if remote_info else None
    if remote_info is not None and remote_info.exists is False:
        if required and not dry_run:
            msg = f"required remote {kind} asset is missing: {source_url}"
            raise FileNotFoundError(msg)
        return AssetTransferResult(
            kind=kind,
            source_href=source_href,
            source_url=source_url,
            local_path=local_path,
            status="remote_missing",
            remote_info=remote_info,
            file_size_bytes=file_size(local_path),
            error=None,
        )

    if dry_run:
        status = "dry_run_existing" if local_file_matches(local_path, expected_size) else "dry_run"
        return AssetTransferResult(
            kind=kind,
            source_href=source_href,
            source_url=source_url,
            local_path=local_path,
            status=status,
            remote_info=remote_info,
            file_size_bytes=file_size(local_path),
            error=None,
        )

    if not force and local_file_matches(local_path, expected_size):
        LOGGER.info("Skipping existing %s asset: %s", kind, local_path)
        return AssetTransferResult(
            kind=kind,
            source_href=source_href,
            source_url=source_url,
            local_path=local_path,
            status="skipped_existing",
            remote_info=remote_info,
            file_size_bytes=file_size(local_path),
            error=None,
        )

    LOGGER.info("Downloading %s asset: %s", kind, source_url)
    download_file(session, source_url, local_path, timeout_seconds, chunk_size_bytes)
    return AssetTransferResult(
        kind=kind,
        source_href=source_href,
        source_url=source_url,
        local_path=local_path,
        status="downloaded",
        remote_info=remote_info,
        file_size_bytes=file_size(local_path),
        error=None,
    )


def ensure_local_read_vrt_available(
    *,
    asset: AefCatalogAsset,
    tiff_transfer: AssetTransferResult,
    vrt_transfer: AssetTransferResult,
    dry_run: bool,
) -> LocalVrtResult:
    """Create or plan a VRT whose source dataset is the downloaded local TIFF."""
    if asset.local_vrt_path is None:
        return empty_local_vrt_result(None, None, "not_configured")

    local_path = local_read_vrt_path(asset.local_vrt_path)
    if dry_run:
        status = "dry_run_existing" if local_path.exists() else "dry_run"
        return LocalVrtResult(
            source_vrt_path=asset.local_vrt_path,
            local_path=local_path,
            status=status,
            file_size_bytes=file_size(local_path),
            error=None,
        )

    if tiff_transfer.local_path is None or not tiff_transfer.local_path.exists():
        return empty_local_vrt_result(asset.local_vrt_path, local_path, "missing_tiff")
    if vrt_transfer.local_path is None or not vrt_transfer.local_path.exists():
        return empty_local_vrt_result(asset.local_vrt_path, local_path, "missing_source_vrt")

    try:
        rewrite_vrt_sources_to_local_tiff(
            source_vrt_path=vrt_transfer.local_path,
            local_tiff_path=tiff_transfer.local_path,
            output_path=local_path,
        )
    except (OSError, ET.ParseError, ValueError) as exc:
        LOGGER.warning("Could not prepare local AEF VRT %s: %s", local_path, exc)
        return LocalVrtResult(
            source_vrt_path=vrt_transfer.local_path,
            local_path=local_path,
            status="error",
            file_size_bytes=file_size(local_path),
            error=str(exc),
        )

    LOGGER.info("Prepared local AEF VRT: %s", local_path)
    return LocalVrtResult(
        source_vrt_path=vrt_transfer.local_path,
        local_path=local_path,
        status="generated",
        file_size_bytes=file_size(local_path),
        error=None,
    )


def local_read_vrt_path(source_vrt_path: Path) -> Path:
    """Return the generated local-read VRT path for an upstream VRT path."""
    return source_vrt_path.with_name(f"{source_vrt_path.stem}.local.vrt")


def rewrite_vrt_sources_to_local_tiff(
    *, source_vrt_path: Path, local_tiff_path: Path, output_path: Path
) -> None:
    """Rewrite VRT source dataset references to point at the downloaded local TIFF."""
    tree = ET.parse(source_vrt_path)
    replaced_count = 0
    for element in tree.iter():
        if local_xml_name(element.tag) not in VRT_SOURCE_ELEMENT_NAMES:
            continue
        if element.text is None or not element.text.strip():
            continue
        element.text = str(local_tiff_path)
        element.set("relativeToVRT", "0")
        replaced_count += 1

    if replaced_count == 0:
        msg = f"VRT contains no source dataset references: {source_vrt_path}"
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def local_xml_name(tag: str) -> str:
    """Return an XML tag's local name without a namespace prefix."""
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[1]
    return tag


def download_file(
    session: requests.Session,
    source_url: str,
    local_path: Path,
    timeout_seconds: float,
    chunk_size_bytes: int,
) -> None:
    """Stream a remote asset to a temporary file before replacing the final path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_name(f"{local_path.name}.part")
    temporary_path.unlink(missing_ok=True)

    with session.get(source_url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    file.write(chunk)

    temporary_path.replace(local_path)


def head_remote_asset(
    session: requests.Session, source_url: str, timeout_seconds: float
) -> RemoteAssetInfo:
    """Check remote asset availability without downloading the body."""
    try:
        response = session.head(source_url, allow_redirects=True, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return RemoteAssetInfo(
            exists=None,
            status_code=None,
            content_length_bytes=None,
            final_url=None,
            error=str(exc),
        )

    return RemoteAssetInfo(
        exists=remote_exists_from_status_code(response.status_code),
        status_code=response.status_code,
        content_length_bytes=content_length_from_headers(response.headers),
        final_url=response.url,
        error=None if response.ok else response.reason,
    )


def remote_exists_from_status_code(status_code: int) -> bool | None:
    """Interpret HEAD status codes without treating every failure as missing."""
    if 200 <= status_code < 400:
        return True
    if status_code in {404, 410}:
        return False
    return None


def content_length_from_headers(
    headers: requests.structures.CaseInsensitiveDict[str],
) -> int | None:
    """Parse a response Content-Length header when the server provides one."""
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def selected_tiff_href(row: dict[str, Any]) -> str:
    """Return the TIFF href from a selected catalog row."""
    value = row.get("asset_tiff_href", row.get("assets.data.href"))
    return require_string(value, "catalog row asset_tiff_href")


def selected_vrt_href(row: dict[str, Any], source_tiff_href: str) -> str | None:
    """Return the VRT href from a selected catalog row or derive the sibling href."""
    value = row.get("asset_vrt_href")
    if isinstance(value, str) and value:
        return value
    return vrt_href_for_tiff(source_tiff_href)


def source_key_for_href(href: str) -> str:
    """Extract the Source Cooperative object key from an s3 or HTTP href."""
    parsed = urlparse(href)
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")
    if parsed.scheme in {"http", "https"}:
        return parsed.path.lstrip("/")
    msg = f"unsupported AEF asset href scheme: {href}"
    raise ValueError(msg)


def download_url_for_href(href: str) -> str:
    """Convert an s3 or HTTP source href into a downloadable HTTP URL."""
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https"}:
        return href
    key = source_key_for_href(href)
    return f"{SOURCE_COOPERATIVE_BASE_URL}/{key}"


def local_path_for_href(href: str, local_mirror_root: Path, s3_prefix: str) -> Path:
    """Map a Source Cooperative asset href to the configured local mirror path."""
    source_key = source_key_for_href(href)
    mirror_prefix = collection_root_prefix(s3_prefix)
    if source_key == mirror_prefix:
        relative_key = ""
    elif source_key.startswith(f"{mirror_prefix}/"):
        relative_key = source_key.removeprefix(f"{mirror_prefix}/")
    else:
        msg = f"source key is not under configured AEF prefix {mirror_prefix!r}: {source_key}"
        raise ValueError(msg)
    return local_mirror_root / Path(PurePosixPath(relative_key))


def collection_root_prefix(s3_prefix: str) -> str:
    """Return the source key prefix that should be stripped below the mirror root."""
    normalized = s3_prefix.strip("/")
    marker_index = normalized.find(SOURCE_COLLECTION_MARKER)
    if marker_index == -1:
        return normalized
    return normalized[:marker_index]


def year_grid_from_source_key(source_key: str) -> tuple[int, str | None]:
    """Parse the annual year and grid from an AEF source object key."""
    parts = PurePosixPath(source_key).parts
    try:
        annual_index = parts.index("annual")
        year = int(parts[annual_index + 1])
        grid = parts[annual_index + 2]
    except (ValueError, IndexError):
        msg = f"could not parse year/grid from AEF source key: {source_key}"
        raise ValueError(msg) from None
    return year, grid


def catalog_bounds_from_row(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Read catalog geometry bounds from the GeoPandas row geometry."""
    geometry = row.get("geometry")
    if isinstance(geometry, BaseGeometry) and not geometry.is_empty:
        return geometry.bounds
    return None


def choose_preferred_read_path(
    tiff_transfer: AssetTransferResult, local_vrt: LocalVrtResult
) -> Path | None:
    """Prefer the generated local-read VRT, otherwise fall back to the local TIFF."""
    if (
        local_vrt.status in {"generated", "dry_run_existing"}
        and local_vrt.local_path is not None
        and local_vrt.local_path.exists()
    ):
        return local_vrt.local_path
    if tiff_transfer.local_path is not None and tiff_transfer.local_path.exists():
        return tiff_transfer.local_path
    return tiff_transfer.local_path


def raster_metadata_for_manifest(preferred_read_path: Path | None, dry_run: bool) -> dict[str, Any]:
    """Return raster metadata for a completed asset or a fast unchecked placeholder."""
    if preferred_read_path is None:
        return blank_raster_metadata("missing")
    if dry_run:
        return blank_raster_metadata("not_checked_dry_run")
    if not preferred_read_path.exists():
        return blank_raster_metadata("missing")
    return inspect_raster_metadata(preferred_read_path)


def inspect_raster_metadata(path: Path) -> dict[str, Any]:
    """Inspect raster metadata with Rasterio for a completed local TIFF or VRT."""
    try:
        with rasterio.open(path) as dataset:
            return {
                "validation_status": "valid",
                "validation_error": None,
                "crs": dataset.crs.to_string() if dataset.crs is not None else None,
                "bounds": {
                    "left": dataset.bounds.left,
                    "bottom": dataset.bounds.bottom,
                    "right": dataset.bounds.right,
                    "top": dataset.bounds.top,
                },
                "shape": {
                    "height": dataset.height,
                    "width": dataset.width,
                },
                "transform_gdal": list(dataset.transform.to_gdal()),
                "band_count": dataset.count,
                "band_names": list(dataset.descriptions),
                "dtypes": list(dataset.dtypes),
                "nodata": dataset.nodata,
            }
    except rasterio.errors.RasterioError as exc:
        metadata = blank_raster_metadata("metadata_error")
        metadata["validation_error"] = str(exc)
        return metadata


def blank_raster_metadata(validation_status: str) -> dict[str, Any]:
    """Build an empty raster metadata block with an explicit validation status."""
    return {
        "validation_status": validation_status,
        "validation_error": None,
        "crs": None,
        "bounds": None,
        "shape": None,
        "transform_gdal": None,
        "band_count": None,
        "band_names": None,
        "dtypes": None,
        "nodata": None,
    }


def build_asset_manifest_record(
    *,
    asset: AefCatalogAsset,
    tiff_transfer: AssetTransferResult,
    vrt_transfer: AssetTransferResult,
    local_vrt: LocalVrtResult,
    preferred_read_path: Path | None,
    raster_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build one JSON-serializable manifest record for a selected AEF tile."""
    return {
        "year": asset.year,
        "grid": asset.grid,
        "catalog_bounds": asset.catalog_bounds,
        "source_href": asset.source_tiff_href,
        "source_tiff_href": asset.source_tiff_href,
        "source_tiff_url": asset.source_tiff_url,
        "source_vrt_href": asset.source_vrt_href,
        "source_vrt_url": asset.source_vrt_url,
        "local_path": str(asset.local_tiff_path),
        "local_tiff_path": str(asset.local_tiff_path),
        "local_vrt_path": str(asset.local_vrt_path) if asset.local_vrt_path else None,
        "local_read_vrt_path": str(local_vrt.local_path) if local_vrt.local_path else None,
        "preferred_read_path": str(preferred_read_path) if preferred_read_path else None,
        "file_size_bytes": file_size(asset.local_tiff_path),
        "validation_status": raster_metadata["validation_status"],
        "raster": raster_metadata,
        "transfers": {
            "tiff": transfer_result_to_dict(tiff_transfer),
            "vrt": transfer_result_to_dict(vrt_transfer),
            "local_vrt": local_vrt_result_to_dict(local_vrt),
        },
    }


def transfer_result_to_dict(result: AssetTransferResult) -> dict[str, Any]:
    """Convert an asset transfer result into JSON-serializable fields."""
    return {
        "kind": result.kind,
        "source_href": result.source_href,
        "source_url": result.source_url,
        "local_path": str(result.local_path) if result.local_path else None,
        "status": result.status,
        "remote": remote_info_to_dict(result.remote_info),
        "file_size_bytes": result.file_size_bytes,
        "error": result.error,
    }


def local_vrt_result_to_dict(result: LocalVrtResult) -> dict[str, Any]:
    """Convert a local VRT preparation result into JSON-serializable fields."""
    return {
        "source_vrt_path": str(result.source_vrt_path) if result.source_vrt_path else None,
        "local_path": str(result.local_path) if result.local_path else None,
        "status": result.status,
        "file_size_bytes": result.file_size_bytes,
        "error": result.error,
    }


def remote_info_to_dict(info: RemoteAssetInfo | None) -> dict[str, Any] | None:
    """Convert remote HEAD metadata into JSON-serializable fields."""
    if info is None:
        return None
    return {
        "exists": info.exists,
        "status_code": info.status_code,
        "content_length_bytes": info.content_length_bytes,
        "final_url": info.final_url,
        "error": info.error,
    }


def build_manifest(
    *,
    download_config: AefDownloadConfig,
    manifest_path: Path,
    dry_run: bool,
    skip_remote_checks: bool,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full AEF tile manifest document."""
    return {
        "command": "download-aef",
        "config_path": str(download_config.config_path),
        "catalog_query": str(download_config.catalog_query),
        "catalog_query_summary": str(download_config.catalog_query_summary),
        "manifest_path": str(manifest_path),
        "local_mirror_root": str(download_config.local_mirror_root),
        "years": list(download_config.years),
        "grid": download_config.grid,
        "dry_run": dry_run,
        "skip_remote_checks": skip_remote_checks,
        "record_count": len(records),
        "records": records,
    }


def update_metadata_summary(summary_path: Path, manifest: dict[str, Any]) -> None:
    """Merge the AEF tile manifest pointer into the shared metadata summary."""
    summary = load_json_object(summary_path) if summary_path.exists() else {}
    records = cast(list[dict[str, Any]], manifest.get("records", []))
    summary["aef_tiles"] = {
        "manifest_path": manifest["manifest_path"],
        "catalog_query": manifest["catalog_query"],
        "record_count": manifest["record_count"],
        "years": [record["year"] for record in records],
        "grid": manifest["grid"],
        "validation_statuses": {
            str(record["year"]): record["validation_status"] for record in records
        },
    }
    write_json(summary_path, summary)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"expected JSON object at {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable indentation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def empty_transfer_result(kind: str, status: str) -> AssetTransferResult:
    """Build an empty transfer result for assets that are not configured."""
    return AssetTransferResult(
        kind=kind,
        source_href=None,
        source_url=None,
        local_path=None,
        status=status,
        remote_info=None,
        file_size_bytes=None,
        error=None,
    )


def empty_local_vrt_result(
    source_vrt_path: Path | None, local_path: Path | None, status: str
) -> LocalVrtResult:
    """Build an empty local VRT result for unavailable source inputs."""
    return LocalVrtResult(
        source_vrt_path=source_vrt_path,
        local_path=local_path,
        status=status,
        file_size_bytes=file_size(local_path),
        error=None,
    )


def local_file_matches(path: Path, expected_size_bytes: int | None) -> bool:
    """Return whether a local file exists and matches the expected size when known."""
    if not path.exists() or not path.is_file():
        return False
    if expected_size_bytes is None:
        return True
    return path.stat().st_size == expected_size_bytes


def file_size(path: Path | None) -> int | None:
    """Return the local file size when a path exists."""
    if path is None or not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def optional_int_value(value: object, name: str) -> int | None:
    """Parse an optional integer-like value from config or catalog data."""
    if value is None:
        return None
    return require_int_value(value, name)


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        msg = f"field must be an integer: {name}"
        raise ValueError(msg) from exc
