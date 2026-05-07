"""Download and inspect Kelpwatch source metadata for workflow configs."""

from __future__ import annotations

import hashlib
import json
import logging
import operator
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsIndex, cast

import numpy as np
import requests
import xarray as xr

from kelp_aef.config import load_yaml_config, require_mapping, require_string

LOGGER = logging.getLogger(__name__)

PASTA_BASE_URL = "https://pasta.lternet.edu/package"
KELPWATCH_SCOPE = "knb-lter-sbc"
KELPWATCH_IDENTIFIER = "74"
KELPWATCH_ENTITY_ID = "c2bea785267fa434c40a22e2239bb337"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024
CHECKSUM_CHUNK_SIZE_BYTES = 16 * 1024 * 1024
DEFAULT_DOWNLOAD_ATTEMPTS = 5
NETCDF_ENGINE = "h5netcdf"
VARIABLE_ATTR_NAMES = (
    "units",
    "long_name",
    "standard_name",
    "description",
    "grid_mapping",
    "_FillValue",
    "missing_value",
)


@dataclass(frozen=True)
class KelpwatchInspectConfig:
    """Resolved config values needed by the Kelpwatch inspection step."""

    config_path: Path
    years: tuple[int, ...]
    target: str
    aggregation: str
    raw_dir: Path
    source_manifest: Path
    metadata_summary: Path


@dataclass(frozen=True)
class KelpwatchSourceMetadata:
    """Source metadata resolved from the EDI/PASTA package endpoints."""

    latest_revision_url: str
    entity_list_url: str
    metadata_url: str
    checksum_url: str
    revision: int
    package_id: str
    entity_id: str
    doi: str | None
    title: str | None
    pub_date: str | None
    object_name: str
    size_bytes: int
    format_name: str | None
    eml_md5: str | None
    pasta_checksum: str | None
    data_url: str
    temporal_begin: str | None
    temporal_end: str | None
    bounds: dict[str, float | None]


@dataclass(frozen=True)
class SourceFileResult:
    """Result of ensuring the resolved Kelpwatch NetCDF exists locally."""

    source_url: str
    local_path: Path
    status: str
    expected_size_bytes: int | None
    file_size_bytes: int | None
    expected_md5: str | None
    observed_md5: str | None
    checksum_status: str
    error: str | None


def load_kelpwatch_inspect_config(config_path: Path) -> KelpwatchInspectConfig:
    """Load Kelpwatch inspection settings from the workflow config."""
    config = load_yaml_config(config_path)
    years_config = require_mapping(config.get("years"), "years")
    labels = require_mapping(config.get("labels"), "labels")
    label_paths = require_mapping(labels.get("paths"), "labels.paths")
    reports = require_mapping(config.get("reports"), "reports")
    report_outputs = require_mapping(reports.get("outputs"), "reports.outputs")

    raw_years = years_config.get("smoke")
    if not isinstance(raw_years, list):
        msg = "config field must be a list of years: years.smoke"
        raise ValueError(msg)
    years = tuple(require_int_value(year, "years.smoke[]") for year in raw_years)

    return KelpwatchInspectConfig(
        config_path=config_path,
        years=years,
        target=require_string(labels.get("target"), "labels.target"),
        aggregation=require_string(labels.get("aggregation"), "labels.aggregation"),
        raw_dir=Path(require_string(label_paths.get("raw_dir"), "labels.paths.raw_dir")),
        source_manifest=Path(
            require_string(label_paths.get("source_manifest"), "labels.paths.source_manifest")
        ),
        metadata_summary=Path(
            require_string(
                report_outputs.get("metadata_summary"), "reports.outputs.metadata_summary"
            )
        ),
    )


def inspect_kelpwatch(
    config_path: Path,
    *,
    dry_run: bool = False,
    manifest_output: Path | None = None,
    force: bool = False,
    skip_checksum: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> int:
    """Run the Kelpwatch source download and metadata inspection step."""
    inspect_config = load_kelpwatch_inspect_config(config_path)
    manifest_path = manifest_output or inspect_config.source_manifest
    LOGGER.info("Resolving latest Kelpwatch EDI package metadata")

    with requests.Session() as session:
        source = resolve_latest_kelpwatch_source(session, timeout_seconds)
        LOGGER.info(
            "Resolved Kelpwatch package %s (%s bytes)",
            source.package_id,
            source.size_bytes,
        )
        transfer = ensure_kelpwatch_source_file(
            source=source,
            raw_dir=inspect_config.raw_dir,
            session=session,
            dry_run=dry_run,
            force=force,
            skip_checksum=skip_checksum,
            timeout_seconds=timeout_seconds,
            chunk_size_bytes=chunk_size_bytes,
        )

    netcdf_metadata = inspect_netcdf_for_manifest(transfer.local_path, dry_run)
    label_source = identify_label_source_variable(
        netcdf_metadata=netcdf_metadata,
        target=inspect_config.target,
        aggregation=inspect_config.aggregation,
    )
    manifest = build_source_manifest(
        inspect_config=inspect_config,
        manifest_path=manifest_path,
        dry_run=dry_run,
        skip_checksum=skip_checksum,
        source=source,
        transfer=transfer,
        netcdf_metadata=netcdf_metadata,
        label_source=label_source,
    )

    write_json(manifest_path, manifest)
    LOGGER.info("Wrote Kelpwatch source manifest: %s", manifest_path)

    if dry_run:
        LOGGER.info("Dry run complete; metadata summary was not updated")
    else:
        update_metadata_summary(inspect_config.metadata_summary, manifest)
        LOGGER.info("Updated metadata summary: %s", inspect_config.metadata_summary)

    return 0


def resolve_latest_kelpwatch_source(
    session: requests.Session, timeout_seconds: float
) -> KelpwatchSourceMetadata:
    """Resolve latest Kelpwatch package metadata from EDI/PASTA endpoints."""
    latest_revision_endpoint = latest_revision_url()
    revision_text = get_text(session, latest_revision_endpoint, timeout_seconds)
    revision = parse_latest_revision(revision_text)
    entity_list_url = data_entity_list_url(revision)
    entity_text = get_text(session, entity_list_url, timeout_seconds)
    entity_id = parse_entity_id(entity_text)
    metadata_url = package_metadata_url(revision)
    metadata_xml = get_text(session, metadata_url, timeout_seconds)
    checksum_url = data_checksum_url(revision, entity_id)
    pasta_checksum = get_text(session, checksum_url, timeout_seconds).strip()

    return parse_eml_source_metadata(
        metadata_xml=metadata_xml,
        latest_revision_url=latest_revision_endpoint,
        entity_list_url=entity_list_url,
        metadata_url=metadata_url,
        checksum_url=checksum_url,
        revision=revision,
        entity_id=entity_id,
        pasta_checksum=pasta_checksum,
    )


def latest_revision_url() -> str:
    """Return the EDI endpoint that resolves the latest package revision."""
    return f"{PASTA_BASE_URL}/eml/{KELPWATCH_SCOPE}/{KELPWATCH_IDENTIFIER}?filter=newest"


def data_entity_list_url(revision: int) -> str:
    """Return the EDI endpoint listing data entity identifiers for a revision."""
    return f"{PASTA_BASE_URL}/data/eml/{KELPWATCH_SCOPE}/{KELPWATCH_IDENTIFIER}/{revision}"


def package_metadata_url(revision: int) -> str:
    """Return the EDI endpoint for revision EML metadata."""
    return f"{PASTA_BASE_URL}/metadata/eml/{KELPWATCH_SCOPE}/{KELPWATCH_IDENTIFIER}/{revision}"


def data_checksum_url(revision: int, entity_id: str) -> str:
    """Return the EDI endpoint for a revision data-entity checksum."""
    return (
        f"{PASTA_BASE_URL}/data/checksum/eml/{KELPWATCH_SCOPE}/"
        f"{KELPWATCH_IDENTIFIER}/{revision}/{entity_id}"
    )


def data_download_url(revision: int, entity_id: str) -> str:
    """Return the EDI endpoint for a revision data-entity download."""
    return (
        f"{PASTA_BASE_URL}/data/eml/{KELPWATCH_SCOPE}/{KELPWATCH_IDENTIFIER}/{revision}/{entity_id}"
    )


def get_text(session: requests.Session, url: str, timeout_seconds: float) -> str:
    """Fetch text from an HTTP endpoint and raise for non-success responses."""
    response = session.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def parse_latest_revision(text: str) -> int:
    """Parse the PASTA latest-revision response as an integer revision."""
    stripped = text.strip()
    try:
        return int(stripped)
    except ValueError:
        msg = f"could not parse Kelpwatch latest revision from response: {stripped!r}"
        raise ValueError(msg) from None


def parse_entity_id(text: str) -> str:
    """Parse the first data entity identifier from the PASTA data list response."""
    entity_ids = [line.strip() for line in text.splitlines() if line.strip()]
    if not entity_ids:
        raise ValueError("Kelpwatch package data list did not contain entity ids")
    if len(entity_ids) > 1:
        LOGGER.warning("Kelpwatch package returned multiple entity ids; using the first")
    return entity_ids[0]


def parse_eml_source_metadata(
    *,
    metadata_xml: str,
    latest_revision_url: str,
    entity_list_url: str,
    metadata_url: str,
    checksum_url: str,
    revision: int,
    entity_id: str,
    pasta_checksum: str | None,
) -> KelpwatchSourceMetadata:
    """Parse source-file metadata from a Kelpwatch EML XML document."""
    root = ET.fromstring(metadata_xml)
    other_entity = find_other_entity(root, entity_id)
    package_id = root.get("packageId") or f"{KELPWATCH_SCOPE}.{KELPWATCH_IDENTIFIER}.{revision}"
    object_name = required_text(other_entity, "objectName")
    size_bytes = parse_required_int(required_text(other_entity, "size"), "size")
    data_url = first_text(other_entity, "url") or data_download_url(revision, entity_id)

    return KelpwatchSourceMetadata(
        latest_revision_url=latest_revision_url,
        entity_list_url=entity_list_url,
        metadata_url=metadata_url,
        checksum_url=checksum_url,
        revision=revision,
        package_id=package_id,
        entity_id=entity_id,
        doi=first_text(root, "alternateIdentifier"),
        title=first_text(root, "title"),
        pub_date=first_text(root, "pubDate"),
        object_name=object_name,
        size_bytes=size_bytes,
        format_name=first_text(other_entity, "formatName"),
        eml_md5=authentication_value(other_entity, "MD5"),
        pasta_checksum=pasta_checksum,
        data_url=data_url,
        temporal_begin=calendar_date_at(root, 0),
        temporal_end=calendar_date_at(root, 1),
        bounds={
            "west": optional_float_text(root, "westBoundingCoordinate"),
            "east": optional_float_text(root, "eastBoundingCoordinate"),
            "north": optional_float_text(root, "northBoundingCoordinate"),
            "south": optional_float_text(root, "southBoundingCoordinate"),
        },
    )


def find_other_entity(root: ET.Element, entity_id: str) -> ET.Element:
    """Find the EML otherEntity element for the resolved Kelpwatch entity id."""
    for element in root.iter():
        if local_xml_name(element.tag) == "otherEntity" and element.get("id") == entity_id:
            return element
    msg = f"EML metadata does not contain expected otherEntity id: {entity_id}"
    raise ValueError(msg)


def first_text(element: ET.Element, name: str) -> str | None:
    """Return the first non-empty descendant text for a local XML tag name."""
    for descendant in element.iter():
        if local_xml_name(descendant.tag) != name:
            continue
        if descendant.text is not None and descendant.text.strip():
            return descendant.text.strip()
    return None


def required_text(element: ET.Element, name: str) -> str:
    """Return required XML text or raise a source metadata error."""
    value = first_text(element, name)
    if value is None:
        msg = f"EML metadata missing required field: {name}"
        raise ValueError(msg)
    return value


def authentication_value(element: ET.Element, method: str) -> str | None:
    """Return an EML authentication value for a given checksum method."""
    method_lower = method.lower()
    for descendant in element.iter():
        if local_xml_name(descendant.tag) != "authentication":
            continue
        if descendant.get("method", "").lower() != method_lower:
            continue
        if descendant.text is not None and descendant.text.strip():
            return descendant.text.strip()
    return None


def calendar_date_at(element: ET.Element, index: int) -> str | None:
    """Return the calendarDate text at the requested document position."""
    dates = [date for date in iter_text_by_name(element, "calendarDate")]
    if index >= len(dates):
        return None
    return dates[index]


def iter_text_by_name(element: ET.Element, name: str) -> Iterator[str]:
    """Yield non-empty descendant text values matching a local XML tag name."""
    for descendant in element.iter():
        if local_xml_name(descendant.tag) != name:
            continue
        if descendant.text is not None and descendant.text.strip():
            yield descendant.text.strip()


def optional_float_text(element: ET.Element, name: str) -> float | None:
    """Parse an optional float from the first descendant text matching a tag name."""
    value = first_text(element, name)
    if value is None:
        return None
    return float(value)


def local_xml_name(tag: str) -> str:
    """Return an XML tag's local name without a namespace prefix."""
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[1]
    return tag


def ensure_kelpwatch_source_file(
    *,
    source: KelpwatchSourceMetadata,
    raw_dir: Path,
    session: requests.Session,
    dry_run: bool,
    force: bool,
    skip_checksum: bool,
    timeout_seconds: float,
    chunk_size_bytes: int,
) -> SourceFileResult:
    """Ensure the resolved Kelpwatch NetCDF exists under the configured raw dir."""
    local_path = raw_dir / source.object_name
    expected_md5 = source.eml_md5
    expected_size = source.size_bytes

    if dry_run:
        status = "dry_run_existing" if local_file_matches(local_path, expected_size) else "dry_run"
        return source_file_result(
            source=source,
            local_path=local_path,
            status=status,
            expected_md5=expected_md5,
            observed_md5=None,
            checksum_status="not_checked_dry_run",
            error=None,
        )

    if not force and local_file_matches(local_path, expected_size):
        LOGGER.info("Skipping existing Kelpwatch NetCDF: %s", local_path)
        status = "skipped_existing"
    else:
        LOGGER.info("Downloading Kelpwatch NetCDF: %s", source.data_url)
        download_file(
            session,
            source.data_url,
            local_path,
            timeout_seconds,
            chunk_size_bytes,
            source.size_bytes,
        )
        status = "downloaded"

    observed_md5, checksum_status = checksum_file(local_path, expected_md5, skip_checksum)
    return source_file_result(
        source=source,
        local_path=local_path,
        status=status,
        expected_md5=expected_md5,
        observed_md5=observed_md5,
        checksum_status=checksum_status,
        error=None,
    )


def source_file_result(
    *,
    source: KelpwatchSourceMetadata,
    local_path: Path,
    status: str,
    expected_md5: str | None,
    observed_md5: str | None,
    checksum_status: str,
    error: str | None,
) -> SourceFileResult:
    """Build a source-file result with current local file size metadata."""
    return SourceFileResult(
        source_url=source.data_url,
        local_path=local_path,
        status=status,
        expected_size_bytes=source.size_bytes,
        file_size_bytes=file_size(local_path),
        expected_md5=expected_md5,
        observed_md5=observed_md5,
        checksum_status=checksum_status,
        error=error,
    )


def checksum_file(
    path: Path, expected_md5: str | None, skip_checksum: bool
) -> tuple[str | None, str]:
    """Return a local MD5 checksum and status for a downloaded source file."""
    if skip_checksum:
        return None, "skipped"
    if expected_md5 is None:
        return None, "missing_expected_md5"
    observed_md5 = md5_file(path)
    if observed_md5 == expected_md5:
        return observed_md5, "valid"
    return observed_md5, "mismatch"


def md5_file(path: Path) -> str:
    """Compute an MD5 checksum for a local file in fixed-size chunks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHECKSUM_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    session: requests.Session,
    source_url: str,
    local_path: Path,
    timeout_seconds: float,
    chunk_size_bytes: int,
    expected_size_bytes: int | None,
) -> None:
    """Stream a remote asset to a temporary file with bounded resume attempts."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = local_path.with_name(f"{local_path.name}.part")

    last_error: BaseException | None = None
    for attempt in range(1, DEFAULT_DOWNLOAD_ATTEMPTS + 1):
        try:
            stream_download_attempt(
                session=session,
                source_url=source_url,
                temporary_path=temporary_path,
                timeout_seconds=timeout_seconds,
                chunk_size_bytes=chunk_size_bytes,
                expected_size_bytes=expected_size_bytes,
                attempt=attempt,
            )
            if temporary_file_is_complete(temporary_path, expected_size_bytes):
                temporary_path.replace(local_path)
                return

            current_size = file_size(temporary_path) or 0
            msg = (
                "download ended before expected size; "
                f"got {current_size}, expected {expected_size_bytes}"
            )
            last_error = RuntimeError(msg)
            LOGGER.warning(
                "Kelpwatch download attempt %s/%s incomplete: %s",
                attempt,
                DEFAULT_DOWNLOAD_ATTEMPTS,
                msg,
            )
        except (OSError, requests.RequestException) as exc:
            last_error = exc
            LOGGER.warning(
                "Kelpwatch download attempt %s/%s failed: %s",
                attempt,
                DEFAULT_DOWNLOAD_ATTEMPTS,
                exc,
            )

    msg = f"failed to download Kelpwatch source after {DEFAULT_DOWNLOAD_ATTEMPTS} attempts"
    raise RuntimeError(msg) from last_error


def stream_download_attempt(
    *,
    session: requests.Session,
    source_url: str,
    temporary_path: Path,
    timeout_seconds: float,
    chunk_size_bytes: int,
    expected_size_bytes: int | None,
    attempt: int,
) -> None:
    """Run one HTTP streaming attempt, resuming from a partial file when possible."""
    resume_from = resumable_file_size(temporary_path, expected_size_bytes)
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else None
    if resume_from > 0:
        LOGGER.info(
            "Resuming Kelpwatch download at byte %s (attempt %s/%s)",
            resume_from,
            attempt,
            DEFAULT_DOWNLOAD_ATTEMPTS,
        )

    with session.get(
        source_url,
        stream=True,
        timeout=timeout_seconds,
        headers=headers,
    ) as response:
        response.raise_for_status()
        mode = download_write_mode(response, temporary_path, resume_from)
        with temporary_path.open(mode) as file:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if chunk:
                    file.write(chunk)


def resumable_file_size(path: Path, expected_size_bytes: int | None) -> int:
    """Return a partial-file resume offset, clearing oversized partial files."""
    current_size = file_size(path) or 0
    if expected_size_bytes is not None and current_size > expected_size_bytes:
        LOGGER.warning("Removing oversized partial Kelpwatch download: %s", path)
        path.unlink(missing_ok=True)
        return 0
    return current_size


def download_write_mode(response: requests.Response, temporary_path: Path, resume_from: int) -> str:
    """Choose append or restart mode based on whether the server honored Range."""
    status_code = getattr(response, "status_code", 200)
    if resume_from > 0 and status_code != 206:
        LOGGER.warning("Server ignored Range request; restarting Kelpwatch download")
        temporary_path.unlink(missing_ok=True)
        return "wb"
    if resume_from > 0:
        return "ab"
    return "wb"


def temporary_file_is_complete(path: Path, expected_size_bytes: int | None) -> bool:
    """Return whether a temporary download is complete enough to promote."""
    if not path.exists() or not path.is_file():
        return False
    if expected_size_bytes is None:
        return True
    return path.stat().st_size == expected_size_bytes


def inspect_netcdf_for_manifest(path: Path, dry_run: bool) -> dict[str, Any]:
    """Inspect a local NetCDF source file or return an explicit unchecked block."""
    if dry_run:
        return blank_netcdf_metadata("not_checked_dry_run")
    if not path.exists():
        return blank_netcdf_metadata("missing")
    try:
        return inspect_netcdf_metadata(path)
    except (OSError, ValueError) as exc:
        metadata = blank_netcdf_metadata("metadata_error")
        metadata["validation_error"] = str(exc)
        return metadata


def inspect_netcdf_metadata(path: Path) -> dict[str, Any]:
    """Inspect NetCDF structure with xarray without loading full variable arrays."""
    with xr.open_dataset(path, engine=NETCDF_ENGINE, decode_cf=False) as dataset:
        variables = [
            variable_summary(str(name), variable) for name, variable in dataset.data_vars.items()
        ]
        coordinates = [
            variable_summary(str(name), coordinate) for name, coordinate in dataset.coords.items()
        ]
        return {
            "validation_status": "valid",
            "validation_error": None,
            "engine": NETCDF_ENGINE,
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": variables,
            "coordinates": coordinates,
            "global_attrs": selected_attrs(dataset.attrs),
            "bounds": infer_coordinate_bounds(dataset),
            "crs": infer_crs(dataset),
        }


def variable_summary(name: str, variable: Any) -> dict[str, Any]:
    """Summarize one xarray variable or coordinate for JSON metadata outputs."""
    return {
        "name": name,
        "dims": list(variable.dims),
        "shape": [int(size) for size in variable.shape],
        "dtype": str(variable.dtype),
        "attrs": selected_attrs(variable.attrs),
    }


def selected_attrs(attrs: dict[Any, Any]) -> dict[str, Any]:
    """Return JSON-friendly selected metadata attributes."""
    selected: dict[str, Any] = {}
    for name, value in attrs.items():
        if name in VARIABLE_ATTR_NAMES or not name.startswith("_"):
            selected[str(name)] = json_friendly_value(value)
    return selected


def json_friendly_value(value: object) -> object:
    """Convert common metadata scalar values into JSON-serializable values."""
    if isinstance(value, np.generic):
        return cast(object, value.item())
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, tuple):
        return [json_friendly_value(item) for item in value]
    if isinstance(value, list):
        return [json_friendly_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def infer_coordinate_bounds(dataset: Any) -> dict[str, dict[str, float | str] | None]:
    """Infer simple min/max bounds from common coordinate names."""
    return {
        "x": coordinate_min_max(dataset, ("x", "lon", "longitude")),
        "y": coordinate_min_max(dataset, ("y", "lat", "latitude")),
    }


def coordinate_min_max(
    dataset: Any, candidate_names: tuple[str, ...]
) -> dict[str, float | str] | None:
    """Return min/max for the first available coordinate in a candidate name list."""
    for name in candidate_names:
        if name not in dataset.coords:
            continue
        values = dataset.coords[name].values
        if values.size == 0:
            continue
        return {
            "coordinate": name,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }
    return None


def infer_crs(dataset: Any) -> str | None:
    """Infer a CRS string from common NetCDF global or grid-mapping attributes."""
    for attrs in [dataset.attrs, *(variable.attrs for variable in dataset.variables.values())]:
        for key in ("crs", "spatial_ref", "epsg_code", "grid_mapping_name"):
            value = attrs.get(key)
            if value is not None:
                return str(value)
    return None


def blank_netcdf_metadata(validation_status: str) -> dict[str, Any]:
    """Build an empty NetCDF metadata block with an explicit validation status."""
    return {
        "validation_status": validation_status,
        "validation_error": None,
        "engine": NETCDF_ENGINE,
        "dimensions": {},
        "variables": [],
        "coordinates": [],
        "global_attrs": {},
        "bounds": None,
        "crs": None,
    }


def identify_label_source_variable(
    *, netcdf_metadata: dict[str, Any], target: str, aggregation: str
) -> dict[str, Any]:
    """Identify the likely NetCDF value field for the configured Kelpwatch target."""
    variable_records = cast(list[dict[str, Any]], netcdf_metadata.get("variables", []))
    candidates = sorted(
        (variable_candidate_record(variable, target, aggregation) for variable in variable_records),
        key=operator.itemgetter("score", "name"),
        reverse=True,
    )
    viable_candidates = [candidate for candidate in candidates if candidate["score"] > 0]
    selected = viable_candidates[0]["name"] if viable_candidates else None
    return {
        "target": target,
        "aggregation": aggregation,
        "selected_variable": selected,
        "candidates": viable_candidates[:5],
    }


def variable_candidate_record(
    variable: dict[str, Any], target: str, aggregation: str
) -> dict[str, Any]:
    """Score one NetCDF variable as a possible label source."""
    name = str(variable.get("name", ""))
    attrs = require_mapping(variable.get("attrs"), "variable.attrs")
    searchable = " ".join([name, *(str(value) for value in attrs.values())]).lower()
    score = 0
    if "kelp" in searchable:
        score += 3
    if "canopy" in searchable:
        score += 2
    if "area" in searchable:
        score += 4
    if "m2" in searchable or "m^2" in searchable:
        score += 1
    if "biomass" in searchable:
        score -= 2
    if name.endswith("_se") or "standard error" in searchable:
        score -= 6
    if target == "kelp_max_y" and aggregation == "annual_max":
        score += 1
    return {
        "name": name,
        "score": score,
        "units": attrs.get("units"),
        "long_name": attrs.get("long_name"),
    }


def build_source_manifest(
    *,
    inspect_config: KelpwatchInspectConfig,
    manifest_path: Path,
    dry_run: bool,
    skip_checksum: bool,
    source: KelpwatchSourceMetadata,
    transfer: SourceFileResult,
    netcdf_metadata: dict[str, Any],
    label_source: dict[str, Any],
) -> dict[str, Any]:
    """Build the full Kelpwatch source manifest document."""
    return {
        "command": "inspect-kelpwatch",
        "config_path": str(inspect_config.config_path),
        "manifest_path": str(manifest_path),
        "raw_dir": str(inspect_config.raw_dir),
        "years": list(inspect_config.years),
        "target": inspect_config.target,
        "aggregation": inspect_config.aggregation,
        "dry_run": dry_run,
        "skip_checksum": skip_checksum,
        "source": source_metadata_to_dict(source),
        "transfer": source_file_result_to_dict(transfer),
        "netcdf": netcdf_metadata,
        "label_source": label_source,
    }


def source_metadata_to_dict(source: KelpwatchSourceMetadata) -> dict[str, Any]:
    """Convert resolved Kelpwatch source metadata into JSON-serializable fields."""
    return {
        "latest_revision_url": source.latest_revision_url,
        "entity_list_url": source.entity_list_url,
        "metadata_url": source.metadata_url,
        "checksum_url": source.checksum_url,
        "revision": source.revision,
        "package_id": source.package_id,
        "entity_id": source.entity_id,
        "doi": source.doi,
        "title": source.title,
        "pub_date": source.pub_date,
        "object_name": source.object_name,
        "size_bytes": source.size_bytes,
        "format_name": source.format_name,
        "eml_md5": source.eml_md5,
        "pasta_checksum": source.pasta_checksum,
        "data_url": source.data_url,
        "temporal_begin": source.temporal_begin,
        "temporal_end": source.temporal_end,
        "bounds": source.bounds,
    }


def source_file_result_to_dict(result: SourceFileResult) -> dict[str, Any]:
    """Convert a source-file result into JSON-serializable fields."""
    return {
        "source_url": result.source_url,
        "local_path": str(result.local_path),
        "status": result.status,
        "expected_size_bytes": result.expected_size_bytes,
        "file_size_bytes": result.file_size_bytes,
        "expected_md5": result.expected_md5,
        "observed_md5": result.observed_md5,
        "checksum_status": result.checksum_status,
        "error": result.error,
    }


def update_metadata_summary(summary_path: Path, manifest: dict[str, Any]) -> None:
    """Merge the Kelpwatch source manifest pointer into the metadata summary."""
    summary = load_json_object(summary_path) if summary_path.exists() else {}
    source = require_mapping(manifest.get("source"), "manifest.source")
    transfer = require_mapping(manifest.get("transfer"), "manifest.transfer")
    netcdf = require_mapping(manifest.get("netcdf"), "manifest.netcdf")
    label_source = require_mapping(manifest.get("label_source"), "manifest.label_source")
    summary["kelpwatch"] = {
        "manifest_path": manifest["manifest_path"],
        "package_id": source.get("package_id"),
        "revision": source.get("revision"),
        "object_name": source.get("object_name"),
        "local_path": transfer.get("local_path"),
        "download_status": transfer.get("status"),
        "checksum_status": transfer.get("checksum_status"),
        "source_temporal_begin": source.get("temporal_begin"),
        "source_temporal_end": source.get("temporal_end"),
        "source_bounds": source.get("bounds"),
        "netcdf_validation_status": netcdf.get("validation_status"),
        "dimensions": netcdf.get("dimensions"),
        "crs": netcdf.get("crs"),
        "bounds": netcdf.get("bounds"),
        "label_source": label_source,
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


def parse_required_int(value: object, name: str) -> int:
    """Validate an integer-like value from source metadata."""
    try:
        return int(str(value).strip())
    except ValueError:
        msg = f"field must be an integer: {name}"
        raise ValueError(msg) from None


def require_int_value(value: object, name: str) -> int:
    """Validate an integer-like value without accepting booleans."""
    if isinstance(value, bool):
        msg = f"field must be an integer, not a boolean: {name}"
        raise ValueError(msg)
    if not hasattr(value, "__index__"):
        msg = f"field must be an integer: {name}"
        raise ValueError(msg)
    return operator.index(cast(SupportsIndex, value))
