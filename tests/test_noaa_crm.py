import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from kelp_aef.domain import noaa_crm
from kelp_aef.domain.noaa_crm import (
    actual_range_from_das,
    download_file,
    download_noaa_crm,
    load_noaa_crm_config,
    local_source_path_for_source_uri,
    query_noaa_crm,
    remote_exists_from_status_code,
)


def test_load_noaa_crm_config_reads_product_registry(tmp_path: Path) -> None:
    """Load the configured NOAA CRM product registry."""
    config_path = write_noaa_crm_config(tmp_path)

    config = load_noaa_crm_config(config_path)

    assert config.source_role == "primary_broad_topo_bathy"
    assert config.query_manifest == tmp_path / "interim/noaa_crm_query_manifest.json"
    assert len(config.products) == 3
    assert config.products[0].product_id == "crm_socal_v2_1as"
    assert config.products[1].data_variable == "z"


def test_local_source_path_for_source_uri_uses_source_basename(tmp_path: Path) -> None:
    """Map a remote CRM NetCDF URL to the configured local source root."""
    source_uri = "https://example.test/thredds/fileServer/crm/crm_vol7_2025.nc"

    local_path = local_source_path_for_source_uri(source_uri, tmp_path / "raw/noaa_crm")

    assert local_path == tmp_path / "raw/noaa_crm/crm_vol7_2025.nc"


def test_remote_exists_from_status_code_keeps_unknown_head_failures_unknown() -> None:
    """Treat only explicit missing HEAD responses as missing remote sources."""
    assert remote_exists_from_status_code(200) is True
    assert remote_exists_from_status_code(302) is True
    assert remote_exists_from_status_code(404) is False
    assert remote_exists_from_status_code(405) is None


def test_actual_range_from_das_parses_noaa_ranges() -> None:
    """Parse OPeNDAP DAS actual_range values from NOAA metadata text."""
    das_text = """
    lon {
        Float64 actual_range -127.0, -120.75;
    }
    lat {
        Int32 actual_range 37, 44;
    }
    z {
        Float64 actual_range -4447.799, 2260.081;
    }
    """

    assert actual_range_from_das(das_text, "lon") == (-127.0, -120.75)
    assert actual_range_from_das(das_text, "lat") == (37.0, 44.0)
    assert actual_range_from_das(das_text, "z") == (-4447.799, 2260.081)


def test_query_noaa_crm_selects_intersecting_california_products(tmp_path: Path) -> None:
    """Select SoCal v2 and Volume 7 because the Monterey fixture crosses 37N."""
    config_path = write_noaa_crm_config(tmp_path)
    manifest_path = tmp_path / "query_manifest.json"

    assert (
        query_noaa_crm(
            config_path,
            manifest_output=manifest_path,
            skip_remote_checks=True,
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    selected_ids = {record["product_id"] for record in manifest["selected_products"]}
    skipped_ids = {record["product_id"] for record in manifest["skipped_products"]}
    assert manifest["query_status"] == "selected_sources"
    assert selected_ids == {"crm_socal_v2_1as", "crm_vol7_2025"}
    assert skipped_ids == {"crm_vol8_2025"}
    assert manifest["selected_product_count"] == 2
    for record in manifest["selected_products"]:
        assert record["coverage_check"]["intersects_region"] is True
        assert record["coverage_check"]["recommended_subset_bounds"] is not None


def test_query_noaa_crm_dry_run_writes_plan_without_remote_metadata(tmp_path: Path) -> None:
    """Dry-run the CRM query without checking remote OPeNDAP metadata."""
    config_path = write_noaa_crm_config(tmp_path)
    manifest_path = tmp_path / "query_dry_run.json"

    assert query_noaa_crm(config_path, dry_run=True, manifest_output=manifest_path) == 0

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dry_run"] is True
    assert manifest["selected_product_count"] == 2
    assert manifest["skipped_product_count"] == 1
    assert manifest["selected_products"][0]["remote_metadata"]["status"] == "not_checked"


def test_download_noaa_crm_dry_run_writes_manifest_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the NOAA CRM source-download planning path without downloads."""
    config_path = write_noaa_crm_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    download_manifest = tmp_path / "download_manifest.json"
    write_query_manifest(query_manifest, tmp_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly tries to download a CRM source."""
        raise AssertionError("download_file should not be called in dry-run mode")

    monkeypatch.setattr(noaa_crm, "download_file", fail_download)

    assert (
        download_noaa_crm(
            config_path,
            dry_run=True,
            query_manifest=query_manifest,
            manifest_output=download_manifest,
        )
        == 0
    )

    manifest = json.loads(download_manifest.read_text())
    record = manifest["records"][0]
    assert manifest["dry_run"] is True
    assert manifest["record_count"] == 2
    assert record["transfer"]["status"] == "dry_run"
    assert record["raster"]["validation_status"] == "not_checked_dry_run"
    assert not Path(record["local_path"]).exists()


def test_download_file_resumes_existing_partial(tmp_path: Path) -> None:
    """Resume a CRM source download from an existing partial file."""
    local_path = tmp_path / "crm_socal_1as_vers2.nc"
    partial_path = tmp_path / "crm_socal_1as_vers2.nc.part"
    partial_path.write_bytes(b"abc")
    session = FakeResumeSession()

    attempt_count = download_file(
        session=session,  # type: ignore[arg-type]
        source_uri="https://example.test/crm_socal_1as_vers2.nc",
        local_path=local_path,
        timeout_seconds=30.0,
        chunk_size_bytes=2,
        max_attempts=1,
        retry_backoff_seconds=0.01,
    )

    assert attempt_count == 1
    assert local_path.read_bytes() == b"abcdef"
    assert not partial_path.exists()
    assert session.request_headers == [{"Range": "bytes=3-"}]


def test_download_noaa_crm_existing_source_updates_metadata_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inspect an existing local CRM source without downloading it."""
    config_path = write_noaa_crm_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    local_path = tmp_path / "raw/domain/noaa_crm/crm_socal_1as_vers2.nc"
    second_local_path = tmp_path / "raw/domain/noaa_crm/crm_vol7_2025.nc"
    write_query_manifest(query_manifest, tmp_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text("already local")
    second_local_path.write_text("already local")

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local source is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    def fake_raster_metadata(path: Path, dry_run: bool) -> dict[str, object]:
        """Return deterministic raster metadata without needing a real NetCDF fixture."""
        assert path.exists()
        assert dry_run is False
        return {"validation_status": "valid", "driver": "NetCDF"}

    monkeypatch.setattr(noaa_crm, "download_file", fail_download)
    monkeypatch.setattr(noaa_crm, "raster_metadata_for_manifest", fake_raster_metadata)

    assert (
        download_noaa_crm(config_path, skip_remote_checks=True, query_manifest=query_manifest) == 0
    )

    manifest = json.loads((tmp_path / "interim/noaa_crm_source_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    first_record = manifest["records"][0]
    assert first_record["transfer"]["status"] == "skipped_existing"
    assert first_record["raster"]["validation_status"] == "valid"
    assert metadata_summary["noaa_crm"]["record_count"] == 2


class FakeResumeSession:
    """Tiny requests.Session stand-in that returns a resumable response."""

    def __init__(self) -> None:
        """Initialize the recorded request list."""
        self.request_headers: list[dict[str, str] | None] = []

    def get(
        self,
        _source_uri: str,
        *,
        stream: bool,
        timeout: float,
        headers: dict[str, str] | None,
    ) -> "FakeResumeResponse":
        """Return a fake HTTP 206 response and record the request headers."""
        assert stream is True
        assert timeout == 30.0
        self.request_headers.append(headers)
        return FakeResumeResponse()


class FakeResumeResponse:
    """Tiny streaming response stand-in for an HTTP range response."""

    status_code = 206
    headers = {"Content-Range": "bytes 3-5/6"}

    def __enter__(self) -> "FakeResumeResponse":
        """Enter the response context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the response context manager."""
        return None

    def raise_for_status(self) -> None:
        """Match requests.Response.raise_for_status for a successful response."""
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        """Return fake chunks for the remaining bytes."""
        assert chunk_size == 2
        return [b"de", b"f"]


def write_noaa_crm_config(tmp_path: Path) -> Path:
    """Write the minimal YAML config needed by the NOAA CRM workflow."""
    geometry_path = tmp_path / "geos/footprint.geojson"
    write_region_geometry(geometry_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
region:
  name: monterey_peninsula
  geometry:
    path: {geometry_path}
domain:
  noaa_crm:
    source_name: NOAA Coastal Relief Model California Mosaic
    source_role: primary_broad_topo_bathy
    source_page_url: https://www.ncei.noaa.gov/products/coastal-relief-model
    local_source_root: {tmp_path / "raw/domain/noaa_crm"}
    query_manifest: {tmp_path / "interim/noaa_crm_query_manifest.json"}
    source_manifest: {tmp_path / "interim/noaa_crm_source_manifest.json"}
    download_mode: full_product_after_manifest_review
    query_padding_degrees: 0.02
    products:
      - product_id: crm_socal_v2_1as
        product_name: NOAA Coastal Relief Model Southern California Version 2
        product_version: "2"
        source_role: primary_broad_topo_bathy
        metadata_url: https://example.test/socal
        thredds_catalog_url: https://example.test/catalog/socal
        source_uri: https://example.test/thredds/fileServer/crm_socal_1as_vers2.nc
        opendap_url: https://example.test/thredds/dodsC/crm_socal_1as_vers2.nc
        local_filename: crm_socal_1as_vers2.nc
        data_variable: Band1
        bounds:
          west: -123.00013888889
          south: 30.999861071109994
          east: -116.00013883288999
          north: 36.99986111911
        horizontal_crs: EPSG:4326
        vertical_datum: mean sea level
        units: meters
        resolution: 1 arc-second
        grid_spacing_arc_seconds: 1.0
        elevation_sign_convention: positive land elevation and negative ocean depth
        data_format: NetCDF
      - product_id: crm_vol7_2025
        product_name: NOAA Coastal Relief Model Volume 7 Central Pacific 2025
        product_version: "2025"
        source_role: primary_broad_topo_bathy
        metadata_url: https://example.test/vol7
        thredds_catalog_url: https://example.test/catalog/vol7
        source_uri: https://example.test/thredds/fileServer/crm_vol7_2025.nc
        opendap_url: https://example.test/thredds/dodsC/crm_vol7_2025.nc
        local_filename: crm_vol7_2025.nc
        data_variable: z
        bounds:
          west: -127.0
          south: 37.0
          east: -120.75
          north: 44.0
        horizontal_crs: EPSG:4326
        vertical_datum: EGM2008
        units: meters
        resolution: 1 arc-second
        grid_spacing_arc_seconds: 1.0
        elevation_sign_convention: positive land elevation and negative ocean depth
        data_format: NetCDF
      - product_id: crm_vol8_2025
        product_name: NOAA Coastal Relief Model Volume 8 Northwest Pacific 2025
        product_version: "2025"
        source_role: future_scaleup_topo_bathy
        metadata_url: https://example.test/vol8
        thredds_catalog_url: https://example.test/catalog/vol8
        source_uri: https://example.test/thredds/fileServer/crm_vol8_2025.nc
        opendap_url: https://example.test/thredds/dodsC/crm_vol8_2025.nc
        local_filename: crm_vol8_2025.nc
        data_variable: z
        bounds:
          west: -127.0
          south: 44.0
          east: -121.75
          north: 49.0
        horizontal_crs: EPSG:4326
        vertical_datum: EGM2008
        units: meters
        resolution: 1 arc-second
        grid_spacing_arc_seconds: 1.0
        elevation_sign_convention: positive land elevation and negative ocean depth
        data_format: NetCDF
reports:
  outputs:
    metadata_summary: {tmp_path / "interim/metadata_summary.json"}
""".lstrip()
    )
    return config_path


def write_region_geometry(path: Path) -> None:
    """Write a tiny Monterey-like footprint fixture that crosses 37N."""
    dataframe = gpd.GeoDataFrame(
        [
            {
                "name": "monterey_peninsula",
                "geometry": Polygon(
                    [
                        (-122.08798, 36.257752),
                        (-121.158794, 36.257752),
                        (-121.158794, 37.006593),
                        (-122.08798, 37.006593),
                    ]
                ),
            }
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_file(path, driver="GeoJSON")


def write_query_manifest(path: Path, tmp_path: Path) -> None:
    """Write a two-product CRM query manifest fixture."""
    payload = {
        "command": "query-noaa-crm",
        "selected_products": [
            {
                "product_id": "crm_socal_v2_1as",
                "product_name": "NOAA Coastal Relief Model Southern California Version 2",
                "source_role": "primary_broad_topo_bathy",
                "source_uri": "https://example.test/thredds/fileServer/crm_socal_1as_vers2.nc",
                "opendap_url": "https://example.test/thredds/dodsC/crm_socal_1as_vers2.nc",
                "local_path": str(tmp_path / "raw/domain/noaa_crm/crm_socal_1as_vers2.nc"),
                "download_mode": "full_product_after_manifest_review",
                "coverage_check": {"intersects_region": True},
                "horizontal_crs": "EPSG:4326",
                "vertical_datum": "mean sea level",
                "units": "meters",
                "resolution": "1 arc-second",
                "grid_spacing_arc_seconds": 1.0,
                "elevation_sign_convention": ("positive land elevation and negative ocean depth"),
            },
            {
                "product_id": "crm_vol7_2025",
                "product_name": "NOAA Coastal Relief Model Volume 7 Central Pacific 2025",
                "source_role": "primary_broad_topo_bathy",
                "source_uri": "https://example.test/thredds/fileServer/crm_vol7_2025.nc",
                "opendap_url": "https://example.test/thredds/dodsC/crm_vol7_2025.nc",
                "local_path": str(tmp_path / "raw/domain/noaa_crm/crm_vol7_2025.nc"),
                "download_mode": "full_product_after_manifest_review",
                "coverage_check": {"intersects_region": True},
                "horizontal_crs": "EPSG:4326",
                "vertical_datum": "EGM2008",
                "units": "meters",
                "resolution": "1 arc-second",
                "grid_spacing_arc_seconds": 1.0,
                "elevation_sign_convention": ("positive land elevation and negative ocean depth"),
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
