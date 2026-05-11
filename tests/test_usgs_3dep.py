import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from kelp_aef.domain import usgs_3dep
from kelp_aef.domain.usgs_3dep import (
    build_tnm_api_request,
    download_usgs_3dep,
    load_usgs_3dep_config,
    local_source_path_for_source_uri,
    product_items_from_api_payload,
    query_usgs_3dep,
    remote_exists_from_status_code,
)


def test_load_usgs_3dep_config_reads_tnm_source(tmp_path: Path) -> None:
    """Load the selected USGS 3DEP source from config."""
    config_path = write_usgs_3dep_config(tmp_path)

    config = load_usgs_3dep_config(config_path)

    assert config.product_id == "USGS:3a81321b-c153-416f-98b7-cc8e5f0e17c3"
    assert config.dataset_name == "National Elevation Dataset (NED) 1/3 arc-second"
    assert config.product_extent == "1 x 1 degree"
    assert config.product_format == "GeoTIFF"
    assert config.selection_policy == "latest_publication_per_1x1_tile"
    assert config.vertical_datum == "NAVD88"
    assert config.query_manifest == tmp_path / "interim/usgs_3dep_query_manifest.json"


def test_build_tnm_api_request_uses_region_bbox_and_product_filter(tmp_path: Path) -> None:
    """Build the TNMAccess request from config metadata and region bounds."""
    config = load_usgs_3dep_config(write_usgs_3dep_config(tmp_path))

    request = build_tnm_api_request(config, (-122.1, 36.2, -121.1, 37.1))

    assert request["endpoint"] == "https://tnmaccess.nationalmap.gov/api/v1/products"
    assert request["params"]["datasets"] == "National Elevation Dataset (NED) 1/3 arc-second"
    assert request["params"]["bbox"] == "-122.1,36.2,-121.1,37.1"
    assert request["params"]["prodExtents"] == "1 x 1 degree"
    assert request["params"]["prodFormats"] == "GeoTIFF"
    assert "National+Elevation+Dataset" in request["url"]


def test_product_items_from_api_payload_reads_items_array() -> None:
    """Extract TNMAccess item records from the response payload."""
    payload = {"items": [product_fixture(), {"not": "downloadable"}, "ignored"]}

    products = product_items_from_api_payload(payload)

    assert len(products) == 2
    assert products[0]["downloadURL"].endswith("USGS_13_n37w122_20250101.tif")


def test_local_source_path_for_source_uri_uses_source_basename(tmp_path: Path) -> None:
    """Map a remote 3DEP raster URL to the configured local source root."""
    source_uri = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
        "n37w122/USGS_13_n37w122.tif"
    )

    local_path = local_source_path_for_source_uri(source_uri, tmp_path / "raw/usgs_3dep")

    assert local_path == tmp_path / "raw/usgs_3dep/USGS_13_n37w122.tif"


def test_remote_exists_from_status_code_keeps_unknown_head_failures_unknown() -> None:
    """Treat only explicit missing HEAD responses as missing remote sources."""
    assert remote_exists_from_status_code(200) is True
    assert remote_exists_from_status_code(302) is True
    assert remote_exists_from_status_code(404) is False
    assert remote_exists_from_status_code(405) is None


def test_query_usgs_3dep_selects_products_from_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Select 3DEP products returned by a TNMAccess metadata query."""
    config_path = write_usgs_3dep_config(tmp_path)
    manifest_path = tmp_path / "query_manifest.json"

    def fake_fetch_tnm_api_status(
        request: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Return a deterministic TNMAccess query response."""
        assert timeout_seconds == 30.0
        return {
            "status": "queried",
            "request_url": request["url"],
            "product_count": 3,
            "total": 3,
            "products": [
                product_fixture(tile="n37w122", date="20240101"),
                product_fixture(tile="n37w122", date="20250101"),
                product_fixture(tile="n37w123", date="20240301"),
            ],
            "error": None,
        }

    monkeypatch.setattr(usgs_3dep, "fetch_tnm_api_status", fake_fetch_tnm_api_status)

    assert query_usgs_3dep(config_path, manifest_output=manifest_path) == 0

    manifest = json.loads(manifest_path.read_text())
    assert manifest["query_status"] == "selected_sources"
    assert manifest["tnm_api"]["product_count"] == 3
    assert manifest["selection_policy"] == "latest_publication_per_1x1_tile"
    assert manifest["selected_artifact_count"] == 2
    artifact = manifest["selected_artifacts"][0]
    assert artifact["artifact_id"] == "USGS_13_n37w122"
    assert artifact["source_uri"].endswith("USGS_13_n37w122_20250101.tif")
    assert artifact["local_path"].endswith("USGS_13_n37w122_20250101.tif")
    assert artifact["source_role"] == "land_dem_fallback"
    assert artifact["coverage_check"]["selection_policy"] == "latest_publication_per_1x1_tile"
    assert artifact["coverage_check"]["intersects_region"] is True


def test_query_usgs_3dep_dry_run_writes_plan_without_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dry-run the 3DEP query without calling TNMAccess."""
    config_path = write_usgs_3dep_config(tmp_path)
    manifest_path = tmp_path / "query_dry_run.json"

    def fail_api_fetch(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly calls TNMAccess."""
        raise AssertionError("fetch_tnm_api_status should not be called in dry-run mode")

    monkeypatch.setattr(usgs_3dep, "fetch_tnm_api_status", fail_api_fetch)

    assert query_usgs_3dep(config_path, dry_run=True, manifest_output=manifest_path) == 0

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dry_run"] is True
    assert manifest["query_status"] == "planned_no_api_request"
    assert manifest["tnm_api"]["status"] == "not_requested_dry_run"
    assert manifest["selected_artifact_count"] == 0


def test_download_usgs_3dep_dry_run_writes_manifest_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the USGS 3DEP source-download planning path without downloads."""
    config_path = write_usgs_3dep_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    download_manifest = tmp_path / "download_manifest.json"
    write_query_manifest(query_manifest, tmp_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly tries to download a raster."""
        raise AssertionError("download_file should not be called in dry-run mode")

    monkeypatch.setattr(usgs_3dep, "download_file", fail_download)

    assert (
        download_usgs_3dep(
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
    assert manifest["record_count"] == 1
    assert record["transfer"]["status"] == "dry_run"
    assert record["raster"]["validation_status"] == "not_checked_dry_run"
    assert not Path(record["local_path"]).exists()


def test_download_usgs_3dep_existing_tiff_updates_metadata_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inspect an existing local 3DEP raster without downloading it."""
    config_path = write_usgs_3dep_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    local_path = tmp_path / "raw/domain/usgs_3dep/USGS_13_n37w122.tif"
    write_query_manifest(query_manifest, tmp_path)
    write_tiny_tiff(local_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local raster is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(usgs_3dep, "download_file", fail_download)

    assert (
        download_usgs_3dep(config_path, skip_remote_checks=True, query_manifest=query_manifest) == 0
    )

    manifest = json.loads((tmp_path / "interim/usgs_3dep_source_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    record = manifest["records"][0]
    assert record["transfer"]["status"] == "skipped_existing"
    assert record["raster"]["validation_status"] == "valid"
    assert record["raster"]["shape"] == {"height": 1, "width": 1}
    assert metadata_summary["usgs_3dep"]["record_count"] == 1


def write_usgs_3dep_config(tmp_path: Path) -> Path:
    """Write the minimal YAML config needed by the USGS 3DEP workflow."""
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
  usgs_3dep:
    source_name: USGS 3DEP 1/3 Arc-Second Digital Elevation Model
    source_role: land_dem_fallback
    product_id: USGS:3a81321b-c153-416f-98b7-cc8e5f0e17c3
    metadata_url: https://data.usgs.gov/datacatalog/data/USGS%3A3a81321b-c153-416f-98b7-cc8e5f0e17c3
    data_catalog_url: https://apps.nationalmap.gov/downloader/
    tnm_api_url: https://tnmaccess.nationalmap.gov/api/v1/products
    dataset_name: National Elevation Dataset (NED) 1/3 arc-second
    product_extent: 1 x 1 degree
    product_format: GeoTIFF
    selection_policy: latest_publication_per_1x1_tile
    local_source_root: {tmp_path / "raw/domain/usgs_3dep"}
    query_manifest: {tmp_path / "interim/usgs_3dep_query_manifest.json"}
    source_manifest: {tmp_path / "interim/usgs_3dep_source_manifest.json"}
    horizontal_crs: EPSG:4269
    vertical_datum: NAVD88
    units: meters
    resolution: 1/3 arc-second
    grid_spacing_arc_seconds: 0.3333333333
    elevation_sign_convention: positive land elevation; land-side topographic DEM fallback only
    data_format: Cloud Optimized GeoTIFF
reports:
  outputs:
    metadata_summary: {tmp_path / "interim/metadata_summary.json"}
""".lstrip()
    )
    return config_path


def write_region_geometry(path: Path) -> None:
    """Write a tiny Monterey-like footprint fixture."""
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


def product_fixture(tile: str = "n37w122", date: str = "20250101") -> dict[str, object]:
    """Return one TNMAccess product fixture for Monterey-like DEM data."""
    return {
        "sourceId": f"sciencebase-{tile}-{date}",
        "title": f"USGS 1/3 Arc Second {tile} {date}",
        "downloadURL": (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/historical/"
            f"{tile}/USGS_13_{tile}_{date}.tif"
        ),
        "boundingBox": {
            "minX": -122.0,
            "minY": 36.0,
            "maxX": -121.0,
            "maxY": 37.0,
        },
        "sizeInBytes": "12345",
        "publicationDate": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "lastUpdated": f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z",
    }


def write_query_manifest(path: Path, tmp_path: Path) -> None:
    """Write a one-raster 3DEP query manifest fixture."""
    local_path = tmp_path / "raw/domain/usgs_3dep/USGS_13_n37w122.tif"
    payload = {
        "command": "query-usgs-3dep",
        "selected_artifacts": [
            {
                "artifact_id": "USGS_13_n37w122",
                "title": "USGS 1/3 Arc Second n37w122",
                "source_uri": (
                    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
                    "TIFF/current/n37w122/USGS_13_n37w122.tif"
                ),
                "local_path": str(local_path),
                "source_role": "land_dem_fallback",
                "bounds": {"west": -122.0, "south": 36.0, "east": -121.0, "north": 37.0},
                "coverage_check": {
                    "intersects_region": True,
                    "selection_method": "tnm_access_bbox_intersection",
                    "region_bounds": {
                        "west": -122.08798,
                        "south": 36.257752,
                        "east": -121.158794,
                        "north": 37.006593,
                    },
                },
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_tiny_tiff(path: Path) -> None:
    """Write a tiny GeoTIFF fixture for fast Rasterio metadata validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs="EPSG:4269",
        transform=from_origin(-122.0, 37.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[1.0]], dtype=np.float32), 1)
