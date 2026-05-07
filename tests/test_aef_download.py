import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from kelp_aef.features import aef_download
from kelp_aef.features.aef_download import (
    download_aef,
    download_url_for_href,
    local_path_for_href,
    remote_exists_from_status_code,
)


def test_download_url_for_s3_href_uses_source_cooperative_https() -> None:
    """Convert Source Cooperative S3 hrefs to public HTTPS download URLs."""
    href = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/tile.tiff"

    assert (
        download_url_for_href(href)
        == "https://data.source.coop/tge-labs/aef/v1/annual/2018/10N/tile.tiff"
    )


def test_local_path_for_href_mirrors_collection_below_local_root(tmp_path: Path) -> None:
    """Mirror Source Cooperative AEF paths below the configured local AEF root."""
    href = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/tile.tiff"

    local_path = local_path_for_href(href, tmp_path / "raw/aef", "tge-labs/aef/v1/annual")

    assert local_path == tmp_path / "raw/aef/v1/annual/2018/10N/tile.tiff"


def test_remote_exists_from_status_code_keeps_head_failures_unknown() -> None:
    """Treat only explicit missing HEAD responses as missing assets."""
    assert remote_exists_from_status_code(200) is True
    assert remote_exists_from_status_code(404) is False
    assert remote_exists_from_status_code(405) is None


def test_download_aef_dry_run_writes_fast_manifest(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Exercise the downloader planning path without remote checks or downloads."""
    catalog_path = tmp_path / "catalog.parquet"
    manifest_path = tmp_path / "dry_manifest.json"
    config_path = write_download_config(tmp_path, catalog_path)
    write_catalog_query(catalog_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly tries to download a file."""
        raise AssertionError("download_file should not be called in dry-run mode")

    monkeypatch.setattr(aef_download, "download_file", fail_download)

    assert (
        download_aef(
            config_path,
            dry_run=True,
            skip_remote_checks=True,
            manifest_output=manifest_path,
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    record = manifest["records"][0]
    assert manifest["dry_run"] is True
    assert record["year"] == 2018
    assert record["transfers"]["tiff"]["status"] == "dry_run"
    assert record["transfers"]["vrt"]["status"] == "not_checked"
    assert record["validation_status"] == "not_checked_dry_run"
    assert not Path(record["local_tiff_path"]).exists()


def test_download_aef_skips_existing_tiff_and_writes_metadata_summary(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Validate a tiny existing GeoTIFF without downloading a remote raster."""
    catalog_path = tmp_path / "catalog.parquet"
    config_path = write_download_config(tmp_path, catalog_path)
    write_catalog_query(catalog_path)
    local_tiff_path = tmp_path / "raw/aef/v1/annual/2018/10N/tile.tiff"
    write_tiny_tiff(local_tiff_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local raster is downloaded again."""
        raise AssertionError("download_file should not be called for an existing file")

    monkeypatch.setattr(aef_download, "download_file", fail_download)

    assert download_aef(config_path, skip_remote_checks=True) == 0

    manifest = json.loads((tmp_path / "interim/aef_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    record = manifest["records"][0]
    assert record["transfers"]["tiff"]["status"] == "skipped_existing"
    assert record["preferred_read_path"] == str(local_tiff_path)
    assert record["validation_status"] == "valid"
    assert record["raster"]["shape"] == {"height": 1, "width": 1}
    assert metadata_summary["aef_tiles"]["record_count"] == 1


def test_download_aef_generates_local_vrt_for_offline_reads(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Rewrite downloaded AEF VRT sources to local TIFF paths before validation."""
    catalog_path = tmp_path / "catalog.parquet"
    config_path = write_download_config(tmp_path, catalog_path)
    write_catalog_query(catalog_path)
    local_tiff_path = tmp_path / "raw/aef/v1/annual/2018/10N/tile.tiff"
    source_vrt_path = tmp_path / "raw/aef/v1/annual/2018/10N/tile.vrt"
    local_read_vrt_path = tmp_path / "raw/aef/v1/annual/2018/10N/tile.local.vrt"
    write_tiny_tiff(local_tiff_path)
    write_remote_source_vrt(source_vrt_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if existing local AEF files are downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(aef_download, "download_file", fail_download)

    assert download_aef(config_path, skip_remote_checks=True) == 0

    manifest = json.loads((tmp_path / "interim/aef_manifest.json").read_text())
    record = manifest["records"][0]
    local_read_vrt_text = local_read_vrt_path.read_text()
    assert record["preferred_read_path"] == str(local_read_vrt_path)
    assert record["local_read_vrt_path"] == str(local_read_vrt_path)
    assert record["transfers"]["local_vrt"]["status"] == "generated"
    assert record["validation_status"] == "valid"
    assert "/vsis3/" not in local_read_vrt_text
    assert str(local_tiff_path) in local_read_vrt_text


def write_catalog_query(path: Path) -> None:
    """Write a one-row GeoParquet catalog query fixture."""
    href = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/tile.tiff"
    dataframe = gpd.GeoDataFrame(
        [
            {
                "query_year": 2018,
                "asset_tiff_href": href,
                "asset_vrt_href": href.replace(".tiff", ".vrt"),
                "geometry": Polygon(
                    [
                        (-122.0, 36.0),
                        (-121.0, 36.0),
                        (-121.0, 37.0),
                        (-122.0, 37.0),
                    ]
                ),
            }
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(path)


def write_download_config(tmp_path: Path, catalog_path: Path) -> Path:
    """Write the minimal YAML config needed by the AEF downloader."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
years:
  smoke: [2018]
features:
  utm_grid: 10N
  s3_prefix: tge-labs/aef/v1/annual
  paths:
    local_mirror_root: {tmp_path / "raw/aef"}
    catalog_query: {catalog_path}
    catalog_query_summary: {tmp_path / "interim/catalog_summary.json"}
    tile_manifest: {tmp_path / "interim/aef_manifest.json"}
reports:
  outputs:
    metadata_summary: {tmp_path / "interim/metadata_summary.json"}
""".lstrip()
    )
    return config_path


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
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0.0, 1.0, 1.0, 1.0),
        nodata=0,
    ) as dataset:
        dataset.write(np.array([[1]], dtype=np.uint8), 1)


def write_remote_source_vrt(path: Path) -> None:
    """Write a tiny VRT fixture that mimics upstream S3-backed AEF VRTs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
<VRTDataset rasterXSize="1" rasterYSize="1">
  <SRS>EPSG:4326</SRS>
  <GeoTransform>0.0, 1.0, 0.0, 1.0, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1">
    <NoDataValue>0</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="0">/vsis3/example-bucket/tile.tiff</SourceFilename>
      <SourceBand>1</SourceBand>
      <SourceProperties RasterXSize="1" RasterYSize="1" DataType="Byte"
        BlockXSize="1" BlockYSize="1"/>
      <SrcRect xOff="0" yOff="0" xSize="1" ySize="1"/>
      <DstRect xOff="0" yOff="0" xSize="1" ySize="1"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""".lstrip()
    )
