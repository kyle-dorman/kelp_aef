import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from kelp_aef.domain import noaa_cudem
from kelp_aef.domain.noaa_cudem import (
    download_noaa_cudem,
    load_noaa_cudem_config,
    local_tile_path_for_source_uri,
    query_noaa_cudem,
    remote_exists_from_status_code,
)


def test_load_noaa_cudem_config_reads_tile_query_source(tmp_path: Path) -> None:
    """Load the selected NOAA CUDEM tile source from config."""
    config_path = write_noaa_cudem_config(tmp_path)

    config = load_noaa_cudem_config(config_path)

    assert config.product_id == "gov.noaa.ngdc.mgg.dem:199919"
    assert (
        config.tile_index_url == "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
        "NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip"
    )
    assert config.vertical_datum == "NAVD88"
    assert config.grid_spacing_arc_seconds == 0.1111111111
    assert config.query_manifest == tmp_path / "interim/noaa_cudem_tile_query_manifest.json"


def test_local_tile_path_for_source_uri_uses_source_basename(tmp_path: Path) -> None:
    """Map a remote tile URL to the configured local tile root."""
    source_uri = "https://example.test/path/to/ncei19_n36x75_w122x25.tif"

    local_path = local_tile_path_for_source_uri(source_uri, tmp_path / "tiles")

    assert local_path == tmp_path / "tiles/ncei19_n36x75_w122x25.tif"


def test_remote_exists_from_status_code_keeps_unknown_head_failures_unknown() -> None:
    """Treat only explicit missing HEAD responses as missing remote sources."""
    assert remote_exists_from_status_code(200) is True
    assert remote_exists_from_status_code(302) is True
    assert remote_exists_from_status_code(404) is False
    assert remote_exists_from_status_code(405) is None


def test_query_noaa_cudem_selects_intersecting_tiles_from_local_index(tmp_path: Path) -> None:
    """Select only CUDEM tile-index rows intersecting the configured footprint."""
    config_path = write_noaa_cudem_config(tmp_path)
    tile_index_path = tmp_path / "tile_index.geojson"
    manifest_path = tmp_path / "query_manifest.json"
    write_tile_index(tile_index_path)

    assert (
        query_noaa_cudem(
            config_path,
            tile_index_path=tile_index_path,
            manifest_output=manifest_path,
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["query_status"] == "selected_tiles"
    assert manifest["selected_tile_count"] == 1
    tile = manifest["selected_tiles"][0]
    assert tile["tile_id"] == "monterey_tile"
    assert tile["source_uri"].endswith("monterey_tile.tif")
    assert tile["local_path"].endswith("monterey_tile.tif")


def test_query_noaa_cudem_dry_run_without_index_writes_plan(tmp_path: Path) -> None:
    """Dry-run the CUDEM query without downloading or reading a tile index."""
    config_path = write_noaa_cudem_config(tmp_path)
    manifest_path = tmp_path / "query_dry_run.json"

    assert query_noaa_cudem(config_path, dry_run=True, manifest_output=manifest_path) == 0

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dry_run"] is True
    assert manifest["query_status"] == "planned_no_index_read"
    assert manifest["tile_index"]["status"] == "dry_run"
    assert manifest["selected_tile_count"] == 0


def test_download_noaa_cudem_dry_run_writes_manifest_without_download(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Exercise the NOAA CUDEM tile-download planning path without downloads."""
    config_path = write_noaa_cudem_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    download_manifest = tmp_path / "download_manifest.json"
    write_query_manifest(query_manifest, tmp_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly tries to download a tile."""
        raise AssertionError("download_file should not be called in dry-run mode")

    monkeypatch.setattr(noaa_cudem, "download_file", fail_download)  # pyright: ignore[reportAttributeAccessIssue]

    assert (
        download_noaa_cudem(
            config_path,
            dry_run=True,
            skip_remote_checks=True,
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


def test_download_noaa_cudem_existing_tile_updates_metadata_summary(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Inspect an existing local CUDEM tile without downloading it."""
    config_path = write_noaa_cudem_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    local_path = tmp_path / "raw/domain/noaa_cudem/tiles/monterey_tile.tif"
    write_query_manifest(query_manifest, tmp_path)
    write_tiny_tiff(local_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local tile is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(noaa_cudem, "download_file", fail_download)  # pyright: ignore[reportAttributeAccessIssue]

    assert (
        download_noaa_cudem(config_path, skip_remote_checks=True, query_manifest=query_manifest)
        == 0
    )

    manifest = json.loads((tmp_path / "interim/noaa_cudem_tile_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    record = manifest["records"][0]
    assert record["transfer"]["status"] == "skipped_existing"
    assert record["raster"]["validation_status"] == "valid"
    assert record["raster"]["shape"] == {"height": 1, "width": 1}
    assert metadata_summary["noaa_cudem"]["record_count"] == 1


def test_download_noaa_cudem_manifest_override_skips_metadata_summary(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    """Avoid pointing canonical metadata at a temporary test manifest override."""
    config_path = write_noaa_cudem_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    manifest_output = tmp_path / "override_manifest.json"
    local_path = tmp_path / "raw/domain/noaa_cudem/tiles/monterey_tile.tif"
    write_query_manifest(query_manifest, tmp_path)
    write_tiny_tiff(local_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local tile is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(noaa_cudem, "download_file", fail_download)  # pyright: ignore[reportAttributeAccessIssue]

    assert (
        download_noaa_cudem(
            config_path,
            skip_remote_checks=True,
            query_manifest=query_manifest,
            manifest_output=manifest_output,
        )
        == 0
    )

    assert manifest_output.exists()
    assert not (tmp_path / "interim/metadata_summary.json").exists()


def write_noaa_cudem_config(tmp_path: Path) -> Path:
    """Write the minimal YAML config needed by the NOAA CUDEM tile workflow."""
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
  noaa_cudem:
    source_name: NOAA NCEI Continuously Updated DEM 1/9 Arc-Second Topobathy Tiles
    source_role: preferred_topo_bathy
    product_id: gov.noaa.ngdc.mgg.dem:199919
    metadata_url: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem%3A199919
    bulk_url: https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/
    tile_index_url: https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip
    url_list_url: https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/urllist8483.txt
    tile_index_path: {tmp_path / "raw/domain/noaa_cudem/tileindex.zip"}
    local_tile_root: {tmp_path / "raw/domain/noaa_cudem/tiles"}
    query_manifest: {tmp_path / "interim/noaa_cudem_tile_query_manifest.json"}
    tile_manifest: {tmp_path / "interim/noaa_cudem_tile_manifest.json"}
    horizontal_crs: EPSG:4269
    vertical_datum: NAVD88
    units: meters
    resolution: 1/9 arc-second
    grid_spacing_arc_seconds: 0.1111111111
    elevation_sign_convention: positive land elevation and negative ocean depth
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


def write_tile_index(path: Path) -> None:
    """Write a small tile-index fixture with one intersecting and one distant tile."""
    dataframe = gpd.GeoDataFrame(
        [
            {
                "tile_id": "monterey_tile",
                "URL": "https://example.test/cudem/monterey_tile.tif",
                "geometry": Polygon(
                    [
                        (-122.1, 36.2),
                        (-121.1, 36.2),
                        (-121.1, 37.1),
                        (-122.1, 37.1),
                    ]
                ),
            },
            {
                "tile_id": "distant_tile",
                "URL": "https://example.test/cudem/distant_tile.tif",
                "geometry": Polygon(
                    [
                        (-120.0, 35.0),
                        (-119.0, 35.0),
                        (-119.0, 36.0),
                        (-120.0, 36.0),
                    ]
                ),
            },
        ],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_file(path, driver="GeoJSON")


def write_query_manifest(path: Path, tmp_path: Path) -> None:
    """Write a one-tile query manifest fixture."""
    local_path = tmp_path / "raw/domain/noaa_cudem/tiles/monterey_tile.tif"
    payload = {
        "command": "query-noaa-cudem",
        "selected_tiles": [
            {
                "tile_id": "monterey_tile",
                "source_uri": "https://example.test/cudem/monterey_tile.tif",
                "local_path": str(local_path),
                "bounds": {"west": -122.1, "south": 36.2, "east": -121.1, "north": 37.1},
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
