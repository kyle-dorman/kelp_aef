import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from kelp_aef.domain import noaa_cusp
from kelp_aef.domain.noaa_cusp import (
    download_noaa_cusp,
    load_noaa_cusp_config,
    local_source_path_for_source_uri,
    query_noaa_cusp,
    remote_exists_from_status_code,
)


def test_load_noaa_cusp_config_reads_source_package(tmp_path: Path) -> None:
    """Load the selected NOAA CUSP source package from config."""
    config_path = write_noaa_cusp_config(tmp_path)

    config = load_noaa_cusp_config(config_path)

    assert config.product_id == "gov.noaa.nmfs.inport:60812"
    assert config.source_uri == "https://geodesy.noaa.gov/dist_shoreline/West.zip"
    assert config.package_region == "West"
    assert config.horizontal_crs == "EPSG:4269"
    assert config.query_manifest == tmp_path / "interim/noaa_cusp_query_manifest.json"


def test_local_source_path_for_source_uri_uses_source_basename(tmp_path: Path) -> None:
    """Map a remote CUSP ZIP URL to the configured local source root."""
    source_uri = "https://example.test/dist_shoreline/West.zip"

    local_path = local_source_path_for_source_uri(source_uri, tmp_path / "raw/noaa_cusp")

    assert local_path == tmp_path / "raw/noaa_cusp/West.zip"


def test_remote_exists_from_status_code_keeps_unknown_head_failures_unknown() -> None:
    """Treat only explicit missing HEAD responses as missing remote sources."""
    assert remote_exists_from_status_code(200) is True
    assert remote_exists_from_status_code(302) is True
    assert remote_exists_from_status_code(404) is False
    assert remote_exists_from_status_code(405) is None


def test_query_noaa_cusp_selects_west_package_for_monterey(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Select the CUSP West package because its bounds intersect Monterey."""
    config_path = write_noaa_cusp_config(tmp_path)
    manifest_path = tmp_path / "query_manifest.json"

    def fail_remote_check(*_args: object, **_kwargs: object) -> None:
        """Fail if skip_remote_checks unexpectedly still checks the remote package."""
        raise AssertionError("head_remote_asset should not be called")

    monkeypatch.setattr(noaa_cusp, "head_remote_asset", fail_remote_check)

    assert (
        query_noaa_cusp(
            config_path,
            manifest_output=manifest_path,
            skip_remote_checks=True,
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["query_status"] == "selected_sources"
    assert manifest["selected_artifact_count"] == 1
    assert manifest["coverage_check"]["intersects_region"] is True
    artifact = manifest["selected_artifacts"][0]
    assert artifact["artifact_id"] == "noaa_cusp_west"
    assert artifact["source_uri"].endswith("West.zip")
    assert artifact["local_path"].endswith("West.zip")
    assert artifact["coverage_check"]["intersects_region"] is True


def test_query_noaa_cusp_dry_run_writes_plan_without_remote_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dry-run the CUSP query without checking or downloading the source package."""
    config_path = write_noaa_cusp_config(tmp_path)
    manifest_path = tmp_path / "query_dry_run.json"

    def fail_remote_check(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly checks the remote package."""
        raise AssertionError("head_remote_asset should not be called in dry-run mode")

    monkeypatch.setattr(noaa_cusp, "head_remote_asset", fail_remote_check)

    assert query_noaa_cusp(config_path, dry_run=True, manifest_output=manifest_path) == 0

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dry_run"] is True
    assert manifest["query_status"] == "selected_sources"
    assert manifest["remote"] is None
    assert manifest["selected_artifact_count"] == 1


def test_download_noaa_cusp_dry_run_writes_manifest_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the NOAA CUSP source-download planning path without downloads."""
    config_path = write_noaa_cusp_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    download_manifest = tmp_path / "download_manifest.json"
    write_query_manifest(query_manifest, tmp_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if dry-run mode unexpectedly tries to download a source package."""
        raise AssertionError("download_file should not be called in dry-run mode")

    monkeypatch.setattr(noaa_cusp, "download_file", fail_download)

    assert (
        download_noaa_cusp(
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
    assert record["vector"]["validation_status"] == "not_checked_dry_run"
    assert not Path(record["local_path"]).exists()


def test_download_noaa_cusp_existing_zip_updates_metadata_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inspect an existing local CUSP ZIP without downloading it."""
    config_path = write_noaa_cusp_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    local_path = tmp_path / "raw/domain/noaa_cusp/West.zip"
    write_query_manifest(query_manifest, tmp_path)
    write_tiny_shapefile_zip(local_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local ZIP is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(noaa_cusp, "download_file", fail_download)

    assert (
        download_noaa_cusp(config_path, skip_remote_checks=True, query_manifest=query_manifest) == 0
    )

    manifest = json.loads((tmp_path / "interim/noaa_cusp_source_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    record = manifest["records"][0]
    assert record["transfer"]["status"] == "skipped_existing"
    assert record["vector"]["validation_status"] == "valid"
    assert record["vector"]["geometry_types"] == ["LineString"]
    assert record["vector"]["feature_count"] == 1
    assert metadata_summary["noaa_cusp"]["record_count"] == 1


def test_download_noaa_cusp_manifest_override_skips_metadata_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Avoid pointing canonical metadata at a temporary CUSP manifest override."""
    config_path = write_noaa_cusp_config(tmp_path)
    query_manifest = tmp_path / "query_manifest.json"
    manifest_output = tmp_path / "override_manifest.json"
    local_path = tmp_path / "raw/domain/noaa_cusp/West.zip"
    write_query_manifest(query_manifest, tmp_path)
    write_tiny_shapefile_zip(local_path)

    def fail_download(*_args: object, **_kwargs: object) -> None:
        """Fail if an existing local ZIP is downloaded again."""
        raise AssertionError("download_file should not be called for existing files")

    monkeypatch.setattr(noaa_cusp, "download_file", fail_download)

    assert (
        download_noaa_cusp(
            config_path,
            skip_remote_checks=True,
            query_manifest=query_manifest,
            manifest_output=manifest_output,
        )
        == 0
    )

    assert manifest_output.exists()
    assert not (tmp_path / "interim/metadata_summary.json").exists()


def write_noaa_cusp_config(tmp_path: Path) -> Path:
    """Write the minimal YAML config needed by the NOAA CUSP workflow."""
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
  noaa_cusp:
    source_name: NOAA NGS Continually Updated Shoreline Product
    source_role: shoreline_side_source
    product_id: gov.noaa.nmfs.inport:60812
    metadata_url: https://www.fisheries.noaa.gov/inport/item/60812
    source_page_url: https://coast.noaa.gov/digitalcoast/data/cusp.html
    viewer_url: https://nsde.ngs.noaa.gov/
    package_region: West
    source_uri: https://geodesy.noaa.gov/dist_shoreline/West.zip
    local_source_root: {tmp_path / "raw/domain/noaa_cusp"}
    query_manifest: {tmp_path / "interim/noaa_cusp_query_manifest.json"}
    source_manifest: {tmp_path / "interim/noaa_cusp_source_manifest.json"}
    horizontal_crs: EPSG:4269
    data_format: ESRI shapefile ZIP
    shoreline_reference: proxy mean high water
    scale: variable; 1:1,000-1:24,000
    geometry_type: line shoreline vectors
    package_bounds:
      west: -125.0
      south: 32.0
      east: -114.0
      north: 49.5
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


def write_query_manifest(path: Path, tmp_path: Path) -> None:
    """Write a one-package CUSP query manifest fixture."""
    local_path = tmp_path / "raw/domain/noaa_cusp/West.zip"
    payload = {
        "command": "query-noaa-cusp",
        "selected_artifacts": [
            {
                "artifact_id": "noaa_cusp_west",
                "package_region": "West",
                "source_uri": "https://geodesy.noaa.gov/dist_shoreline/West.zip",
                "local_path": str(local_path),
                "bounds": {"west": -125.0, "south": 32.0, "east": -114.0, "north": 49.5},
                "coverage_check": {
                    "intersects_region": True,
                    "selection_method": "configured_package_bounds_intersection",
                    "package_bounds": {
                        "west": -125.0,
                        "south": 32.0,
                        "east": -114.0,
                        "north": 49.5,
                    },
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


def write_tiny_shapefile_zip(path: Path) -> None:
    """Write a tiny zipped shapefile fixture for fast GeoPandas validation."""
    shapefile_dir = path.parent / "shapefile_fixture"
    shapefile_path = shapefile_dir / "West.shp"
    dataframe = gpd.GeoDataFrame(
        [
            {
                "source": "fixture",
                "geometry": LineString([(-122.0, 36.5), (-121.9, 36.6)]),
            }
        ],
        crs="EPSG:4269",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    shapefile_dir.mkdir(parents=True, exist_ok=True)
    dataframe.to_file(shapefile_path, driver="ESRI Shapefile")
    with zipfile.ZipFile(path, "w") as archive:
        for member in shapefile_dir.iterdir():
            archive.write(member, arcname=member.name)
