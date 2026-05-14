import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from kelp_aef import main
from kelp_aef.domain.crm_alignment import (
    load_crm_alignment_config,
    load_static_target_grid,
)


def test_load_crm_alignment_config_reads_fast_outputs(tmp_path: Path) -> None:
    """Load CRM alignment paths, product boundary, and fast windows from config."""
    fixture = write_crm_alignment_fixture(tmp_path)

    config = load_crm_alignment_config(fixture["config_path"], fast=True)

    assert config.output_table == fixture["fast_output"]
    assert config.product_boundary.latitude == 37.0
    assert config.product_boundary.south_product_id == "crm_socal_v2_1as"
    assert config.row_window == (0, 2)
    assert config.col_window == (0, 2)


def test_load_static_target_grid_deduplicates_years_and_applies_fast_window(
    tmp_path: Path,
) -> None:
    """Derive one static target-grid row per cell from the full-grid artifact."""
    fixture = write_crm_alignment_fixture(tmp_path)
    config = load_crm_alignment_config(fixture["config_path"], fast=True)

    target_grid = load_static_target_grid(config)

    assert len(target_grid) == 2
    assert target_grid["aef_grid_cell_id"].to_list() == [0, 11]
    assert target_grid["aef_grid_row"].to_list() == [0, 1]


def test_align_noaa_crm_fast_writes_support_and_qa_tables(tmp_path: Path) -> None:
    """Run the package CLI fast path and verify CRM plus QA source outputs."""
    fixture = write_crm_alignment_fixture(tmp_path)

    assert main(["align-noaa-crm", "--config", str(fixture["config_path"]), "--fast"]) == 0

    aligned = pd.read_parquet(fixture["fast_output"]).sort_values("aef_grid_cell_id")
    manifest = json.loads(fixture["fast_manifest"].read_text())
    summary = pd.read_csv(fixture["fast_summary"])
    comparison = pd.read_csv(fixture["fast_comparison"])

    assert len(aligned) == 2
    assert aligned["crm_source_product_id"].to_list() == [
        "crm_socal_v2_1as",
        "crm_vol7_2025",
    ]
    assert aligned["crm_elevation_m"].to_list() == pytest.approx([-12.0, -34.0])
    assert aligned["crm_depth_m"].to_list() == pytest.approx([12.0, 34.0])
    assert set(aligned["cudem_value_status"]) == {"valid"}
    assert set(aligned["usgs_3dep_value_status"]) == {"valid"}
    assert manifest["fast"] is True
    assert manifest["target_grid"]["row_count"] == 2
    assert manifest["qa_source_status"]["noaa_cusp"]["alignment_role"] == (
        "shoreline_vector_validation_only"
    )
    assert (
        int(summary.query("source == 'noaa_crm' and metric == 'valid_cells'")["value"].iloc[0]) == 2
    )
    assert "crm_vs_cudem" in set(comparison["source_pair"])


def test_align_noaa_crm_does_not_require_unused_north_product(tmp_path: Path) -> None:
    """Verify south-only target grids do not require the north CRM source file."""
    fixture = write_crm_alignment_fixture(
        tmp_path,
        include_north_source=False,
        include_north_target=False,
    )

    assert main(["align-noaa-crm", "--config", str(fixture["config_path"]), "--fast"]) == 0

    aligned = pd.read_parquet(fixture["fast_output"]).sort_values("aef_grid_cell_id")
    assert len(aligned) == 2
    assert set(aligned["crm_source_product_id"]) == {"crm_socal_v2_1as"}
    assert set(aligned["crm_value_status"]) == {"valid"}


def write_crm_alignment_fixture(
    tmp_path: Path,
    *,
    include_north_source: bool = True,
    include_north_target: bool = True,
) -> dict[str, Path]:
    """Write a complete tiny CRM alignment fixture under a temp directory."""
    target_grid = tmp_path / "interim/full_grid.parquet"
    source_manifest = tmp_path / "interim/noaa_crm_source_manifest.json"
    query_manifest = tmp_path / "interim/noaa_crm_query_manifest.json"
    cudem_manifest = tmp_path / "interim/noaa_cudem_tile_manifest.json"
    usgs_manifest = tmp_path / "interim/usgs_3dep_source_manifest.json"
    cusp_manifest = tmp_path / "interim/noaa_cusp_source_manifest.json"
    socal_path = tmp_path / "raw/noaa_crm/crm_socal_1as_vers2.nc"
    vol7_path = tmp_path / "raw/noaa_crm/crm_vol7_2025.nc"
    cudem_path = tmp_path / "raw/noaa_cudem/cudem.tif"
    usgs_path = tmp_path / "raw/usgs_3dep/usgs.tif"
    fast_output = tmp_path / "interim/aligned_noaa_crm.fast.parquet"
    fast_manifest = tmp_path / "interim/aligned_noaa_crm.fast_manifest.json"
    fast_summary = tmp_path / "tables/aligned_noaa_crm.fast_summary.csv"
    fast_comparison = tmp_path / "tables/aligned_domain_source_comparison.fast.csv"
    geometry_path = tmp_path / "geos/footprint.geojson"
    config_path = tmp_path / "config.yaml"

    write_target_grid_table(target_grid, include_north_target=include_north_target)
    write_crm_source(socal_path, variable="Band1", value=-12.0, lat_value=36.9)
    if include_north_source:
        write_crm_source(vol7_path, variable="z", value=-34.0, lat_value=37.1)
    write_qa_raster(cudem_path, value=-11.0)
    write_qa_raster(usgs_path, value=7.0)
    write_region_geometry(geometry_path)
    write_crm_source_manifest(source_manifest, socal_path, vol7_path, include_north_source)
    query_manifest.write_text(json.dumps({"selected_products": []}))
    write_raster_manifest(cudem_manifest, "tile_id", "cudem_tile", cudem_path)
    write_raster_manifest(usgs_manifest, "artifact_id", "usgs_tile", usgs_path)
    write_cusp_manifest(cusp_manifest)
    write_crm_alignment_config(
        config_path=config_path,
        geometry_path=geometry_path,
        target_grid=target_grid,
        source_manifest=source_manifest,
        query_manifest=query_manifest,
        cudem_manifest=cudem_manifest,
        usgs_manifest=usgs_manifest,
        cusp_manifest=cusp_manifest,
        fast_output=fast_output,
        fast_manifest=fast_manifest,
        fast_summary=fast_summary,
        fast_comparison=fast_comparison,
    )
    return {
        "config_path": config_path,
        "fast_output": fast_output,
        "fast_manifest": fast_manifest,
        "fast_summary": fast_summary,
        "fast_comparison": fast_comparison,
    }


def write_target_grid_table(path: Path, *, include_north_target: bool = True) -> None:
    """Write duplicate years of a small full-grid alignment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in (2018, 2019):
        rows.extend(
            [
                {
                    "year": year,
                    "aef_grid_row": 0,
                    "aef_grid_col": 0,
                    "aef_grid_cell_id": 0,
                    "longitude": -122.0,
                    "latitude": 36.9,
                },
                {
                    "year": year,
                    "aef_grid_row": 1,
                    "aef_grid_col": 1,
                    "aef_grid_cell_id": 11,
                    "longitude": -122.0,
                    "latitude": 37.1 if include_north_target else 36.9,
                },
                {
                    "year": year,
                    "aef_grid_row": 2,
                    "aef_grid_col": 2,
                    "aef_grid_cell_id": 22,
                    "longitude": -121.9,
                    "latitude": 36.9,
                },
            ]
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_crm_source(path: Path, *, variable: str, value: float, lat_value: float) -> None:
    """Write a tiny xarray NetCDF CRM source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    longitudes = np.asarray([-122.1, -122.0, -121.9], dtype=np.float64)
    latitudes = np.asarray([lat_value - 0.1, lat_value, lat_value + 0.1], dtype=np.float64)
    values = np.full((3, 3), value, dtype=np.float32)
    dataset = xr.Dataset(
        {variable: (("lat", "lon"), values)},
        coords={"lat": latitudes, "lon": longitudes},
    )
    dataset.to_netcdf(path, engine="scipy")


def write_qa_raster(path: Path, *, value: float) -> None:
    """Write a small QA GeoTIFF covering the fixture target cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, 4), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-122.2, 37.2, 0.1, 0.1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)


def write_region_geometry(path: Path) -> None:
    """Write a minimal GeoJSON footprint for config validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-122.2, 36.8],
                                    [-121.8, 36.8],
                                    [-121.8, 37.2],
                                    [-122.2, 37.2],
                                    [-122.2, 36.8],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )


def write_crm_source_manifest(
    path: Path,
    socal_path: Path,
    vol7_path: Path,
    include_north_source: bool,
) -> None:
    """Write the source manifest consumed by the CRM alignment command."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "product_id": "crm_socal_v2_1as",
            "product_name": "NOAA Coastal Relief Model Southern California Version 2",
            "local_path": str(socal_path),
            "vertical_datum": "mean sea level",
        }
    ]
    if include_north_source:
        records.append(
            {
                "product_id": "crm_vol7_2025",
                "product_name": "NOAA Coastal Relief Model Volume 7 Central Pacific 2025",
                "local_path": str(vol7_path),
                "vertical_datum": "EGM2008",
            }
        )
    payload = {"records": records}
    path.write_text(json.dumps(payload))


def write_raster_manifest(path: Path, id_field: str, source_id: str, raster_path: Path) -> None:
    """Write a one-record raster QA source manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [
            {
                id_field: source_id,
                "local_path": str(raster_path),
                "raster": {"validation_status": "valid"},
            }
        ]
    }
    path.write_text(json.dumps(payload))


def write_cusp_manifest(path: Path) -> None:
    """Write a CUSP vector manifest fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [
            {
                "artifact_id": "noaa_cusp_west",
                "local_path": str(path.parent / "West.zip"),
                "vector": {
                    "validation_status": "valid",
                    "crs": "EPSG:4269",
                    "feature_count": 82942,
                    "geometry_types": ["LineString"],
                    "bounds": {
                        "west": -125.0,
                        "south": 32.0,
                        "east": -114.0,
                        "north": 49.5,
                    },
                },
            }
        ]
    }
    path.write_text(json.dumps(payload))


def write_crm_alignment_config(
    *,
    config_path: Path,
    geometry_path: Path,
    target_grid: Path,
    source_manifest: Path,
    query_manifest: Path,
    cudem_manifest: Path,
    usgs_manifest: Path,
    cusp_manifest: Path,
    fast_output: Path,
    fast_manifest: Path,
    fast_summary: Path,
    fast_comparison: Path,
) -> None:
    """Write the minimal YAML config needed by CRM alignment tests."""
    config_path.write_text(
        f"""
years:
  smoke: [2018, 2019]
region:
  name: monterey_peninsula
  geometry:
    path: {geometry_path}
alignment:
  full_grid:
    output_table: {target_grid}
    output_manifest: {config_path.parent / "interim/full_grid_manifest.json"}
    summary_table: {config_path.parent / "tables/full_grid_summary.csv"}
    fast:
      years: [2018]
      row_window: [0, 2]
      col_window: [0, 2]
domain:
  noaa_crm:
    source_name: NOAA Coastal Relief Model California Mosaic
    source_role: primary_broad_topo_bathy
    source_page_url: https://example.test/crm
    local_source_root: {config_path.parent / "raw/noaa_crm"}
    query_manifest: {query_manifest}
    source_manifest: {source_manifest}
    download_mode: full_product_after_manifest_review
    query_padding_degrees: 0.02
    alignment:
      source_manifest: {source_manifest}
      query_manifest: {query_manifest}
      cudem_tile_manifest: {cudem_manifest}
      usgs_3dep_source_manifest: {usgs_manifest}
      cusp_source_manifest: {cusp_manifest}
      target_grid_table: {target_grid}
      target_grid_manifest: {config_path.parent / "interim/full_grid_manifest.json"}
      output_table: {config_path.parent / "interim/aligned_noaa_crm.parquet"}
      output_manifest: {config_path.parent / "interim/aligned_noaa_crm_manifest.json"}
      qa_summary_table: {config_path.parent / "tables/aligned_noaa_crm_summary.csv"}
      comparison_table: {config_path.parent / "tables/aligned_domain_source_comparison.csv"}
      resampling_method: nearest_center_sample
      row_chunk_size: 2
      product_boundary:
        latitude: 37.0
        south_product_id: crm_socal_v2_1as
        north_product_id: crm_vol7_2025
      fast:
        output_table: {fast_output}
        output_manifest: {fast_manifest}
        qa_summary_table: {fast_summary}
        comparison_table: {fast_comparison}
    products:
      - product_id: crm_socal_v2_1as
        product_name: NOAA Coastal Relief Model Southern California Version 2
        product_version: "2"
        source_role: primary_broad_topo_bathy
        metadata_url: https://example.test/socal
        thredds_catalog_url: https://example.test/catalog/socal
        source_uri: https://example.test/crm_socal_1as_vers2.nc
        opendap_url: https://example.test/opendap/socal
        local_filename: crm_socal_1as_vers2.nc
        data_variable: Band1
        bounds:
          west: -123.0
          south: 31.0
          east: -116.0
          north: 37.0
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
        source_uri: https://example.test/crm_vol7_2025.nc
        opendap_url: https://example.test/opendap/vol7
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
reports:
  outputs:
    metadata_summary: {config_path.parent / "interim/metadata_summary.json"}
""".lstrip()
    )
