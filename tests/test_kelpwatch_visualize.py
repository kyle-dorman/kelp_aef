import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from kelp_aef import main


def test_visualize_kelpwatch_cli_writes_qa_artifacts(tmp_path: Path) -> None:
    """Verify visualize-kelpwatch writes static, GIS, table, and HTML QA artifacts."""
    netcdf_path = tmp_path / "raw/kelpwatch/kelpwatch.nc"
    footprint_path = tmp_path / "geos/footprint.geojson"
    manifest_path = tmp_path / "interim/kelpwatch_manifest.json"
    config_path = write_visualize_config(tmp_path, footprint_path, manifest_path)
    write_tiny_visualize_netcdf(netcdf_path)
    write_footprint_geojson(footprint_path)
    write_source_manifest(manifest_path, netcdf_path)

    assert (
        main(
            [
                "visualize-kelpwatch",
                "--config",
                str(config_path),
                "--preview-max-pixels",
                "4",
            ]
        )
        == 0
    )

    assert (tmp_path / "figures/kelpwatch_test_monterey_annual_max_qa.png").is_file()
    assert (tmp_path / "figures/kelpwatch_test_monterey_quarterly_timeseries_qa.png").is_file()
    assert (tmp_path / "figures/kelpwatch_test_monterey_interactive_qa.html").is_file()
    assert (tmp_path / "interim/kelpwatch_test_monterey_source_qa.json").is_file()

    csv_path = tmp_path / "tables/kelpwatch_test_monterey_source_qa.csv"
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["year"] for row in rows] == ["2018", "2019"]
    assert rows[0]["pixel_count"] == "4"
    assert rows[0]["valid_count"] == "4"
    assert rows[0]["nonzero_count"] == "4"

    html = (tmp_path / "figures/kelpwatch_test_monterey_interactive_qa.html").read_text()
    assert 'data-year="2018"' in html
    assert 'data-year="2019"' in html

    with rasterio.open(
        tmp_path / "interim/kelpwatch_test_monterey_annual_max_2018_2019.tif"
    ) as dataset:
        assert dataset.count == 2
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.descriptions == ("2018", "2019")
        assert dataset.read(1).shape == (2, 2)


def write_visualize_config(tmp_path: Path, footprint_path: Path, manifest_path: Path) -> Path:
    """Write the minimal workflow config used by the visualize-kelpwatch test."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
years:
  smoke: [2018, 2019]
region:
  name: test_monterey
  crs: EPSG:4326
  geometry:
    path: {footprint_path}
labels:
  target: kelp_max_y
  aggregation: annual_max
  paths:
    source_manifest: {manifest_path}
reports:
  figures_dir: {tmp_path / "figures"}
  tables_dir: {tmp_path / "tables"}
""".lstrip()
    )
    return config_path


def write_tiny_visualize_netcdf(path: Path) -> None:
    """Write a tiny station-layout Kelpwatch-like NetCDF fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "longitude": (
                ("station",),
                np.array([-122.0, -121.9, -122.0, -121.9], dtype=np.float64),
                {
                    "units": "degrees_east",
                    "axis": "X",
                    "coordinate_reference_frame": "urn:ogc:crs:EPSG:4326",
                },
            ),
            "latitude": (
                ("station",),
                np.array([36.55, 36.55, 36.45, 36.45], dtype=np.float64),
                {
                    "units": "degrees_north",
                    "axis": "Y",
                    "coordinate_reference_frame": "urn:ogc:crs:EPSG:4326",
                },
            ),
            "area": (
                ("time", "station"),
                np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [5.0, 0.0, np.nan, 8.0],
                        [1.0, 0.0, 0.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
                {"units": "m2", "long_name": "kelp canopy area"},
            ),
        },
        coords={
            "time": np.array(["2018-01-01", "2018-04-01", "2019-01-01"], dtype="datetime64[D]"),
            "station": np.arange(4, dtype=np.int32),
        },
        attrs={"crs": "EPSG:4326"},
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def write_footprint_geojson(path: Path) -> None:
    """Write a tiny GeoJSON footprint that covers the NetCDF fixture cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.05, 36.4],
                            [-121.85, 36.4],
                            [-121.85, 36.6],
                            [-122.05, 36.6],
                            [-122.05, 36.4],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload))


def write_source_manifest(path: Path, netcdf_path: Path) -> None:
    """Write a Kelpwatch source manifest pointing at the fixture NetCDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transfer": {
            "local_path": str(netcdf_path),
        },
        "label_source": {
            "selected_variable": "area",
        },
    }
    path.write_text(json.dumps(payload))
