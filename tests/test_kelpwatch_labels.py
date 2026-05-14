import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from kelp_aef import main


def test_build_labels_cli_writes_annual_label_artifacts(tmp_path: Path) -> None:
    """Verify build-labels writes Kelpwatch-native annual labels and QA artifacts."""
    netcdf_path = tmp_path / "raw/kelpwatch/kelpwatch.nc"
    footprint_path = tmp_path / "geos/footprint.geojson"
    manifest_path = tmp_path / "interim/kelpwatch_manifest.json"
    labels_path = tmp_path / "interim/labels_annual.parquet"
    summary_path = tmp_path / "tables/custom_labels_annual_summary.csv"
    label_manifest_path = tmp_path / "interim/custom_labels_annual_manifest.json"
    config_path = write_label_config(
        tmp_path,
        footprint_path,
        manifest_path,
        labels_path,
        summary_path,
        label_manifest_path,
    )
    write_tiny_label_netcdf(netcdf_path)
    write_footprint_geojson(footprint_path)
    write_source_manifest(manifest_path, netcdf_path)

    assert main(["build-labels", "--config", str(config_path)]) == 0

    labels = pd.read_parquet(labels_path)
    assert len(labels) == 8
    assert set(labels["year"]) == {2018, 2019}
    assert set(labels["kelpwatch_station_id"]) == {0, 1, 2, 3}
    assert summary_path.is_file()
    assert label_manifest_path.is_file()

    station0_2018 = labels.query("year == 2018 and kelpwatch_station_id == 0").iloc[0]
    assert station0_2018["area_q1"] == 0.0
    assert station0_2018["area_q2"] == 9.0
    assert station0_2018["area_q3"] == 45.0
    assert station0_2018["area_q4"] == 90.0
    assert station0_2018["kelp_max_y"] == 90.0
    assert station0_2018["kelp_fraction_y"] == 0.1
    assert station0_2018["max_area_quarter"] == 4
    assert bool(station0_2018["kelp_present_ge_10pct_y"]) is True

    station2_2018 = labels.query("year == 2018 and kelpwatch_station_id == 2").iloc[0]
    assert pd.isna(station2_2018["kelp_max_y"])
    assert station2_2018["valid_quarter_count"] == 0
    assert pd.isna(station2_2018["max_area_quarter"])

    with summary_path.open(newline="") as file:
        summary_rows = list(csv.DictReader(file))
    assert [row["year"] for row in summary_rows] == ["2018", "2019"]
    assert summary_rows[0]["row_count"] == "4"
    assert summary_rows[0]["valid_count"] == "3"
    assert summary_rows[0]["ge_10pct_count"] == "2"

    label_manifest = json.loads(label_manifest_path.read_text())
    assert label_manifest["label_variable"] == "area"
    assert label_manifest["spatial"]["support"] == "kelpwatch_30m_station"
    assert label_manifest["summary_table"] == str(summary_path)


def write_label_config(
    tmp_path: Path,
    footprint_path: Path,
    manifest_path: Path,
    labels_path: Path,
    summary_path: Path,
    label_manifest_path: Path,
) -> Path:
    """Write the minimal workflow config used by the build-labels test."""
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
    annual_labels: {labels_path}
    annual_label_summary: {summary_path}
    annual_label_manifest: {label_manifest_path}
reports:
  tables_dir: {tmp_path / "tables"}
""".lstrip()
    )
    return config_path


def write_tiny_label_netcdf(path: Path) -> None:
    """Write a tiny station-layout Kelpwatch NetCDF fixture for label derivation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "longitude": (
                ("station",),
                np.array([-122.0, -121.9, -122.0, -121.9, -121.0], dtype=np.float64),
                {
                    "units": "degrees_east",
                    "axis": "X",
                    "coordinate_reference_frame": "urn:ogc:crs:EPSG:4326",
                },
            ),
            "latitude": (
                ("station",),
                np.array([36.55, 36.55, 36.45, 36.45, 36.45], dtype=np.float64),
                {
                    "units": "degrees_north",
                    "axis": "Y",
                    "coordinate_reference_frame": "urn:ogc:crs:EPSG:4326",
                },
            ),
            "year": (("time",), np.array([2018, 2018, 2018, 2018, 2019], dtype=np.int16)),
            "quarter": (("time",), np.array([1, 2, 3, 4, 1], dtype=np.int16)),
            "area": (
                ("time", "station"),
                np.array(
                    [
                        [0.0, 0.0, -1.0, 100.0, 999.0],
                        [9.0, 0.0, -1.0, 200.0, 999.0],
                        [45.0, 0.0, -1.0, -1.0, 999.0],
                        [90.0, 0.0, -1.0, 50.0, 999.0],
                        [10.0, 0.0, 30.0, 40.0, 999.0],
                    ],
                    dtype=np.float32,
                ),
                {"units": "m^2 canopy/900m^2 pixel", "long_name": "kelp area"},
            ),
        },
        coords={
            "time": np.array(
                ["2018-02-15", "2018-05-15", "2018-08-15", "2018-11-15", "2019-02-15"],
                dtype="datetime64[D]",
            ),
            "station": np.arange(5, dtype=np.int32),
        },
        attrs={"crs": "EPSG:4326"},
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def write_footprint_geojson(path: Path) -> None:
    """Write a tiny GeoJSON footprint that covers four fixture stations."""
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
        "source": {
            "package_id": "knb-lter-sbc.74.32",
            "revision": 32,
            "object_name": "kelpwatch.nc",
        },
        "transfer": {
            "local_path": str(netcdf_path),
        },
        "label_source": {
            "selected_variable": "area",
        },
    }
    path.write_text(json.dumps(payload))
