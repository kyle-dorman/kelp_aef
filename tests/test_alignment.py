import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from kelp_aef import main


def test_align_cli_writes_feature_label_table(tmp_path: Path) -> None:
    """Verify align writes parquet features, summary diagnostics, and a manifest."""
    fixture = write_alignment_fixture(tmp_path, method="mean_10m_to_kelpwatch_30m")

    assert main(["align", "--config", str(fixture["config_path"])]) == 0

    aligned = pd.read_parquet(fixture["output_table"])
    center_row = aligned.query("kelpwatch_station_id == 10").iloc[0]
    edge_row = aligned.query("kelpwatch_station_id == 11").iloc[0]
    expected_center_a00, expected_center_a01 = expected_center_means()

    assert len(aligned) == 2
    assert center_row["A00"] == pytest.approx(expected_center_a00)
    assert center_row["A01"] == pytest.approx(expected_center_a01)
    assert center_row["aef_expected_pixel_count"] == 9
    assert center_row["aef_valid_pixel_count"] == 8
    assert center_row["aef_missing_pixel_count"] == 1
    assert edge_row["aef_expected_pixel_count"] == 4
    assert edge_row["aef_valid_pixel_count"] == 4
    assert fixture["summary_table"].is_file()

    manifest = json.loads(fixture["output_manifest"].read_text())
    assert manifest["command"] == "align"
    assert manifest["row_count"] == 2
    assert manifest["support_cells_per_side"] == 3
    assert manifest["bands"] == ["A00", "A01"]


def test_align_cli_rasterio_average_writes_fast_comparison(tmp_path: Path) -> None:
    """Verify Rasterio average alignment coarsens AEF cells and compares fast output."""
    fixture = write_alignment_fixture(tmp_path, method="rasterio_average_10m_to_30m")

    assert main(["align", "--config", str(fixture["config_path"]), "--fast"]) == 0

    aligned = pd.read_parquet(fixture["fast_output_table"])
    center_row = aligned.query("kelpwatch_station_id == 10").iloc[0]
    edge_row = aligned.query("kelpwatch_station_id == 11").iloc[0]
    comparison = pd.read_csv(fixture["fast_comparison_table"])

    assert center_row["A00"] == pytest.approx(expected_center_means()[0])
    assert center_row["aef_expected_pixel_count"] == 9
    assert center_row["aef_valid_pixel_count"] == 8
    assert edge_row["A00"] == pytest.approx(expected_upper_left_average())
    assert edge_row["aef_expected_pixel_count"] == 9
    assert edge_row["aef_valid_pixel_count"] == 9
    assert comparison.loc[0, "candidate_method"] == "rasterio_average_10m_to_30m"
    assert comparison.loc[0, "reference_method"] == "station_centered_3x3_mean"


def test_align_cli_fast_path_limits_stations_and_uses_fast_outputs(tmp_path: Path) -> None:
    """Verify --fast uses configured fast artifacts and a spatial station cap."""
    fixture = write_alignment_fixture(tmp_path, method="mean_10m_to_kelpwatch_30m")

    assert (
        main(["align", "--config", str(fixture["config_path"]), "--fast", "--max-stations", "1"])
        == 0
    )

    aligned = pd.read_parquet(fixture["fast_output_table"])
    manifest = json.loads(fixture["fast_output_manifest"].read_text())
    assert len(aligned) == 1
    assert aligned["kelpwatch_station_id"].nunique() == 1
    assert fixture["fast_summary_table"].is_file()
    assert manifest["fast"] is True
    assert manifest["max_stations"] == 1


def write_alignment_fixture(tmp_path: Path, *, method: str) -> dict[str, Path]:
    """Write a complete synthetic config, label table, manifest, and AEF raster."""
    labels_path = tmp_path / "interim/labels_annual.parquet"
    label_manifest_path = tmp_path / "interim/labels_annual_manifest.json"
    raster_path = tmp_path / "raw/aef/2018/tile.tif"
    tile_manifest_path = tmp_path / "interim/aef_manifest.json"
    output_table = tmp_path / "interim/aligned_training_table.parquet"
    output_manifest = tmp_path / "interim/aligned_training_table_manifest.json"
    summary_table = tmp_path / "tables/aligned_training_table_summary.csv"
    fast_output_table = tmp_path / "interim/aligned_training_table.fast.parquet"
    fast_output_manifest = tmp_path / "interim/aligned_training_table.fast_manifest.json"
    fast_summary_table = tmp_path / "tables/aligned_training_table.fast_summary.csv"
    fast_comparison_table = tmp_path / "tables/aligned_training_table.fast_method_comparison.csv"
    config_path = tmp_path / "config.yaml"
    write_label_table(labels_path)
    write_label_manifest(label_manifest_path)
    write_aef_raster(raster_path)
    write_tile_manifest(tile_manifest_path, raster_path)
    config_path.write_text(
        f"""
years:
  smoke: [2018]
labels:
  native_resolution_m: 3
  paths:
    annual_labels: {labels_path}
    annual_label_manifest: {label_manifest_path}
features:
  bands: A00-A01
  native_resolution_m: 1
  paths:
    tile_manifest: {tile_manifest_path}
alignment:
  method: {method}
  output_table: {output_table}
  output_manifest: {output_manifest}
  summary_table: {summary_table}
  fast:
    years: [2018]
    max_stations: 2
    output_table: {fast_output_table}
    output_manifest: {fast_output_manifest}
    summary_table: {fast_summary_table}
    comparison_table: {fast_comparison_table}
reports:
  tables_dir: {tmp_path / "tables"}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "output_table": output_table,
        "output_manifest": output_manifest,
        "summary_table": summary_table,
        "fast_output_table": fast_output_table,
        "fast_output_manifest": fast_output_manifest,
        "fast_summary_table": fast_summary_table,
        "fast_comparison_table": fast_comparison_table,
    }


def write_label_table(path: Path) -> None:
    """Write two station labels whose centers map to known raster pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame(
        {
            "year": [2018, 2018],
            "kelpwatch_station_id": [10, 11],
            "longitude": [4.5, 0.5],
            "latitude": [4.5, 8.5],
            "kelp_max_y": [90.0, 0.0],
            "kelp_fraction_y": [0.1, 0.0],
            "kelp_present_gt0_y": [True, False],
            "kelp_present_ge_1pct_y": [True, False],
            "kelp_present_ge_5pct_y": [True, False],
            "kelp_present_ge_10pct_y": [True, False],
        }
    )
    labels.to_parquet(path, index=False)


def write_label_manifest(path: Path) -> None:
    """Write the label manifest fields used by alignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spatial": {"crs": "EPSG:4326"}}))


def write_aef_raster(path: Path) -> None:
    """Write a tiny two-band raster with one NoData pixel in the center support."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(9, dtype=np.int16)[:, None]
    cols = np.arange(9, dtype=np.int16)[None, :]
    a00 = rows * 10 + cols
    a01 = a00 + 100
    a00[3, 3] = -999
    a01[3, 3] = -999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=9,
        width=9,
        count=2,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(0.0, 9.0, 1.0, 1.0),
        nodata=-999,
    ) as dataset:
        dataset.write(a00, 1)
        dataset.write(a01, 2)
        dataset.set_band_description(1, "A00")
        dataset.set_band_description(2, "A01")


def write_tile_manifest(path: Path, raster_path: Path) -> None:
    """Write the AEF tile manifest fields consumed by alignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": [
            {
                "year": 2018,
                "preferred_read_path": str(raster_path),
                "source_href": "s3://example/aef/2018/tile.tif",
                "raster": {
                    "crs": "EPSG:4326",
                    "band_count": 2,
                    "band_names": ["A00", "A01"],
                },
            }
        ]
    }
    path.write_text(json.dumps(payload))


def expected_center_means() -> tuple[float, float]:
    """Return expected means for the center station after all-band NoData masking."""
    rows = np.arange(9, dtype=float)[:, None]
    cols = np.arange(9, dtype=float)[None, :]
    a00 = rows * 10 + cols
    support = a00[3:6, 3:6]
    valid_mask = np.ones((3, 3), dtype=bool)
    valid_mask[0, 0] = False
    return float(support[valid_mask].mean()), float((support + 100)[valid_mask].mean())


def expected_upper_left_average() -> float:
    """Return the expected A00 average for the upper-left AEF-aligned 3x3 cell."""
    rows = np.arange(9, dtype=float)[:, None]
    cols = np.arange(9, dtype=float)[None, :]
    a00 = rows * 10 + cols
    return float(a00[0:3, 0:3].mean())
