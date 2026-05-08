import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from kelp_aef import main


def test_align_full_grid_fast_writes_background_and_observed_rows(tmp_path: Path) -> None:
    """Verify full-grid alignment retains assumed background and observed labels."""
    fixture = write_full_grid_fixture(tmp_path)

    assert main(["align-full-grid", "--config", str(fixture["config_path"]), "--fast"]) == 0

    full_grid = pd.read_parquet(fixture["fast_full_grid"])
    sample = pd.read_parquet(fixture["fast_sample"])
    manifest = json.loads(fixture["fast_manifest"].read_text())

    assert len(full_grid) == 4
    assert set(full_grid["label_source"]) == {"kelpwatch_station", "assumed_background"}
    assert int((full_grid["label_source"] == "kelpwatch_station").sum()) == 2
    assert int((full_grid["label_source"] == "assumed_background").sum()) == 2
    assert set(sample["label_source"]) == {"kelpwatch_station", "assumed_background"}
    assert (sample["sample_weight"] >= 1.0).all()
    assert manifest["label_source_counts"]["2018:kelpwatch_station"] == 2
    assert manifest["label_source_counts"]["2018:assumed_background"] == 2

    observed = full_grid.query("label_source == 'kelpwatch_station'").sort_values("kelp_max_y")
    assert observed["kelp_max_y"].to_list() == [0.0, 90.0]
    assert observed.iloc[1]["A00"] == pytest.approx(
        np.mean([[0, 1, 2], [10, 11, 12], [20, 21, 22]])
    )


def write_full_grid_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write a tiny full-grid config, annual labels, manifest, and AEF raster."""
    labels_path = tmp_path / "interim/labels_annual.parquet"
    label_manifest_path = tmp_path / "interim/labels_annual_manifest.json"
    raster_path = tmp_path / "raw/aef/2018/tile.tif"
    tile_manifest_path = tmp_path / "interim/aef_manifest.json"
    full_grid = tmp_path / "interim/full_grid.parquet"
    fast_full_grid = tmp_path / "interim/full_grid.fast.parquet"
    fast_manifest = tmp_path / "interim/full_grid.fast_manifest.json"
    sample = tmp_path / "interim/sample.parquet"
    fast_sample = tmp_path / "interim/sample.fast.parquet"
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
  full_grid:
    output_table: {full_grid}
    output_manifest: {tmp_path / "interim/full_grid_manifest.json"}
    summary_table: {tmp_path / "tables/full_grid_summary.csv"}
    target_row_chunk_size: 2
    fast:
      years: [2018]
      row_window: [0, 2]
      col_window: [0, 2]
      output_table: {fast_full_grid}
      output_manifest: {fast_manifest}
      summary_table: {tmp_path / "tables/full_grid.fast_summary.csv"}
  background_sample:
    output_table: {sample}
    output_manifest: {tmp_path / "interim/sample_manifest.json"}
    summary_table: {tmp_path / "tables/sample_summary.csv"}
    background_rows_per_year: 100
    random_seed: 13
    include_all_kelpwatch_observed: true
    sample_weight_column: sample_weight
""".lstrip()
    )
    return {
        "config_path": config_path,
        "fast_full_grid": fast_full_grid,
        "fast_manifest": fast_manifest,
        "fast_sample": fast_sample,
    }


def write_label_table(path: Path) -> None:
    """Write one positive and one zero Kelpwatch station label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "year": [2018, 2018],
            "kelpwatch_station_id": [10, 11],
            "longitude": [1.5, 4.5],
            "latitude": [4.5, 1.5],
            "kelp_max_y": [90.0, 0.0],
            "kelp_fraction_y": [0.1, 0.0],
            "area_q1": [0.0, 0.0],
            "area_q2": [90.0, 0.0],
            "area_q3": [45.0, 0.0],
            "area_q4": [20.0, 0.0],
            "max_area_quarter": [2, 1],
            "valid_quarter_count": [4, 4],
            "nonzero_quarter_count": [3, 0],
            "kelp_present_gt0_y": [True, False],
            "kelp_present_ge_1pct_y": [True, False],
            "kelp_present_ge_5pct_y": [True, False],
            "kelp_present_ge_10pct_y": [True, False],
        }
    ).to_parquet(path, index=False)


def write_label_manifest(path: Path) -> None:
    """Write the label CRS manifest used by full-grid alignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spatial": {"crs": "EPSG:4326"}}))


def write_aef_raster(path: Path) -> None:
    """Write a 6x6 two-band raster that aggregates to a 2x2 target grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(6, dtype=np.int16)[:, None]
    cols = np.arange(6, dtype=np.int16)[None, :]
    a00 = rows * 10 + cols
    a01 = a00 + 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=2,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(0.0, 6.0, 1.0, 1.0),
        nodata=-999,
    ) as dataset:
        dataset.write(a00, 1)
        dataset.write(a01, 2)
        dataset.set_band_description(1, "A00")
        dataset.set_band_description(2, "A01")


def write_tile_manifest(path: Path, raster_path: Path) -> None:
    """Write the AEF tile manifest consumed by full-grid alignment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
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
        )
    )
