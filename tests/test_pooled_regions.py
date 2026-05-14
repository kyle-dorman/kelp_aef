import json
from pathlib import Path

import pandas as pd

from kelp_aef import main


def test_build_pooled_region_sample_writes_region_keys(tmp_path: Path) -> None:
    """Verify pooled sample construction preserves source-region identity."""
    monterey = tmp_path / "monterey_sample.parquet"
    big_sur = tmp_path / "big_sur_sample.parquet"
    config = tmp_path / "config.yaml"
    output_table = tmp_path / "interim/pooled.parquet"
    split_manifest = tmp_path / "interim/pooled_split.parquet"
    manifest = tmp_path / "interim/pooled_manifest.json"
    summary = tmp_path / "reports/pooled_summary.csv"

    write_sample(monterey, longitude=-121.9)
    write_sample(big_sur, longitude=-121.4)
    config.write_text(
        f"""
features:
  bands: A00-A01
splits:
  train_years: [2018, 2019, 2020]
  validation_years: [2021]
  test_years: [2022]
models:
  baselines:
    target: kelp_fraction_y
pooled_region_sample:
  regions:
    monterey:
      input_table: {monterey}
    big_sur:
      input_table: {big_sur}
  output_table: {output_table}
  split_manifest: {split_manifest}
  manifest: {manifest}
  summary_table: {summary}
""".lstrip()
    )

    assert main(["build-pooled-region-sample", "--config", str(config)]) == 0

    pooled = pd.read_parquet(output_table)
    splits = pd.read_parquet(split_manifest)
    manifest_payload = json.loads(manifest.read_text())
    summary_rows = pd.read_csv(summary)

    assert set(pooled["source_region"]) == {"monterey", "big_sur"}
    assert not splits.duplicated(["source_region", "year", "aef_grid_cell_id"]).any()
    assert set(splits["split"]) == {"train", "validation", "test"}
    assert manifest_payload["region_metadata_used_as_predictor"] is False
    assert {"monterey", "big_sur"} <= set(summary_rows["source_region"])


def test_compare_training_regimes_normalizes_primary_rows(tmp_path: Path) -> None:
    """Verify combined comparison rows use configured canonical regime labels."""
    raw_a = tmp_path / "a.csv"
    raw_b = tmp_path / "b.csv"
    config = tmp_path / "config.yaml"
    comparison = tmp_path / "combined.csv"
    primary = tmp_path / "primary.csv"
    manifest = tmp_path / "manifest.json"
    write_comparison(raw_a, training_regime="monterey_transfer")
    write_comparison(raw_b, training_regime="pooled_monterey_big_sur")
    config.write_text(
        f"""
training_regime_comparison:
  model_comparison: {comparison}
  primary_summary: {primary}
  manifest: {manifest}
  inputs:
    transfer:
      path: {raw_a}
      training_regime: monterey_only
      model_origin_region: monterey
      evaluation_region: big_sur
    pooled:
      path: {raw_b}
      training_regime: pooled_monterey_big_sur
      model_origin_region: monterey_big_sur
      evaluation_region: big_sur
""".lstrip()
    )

    assert main(["compare-training-regimes", "--config", str(config)]) == 0

    combined = pd.read_csv(comparison)
    primary_rows = pd.read_csv(primary)
    manifest_payload = json.loads(manifest.read_text())

    assert set(combined["training_regime"]) == {"monterey_only", "pooled_monterey_big_sur"}
    assert set(primary_rows["year"]) == {2022}
    assert manifest_payload["raw_transfer_rows_normalized"] is True


def write_sample(path: Path, *, longitude: float) -> None:
    """Write a tiny five-year retained-domain sample for one region."""
    rows = []
    for year in range(2018, 2023):
        rows.append(
            {
                "year": year,
                "kelpwatch_station_id": 1,
                "longitude": longitude,
                "latitude": 36.0,
                "kelp_fraction_y": 0.2 if year == 2022 else 0.0,
                "kelp_max_y": 180.0 if year == 2022 else 0.0,
                "A00": 0.1,
                "A01": 0.2,
                "aef_grid_cell_id": 1,
                "label_source": "kelpwatch_station",
                "is_kelpwatch_observed": True,
                "sample_weight": 1.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_comparison(path: Path, *, training_regime: str) -> None:
    """Write a minimal training-regime comparison CSV."""
    frame = pd.DataFrame(
        [
            {
                "training_regime": training_regime,
                "model_origin_region": "raw",
                "evaluation_region": "raw",
                "model_name": "ridge_regression",
                "model_family": "aef_ridge",
                "composition_policy": "",
                "split": "test",
                "year": 2022,
                "mask_status": "plausible_kelp_domain",
                "evaluation_scope": "full_grid_masked",
                "label_source": "all",
                "row_count": 10,
                "mae": 0.1,
                "rmse": 0.2,
                "r2": 0.3,
                "f1_ge_10pct": 0.4,
                "observed_canopy_area": 100.0,
                "predicted_canopy_area": 90.0,
                "area_pct_bias": -0.1,
                "source_table": "raw.csv",
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
