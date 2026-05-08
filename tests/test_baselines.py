import json
from pathlib import Path

import numpy as np
import pandas as pd

from kelp_aef import main


def test_train_baselines_writes_artifacts_and_selects_alpha(tmp_path: Path) -> None:
    """Verify train-baselines writes split, model, prediction, metric, and manifest artifacts."""
    fixture = write_baseline_fixture(tmp_path)

    assert main(["train-baselines", "--config", str(fixture["config_path"])]) == 0

    split_manifest = pd.read_parquet(fixture["split_manifest"])
    predictions = pd.read_parquet(fixture["predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    manifest = json.loads(fixture["manifest"].read_text())

    assert fixture["model"].is_file()
    assert len(split_manifest) == 9
    assert int((~split_manifest["used_for_training_eval"]).sum()) == 1
    assert manifest["dropped_counts_by_split"] == {"train": 0, "validation": 0, "test": 1}
    assert manifest["selected_alpha"] == 0.01
    assert manifest["retained_row_count"] == 8
    assert len(predictions) == 16
    assert set(predictions["model_name"]) == {"no_skill_train_mean", "ridge_regression"}
    assert set(metrics["model_name"]) == {"no_skill_train_mean", "ridge_regression"}
    assert set(metrics["split"]) == {"train", "validation", "test"}

    train_mean = np.mean([0.0, 0.1, 0.2])
    no_skill = predictions.query("model_name == 'no_skill_train_mean' and split == 'train'")
    assert np.allclose(no_skill["pred_kelp_fraction_y"], train_mean)


def write_baseline_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write a synthetic aligned table and minimal baseline training config."""
    aligned_table = tmp_path / "interim/aligned_training_table.parquet"
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    model = tmp_path / "models/baselines/ridge_kelp_fraction.joblib"
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    manifest = tmp_path / "interim/baseline_eval_manifest.json"
    config_path = tmp_path / "config.yaml"
    write_aligned_table(aligned_table)
    config_path.write_text(
        f"""
alignment:
  output_table: {aligned_table}
features:
  bands: A00-A01
splits:
  train_years: [2018]
  validation_years: [2019]
  test_years: [2020]
  output_manifest: {split_manifest}
models:
  output_dir: {tmp_path / "models"}
  baselines:
    target: kelp_fraction_y
    features: A00-A01
    alpha_grid: [0.01, 100.0]
    drop_missing_features: true
    ridge_model: {model}
    predictions: {predictions}
    metrics: {metrics}
    manifest: {manifest}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "split_manifest": split_manifest,
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "manifest": manifest,
    }


def write_aligned_table(path: Path) -> None:
    """Write a tiny aligned feature/label table with one missing test feature row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for year, split_offset in [(2018, 0.0), (2019, 0.0), (2020, 0.0)]:
        for index in range(3):
            feature_value = float(index + split_offset)
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": year * 10 + index,
                    "longitude": -122.0 + index * 0.001,
                    "latitude": 36.0 + index * 0.001,
                    "kelp_fraction_y": feature_value / 10.0,
                    "kelp_max_y": feature_value / 10.0 * 900.0,
                    "A00": feature_value,
                    "A01": feature_value * 2.0,
                }
            )
    rows[-1]["A00"] = np.nan
    pd.DataFrame(rows).to_parquet(path, index=False)
