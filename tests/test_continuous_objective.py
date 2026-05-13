import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main


def test_train_continuous_objective_writes_capped_weight_artifacts(tmp_path: Path) -> None:
    """Verify the capped-weight continuous objective writes model and report artifacts."""
    fixture = write_continuous_objective_fixture(tmp_path)

    assert (
        main(
            [
                "train-continuous-objective",
                "--config",
                str(fixture["config_path"]),
                "--experiment",
                "capped-weight",
            ]
        )
        == 0
    )

    sample_predictions = pd.read_parquet(fixture["sample_predictions"])
    full_grid_predictions = pd.read_parquet(fixture["full_grid_predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    area = pd.read_csv(fixture["area_calibration"])
    leakage = pd.read_csv(fixture["assumed_background_leakage"])
    manifest = json.loads(fixture["manifest"].read_text())

    assert fixture["model"].is_file()
    assert fixture["manifest"].is_file()
    assert set(sample_predictions["model_name"]) == {"ridge_capped_weight"}
    assert set(full_grid_predictions["model_family"]) == {"continuous_objective"}
    observed = sample_predictions.loc[sample_predictions["is_kelpwatch_observed"]]
    background = sample_predictions.loc[sample_predictions["label_source"] == "assumed_background"]
    assert set(observed["fit_weight"]) == {1.0}
    assert float(background["fit_weight"].max()) == 5.0
    assert set(metrics["weighted"].astype(str).str.lower()) == {"false"}
    assert {"all", "kelpwatch_station", "assumed_background"} <= set(metrics["label_source"])
    assert {"all", "kelpwatch_station", "assumed_background"} <= set(area["label_source"])
    assert set(area["evaluation_scope"]) == {"full_grid_prediction"}
    assert set(leakage["label_source"]) == {"assumed_background"}
    assert int(manifest["row_counts"]["full_grid_prediction_rows"]) == len(full_grid_predictions)
    assert manifest["fit_weight_cap"] == 5.0
    assert manifest["evaluation_metrics_weighted"] is False
    assert manifest["test_rows_used_for_training_or_model_selection"] is False
    assert manifest["selected_alpha"] in {0.01, 1.0}


def write_continuous_objective_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write synthetic inputs and config for a continuous-objective command run."""
    input_table = tmp_path / "interim/aligned_background_sample_training_table.masked.parquet"
    full_grid_table = tmp_path / "interim/aligned_full_grid_training_table.parquet"
    model = tmp_path / "models/continuous_objective/ridge_capped_weight.joblib"
    sample_predictions = (
        tmp_path / "processed/continuous_objective_capped_weight_sample_predictions.parquet"
    )
    full_grid_predictions = (
        tmp_path / "processed/continuous_objective_capped_weight_full_grid_predictions.parquet"
    )
    metrics = tmp_path / "reports/tables/continuous_objective_capped_weight_metrics.csv"
    area_calibration = (
        tmp_path / "reports/tables/continuous_objective_capped_weight_area_calibration.csv"
    )
    leakage = (
        tmp_path
        / "reports/tables/continuous_objective_capped_weight_assumed_background_leakage.csv"
    )
    manifest = tmp_path / "interim/continuous_objective_capped_weight_manifest.json"
    config_path = tmp_path / "config.yaml"
    write_objective_table(input_table)
    write_objective_table(full_grid_table)
    config_path.write_text(
        f"""
features:
  bands: A00-A01
splits:
  train_years: [2018]
  validation_years: [2019]
  test_years: [2020]
models:
  baselines:
    sample_policy: crm_stratified_mask_first_sample
    input_table: {input_table}
    inference_table: {full_grid_table}
    target: kelp_fraction_y
    features: A00-A01
  continuous_objective:
    sample_policy: crm_stratified_mask_first_sample
    input_table: {input_table}
    inference_table: {full_grid_table}
    target: kelp_fraction_y
    target_area_column: kelp_max_y
    features: A00-A01
    alpha_grid: [0.01, 1.0]
    drop_missing_features: true
    sample_weight_column: sample_weight
    batch_size: 3
    experiments:
      capped-weight:
        model_name: ridge_capped_weight
        objective_policy: capped_weighted_ridge
        fit_weight_policy: capped_assumed_background_sample_weight
        fit_weight_cap: 5.0
        model: {model}
        sample_predictions: {sample_predictions}
        full_grid_predictions: {full_grid_predictions}
        metrics: {metrics}
        area_calibration: {area_calibration}
        assumed_background_leakage: {leakage}
        manifest: {manifest}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "model": model,
        "sample_predictions": sample_predictions,
        "full_grid_predictions": full_grid_predictions,
        "metrics": metrics,
        "area_calibration": area_calibration,
        "assumed_background_leakage": leakage,
        "manifest": manifest,
    }


def write_objective_table(path: Path) -> None:
    """Write a tiny feature table with station and assumed-background rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for year, year_shift in [(2018, 0.0), (2019, 0.02), (2020, 0.04)]:
        rows.extend(
            [
                objective_row(year, 0, 0.55 + year_shift, True, 1.0),
                objective_row(year, 1, 0.05, True, 1.0),
                objective_row(year, 2, 0.0, False, 20.0),
                objective_row(year, 3, 0.0, False, 2.0),
            ]
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def objective_row(
    year: int,
    cell_id: int,
    fraction: float,
    observed: bool,
    sample_weight: float,
) -> dict[str, object]:
    """Build one synthetic model-input row."""
    return {
        "year": year,
        "kelpwatch_station_id": year * 10 + cell_id,
        "longitude": -122.0 + cell_id * 0.001,
        "latitude": 36.0 + cell_id * 0.001,
        "kelp_fraction_y": fraction,
        "kelp_max_y": fraction * 900.0,
        "A00": fraction + cell_id * 0.01,
        "A01": np.sin(fraction + cell_id),
        "aef_grid_cell_id": cell_id,
        "aef_grid_row": cell_id,
        "aef_grid_col": cell_id,
        "label_source": "kelpwatch_station" if observed else "assumed_background",
        "is_kelpwatch_observed": observed,
        "kelpwatch_station_count": 1 if observed else 0,
        "sample_weight": sample_weight,
    }
