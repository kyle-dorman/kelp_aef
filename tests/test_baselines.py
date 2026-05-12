import json
from pathlib import Path

import numpy as np
import pandas as pd

from kelp_aef import main
from kelp_aef.evaluation.baselines import write_reference_area_calibration


def test_train_baselines_writes_artifacts_and_selects_alpha(tmp_path: Path) -> None:
    """Verify train-baselines writes split, model, prediction, metric, and manifest artifacts."""
    fixture = write_baseline_fixture(tmp_path)

    assert main(["train-baselines", "--config", str(fixture["config_path"])]) == 0

    split_manifest = pd.read_parquet(fixture["split_manifest"])
    predictions = pd.read_parquet(fixture["predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    manifest = json.loads(fixture["manifest"].read_text())
    fallback_summary = pd.read_csv(fixture["fallback_summary"])

    assert fixture["model"].is_file()
    assert fixture["geographic_model"].is_file()
    assert len(split_manifest) == 9
    assert int((~split_manifest["used_for_training_eval"]).sum()) == 1
    assert manifest["dropped_counts_by_split"] == {"train": 0, "validation": 0, "test": 1}
    assert manifest["selected_alpha"] == 0.01
    assert manifest["geographic_selected_alpha"] == 0.01
    assert manifest["retained_row_count"] == 8
    assert len(predictions) == 37
    assert set(predictions["model_name"]) == {
        "no_skill_train_mean",
        "ridge_regression",
        "previous_year_annual_max",
        "grid_cell_climatology",
        "geographic_ridge_lon_lat_year",
    }
    assert set(metrics["model_name"]) == set(predictions["model_name"])
    assert set(metrics["split"]) == {"train", "validation", "test"}

    train_mean = np.mean([0.0, 0.1, 0.2])
    no_skill = predictions.query("model_name == 'no_skill_train_mean' and split == 'train'")
    assert np.allclose(no_skill["pred_kelp_fraction_y"], train_mean)
    previous_validation = predictions.query(
        "model_name == 'previous_year_annual_max' and split == 'validation'"
    ).sort_values("aef_grid_cell_id")
    assert np.allclose(previous_validation["pred_kelp_fraction_y"], [0.0, 0.1, 0.2])
    climatology_test = predictions.query(
        "model_name == 'grid_cell_climatology' and split == 'test'"
    ).sort_values("aef_grid_cell_id")
    assert np.allclose(climatology_test["pred_kelp_fraction_y"], [0.0, 0.1])
    assert "missing_previous_year" in set(fallback_summary["fallback_reason"])
    assert "cell_training_mean" in set(fallback_summary["fallback_reason"])


def test_predict_full_grid_streams_trained_ridge_predictions(tmp_path: Path) -> None:
    """Verify predict-full-grid writes streamed ridge predictions with provenance."""
    fixture = write_baseline_fixture(tmp_path, include_full_grid=True)

    assert main(["train-baselines", "--config", str(fixture["config_path"])]) == 0
    assert main(["predict-full-grid", "--config", str(fixture["config_path"])]) == 0

    predictions = pd.read_parquet(fixture["full_predictions"])
    manifest = json.loads(fixture["prediction_manifest"].read_text())
    fallback_summary = pd.read_csv(fixture["fallback_summary"])

    assert len(predictions) == 9
    assert set(predictions["model_name"]) == {"ridge_regression"}
    assert {"aef_grid_cell_id", "label_source", "is_kelpwatch_observed"}.issubset(
        predictions.columns
    )
    assert manifest["row_count"] == 9
    assert manifest["part_count"] >= 1
    assert manifest["prediction_models"] == ["ridge_regression"]
    assert set(fallback_summary["evaluation_scope"]) == {"background_inclusive_sample"}


def test_reference_area_calibration_writes_compact_full_grid_rows(tmp_path: Path) -> None:
    """Verify reference baselines write aggregate full-grid calibration rows only."""
    fixture = write_baseline_fixture(tmp_path, include_full_grid=True)

    assert main(["train-baselines", "--config", str(fixture["config_path"])]) == 0
    rows = write_reference_area_calibration(fixture["config_path"])

    calibration = pd.read_csv(fixture["area_calibration"])
    assert fixture["area_calibration"].is_file()
    assert len(rows) == len(calibration)
    assert len(calibration) == 42
    assert set(calibration["model_name"]) == {
        "no_skill_train_mean",
        "ridge_regression",
        "previous_year_annual_max",
        "grid_cell_climatology",
        "geographic_ridge_lon_lat_year",
    }
    assert set(calibration["label_source"]) == {
        "all",
        "assumed_background",
        "kelpwatch_station",
    }
    assert "test" in set(calibration["split"])


def test_reference_area_calibration_filters_to_reporting_domain_mask(tmp_path: Path) -> None:
    """Verify compact full-grid calibration uses the configured reporting mask."""
    fixture = write_baseline_fixture(
        tmp_path,
        include_full_grid=True,
        include_domain_mask=True,
    )

    assert main(["train-baselines", "--config", str(fixture["config_path"])]) == 0
    rows = write_reference_area_calibration(fixture["config_path"])

    calibration = pd.read_csv(fixture["area_calibration"])
    ridge_test_all = calibration.query(
        "model_name == 'ridge_regression' and split == 'test' and label_source == 'all'"
    ).iloc[0]
    assert len(rows) == len(calibration)
    assert int(ridge_test_all["row_count"]) == 2
    assert set(calibration["mask_status"]) == {"plausible_kelp_domain"}
    assert set(calibration["evaluation_scope"]) == {"full_grid_masked"}


def write_baseline_fixture(
    tmp_path: Path,
    *,
    include_full_grid: bool = False,
    include_domain_mask: bool = False,
) -> dict[str, Path]:
    """Write a synthetic aligned table and minimal baseline training config."""
    aligned_table = tmp_path / "interim/aligned_training_table.parquet"
    full_grid_table = tmp_path / "interim/aligned_full_grid_training_table.parquet"
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    model = tmp_path / "models/baselines/ridge_kelp_fraction.joblib"
    sample_predictions = tmp_path / "processed/baseline_sample_predictions.parquet"
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    prediction_manifest = tmp_path / "interim/baseline_prediction_manifest.json"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    fallback_summary = tmp_path / "reports/tables/reference_baseline_fallback_summary.csv"
    area_calibration = tmp_path / (
        "reports/tables/reference_baseline_area_calibration.masked.csv"
        if include_domain_mask
        else "reports/tables/reference_baseline_area_calibration.csv"
    )
    unmasked_area_calibration = tmp_path / "reports/tables/reference_baseline_area_calibration.csv"
    domain_mask = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    domain_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    manifest = tmp_path / "interim/baseline_eval_manifest.json"
    geographic_model = tmp_path / "models/baselines/geographic_ridge_lon_lat_year.joblib"
    config_path = tmp_path / "config.yaml"
    write_aligned_table(aligned_table)
    if include_full_grid:
        write_aligned_table(full_grid_table)
    if include_domain_mask:
        write_baseline_domain_mask(domain_mask, domain_manifest)
    domain_mask_config = (
        f"""
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
"""
        if include_domain_mask
        else ""
    )
    masked_area_output = (
        f"    reference_baseline_area_calibration_masked: {area_calibration}\n"
        if include_domain_mask
        else ""
    )
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
    input_table: {aligned_table if include_full_grid else ""}
    inference_table: {full_grid_table if include_full_grid else ""}
    target: kelp_fraction_y
    features: A00-A01
    alpha_grid: [0.01, 100.0]
    drop_missing_features: true
    use_sample_weight: {str(include_full_grid).lower()}
    sample_weight_column: sample_weight
    ridge_model: {model}
    geographic_model: {geographic_model}
    sample_predictions: {sample_predictions}
    predictions: {predictions}
    prediction_manifest: {prediction_manifest}
    metrics: {metrics}
    manifest: {manifest}
reports:
{domain_mask_config}
  outputs:
    reference_baseline_fallback_summary: {fallback_summary}
    reference_baseline_area_calibration: {unmasked_area_calibration}
{masked_area_output}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "split_manifest": split_manifest,
        "model": model,
        "geographic_model": geographic_model,
        "predictions": sample_predictions,
        "full_predictions": predictions,
        "prediction_manifest": prediction_manifest,
        "metrics": metrics,
        "fallback_summary": fallback_summary,
        "area_calibration": area_calibration,
        "manifest": manifest,
    }


def write_aligned_table(path: Path) -> None:
    """Write a tiny aligned feature/label table with one missing test feature row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
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
            rows[-1].update(
                {
                    "aef_grid_cell_id": index,
                    "aef_grid_row": index,
                    "aef_grid_col": index,
                    "label_source": "kelpwatch_station" if index == 0 else "assumed_background",
                    "is_kelpwatch_observed": index == 0,
                    "kelpwatch_station_count": 1 if index == 0 else 0,
                    "sample_weight": 1.0 if index == 0 else 5.0,
                }
            )
    rows[-1]["A00"] = np.nan
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_baseline_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a tiny mask that retains two of three fixture target-grid cells."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [0, 1, 2],
            "is_plausible_kelp_domain": [True, True, False],
            "domain_mask_reason": ["retained", "retained", "dropped_too_deep"],
            "domain_mask_detail": ["fixture"] * 3,
            "domain_mask_version": ["test_mask_v1"] * 3,
        }
    ).to_parquet(mask_path, index=False)
    manifest_path.write_text(json.dumps({"mask_version": "test_mask_v1"}))
