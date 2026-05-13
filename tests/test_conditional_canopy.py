import json
from pathlib import Path

import numpy as np
import pandas as pd

from kelp_aef import main


def test_train_conditional_canopy_writes_artifacts(tmp_path: Path) -> None:
    """Verify train-conditional-canopy writes positive-only amount diagnostics."""
    fixture = write_conditional_canopy_fixture(tmp_path)

    assert main(["train-conditional-canopy", "--config", str(fixture["config_path"])]) == 0

    predictions = pd.read_parquet(fixture["sample_predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    residuals = pd.read_csv(fixture["positive_residuals"])
    comparison = pd.read_csv(fixture["model_comparison"])
    full_grid = pd.read_csv(fixture["full_grid_likely_positive_summary"])
    manifest = json.loads(fixture["manifest"].read_text())

    assert fixture["model"].is_file()
    assert fixture["residual_figure"].is_file()
    assert {"conditional_observed_positive_sample", "conditional_likely_positive_diagnostic"} <= (
        set(predictions["evaluation_scope"])
    )
    assert set(metrics["evaluation_scope"]) == {"conditional_observed_positive_sample"}
    assert {"annual_max_ge_50pct", "near_saturated_ge_810m2"} <= set(residuals["observed_bin"])
    assert {"ridge_positive_annual_max", "ridge_regression"} <= set(comparison["model_name"])
    assert set(full_grid["likely_positive_threshold_policy"]) == {"validation_max_f1_calibrated"}
    assert manifest["selection_split"] == "validation"
    assert not manifest["test_rows_used_for_training_or_selection"]
    assert not manifest["full_grid_hurdle_predictions_written"]
    assert manifest["row_counts"]["train_observed_positive_rows"] == 9


def test_train_conditional_canopy_writes_crm_stratified_reuse_manifest(
    tmp_path: Path,
) -> None:
    """Verify CRM-stratified conditional sidecar reuses the observed-positive model."""
    fixture = write_conditional_canopy_fixture(tmp_path, include_sidecar=True)

    assert main(["train-conditional-canopy", "--config", str(fixture["config_path"])]) == 0

    manifest = json.loads(fixture["sidecar_reuse_manifest"].read_text())
    summary = pd.read_csv(fixture["sidecar_full_grid_likely_positive_summary"])

    assert manifest["sample_policy"] == "crm_stratified_background_sample"
    assert manifest["conditional_model_reused"]
    assert manifest["support_sets_match"]
    assert set(summary["likely_positive_threshold_policy"]) == {"validation_max_f1_calibrated"}


def write_conditional_canopy_fixture(
    tmp_path: Path,
    *,
    include_sidecar: bool = False,
) -> dict[str, Path]:
    """Write synthetic conditional-canopy inputs and config."""
    sample = tmp_path / "interim/aligned_background_sample_training_table.masked.parquet"
    sidecar_sample = (
        tmp_path / "interim/aligned_background_sample_training_table.crm_stratified.masked.parquet"
    )
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    baseline_predictions = tmp_path / "processed/baseline_sample_predictions.parquet"
    calibrated_sample = tmp_path / "processed/binary_presence_calibrated_sample_predictions.parquet"
    calibrated_full_grid = (
        tmp_path / "reports/tables/binary_presence_calibrated_full_grid_area_summary.csv"
    )
    model = tmp_path / "models/conditional_canopy/ridge_positive_annual_max.joblib"
    sample_predictions = tmp_path / "processed/conditional_canopy_sample_predictions.parquet"
    metrics = tmp_path / "reports/tables/conditional_canopy_metrics.csv"
    positive_residuals = tmp_path / "reports/tables/conditional_canopy_positive_residuals.csv"
    model_comparison = tmp_path / "reports/tables/conditional_canopy_model_comparison.csv"
    full_grid_likely_positive_summary = (
        tmp_path / "reports/tables/conditional_canopy_full_grid_likely_positive_summary.csv"
    )
    sidecar_full_grid_likely_positive_summary = (
        tmp_path
        / "reports/tables/conditional_canopy_full_grid_likely_positive_summary.crm_stratified.csv"
    )
    sidecar_reuse_manifest = (
        tmp_path / "interim/conditional_canopy_reuse_manifest.crm_stratified.json"
    )
    residual_figure = tmp_path / "reports/figures/conditional_canopy_positive_residuals.png"
    manifest = tmp_path / "interim/conditional_canopy_manifest.json"
    config_path = tmp_path / "config.yaml"
    write_conditional_rows(sample)
    if include_sidecar:
        write_conditional_rows(sidecar_sample)
    write_conditional_split_manifest(split_manifest)
    write_conditional_baseline_predictions(baseline_predictions)
    write_calibrated_sample_predictions(calibrated_sample)
    write_calibrated_full_grid_summary(calibrated_full_grid)
    sidecar_config = (
        f"""
    sidecars:
      crm_stratified:
        sample_policy: crm_stratified_background_sample
        reuse_model: true
        input_table: {sidecar_sample}
        calibrated_binary_sample_predictions: {calibrated_sample}
        calibrated_binary_full_grid_area_summary: {calibrated_full_grid}
        full_grid_likely_positive_summary: {sidecar_full_grid_likely_positive_summary}
        reuse_manifest: {sidecar_reuse_manifest}
"""
        if include_sidecar
        else ""
    )
    config_path.write_text(
        f"""
features:
  bands: A00-A01
alignment:
  output_table: {sample}
splits:
  train_years: [2018, 2019, 2020]
  validation_years: [2021]
  test_years: [2022]
  output_manifest: {split_manifest}
models:
  baselines:
    sample_predictions: {baseline_predictions}
  binary_presence:
    calibration:
      calibrated_sample_predictions: {calibrated_sample}
      full_grid_area_summary: {calibrated_full_grid}
  conditional_canopy:
    input_table: {sample}
    target: kelp_fraction_y
    target_area_column: kelp_max_y
    positive_target_label: annual_max_ge_10pct
    positive_target_threshold_fraction: 0.10
    positive_target_threshold_area: 90.0
    positive_support_policy: observed_positive_train_validation_test
    likely_positive_policy: calibrated_probability_ge_validation_max_f1
    likely_positive_threshold_policy: validation_max_f1_calibrated
    features: A00-A01
    alpha_grid: [0.1, 1.0]
    model: {model}
    sample_predictions: {sample_predictions}
    metrics: {metrics}
    positive_residuals: {positive_residuals}
    model_comparison: {model_comparison}
    full_grid_likely_positive_summary: {full_grid_likely_positive_summary}
    residual_figure: {residual_figure}
    manifest: {manifest}
{sidecar_config}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "model": model,
        "sample_predictions": sample_predictions,
        "metrics": metrics,
        "positive_residuals": positive_residuals,
        "model_comparison": model_comparison,
        "full_grid_likely_positive_summary": full_grid_likely_positive_summary,
        "residual_figure": residual_figure,
        "manifest": manifest,
        "sidecar_full_grid_likely_positive_summary": sidecar_full_grid_likely_positive_summary,
        "sidecar_reuse_manifest": sidecar_reuse_manifest,
    }


def write_conditional_rows(path: Path) -> None:
    """Write tiny feature/label rows for conditional model tests."""
    rows: list[dict[str, object]] = []
    for year in (2018, 2019, 2020, 2021, 2022):
        for cell_id, fraction in enumerate((0.0, 0.12, 0.55, 0.95)):
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": cell_id if cell_id else np.nan,
                    "longitude": -122.0 + cell_id * 0.001,
                    "latitude": 36.0 + cell_id * 0.001,
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "A00": fraction + (year - 2018) * 0.002,
                    "A01": fraction * 0.5,
                    "aef_grid_cell_id": cell_id,
                    "aef_grid_row": cell_id,
                    "aef_grid_col": cell_id,
                    "label_source": "kelpwatch_station" if cell_id else "assumed_background",
                    "is_kelpwatch_observed": cell_id > 0,
                    "is_plausible_kelp_domain": True,
                    "domain_mask_reason": "retained_shallow_depth",
                    "domain_mask_detail": "fixture",
                    "domain_mask_version": "test_mask_v1",
                    "depth_bin": "shallow_depth",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_conditional_split_manifest(path: Path) -> None:
    """Write split labels matching synthetic conditional rows."""
    split_by_year = {
        2018: "train",
        2019: "train",
        2020: "train",
        2021: "validation",
        2022: "test",
    }
    rows = [
        {
            "year": year,
            "aef_grid_cell_id": cell_id,
            "split": split,
            "used_for_training_eval": True,
        }
        for year, split in split_by_year.items()
        for cell_id in range(4)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_conditional_baseline_predictions(path: Path) -> None:
    """Write ridge sample predictions that underpredict positive canopy."""
    split_by_year = {
        2018: "train",
        2019: "train",
        2020: "train",
        2021: "validation",
        2022: "test",
    }
    rows: list[dict[str, object]] = []
    for year, split in split_by_year.items():
        for cell_id, fraction in enumerate((0.0, 0.12, 0.55, 0.95)):
            predicted = min(1.0, fraction * 0.5)
            rows.append(
                {
                    "model_name": "ridge_regression",
                    "split": split,
                    "year": year,
                    "label_source": "kelpwatch_station" if cell_id else "assumed_background",
                    "is_kelpwatch_observed": cell_id > 0,
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "pred_kelp_fraction_y_clipped": predicted,
                    "pred_kelp_max_y": predicted * 900.0,
                    "aef_grid_cell_id": cell_id,
                    "is_plausible_kelp_domain": True,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_calibrated_sample_predictions(path: Path) -> None:
    """Write calibrated binary sample predictions for likely-positive diagnostics."""
    rows: list[dict[str, object]] = []
    for year in (2018, 2019, 2020, 2021, 2022):
        for cell_id, probability in enumerate((0.8, 0.7, 0.9, 0.95)):
            rows.append(
                {
                    "year": year,
                    "aef_grid_cell_id": cell_id,
                    "calibrated_binary_probability": probability,
                    "calibrated_probability_threshold": 0.4,
                    "calibrated_pred_binary_class": probability >= 0.4,
                    "calibrated_threshold_policy": "validation_max_f1_calibrated",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_calibrated_full_grid_summary(path: Path) -> None:
    """Write compact calibrated full-grid area summary rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "probability_source": "platt_calibrated",
                "threshold_policy": "validation_max_f1_calibrated",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "mask_status": "plausible_kelp_domain",
                "evaluation_scope": "full_grid_masked",
                "probability_threshold": 0.4,
                "row_count": 10,
                "predicted_positive_count": 4,
                "predicted_positive_area_m2": 3600.0,
                "observed_positive_count": 3,
                "observed_positive_rate": 0.3,
                "assumed_background_count": 7,
                "assumed_background_predicted_positive_count": 1,
            }
        ]
    ).to_csv(path, index=False)
