import json
from pathlib import Path

import numpy as np
import pandas as pd

from kelp_aef import main
from kelp_aef.evaluation.binary_presence import (
    BinaryPresenceConfig,
    build_binary_target,
    select_validation_threshold,
)


def test_train_binary_presence_writes_artifacts(tmp_path: Path) -> None:
    """Verify train-binary-presence writes model, predictions, metrics, and summaries."""
    fixture = write_binary_presence_fixture(tmp_path)

    assert main(["train-binary-presence", "--config", str(fixture["config_path"])]) == 0

    sample_predictions = pd.read_parquet(fixture["sample_predictions"])
    full_grid_predictions = pd.read_parquet(fixture["full_grid_predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    threshold_selection = pd.read_csv(fixture["threshold_selection"])
    area_summary = pd.read_csv(fixture["full_grid_area_summary"])
    comparison = pd.read_csv(fixture["thresholded_model_comparison"])
    manifest = json.loads(fixture["prediction_manifest"].read_text())

    assert fixture["model"].is_file()
    assert fixture["precision_recall_figure"].is_file()
    assert fixture["map_figure"].is_file()
    assert set(sample_predictions["target_label"]) == {"annual_max_ge_10pct"}
    assert sample_predictions["pred_binary_probability"].between(0, 1).all()
    assert {"pred_binary_class", "domain_mask_reason", "depth_bin"}.issubset(
        sample_predictions.columns
    )
    assert set(full_grid_predictions["is_plausible_kelp_domain"]) == {True}
    assert (
        int(area_summary.query("split == 'test' and label_source == 'all'")["row_count"].iloc[0])
        == 3
    )
    assert bool(threshold_selection["selected_threshold"].any())
    assert {"validation", "test"} <= set(metrics["split"])
    assert {"all", "assumed_background", "kelpwatch_station"} <= set(metrics["label_source"])
    assert {"logistic_annual_max_ge_10pct", "ridge_regression"} <= set(comparison["model_name"])
    assert {"balanced_binary", "thresholded_continuous_baseline"} <= set(comparison["model_family"])
    assert manifest["target_label"] == "annual_max_ge_10pct"
    assert manifest["selection_split"] == "validation"
    assert manifest["full_grid_row_count"] == 15
    assert manifest["thresholded_model_comparison_row_count"] > 0


def test_calibrate_binary_presence_writes_artifacts(tmp_path: Path) -> None:
    """Verify calibrate-binary-presence writes calibration artifacts from validation rows."""
    fixture = write_binary_presence_fixture(tmp_path)

    assert main(["train-binary-presence", "--config", str(fixture["config_path"])]) == 0
    assert main(["calibrate-binary-presence", "--config", str(fixture["config_path"])]) == 0

    calibrated = pd.read_parquet(fixture["calibrated_sample_predictions"])
    metrics = pd.read_csv(fixture["calibration_metrics"])
    thresholds = pd.read_csv(fixture["calibrated_threshold_selection"])
    area_summary = pd.read_csv(fixture["calibrated_full_grid_area_summary"])
    manifest = json.loads(fixture["calibration_manifest"].read_text())

    assert fixture["calibration_model"].is_file()
    assert fixture["calibration_curve_figure"].is_file()
    assert fixture["calibrated_threshold_figure"].is_file()
    assert calibrated["calibrated_binary_probability"].between(0, 1).all()
    assert {"raw_logistic", "platt_calibrated"} <= set(metrics["probability_source"])
    assert set(metrics["evaluation_split"]) == {"test"}
    assert set(thresholds["calibration_year"]) == {2021}
    assert bool(thresholds["selected_threshold"].any())
    assert {
        "p1_18_validation_raw_threshold",
        "validation_max_f1_calibrated",
        "validation_prevalence_match_calibrated",
    } <= set(area_summary["threshold_policy"])
    assert manifest["command"] == "calibrate-binary-presence"
    assert manifest["calibration_year"] == 2021
    assert manifest["evaluation_year"] == 2022
    assert manifest["calibration_includes_assumed_background_negatives"]
    assert "river mouth" in manifest["qa_notes"][0]


def test_train_binary_presence_writes_crm_stratified_sidecar(tmp_path: Path) -> None:
    """Verify optional CRM-stratified binary sidecar outputs and comparison."""
    fixture = write_binary_presence_fixture(tmp_path, include_sidecar=True)

    assert main(["train-binary-presence", "--config", str(fixture["config_path"])]) == 0

    sidecar_predictions = pd.read_parquet(fixture["sidecar_sample_predictions"])
    comparison = pd.read_csv(fixture["sidecar_comparison"])
    manifest = json.loads(fixture["sidecar_prediction_manifest"].read_text())

    assert fixture["sidecar_model"].is_file()
    assert sidecar_predictions["pred_binary_probability"].between(0, 1).all()
    assert set(comparison["sampling_policy"]) == {
        "current_masked_sample",
        "crm_stratified",
    }
    assert "full_grid_assumed_background_stratum" in set(comparison["comparison_scope"])
    assert manifest["sample_policy"] == "crm_stratified"


def test_binary_target_uses_10pct_annual_max_rule() -> None:
    """Verify target construction uses `>= 10%` rather than positive canopy."""
    target = build_binary_target(pd.Series([0.0, 0.099, 0.10, 0.5]), 0.10)

    assert target.tolist() == [False, False, True, True]


def test_validation_threshold_handles_no_positive_rows(tmp_path: Path) -> None:
    """Verify validation threshold selection remains explicit with no positives."""
    config = binary_presence_config(tmp_path)
    validation_rows = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "year": [2021, 2021],
            "kelp_fraction_y": [0.0, 0.0],
            "kelp_max_y": [0.0, 0.0],
            "binary_observed_y": [False, False],
            "label_source": ["assumed_background", "kelpwatch_station"],
        }
    )

    selection = select_validation_threshold(
        validation_rows,
        np.array([0.2, 0.8], dtype=float),
        config,
    )

    assert selection.threshold == 0.5
    assert selection.status == "no_valid_validation_threshold"
    assert not any(row["selected_threshold"] for row in selection.rows)


def write_binary_presence_fixture(
    tmp_path: Path, *, include_sidecar: bool = False
) -> dict[str, Path]:
    """Write synthetic binary-presence inputs and config."""
    sample = tmp_path / "interim/aligned_background_sample_training_table.masked.parquet"
    sidecar_sample = (
        tmp_path / "interim/aligned_background_sample_training_table.crm_stratified.masked.parquet"
    )
    full_grid = tmp_path / "interim/aligned_full_grid_training_table.parquet"
    baseline_predictions = tmp_path / "processed/baseline_sample_predictions.parquet"
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    domain_mask = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    domain_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    model = tmp_path / "models/binary_presence/logistic_annual_max_ge_10pct.joblib"
    sample_predictions = tmp_path / "processed/binary_presence_sample_predictions.parquet"
    full_grid_predictions = tmp_path / "processed/binary_presence_full_grid_predictions.parquet"
    metrics = tmp_path / "reports/tables/binary_presence_metrics.csv"
    threshold_selection = tmp_path / "reports/tables/binary_presence_threshold_selection.csv"
    full_grid_area_summary = tmp_path / "reports/tables/binary_presence_full_grid_area_summary.csv"
    thresholded_model_comparison = (
        tmp_path / "reports/tables/binary_presence_thresholded_model_comparison.csv"
    )
    prediction_manifest = tmp_path / "interim/binary_presence_prediction_manifest.json"
    precision_recall_figure = tmp_path / "reports/figures/binary_presence_precision_recall.png"
    map_figure = tmp_path / "reports/figures/binary_presence_2022_map.png"
    calibration_model = (
        tmp_path / "models/binary_presence/logistic_annual_max_ge_10pct_calibration.joblib"
    )
    calibrated_sample_predictions = (
        tmp_path / "processed/binary_presence_calibrated_sample_predictions.parquet"
    )
    calibration_metrics = tmp_path / "reports/tables/binary_presence_calibration_metrics.csv"
    calibrated_threshold_selection = (
        tmp_path / "reports/tables/binary_presence_calibrated_threshold_selection.csv"
    )
    calibrated_full_grid_area_summary = (
        tmp_path / "reports/tables/binary_presence_calibrated_full_grid_area_summary.csv"
    )
    calibration_curve_figure = tmp_path / "reports/figures/binary_presence_calibration_curve.png"
    calibrated_threshold_figure = (
        tmp_path / "reports/figures/binary_presence_calibrated_thresholds.png"
    )
    calibration_manifest = tmp_path / "interim/binary_presence_calibration_manifest.json"
    sidecar_model = (
        tmp_path / "models/binary_presence/logistic_annual_max_ge_10pct.crm_stratified.joblib"
    )
    sidecar_sample_predictions = (
        tmp_path / "processed/binary_presence_sample_predictions.crm_stratified.parquet"
    )
    sidecar_full_grid_predictions = (
        tmp_path / "processed/binary_presence_full_grid_predictions.crm_stratified.parquet"
    )
    sidecar_metrics = tmp_path / "reports/tables/binary_presence_metrics.crm_stratified.csv"
    sidecar_threshold_selection = (
        tmp_path / "reports/tables/binary_presence_threshold_selection.crm_stratified.csv"
    )
    sidecar_full_grid_area_summary = (
        tmp_path / "reports/tables/binary_presence_full_grid_area_summary.crm_stratified.csv"
    )
    sidecar_thresholded_model_comparison = (
        tmp_path / "reports/tables/binary_presence_thresholded_model_comparison.crm_stratified.csv"
    )
    sidecar_prediction_manifest = (
        tmp_path / "interim/binary_presence_prediction_manifest.crm_stratified.json"
    )
    sidecar_precision_recall_figure = (
        tmp_path / "reports/figures/binary_presence_precision_recall.crm_stratified.png"
    )
    sidecar_map_figure = tmp_path / "reports/figures/binary_presence_2022_map.crm_stratified.png"
    sidecar_comparison = tmp_path / "reports/tables/binary_presence_crm_stratified_comparison.csv"
    config_path = tmp_path / "config.yaml"

    write_binary_rows(sample, years=(2018, 2019, 2020, 2021, 2022), include_mask=True)
    if include_sidecar:
        write_binary_rows(
            sidecar_sample,
            years=(2018, 2019, 2020, 2021, 2022),
            include_mask=True,
        )
    write_binary_rows(full_grid, years=(2018, 2019, 2020, 2021, 2022), include_mask=False)
    write_thresholded_baseline_predictions(
        baseline_predictions,
        years=(2018, 2019, 2020, 2021, 2022),
    )
    write_binary_split_manifest(split_manifest)
    write_binary_domain_mask(domain_mask, domain_manifest)
    sidecar_config = (
        f"""
    sidecars:
      crm_stratified:
        input_table: {sidecar_sample}
        model: {sidecar_model}
        sample_predictions: {sidecar_sample_predictions}
        full_grid_predictions: {sidecar_full_grid_predictions}
        metrics: {sidecar_metrics}
        threshold_selection: {sidecar_threshold_selection}
        full_grid_area_summary: {sidecar_full_grid_area_summary}
        thresholded_model_comparison: {sidecar_thresholded_model_comparison}
        prediction_manifest: {sidecar_prediction_manifest}
        precision_recall_figure: {sidecar_precision_recall_figure}
        map_figure: {sidecar_map_figure}
        comparison_table: {sidecar_comparison}
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
    input_table: {sample}
    inference_table: {full_grid}
    target_label: annual_max_ge_10pct
    target_column: kelp_fraction_y
    target_threshold_fraction: 0.10
    target_threshold_area: 90.0
    features: A00-A01
    class_weight: balanced
    c_grid: [1.0]
    max_iter: 500
    model: {model}
    sample_predictions: {sample_predictions}
    full_grid_predictions: {full_grid_predictions}
    metrics: {metrics}
    threshold_selection: {threshold_selection}
    full_grid_area_summary: {full_grid_area_summary}
    thresholded_model_comparison: {thresholded_model_comparison}
    prediction_manifest: {prediction_manifest}
    precision_recall_figure: {precision_recall_figure}
    map_figure: {map_figure}
    calibration:
      method: platt
      calibration_split: validation
      calibration_year: 2021
      evaluation_split: test
      evaluation_year: 2022
      input_sample_predictions: {sample_predictions}
      input_full_grid_predictions: {full_grid_predictions}
      model: {calibration_model}
      calibrated_sample_predictions: {calibrated_sample_predictions}
      metrics: {calibration_metrics}
      threshold_selection: {calibrated_threshold_selection}
      full_grid_area_summary: {calibrated_full_grid_area_summary}
      calibration_curve_figure: {calibration_curve_figure}
      threshold_figure: {calibrated_threshold_figure}
      manifest: {calibration_manifest}
      include_prevalence_match: true
      reliability_bin_count: 5
{sidecar_config}
reports:
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "model": model,
        "sample_predictions": sample_predictions,
        "full_grid_predictions": full_grid_predictions,
        "metrics": metrics,
        "threshold_selection": threshold_selection,
        "full_grid_area_summary": full_grid_area_summary,
        "thresholded_model_comparison": thresholded_model_comparison,
        "prediction_manifest": prediction_manifest,
        "precision_recall_figure": precision_recall_figure,
        "map_figure": map_figure,
        "calibration_model": calibration_model,
        "calibrated_sample_predictions": calibrated_sample_predictions,
        "calibration_metrics": calibration_metrics,
        "calibrated_threshold_selection": calibrated_threshold_selection,
        "calibrated_full_grid_area_summary": calibrated_full_grid_area_summary,
        "calibration_curve_figure": calibration_curve_figure,
        "calibrated_threshold_figure": calibrated_threshold_figure,
        "calibration_manifest": calibration_manifest,
        "sidecar_model": sidecar_model,
        "sidecar_sample_predictions": sidecar_sample_predictions,
        "sidecar_full_grid_predictions": sidecar_full_grid_predictions,
        "sidecar_metrics": sidecar_metrics,
        "sidecar_threshold_selection": sidecar_threshold_selection,
        "sidecar_full_grid_area_summary": sidecar_full_grid_area_summary,
        "sidecar_thresholded_model_comparison": sidecar_thresholded_model_comparison,
        "sidecar_prediction_manifest": sidecar_prediction_manifest,
        "sidecar_precision_recall_figure": sidecar_precision_recall_figure,
        "sidecar_map_figure": sidecar_map_figure,
        "sidecar_comparison": sidecar_comparison,
    }


def write_binary_rows(path: Path, *, years: tuple[int, ...], include_mask: bool) -> None:
    """Write tiny feature/label rows for binary model tests."""
    rows: list[dict[str, object]] = []
    for year in years:
        for cell_id, fraction in enumerate((0.0, 0.04, 0.35, 0.02)):
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": cell_id if cell_id in {1, 2} else np.nan,
                    "longitude": -122.0 + cell_id * 0.001,
                    "latitude": 36.0 + cell_id * 0.001,
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "A00": fraction + (year - 2018) * 0.001,
                    "A01": fraction * 2.0,
                    "aef_grid_cell_id": cell_id,
                    "aef_grid_row": cell_id,
                    "aef_grid_col": cell_id,
                    "label_source": "kelpwatch_station"
                    if cell_id in {1, 2}
                    else "assumed_background",
                    "is_kelpwatch_observed": cell_id in {1, 2},
                    "kelpwatch_station_count": 1 if cell_id in {1, 2} else 0,
                    "sample_weight": 1.0,
                }
            )
            if include_mask:
                rows[-1].update(
                    {
                        "is_plausible_kelp_domain": True,
                        "domain_mask_reason": "retained_shallow_depth",
                        "domain_mask_detail": "fixture",
                        "domain_mask_version": "test_mask_v1",
                        "depth_bin": "shallow_depth",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_thresholded_baseline_predictions(path: Path, *, years: tuple[int, ...]) -> None:
    """Write tiny continuous-baseline predictions for thresholded comparison tests."""
    split_by_year = {
        2018: "train",
        2019: "train",
        2020: "train",
        2021: "validation",
        2022: "test",
    }
    rows: list[dict[str, object]] = []
    for year in years:
        for cell_id, fraction in enumerate((0.0, 0.04, 0.35, 0.02)):
            rows.append(
                {
                    "model_name": "ridge_regression",
                    "split": split_by_year[year],
                    "year": year,
                    "label_source": "kelpwatch_station"
                    if cell_id in {1, 2}
                    else "assumed_background",
                    "is_kelpwatch_observed": cell_id in {1, 2},
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "pred_kelp_fraction_y_clipped": fraction + (0.03 if cell_id == 1 else 0.0),
                    "is_plausible_kelp_domain": True,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_binary_split_manifest(path: Path) -> None:
    """Write split labels matching the synthetic binary rows."""
    split_by_year = {
        2018: "train",
        2019: "train",
        2020: "train",
        2021: "validation",
        2022: "test",
    }
    rows = []
    for year, split in split_by_year.items():
        for cell_id in range(4):
            rows.append(
                {
                    "year": year,
                    "aef_grid_cell_id": cell_id,
                    "split": split,
                    "used_for_training_eval": True,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_binary_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a domain mask retaining three of four fixture cells."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [0, 1, 2, 3],
            "is_plausible_kelp_domain": [True, True, True, False],
            "domain_mask_reason": [
                "retained_shallow_depth",
                "retained_shallow_depth",
                "retained_shallow_depth",
                "dropped_too_deep",
            ],
            "domain_mask_detail": ["fixture"] * 4,
            "domain_mask_version": ["test_mask_v1"] * 4,
            "depth_bin": ["shallow_depth", "shallow_depth", "shallow_depth", "deep_water"],
        }
    ).to_parquet(mask_path, index=False)
    manifest_path.write_text(json.dumps({"mask_version": "test_mask_v1"}))


def binary_presence_config(tmp_path: Path) -> BinaryPresenceConfig:
    """Build a minimal in-memory binary-presence config for unit tests."""
    path = tmp_path / "placeholder"
    return BinaryPresenceConfig(
        config_path=path / "config.yaml",
        sample_policy="current_masked_sample",
        input_table_path=path / "sample.parquet",
        split_manifest_path=path / "split.parquet",
        inference_table_path=path / "full.parquet",
        model_output_path=path / "model.joblib",
        sample_predictions_path=path / "sample_predictions.parquet",
        full_grid_predictions_path=path / "full_predictions.parquet",
        metrics_path=path / "metrics.csv",
        threshold_selection_path=path / "thresholds.csv",
        full_grid_area_summary_path=path / "area.csv",
        thresholded_model_comparison_path=path / "comparison.csv",
        prediction_manifest_path=path / "manifest.json",
        precision_recall_figure_path=path / "figure.png",
        map_figure_path=path / "map.png",
        baseline_sample_predictions_path=None,
        target_label="annual_max_ge_10pct",
        target_column="kelp_fraction_y",
        target_threshold_fraction=0.10,
        target_threshold_area=90.0,
        feature_columns=("A00", "A01"),
        train_years=(2018, 2019, 2020),
        validation_years=(2021,),
        test_years=(2022,),
        class_weight="balanced",
        c_grid=(1.0,),
        max_iter=500,
        drop_missing_features=True,
        allow_missing_split_manifest_rows=False,
        reporting_domain_mask=None,
    )
