import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main


def test_analyze_model_writes_report_artifacts(tmp_path: Path) -> None:
    """Verify analyze-model writes reports, tables, figures, and manifest."""
    fixture = write_model_analysis_fixture(tmp_path)

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    for key in (
        "report",
        "html_report",
        "pdf_report",
        "manifest",
        "stage_distribution",
        "target_framing",
        "residual_by_bin",
        "residual_by_persistence",
        "prediction_distribution",
        "threshold_sensitivity",
        "spatial_readiness",
        "feature_separability",
        "phase1_decision",
        "phase1_model_comparison",
        "reference_area_calibration",
        "data_health",
        "quarter_mapping",
        "label_distribution_figure",
        "observed_predicted_figure",
        "residual_by_bin_figure",
        "observed_900_figure",
        "residual_by_persistence_figure",
        "alternative_targets_figure",
        "feature_projection_figure",
        "spatial_readiness_figure",
    ):
        assert fixture[key].is_file()

    stage_distribution = pd.read_csv(fixture["stage_distribution"])
    retained_2022 = stage_distribution.query(
        "stage == 'retained_model_rows' and split == 'test' and year == 2022"
    ).iloc[0]
    assert int(retained_2022["zero_count"]) == 1
    assert int(retained_2022["saturated_count"]) == 1

    thresholds = pd.read_csv(fixture["threshold_sensitivity"])
    assert set(thresholds["threshold_fraction"]) >= {0.0, 0.1, 0.9}

    residual_bins = pd.read_csv(fixture["residual_by_bin"])
    test_bins = residual_bins.query("split == 'test' and year == 2022")["observed_bin"].tolist()
    assert test_bins == ["000_zero", "(90, 450]", "(810, 900]"]

    decisions = pd.read_csv(fixture["phase1_decision"])
    assert "Derived-label Phase 1" in set(decisions["branch"])
    assert "Baseline-hardening Phase 1" in set(decisions["branch"])

    model_comparison = pd.read_csv(fixture["phase1_model_comparison"])
    assert {"no_skill_train_mean", "ridge_regression"} <= set(model_comparison["model_name"])
    assert "full_grid_prediction" in set(model_comparison["evaluation_scope"])
    assert set(model_comparison["mask_status"]) == {"unmasked"}

    data_health = pd.read_csv(fixture["data_health"])
    missing_feature_rows = data_health.query(
        "check_name == 'missing_feature_drop_rate' and split == 'validation' and year == 2021"
    )
    assert int(missing_feature_rows.iloc[0]["row_count"]) == 1

    quarter_mapping = pd.read_csv(fixture["quarter_mapping"])
    assert set(quarter_mapping["derived_quarter"]) == {1, 2, 3, 4}

    manifest = json.loads(fixture["manifest"].read_text())
    report = fixture["report"].read_text()
    html_report = fixture["html_report"].read_text()
    assert manifest["command"] == "analyze-model"
    assert manifest["row_counts"]["model_predictions"] == 6
    assert manifest["outputs"]["html_report"] == str(fixture["html_report"])
    assert manifest["outputs"]["pdf_report"] == str(fixture["pdf_report"])
    assert fixture["pdf_report"].stat().st_size > 0
    assert "Phase 1 Harness Status" in report
    assert "Model Comparison" in report
    assert "Reference Baseline Ranking" in report
    assert "Phase 1 Coverage Gaps" in report
    assert "Phase 0 Decision Evidence" not in report
    assert "failed ridge objective" in report
    assert "sampling/objective calibration problem" in report
    assert "previous-year persistence" in report
    assert "Observed, Predicted, And Error Map" in report
    assert "![Observed, predicted, and residual map]" in report
    assert "ridge_2022_residual_interactive.html" in report
    assert "Alternative Target-Framing Findings" not in report
    assert "<h1>Monterey Phase 1 Model Analysis</h1>" in html_report
    assert "<h2>Model Comparison</h2>" in html_report
    assert 'src="data:image/png;base64,' in html_report


def write_model_analysis_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write synthetic model-analysis inputs and configured output paths."""
    labels = tmp_path / "interim/labels_annual.parquet"
    label_manifest = tmp_path / "interim/labels_annual_manifest.json"
    aligned = tmp_path / "interim/aligned_training_table.parquet"
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    paths = output_paths(tmp_path)
    write_labels(labels)
    write_aligned(aligned)
    write_split_manifest(split_manifest)
    write_predictions(predictions)
    write_metrics(metrics)
    label_manifest.parent.mkdir(parents=True, exist_ok=True)
    label_manifest.write_text(json.dumps({"spatial": {"crs": "EPSG:4326"}}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        config_text(
            tmp_path, labels, label_manifest, aligned, split_manifest, predictions, metrics, paths
        )
    )
    return {"config_path": config_path, **paths}


def output_paths(tmp_path: Path) -> dict[str, Path]:
    """Return all expected output paths for the synthetic config."""
    return {
        "report": tmp_path / "reports/model_analysis/monterey_phase1_model_analysis.md",
        "html_report": tmp_path / "reports/model_analysis/monterey_phase1_model_analysis.html",
        "pdf_report": tmp_path / "reports/model_analysis/monterey_phase1_model_analysis.pdf",
        "manifest": tmp_path / "interim/model_analysis_manifest.json",
        "stage_distribution": tmp_path
        / "reports/tables/model_analysis_label_distribution_by_stage.csv",
        "target_framing": tmp_path / "reports/tables/model_analysis_target_framing_summary.csv",
        "residual_by_bin": tmp_path / "reports/tables/model_analysis_residual_by_observed_bin.csv",
        "residual_by_persistence": tmp_path
        / "reports/tables/model_analysis_residual_by_persistence.csv",
        "prediction_distribution": tmp_path
        / "reports/tables/model_analysis_prediction_distribution.csv",
        "threshold_sensitivity": tmp_path
        / "reports/tables/model_analysis_threshold_sensitivity.csv",
        "spatial_readiness": tmp_path
        / "reports/tables/model_analysis_spatial_holdout_readiness.csv",
        "feature_separability": tmp_path / "reports/tables/model_analysis_feature_separability.csv",
        "phase1_decision": tmp_path / "reports/tables/model_analysis_phase1_decision_matrix.csv",
        "phase1_model_comparison": tmp_path
        / "reports/tables/model_analysis_phase1_model_comparison.csv",
        "reference_area_calibration": tmp_path
        / "reports/tables/reference_baseline_area_calibration.csv",
        "data_health": tmp_path / "reports/tables/model_analysis_data_health.csv",
        "quarter_mapping": tmp_path / "reports/tables/model_analysis_quarter_mapping.csv",
        "label_distribution_figure": tmp_path
        / "reports/figures/model_analysis_label_distribution_by_stage.png",
        "observed_predicted_figure": tmp_path
        / "reports/figures/model_analysis_observed_vs_predicted_distribution.png",
        "residual_by_bin_figure": tmp_path
        / "reports/figures/model_analysis_residual_by_observed_bin.png",
        "observed_900_figure": tmp_path
        / "reports/figures/model_analysis_observed_900_predictions.png",
        "residual_by_persistence_figure": tmp_path
        / "reports/figures/model_analysis_residual_by_persistence.png",
        "alternative_targets_figure": tmp_path
        / "reports/figures/model_analysis_alternative_target_framings.png",
        "feature_projection_figure": tmp_path
        / "reports/figures/model_analysis_feature_projection.png",
        "spatial_readiness_figure": tmp_path
        / "reports/figures/model_analysis_spatial_holdout_readiness.png",
    }


def config_text(
    tmp_path: Path,
    labels: Path,
    label_manifest: Path,
    aligned: Path,
    split_manifest: Path,
    predictions: Path,
    metrics: Path,
    paths: dict[str, Path],
) -> str:
    """Build a minimal workflow config for model analysis."""
    return f"""
data_root: {tmp_path}
labels:
  paths:
    annual_labels: {labels}
    annual_label_manifest: {label_manifest}
features:
  bands: A00-A01
alignment:
  output_table: {aligned}
splits:
  output_manifest: {split_manifest}
models:
  baselines:
    features: A00-A01
    predictions: {predictions}
    metrics: {metrics}
reports:
  figures_dir: {tmp_path / "reports/figures"}
  tables_dir: {tmp_path / "reports/tables"}
  model_analysis:
    model_name: ridge_regression
    split: test
    year: 2022
    observed_area_bins: [0, 90, 450, 810, 900]
    threshold_fractions: [0, 0.1, 0.9]
    latitude_band_count: 2
    max_projection_rows: 10
    fall_quarter: 4
    winter_quarter: 1
  outputs:
    model_analysis_report: {paths["report"]}
    model_analysis_html_report: {paths["html_report"]}
    model_analysis_pdf_report: {paths["pdf_report"]}
    model_analysis_manifest: {paths["manifest"]}
    model_analysis_label_distribution_by_stage: {paths["stage_distribution"]}
    model_analysis_target_framing_summary: {paths["target_framing"]}
    model_analysis_residual_by_observed_bin: {paths["residual_by_bin"]}
    model_analysis_residual_by_persistence: {paths["residual_by_persistence"]}
    model_analysis_prediction_distribution: {paths["prediction_distribution"]}
    model_analysis_threshold_sensitivity: {paths["threshold_sensitivity"]}
    model_analysis_spatial_holdout_readiness: {paths["spatial_readiness"]}
    model_analysis_feature_separability: {paths["feature_separability"]}
    model_analysis_phase1_decision_matrix: {paths["phase1_decision"]}
    model_analysis_phase1_model_comparison: {paths["phase1_model_comparison"]}
    reference_baseline_area_calibration: {paths["reference_area_calibration"]}
    model_analysis_data_health: {paths["data_health"]}
    model_analysis_quarter_mapping: {paths["quarter_mapping"]}
    model_analysis_label_distribution_figure: {paths["label_distribution_figure"]}
    model_analysis_observed_predicted_figure: {paths["observed_predicted_figure"]}
    model_analysis_residual_by_bin_figure: {paths["residual_by_bin_figure"]}
    model_analysis_observed_900_figure: {paths["observed_900_figure"]}
    model_analysis_residual_by_persistence_figure: {paths["residual_by_persistence_figure"]}
    model_analysis_alternative_targets_figure: {paths["alternative_targets_figure"]}
    model_analysis_feature_projection_figure: {paths["feature_projection_figure"]}
    model_analysis_spatial_readiness_figure: {paths["spatial_readiness_figure"]}
""".lstrip()


def write_labels(path: Path) -> None:
    """Write annual labels with zero, positive, and saturated cases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        label_row(2021, 1, 0.0, [0.0, 0.0, 0.0, 0.0]),
        label_row(2021, 2, 450.0, [0.0, 450.0, 450.0, 0.0]),
        label_row(2021, 3, 900.0, [900.0, 900.0, 900.0, 900.0]),
        label_row(2022, 1, 0.0, [0.0, 0.0, 0.0, 0.0]),
        label_row(2022, 2, 450.0, [0.0, 450.0, 450.0, 0.0]),
        label_row(2022, 3, 900.0, [900.0, 900.0, 900.0, 900.0]),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def label_row(year: int, station_id: int, area: float, quarters: list[float]) -> dict[str, object]:
    """Build one annual-label row."""
    return {
        "year": year,
        "kelpwatch_station_id": station_id,
        "longitude": -122.0 + station_id * 0.001,
        "latitude": 36.0 + station_id * 0.01,
        "kelp_max_y": area,
        "kelp_fraction_y": area / 900.0,
        "area_q1": quarters[0],
        "area_q2": quarters[1],
        "area_q3": quarters[2],
        "area_q4": quarters[3],
        "valid_quarter_count": 4,
        "nonzero_quarter_count": sum(value > 0 for value in quarters),
    }


def write_aligned(path: Path) -> None:
    """Write aligned labels with two simple AEF feature columns."""
    labels = pd.DataFrame(
        [
            {**label_row(2021, 1, 0.0, [0.0, 0.0, 0.0, 0.0]), "A00": 0.0, "A01": 0.0},
            {**label_row(2021, 2, 450.0, [0.0, 450.0, 450.0, 0.0]), "A00": 0.5, "A01": 0.2},
            {**label_row(2021, 3, 900.0, [900.0, 900.0, 900.0, 900.0]), "A00": 1.0, "A01": 0.9},
            {**label_row(2022, 1, 0.0, [0.0, 0.0, 0.0, 0.0]), "A00": 0.0, "A01": 0.0},
            {**label_row(2022, 2, 450.0, [0.0, 450.0, 450.0, 0.0]), "A00": 0.5, "A01": 0.2},
            {**label_row(2022, 3, 900.0, [900.0, 900.0, 900.0, 900.0]), "A00": 1.0, "A01": 0.9},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)


def write_split_manifest(path: Path) -> None:
    """Write a split manifest with one dropped missing-feature row."""
    rows = []
    for year, split in [(2021, "validation"), (2022, "test")]:
        for station_id, fraction in [(1, 0.0), (2, 0.5), (3, 1.0)]:
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": station_id,
                    "longitude": -122.0 + station_id * 0.001,
                    "latitude": 36.0 + station_id * 0.01,
                    "kelp_fraction_y": fraction,
                    "split": split,
                    "has_complete_features": not (year == 2021 and station_id == 2),
                    "has_target": True,
                    "used_for_training_eval": not (year == 2021 and station_id == 2),
                    "drop_reason": "missing_features" if year == 2021 and station_id == 2 else "",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_predictions(path: Path) -> None:
    """Write ridge prediction rows with saturated underprediction."""
    rows = []
    for year, split in [(2021, "validation"), (2022, "test")]:
        for station_id, observed, predicted in [(1, 0.0, 0.05), (2, 0.5, 0.4), (3, 1.0, 0.55)]:
            observed_area = observed * 900.0
            predicted_area = predicted * 900.0
            rows.append(
                {
                    "year": year,
                    "split": split,
                    "kelpwatch_station_id": station_id,
                    "longitude": -122.0 + station_id * 0.001,
                    "latitude": 36.0 + station_id * 0.01,
                    "kelp_fraction_y": observed,
                    "kelp_max_y": observed_area,
                    "model_name": "ridge_regression",
                    "pred_kelp_fraction_y_clipped": predicted,
                    "pred_kelp_max_y": predicted_area,
                    "residual_kelp_max_y": observed_area - predicted_area,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_metrics(path: Path) -> None:
    """Write minimal ridge metrics for report text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "no_skill_train_mean",
                "split": "test",
                "mae": 0.3,
                "rmse": 0.4,
                "r2": 0.0,
                "spearman": float("nan"),
                "area_pct_bias": -0.11,
                "f1_ge_10pct": 0.4,
            },
            {
                "model_name": "ridge_regression",
                "split": "test",
                "mae": 0.1,
                "rmse": 0.2,
                "r2": 0.5,
                "spearman": 0.7,
                "area_pct_bias": -0.1,
                "f1_ge_10pct": 0.8,
            },
        ]
    ).to_csv(path, index=False)
