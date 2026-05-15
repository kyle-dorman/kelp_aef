import base64
import json
from pathlib import Path
from types import SimpleNamespace

import joblib  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import pytest

from kelp_aef import main
from kelp_aef.evaluation.component_failure import add_component_spatial_context
from kelp_aef.evaluation.model_analysis import (
    BalanceSource,
    binary_threshold_definitions,
    build_binary_threshold_prevalence,
    build_binary_threshold_recommendation,
    build_class_balance_by_split,
    conditional_likely_positive_sampling_rows,
    pooled_context_value_label,
    pooled_context_value_sort_key,
    sampling_policy_comparison_markdown,
)


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
        "all_model_sampling_policy_comparison",
        "reference_area_calibration",
        "data_health",
        "class_balance_by_split",
        "target_balance_by_label_source",
        "background_rate_summary",
        "binary_threshold_prevalence",
        "binary_threshold_comparison",
        "binary_threshold_recommendation",
        "residual_domain_context",
        "residual_by_mask_reason",
        "residual_by_depth_bin",
        "top_residual_context",
        "quarter_mapping",
        "label_distribution_figure",
        "observed_predicted_figure",
        "pixel_skill_area_calibration_figure",
        "residual_by_bin_figure",
        "observed_900_figure",
        "residual_by_persistence_figure",
        "alternative_targets_figure",
        "feature_projection_figure",
        "spatial_readiness_figure",
        "class_balance_figure",
        "binary_threshold_comparison_figure",
        "residual_domain_context_figure",
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

    all_model_comparison = pd.read_csv(fixture["all_model_sampling_policy_comparison"])
    assert "sample_policy" in all_model_comparison.columns
    assert {"crm_stratified_mask_first_sample"} == set(all_model_comparison["sample_policy"])
    assert {"continuous_baseline", "binary_presence"} <= set(all_model_comparison["model_family"])

    data_health = pd.read_csv(fixture["data_health"])
    missing_feature_rows = data_health.query(
        "check_name == 'missing_feature_drop_rate' and split == 'validation' and year == 2021"
    )
    assert int(missing_feature_rows.iloc[0]["row_count"]) == 1

    class_balance = pd.read_csv(fixture["class_balance_by_split"])
    required_balance_columns = {
        "data_scope",
        "mask_status",
        "evaluation_scope",
        "split",
        "year",
        "label_source",
        "positive_rate",
        "high_canopy_rate",
        "saturated_rate",
        "assumed_background_rate",
    }
    assert required_balance_columns <= set(class_balance.columns)
    test_balance = class_balance.query(
        "data_scope == 'full_grid_prediction' "
        "and split == 'test' "
        "and year == 2022 "
        "and label_source == 'all'"
    ).iloc[0]
    assert float(test_balance["positive_rate"]) == 2 / 3
    assert float(test_balance["high_canopy_rate"]) == 2 / 3

    threshold_comparison = pd.read_csv(fixture["binary_threshold_comparison"])
    assert {"annual_max_gt0", "annual_max_ge_1pct", "annual_max_ge_5pct"} <= set(
        threshold_comparison["threshold_label"]
    )
    validation_candidates = threshold_comparison.query(
        "split == 'validation' and year == 2021 and label_source == 'all'"
    )
    assert set(validation_candidates["threshold_fraction"]) >= {0.0, 0.01, 0.05, 0.10}
    recommendation = pd.read_csv(fixture["binary_threshold_recommendation"])
    assert set(recommendation["selection_split"]) == {"validation"}
    assert set(recommendation["selection_year"]) == {2021}
    assert bool(recommendation["selected_candidate"].any())

    quarter_mapping = pd.read_csv(fixture["quarter_mapping"])
    assert set(quarter_mapping["derived_quarter"]) == {1, 2, 3, 4}

    manifest = json.loads(fixture["manifest"].read_text())
    report = fixture["report"].read_text()
    html_report = fixture["html_report"].read_text()
    assert manifest["command"] == "analyze-model"
    assert manifest["row_counts"]["model_predictions"] == 6
    assert manifest["outputs"]["html_report"] == str(fixture["html_report"])
    assert manifest["outputs"]["pdf_report"] == str(fixture["pdf_report"])
    assert manifest["outputs"]["pixel_skill_area_calibration_figure"] == str(
        fixture["pixel_skill_area_calibration_figure"]
    )
    assert fixture["pdf_report"].stat().st_size > 0
    assert "Current Default Policy And Data Scope" in report
    assert "2022 Retained-Domain Model Scoreboard" in report
    assert "What Improved Since Ridge" in report
    assert "Remaining Failure Modes" in report
    assert "Phase 1 Closeout Decision" in report
    assert "Phase 1 Harness Status" not in report
    assert "Reference Baseline Ranking" not in report
    assert "Phase 1 Coverage Gaps" not in report
    assert "Phase 0 Decision Evidence" not in report
    assert "previous-year persistence" in report
    assert "Observed, Predicted, And Error Map" in report
    assert "![Pixel skill and area calibration]" in report
    assert "mask-aware residual rows" in report
    assert "Annual-max class balance" in report
    assert "Threshold, Calibration, And Amount Diagnostics" in report
    assert "Default sampling policy: `crm_stratified_mask_first_sample`" in report
    assert "Selected Phase 1 AEF policy" in report
    assert "Next modeling task" not in report
    assert "next steps" not in report.lower()
    assert "docs/phase1_crm_stratified_sampling_policy_decision.md" in report
    assert "CRM-Stratified Sampling Policy Comparison" not in report
    assert "**Continuous canopy and area**" not in report
    assert "**Binary presence and support**" not in report
    assert "Thresholded baseline comparison" in report
    assert "Binary presence 2022 map" in report
    assert "Platt scaling" in report
    assert "river-mouth false-positive cluster" in report
    assert "class-weighted logistic regression" in report
    assert "validation` rows from `2021`" in report
    assert "![Observed, predicted, and residual map]" in report
    assert "ridge_2022_residual_interactive.html" in report
    assert "Alternative Target-Framing Findings" not in report
    assert "<h1>Monterey Phase 1 Model Analysis</h1>" in html_report
    assert "<h2>2022 Retained-Domain Model Scoreboard</h2>" in html_report
    assert "<h2>Phase 1 Closeout Decision</h2>" in html_report
    assert 'src="data:image/png;base64,' in html_report


def test_analyze_model_writes_phase2_report_sections(tmp_path: Path) -> None:
    """Verify Phase 2 report mode adds training-regime and binary-support sections."""
    fixture = write_model_analysis_fixture(
        tmp_path,
        include_domain_mask=True,
        include_training_mask_columns=True,
        include_reference_area_calibration=True,
    )
    phase2_primary = tmp_path / "reports/tables/phase2_training_regime_primary.csv"
    phase2_model_comparison = tmp_path / "reports/tables/phase2_training_regime_all.csv"
    phase2_manifest = tmp_path / "interim/phase2_training_regime_manifest.json"
    binary_summary = tmp_path / "reports/tables/phase2_binary_support.csv"
    binary_predictions = tmp_path / "processed/phase2_binary_predictions.parquet"
    calibration_model = tmp_path / "models/phase2_binary_calibration.joblib"
    write_phase2_training_regime_rows(phase2_primary)
    write_phase2_training_regime_rows(phase2_model_comparison)
    phase2_manifest.parent.mkdir(parents=True, exist_ok=True)
    phase2_manifest.write_text(json.dumps({"command": "compare-training-regimes"}))
    write_phase2_binary_predictions(binary_predictions)
    write_phase2_calibration_payload(calibration_model)
    append_phase2_config(
        fixture["config_path"],
        phase2_primary=phase2_primary,
        phase2_model_comparison=phase2_model_comparison,
        phase2_manifest=phase2_manifest,
        binary_summary=binary_summary,
        binary_predictions=binary_predictions,
        calibration_model=calibration_model,
    )
    append_binary_hex_map_config(
        fixture["config_path"],
        paths=fixture,
        binary_predictions=binary_predictions,
        calibration_model=calibration_model,
    )

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    report = fixture["report"].read_text()
    html_report = fixture["html_report"].read_text()
    binary_rows = pd.read_csv(binary_summary)
    hex_rows = pd.read_csv(fixture["binary_hex_table"])
    manifest = json.loads(fixture["manifest"].read_text())
    hex_manifest = json.loads(fixture["binary_hex_manifest"].read_text())
    assert "# Big Sur Phase 2 Model Analysis" in report
    assert "Phase 2 Training-Regime Gate" in report
    assert "Pooled Diagnostic Scope" in report
    assert "Compact Baseline Grounding" in report
    assert "![Compact baseline grounding]" in report
    assert "**Binary support**" in report
    assert "**Continuous amount**" in report
    assert "| Binary model | Pooled calibrated support |" in report
    assert "| Train mean |" not in report
    assert "| Geographic ridge |" not in report
    assert "| AEF ridge |" not in report
    assert "| Hurdle hard gate |" not in report
    assert "| Training regime | Rows | Observed positive |" not in report
    assert "Observed area (M m2)" not in report
    assert "Pooled Binary Support Failures" not in report
    assert "Pooled Amount And Hurdle Failures" in report
    assert "Attribution Diagnostics" not in report
    assert "Column Definitions" in report
    assert "Canopy Amount And Hurdle Calibration" not in report
    assert "Binary Support Transfer" not in report
    assert "Pooled binary presence 1 km hex map" not in report
    assert "Full Training-Regime Tables" not in report
    assert "Pooled Data Distribution" not in report
    assert "Kelpwatch-style annual maximum reproduction" in report
    assert "Phase 1 Closeout Decision" not in report
    assert report.index("Phase 2 Training-Regime Gate") < report.index(
        "Compact Baseline Grounding"
    )
    assert "<title>Big Sur Phase 2 Model Analysis</title>" in html_report
    assert set(binary_rows["evaluation_region"]) == {"big_sur", "monterey"}
    assert set(binary_rows["training_regime"]) == {
        "big_sur_only",
        "monterey_only",
        "pooled_monterey_big_sur",
    }
    assert float(binary_rows.iloc[0]["f1"]) == 0.8
    assert set(hex_rows["evaluation_region"]) == {"big_sur"}
    assert fixture["phase2_baseline_grounding_figure"].stat().st_size > 0
    assert fixture["binary_hex_figure"].stat().st_size > 0
    assert hex_manifest["hex_definition"]["flat_to_flat_diameter_m"] == 1000.0
    assert manifest["phase2"]["outputs"]["binary_support_primary_summary"] == str(binary_summary)
    assert manifest["phase2"]["outputs"]["phase2_baseline_grounding_figure"] == str(
        fixture["phase2_baseline_grounding_figure"]
    )
    assert manifest["phase2"]["outputs"]["pooled_binary_presence_hex_table"] == str(
        fixture["binary_hex_table"]
    )
    assert manifest["row_counts"]["phase2_binary_support"] == 6
    assert manifest["row_counts"]["pooled_binary_presence_hex"] == len(hex_rows)


def test_analyze_model_writes_component_failure_outputs(tmp_path: Path) -> None:
    """Verify Phase 2 component-failure diagnostics are report-visible."""
    fixture = write_model_analysis_fixture(
        tmp_path,
        include_domain_mask=True,
        include_training_mask_columns=True,
        include_reference_area_calibration=True,
    )
    phase2_primary = tmp_path / "reports/tables/phase2_training_regime_primary.csv"
    phase2_model_comparison = tmp_path / "reports/tables/phase2_training_regime_all.csv"
    phase2_manifest = tmp_path / "interim/phase2_training_regime_manifest.json"
    binary_summary = tmp_path / "reports/tables/phase2_binary_support.csv"
    binary_predictions = tmp_path / "processed/phase2_binary_predictions.parquet"
    calibration_model = tmp_path / "models/phase2_binary_calibration.joblib"
    component_hurdle = tmp_path / "processed/component_hurdle_predictions.parquet"
    write_phase2_training_regime_rows(phase2_primary)
    write_phase2_training_regime_rows(phase2_model_comparison)
    phase2_manifest.parent.mkdir(parents=True, exist_ok=True)
    phase2_manifest.write_text(json.dumps({"command": "compare-training-regimes"}))
    write_phase2_binary_predictions(binary_predictions)
    write_phase2_calibration_payload(calibration_model)
    write_component_failure_hurdle_predictions(component_hurdle)
    append_phase2_config(
        fixture["config_path"],
        phase2_primary=phase2_primary,
        phase2_model_comparison=phase2_model_comparison,
        phase2_manifest=phase2_manifest,
        binary_summary=binary_summary,
        binary_predictions=binary_predictions,
        calibration_model=calibration_model,
    )
    append_component_failure_config(
        fixture["config_path"],
        paths=fixture,
        hurdle_predictions=component_hurdle,
        label_path=tmp_path / "interim/labels_annual.parquet",
    )

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    report = fixture["report"].read_text()
    summary = pd.read_csv(fixture["component_failure_summary"])
    by_spatial = pd.read_csv(fixture["component_failure_by_spatial"])
    by_model = pd.read_csv(fixture["component_failure_by_model"])
    edge = pd.read_csv(fixture["component_failure_edge"])
    manifest = json.loads(fixture["manifest"].read_text())
    sidecar_manifest = json.loads(fixture["component_failure_manifest"].read_text())
    assert "Deep Component-Failure Analysis" not in report
    assert "Full Deep Component-Failure Analysis" not in report
    assert "Phase 2 component-failure summary" in report
    assert set(summary["context_id"]) == {"fixture_component_context"}
    assert int(summary.iloc[0]["support_miss_positive_count"]) == 1
    assert int(summary.iloc[0]["support_leakage_zero_count"]) == 1
    assert int(summary.iloc[0]["detected_observed_positive_count"]) == 1
    assert float(summary.iloc[0]["amount_underprediction_detected_positive_rate"]) == 1.0
    assert float(summary.iloc[0]["composition_shrinkage_rate"]) == 1.0
    assert "zero_adjacent_to_positive" in set(by_spatial["edge_class"])
    assert "support_miss_positive" in set(by_model["component_failure_class"])
    assert int(edge.iloc[0]["false_negative_count"]) == 1
    assert float(edge.iloc[0]["fn_positive_edge_rate"]) == 1.0
    assert float(edge.iloc[0]["fp_predicted_edge_rate"]) == 1.0
    assert float(edge.iloc[0]["fp_adjacent_observed_rate"]) == 1.0
    assert float(edge.iloc[0]["fp_near_observed_rate"]) == 0.0
    assert float(edge.iloc[0]["fp_far_from_observed_rate"]) == 0.0
    assert float(edge.iloc[0]["fp_adjacent_or_near_positive_rate"]) == 1.0
    assert manifest["row_counts"]["component_failure_summary"] == 1
    assert sidecar_manifest["definitions"]["binary_target"] == "annual_max_ge_10pct"


def test_analyze_model_writes_pooled_context_outputs(tmp_path: Path) -> None:
    """Verify Phase 2 pooled diagnostics align binary, ridge, and hurdle rows."""
    fixture = write_model_analysis_fixture(
        tmp_path,
        include_domain_mask=True,
        include_training_mask_columns=True,
        include_reference_area_calibration=True,
    )
    phase2_primary = tmp_path / "reports/tables/phase2_training_regime_primary.csv"
    phase2_model_comparison = tmp_path / "reports/tables/phase2_training_regime_all.csv"
    phase2_manifest = tmp_path / "interim/phase2_training_regime_manifest.json"
    binary_summary = tmp_path / "reports/tables/phase2_binary_support.csv"
    binary_predictions = tmp_path / "processed/phase2_binary_predictions.parquet"
    calibration_model = tmp_path / "models/phase2_binary_calibration.joblib"
    pooled_baseline = tmp_path / "processed/pooled_baseline_predictions.parquet"
    pooled_binary = tmp_path / "processed/pooled_binary_predictions.parquet"
    pooled_hurdle = tmp_path / "processed/pooled_hurdle_predictions.parquet"
    write_phase2_training_regime_rows(phase2_primary)
    write_phase2_training_regime_rows(phase2_model_comparison)
    phase2_manifest.parent.mkdir(parents=True, exist_ok=True)
    phase2_manifest.write_text(json.dumps({"command": "compare-training-regimes"}))
    write_phase2_binary_predictions(binary_predictions)
    write_phase2_calibration_payload(calibration_model)
    write_pooled_context_baseline_predictions(pooled_baseline)
    write_pooled_context_binary_predictions(pooled_binary)
    write_component_failure_hurdle_predictions(pooled_hurdle)
    append_phase2_config(
        fixture["config_path"],
        phase2_primary=phase2_primary,
        phase2_model_comparison=phase2_model_comparison,
        phase2_manifest=phase2_manifest,
        binary_summary=binary_summary,
        binary_predictions=binary_predictions,
        calibration_model=calibration_model,
    )
    append_pooled_context_config(
        fixture["config_path"],
        paths=fixture,
        baseline_predictions=pooled_baseline,
        binary_predictions=pooled_binary,
        calibration_model=calibration_model,
        hurdle_predictions=pooled_hurdle,
        label_path=tmp_path / "interim/labels_annual.parquet",
    )

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    report = fixture["report"].read_text()
    performance = pd.read_csv(fixture["pooled_context_performance"])
    binary = pd.read_csv(fixture["pooled_binary_context"])
    amount = pd.read_csv(fixture["pooled_amount_context"])
    distribution = pd.read_csv(fixture["pooled_prediction_distribution"])
    manifest = json.loads(fixture["manifest"].read_text())
    sidecar_manifest = json.loads(fixture["pooled_context_manifest"].read_text())
    assert "Pooled Context Diagnostics" in report
    assert "![Pooled context metric breakdown]" in report
    assert "binary panels show F1" in report
    assert "ridge panels show RMSE" in report
    assert "observed-positive rows in each bin" in report
    assert (
        "| Model | Role | F1 >=10% | Precision | Recall | "
        "Predicted positive | FP | FN |"
    ) in report
    assert "| Binary model | Pooled calibrated support |" in report
    assert "| Train mean |" not in report
    assert "| Geographic ridge |" not in report
    assert "| AEF ridge |" not in report
    assert "| Hurdle hard gate |" not in report
    assert "| Training regime | Rows | Observed positive |" not in report
    assert "| Model | Role | Predicted area (M m2) | RMSE | Area bias |" in report
    assert "Observed area (M m2)" not in report
    assert "Pooled Binary Support Failures" not in report
    assert "Attribution Diagnostics" not in report
    assert "Pooled Data Distribution" not in report
    assert "![Pooled prediction distribution]" not in report
    assert "expected-value hurdle panels show RMSE" not in report
    assert "Overall prediction-distribution summaries" not in report
    assert fixture["pooled_context_metric_breakdown_figure"].stat().st_size > 0
    assert fixture["phase2_baseline_grounding_figure"].stat().st_size > 0
    assert fixture["pooled_prediction_distribution_figure"].stat().st_size > 0
    assert {"binary", "ridge", "hurdle_expected_value"} <= set(performance["model_surface"])
    assert {
        "observed_annual_max_bin",
        "annual_mean_canopy_area_bin",
        "crm_depth_m_bin",
        "component_failure_class",
    } <= set(amount["context_type"])
    overall_hurdle = amount.query(
        "context_type == 'overall' and model_surface == 'hurdle_expected_value'"
    ).iloc[0]
    assert int(overall_hurdle["amount_rate_denominator_count"]) == 1
    assert float(overall_hurdle["amount_under_rate"]) == 1.0
    assert float(overall_hurdle["composition_shrink_rate"]) == 1.0
    overall_binary = binary.query("context_type == 'overall'").iloc[0]
    assert int(overall_binary["false_positive_count"]) == 1
    assert int(overall_binary["false_negative_count"]) == 1
    assert "observed_ge_450m2_prediction_p95" in distribution.columns
    assert manifest["phase2"]["outputs"]["pooled_context_model_performance"] == str(
        fixture["pooled_context_performance"]
    )
    assert manifest["phase2"]["outputs"]["pooled_context_metric_breakdown_figure"] == str(
        fixture["pooled_context_metric_breakdown_figure"]
    )
    assert (
        sidecar_manifest["definitions"]["amount_rate_denominator"]
        == "observed-positive rows where calibrated binary support is detected"
    )


def test_pooled_context_value_sort_key_uses_explicit_orders() -> None:
    """Verify pooled-context plot labels do not fall back to alphabetical order."""
    area_values = ["[225, 450)", "0", "900", "[90, 225)", "(0, 90)", "[810, 900)"]
    assert sorted(
        area_values,
        key=lambda value: pooled_context_value_sort_key("observed_annual_max_bin", value),
    ) == ["0", "(0, 90)", "[90, 225)", "[225, 450)", "[810, 900)", "900"]

    previous_values = ["new_ge_10pct", "stable_zero_or_background", "lost_ge_10pct"]
    assert sorted(
        previous_values,
        key=lambda value: pooled_context_value_sort_key("previous_year_class", value),
    ) == ["stable_zero_or_background", "lost_ge_10pct", "new_ge_10pct"]

    failure_values = [
        "composition_shrinkage",
        "near_correct",
        "amount_underprediction_detected_positive",
        "support_miss_positive",
    ]
    assert sorted(
        failure_values,
        key=lambda value: pooled_context_value_sort_key("component_failure_class", value),
    ) == [
        "near_correct",
        "support_miss_positive",
        "amount_underprediction_detected_positive",
        "composition_shrinkage",
    ]
    assert (
        pooled_context_value_label(
            "component_failure_class",
            "amount_underprediction_detected_positive",
        )
        == "amount under"
    )


def test_phase2_diagnostics_cache_preserves_report_rows(tmp_path: Path) -> None:
    """Verify cached Phase 2 diagnostics preserve uncached diagnostic rows."""
    fixture = write_phase2_diagnostics_cache_fixture(tmp_path)

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    component_summary = pd.read_csv(fixture["component_failure_summary"])
    pooled_performance = pd.read_csv(fixture["pooled_context_performance"])

    assert main(["build-phase2-diagnostics", "--config", str(fixture["config_path"])]) == 0

    assert (fixture["phase2_component_frame_cache"] / "fixture_component_context.parquet").exists()
    assert (
        fixture["phase2_pooled_context_frame_cache"] / "pooled_fixture_context.parquet"
    ).exists()
    cache_manifest = json.loads(fixture["phase2_diagnostics_cache_manifest"].read_text())
    assert cache_manifest["status"] == "rebuilt"
    assert cache_manifest["row_counts"]["component_failure_summary"] == 1
    assert cache_manifest["row_counts"]["pooled_context_model_performance"] > 0

    pd.testing.assert_frame_equal(
        component_summary,
        pd.read_csv(fixture["component_failure_summary"]),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pooled_performance,
        pd.read_csv(fixture["pooled_context_performance"]),
        check_dtype=False,
    )

    assert (
        main(
            [
                "analyze-model",
                "--config",
                str(fixture["config_path"]),
                "--reuse-phase2-diagnostics",
            ]
        )
        == 0
    )

    manifest = json.loads(fixture["manifest"].read_text())
    report = fixture["report"].read_text()
    assert manifest["phase2"]["diagnostics_cache"]["status"] == "reused"
    assert fixture["pooled_mean_max_binary_f1_figure"].stat().st_size > 0
    assert "annual max bins with annual mean canopy bins" in report
    assert "![Pooled annual mean versus max binary F1]" in report
    pd.testing.assert_frame_equal(
        component_summary,
        pd.read_csv(fixture["component_failure_summary"]),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pooled_performance,
        pd.read_csv(fixture["pooled_context_performance"]),
        check_dtype=False,
    )


def test_phase2_diagnostics_cache_rejects_stale_manifest(tmp_path: Path) -> None:
    """Verify report-iteration mode rejects stale Phase 2 diagnostics caches."""
    fixture = write_phase2_diagnostics_cache_fixture(tmp_path)

    assert main(["build-phase2-diagnostics", "--config", str(fixture["config_path"])]) == 0
    fixture["config_path"].write_text(
        fixture["config_path"].read_text() + "\n# invalidate diagnostics cache\n"
    )

    with pytest.raises(ValueError, match="Phase 2 diagnostics cache is not fresh"):
        main(
            [
                "analyze-model",
                "--config",
                str(fixture["config_path"]),
                "--reuse-phase2-diagnostics",
            ]
        )


def test_component_spatial_context_classifies_edges_and_distances() -> None:
    """Verify retained-grid neighborhoods classify interiors, edges, and exterior rings."""
    rows = []
    for row in range(9):
        for col in range(9):
            observed = 3 <= row <= 5 and 3 <= col <= 5
            rows.append(
                {
                    "aef_grid_row": row,
                    "aef_grid_col": col,
                    "observed_binary_positive": observed,
                    "predicted_binary_positive": observed,
                }
            )
    frame = pd.DataFrame(rows)

    spatial = add_component_spatial_context(frame, 30.0)

    center = spatial.query("aef_grid_row == 4 and aef_grid_col == 4").iloc[0]
    edge = spatial.query("aef_grid_row == 3 and aef_grid_col == 3").iloc[0]
    adjacent = spatial.query("aef_grid_row == 2 and aef_grid_col == 2").iloc[0]
    near = spatial.query("aef_grid_row == 1 and aef_grid_col == 1").iloc[0]
    far = spatial.query("aef_grid_row == 0 and aef_grid_col == 0").iloc[0]
    assert center["edge_class"] == "positive_interior"
    assert edge["edge_class"] == "positive_edge"
    assert adjacent["edge_class"] == "zero_adjacent_to_positive"
    assert near["edge_class"] == "near_positive_exterior"
    assert far["edge_class"] == "far_zero_exterior"
    assert center["observed_component_area_m2"] == 9 * 900.0
    assert far["distance_to_observed_positive_cells"] > near["distance_to_observed_positive_cells"]


def test_sampling_policy_audit_helper_filters_to_trained_model_tables(tmp_path: Path) -> None:
    """Verify the legacy sampling-policy audit helper filters to trained-model rows."""
    config = SimpleNamespace(
        all_model_sampling_policy_comparison_path=tmp_path / "all_models.csv",
        analysis_split="test",
        analysis_year=2022,
    )
    rows: list[dict[str, object]] = [
        {
            "sample_policy": "current_masked_sample",
            "model_family": "continuous_baseline",
            "model_name": "no_skill_train_mean",
            "artifact_kind": "full_grid_area_calibration",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "rmse": 0.8,
        },
        {
            "sample_policy": "current_masked_sample",
            "model_family": "continuous_baseline",
            "model_name": "ridge_regression",
            "artifact_kind": "full_grid_area_calibration",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "rmse": 0.04,
            "f1_ge_10pct": 0.3,
            "predicted_canopy_area": 12_000_000.0,
            "area_pct_bias": 0.2,
        },
        {
            "sample_policy": "current_masked_sample",
            "model_family": "continuous_baseline",
            "model_name": "ridge_regression",
            "artifact_kind": "full_grid_area_calibration",
            "split": "test",
            "year": 2022,
            "label_source": "assumed_background",
            "predicted_canopy_area": 8_000_000.0,
        },
        {
            "sample_policy": "crm_stratified_background_sample",
            "model_family": "continuous_baseline",
            "model_name": "ridge_regression",
            "artifact_kind": "full_grid_area_calibration",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "rmse": 0.03,
            "f1_ge_10pct": 0.5,
            "predicted_canopy_area": 10_000_000.0,
            "area_pct_bias": 0.1,
        },
        {
            "sample_policy": "current_masked_sample",
            "model_family": "binary_presence",
            "model_name": "logistic_annual_max_ge_10pct",
            "artifact_kind": "binary_calibrated_full_grid_metric",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "probability_source": "platt_calibrated",
            "threshold_policy": "validation_max_f1_calibrated",
            "probability_threshold": 0.4,
            "auprc": 0.65,
            "f1": 0.6,
            "precision": 0.5,
            "recall": 0.75,
        },
        {
            "sample_policy": "current_masked_sample",
            "model_family": "binary_presence",
            "model_name": "logistic_annual_max_ge_10pct",
            "artifact_kind": "binary_calibrated_sample_metric",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "probability_source": "platt_calibrated",
            "threshold_policy": "validation_max_f1_calibrated",
            "probability_threshold": 0.4,
            "auprc": 0.9,
            "f1": 0.8,
            "precision": 0.7,
            "recall": 0.9,
        },
        {
            "sample_policy": "current_masked_sample",
            "model_family": "binary_presence",
            "model_name": "logistic_annual_max_ge_10pct",
            "artifact_kind": "binary_calibrated_full_grid_area",
            "split": "test",
            "year": 2022,
            "label_source": "all",
            "probability_source": "platt_calibrated",
            "threshold_policy": "validation_max_f1_calibrated",
            "probability_threshold": 0.4,
            "predicted_positive_area_m2": 8_500_000.0,
            "assumed_background_predicted_positive_rate": 0.002,
        },
    ]

    markdown = sampling_policy_comparison_markdown(rows, config)

    assert "| Model | Policy | RMSE | F1 >=10% | Area (M m2) |" in markdown
    assert "| Policy | Threshold | AUPRC | F1 | Precision | Recall |" in markdown
    assert "Assumed-bg FP rate" in markdown
    assert "AEF ridge regression" in markdown
    assert "CRM-stratified background" in markdown
    assert "| Current masked sample | 0.40 | 0.650 | 0.600 | 0.500 | 0.750 |" in markdown
    assert "no_skill_train_mean" not in markdown
    assert "previous_year_annual_max" not in markdown
    assert "grid_cell_climatology" not in markdown


def test_conditional_likely_positive_sampling_rows_use_sidecar_policy(tmp_path: Path) -> None:
    """Verify conditional sidecar summaries keep their CRM sample-policy identity."""
    source_path = tmp_path / "conditional_canopy_full_grid_likely_positive_summary.crm.csv"

    rows = conditional_likely_positive_sampling_rows(
        [
            {
                "conditional_model_name": "ridge_positive_annual_max",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "mask_status": "plausible_kelp_domain",
                "evaluation_scope": "conditional_likely_positive_diagnostic",
                "probability_source": "platt_calibrated",
                "likely_positive_threshold_policy": "validation_max_f1_calibrated",
                "probability_threshold": 0.35,
                "row_count": 999519,
                "likely_positive_rate": 0.00813,
                "likely_positive_cell_area_m2": 7313400.0,
                "assumed_background_count": 969365,
                "assumed_background_likely_positive_rate": 0.000663,
            }
        ],
        sample_policy="crm_stratified_background_sample",
        source_path=source_path,
    )

    assert rows[0]["sample_policy"] == "crm_stratified_background_sample"
    assert rows[0]["model_family"] == "conditional_canopy"
    assert rows[0]["artifact_kind"] == "conditional_likely_positive_full_grid_summary"
    assert rows[0]["probability_threshold"] == 0.35
    assert rows[0]["predicted_positive_area_m2"] == 7313400.0
    assert rows[0]["source_path"] == str(source_path)


def test_analyze_model_treats_domain_mask_as_primary_full_grid_scope(
    tmp_path: Path,
) -> None:
    """Verify model-analysis labels masked full-grid rows as the primary scope."""
    fixture = write_model_analysis_fixture(
        tmp_path,
        include_domain_mask=True,
        include_training_mask_columns=True,
        include_reference_area_calibration=True,
    )

    assert main(["analyze-model", "--config", str(fixture["config_path"])]) == 0

    model_comparison = pd.read_csv(fixture["phase1_model_comparison"])
    sample_rows = model_comparison.query("evaluation_scope == 'background_inclusive_sample'")
    assert set(sample_rows["mask_status"]) == {"plausible_kelp_domain"}
    full_grid = model_comparison.query("evaluation_scope == 'full_grid_masked'")
    assert set(full_grid["mask_status"]) == {"plausible_kelp_domain"}
    ridge_all = full_grid.query("model_name == 'ridge_regression' and label_source == 'all'")
    assert len(ridge_all) == 1
    assert int(full_grid.iloc[0]["row_count"]) == 2

    manifest = json.loads(fixture["manifest"].read_text())
    report = fixture["report"].read_text()
    residual_context = pd.read_csv(fixture["residual_domain_context"])
    mask_reason = pd.read_csv(fixture["residual_by_mask_reason"])
    depth_bins = pd.read_csv(fixture["residual_by_depth_bin"])
    top_context = pd.read_csv(fixture["top_residual_context"])
    class_balance = pd.read_csv(fixture["class_balance_by_split"])
    assert manifest["mask_status"] == "plausible_kelp_domain"
    assert manifest["evaluation_scope"] == "full_grid_masked"
    assert "plausible_kelp_domain" in report
    assert "| AEF ridge regression | AEF continuous baseline |" in report
    assert set(residual_context["mask_status"]) == {"plausible_kelp_domain"}
    assert "dropped_too_deep" not in set(residual_context["domain_mask_reason"])
    assert {
        "observed_zero_false_positive",
        "high_canopy_underprediction",
    } <= set(residual_context["residual_class"])
    assert set(mask_reason["domain_mask_reason"]) == {
        "retained_ambiguous_coast",
        "retained_shallow_depth",
    }
    assert set(depth_bins["depth_bin"]) == {"ambiguous_coast", "shallow_depth"}
    assert set(top_context["domain_mask_reason"]) == {
        "retained_ambiguous_coast",
        "retained_shallow_depth",
    }
    assert {"crm_depth_m", "crm_elevation_m", "depth_bin", "elevation_bin"} <= set(
        top_context.columns
    )
    full_grid_balance = class_balance.query(
        "data_scope == 'full_grid_masked' "
        "and split == 'test' "
        "and year == 2022 "
        "and label_source == 'all'"
    ).iloc[0]
    assert full_grid_balance["mask_status"] == "plausible_kelp_domain"
    assert full_grid_balance["evaluation_scope"] == "full_grid_masked"


def reference_comparison_row(
    model_name: str,
    *,
    rmse: float,
    area_pct_bias: float,
) -> dict[str, object]:
    """Build one primary comparison row for report-helper assertions."""
    return {
        "model_name": model_name,
        "split": "test",
        "year": 2022,
        "mask_status": "unmasked",
        "evaluation_scope": "full_grid_prediction",
        "label_source": "all",
        "row_count": 100,
        "mae": rmse / 2,
        "rmse": rmse,
        "r2": 0.5,
        "spearman": 0.5,
        "f1_ge_10pct": 0.7,
        "observed_canopy_area": 1000.0,
        "predicted_canopy_area": 1000.0 * (1.0 + area_pct_bias),
        "area_pct_bias": area_pct_bias,
    }


def test_class_balance_rows_cover_grouped_rates_and_metadata() -> None:
    """Verify grouped balance rows include rates, label sources, and mask metadata."""
    frame = pd.DataFrame(
        [
            balance_row("test", 2022, "assumed_background", None, 0.0),
            balance_row("test", 2022, "kelpwatch_station", 1, 0.0),
            balance_row("test", 2022, "kelpwatch_station", 2, 0.01),
            balance_row("test", 2022, "kelpwatch_station", 3, 0.50),
            balance_row("test", 2022, "kelpwatch_station", 4, 1.00),
        ]
    )
    source = BalanceSource(
        data_scope="model_input_sample",
        mask_status="plausible_kelp_domain",
        evaluation_scope="model_input_sample",
        frame=frame,
    )

    rows = build_class_balance_by_split([source])

    all_row = next(row for row in rows if row["label_source"] == "all")
    assert all_row["mask_status"] == "plausible_kelp_domain"
    assert all_row["evaluation_scope"] == "model_input_sample"
    assert all_row["row_count"] == 5
    assert all_row["station_count"] == 4
    assert all_row["positive_count"] == 3
    assert all_row["positive_ge_1pct_count"] == 3
    assert all_row["positive_ge_10pct_count"] == 2
    assert all_row["high_canopy_count"] == 2
    assert all_row["saturated_count"] == 1
    assert all_row["assumed_background_count"] == 1
    assert all_row["positive_rate"] == 3 / 5
    assert all_row["high_canopy_rate"] == 2 / 5
    assert all_row["assumed_background_rate"] == 1 / 5


def test_class_balance_rows_handle_no_positive_or_high_canopy() -> None:
    """Verify balance rows remain numeric when a group has no positive canopy."""
    frame = pd.DataFrame(
        [
            balance_row("validation", 2021, "assumed_background", None, 0.0),
            balance_row("validation", 2021, "kelpwatch_station", 1, 0.0),
        ]
    )
    source = BalanceSource(
        data_scope="split_manifest_retained",
        mask_status="plausible_kelp_domain",
        evaluation_scope="split_manifest_retained",
        frame=frame,
    )

    rows = build_class_balance_by_split([source])

    all_row = next(row for row in rows if row["label_source"] == "all")
    assert all_row["zero_count"] == 2
    assert all_row["positive_count"] == 0
    assert all_row["high_canopy_count"] == 0
    assert all_row["saturated_count"] == 0
    assert all_row["positive_rate"] == 0
    assert all_row["high_canopy_rate"] == 0


def test_binary_threshold_definitions_include_required_candidates() -> None:
    """Verify annual-max threshold labels and roles are stable."""
    definitions = binary_threshold_definitions((0.50, 0.90))

    by_label = {definition.label: definition for definition in definitions}

    assert {"annual_max_gt0", "annual_max_ge_1pct", "annual_max_ge_5pct"} <= set(by_label)
    assert by_label["annual_max_gt0"].operator == ">"
    assert by_label["annual_max_ge_10pct"].role == "selection_candidate"
    assert by_label["annual_max_ge_50pct"].role == "diagnostic"


def test_binary_threshold_prevalence_groups_rates_and_label_sources() -> None:
    """Verify threshold prevalence rows preserve grouping metadata and rates."""
    frame = pd.DataFrame(
        [
            balance_row("validation", 2021, "assumed_background", None, 0.0),
            balance_row("validation", 2021, "kelpwatch_station", 1, 0.02),
            balance_row("validation", 2021, "kelpwatch_station", 2, 0.06),
        ]
    )
    source = BalanceSource(
        data_scope="model_input_sample",
        mask_status="plausible_kelp_domain",
        evaluation_scope="model_input_sample",
        frame=frame,
    )
    definitions = tuple(
        definition
        for definition in binary_threshold_definitions(())
        if definition.fraction in {0.0, 0.05}
    )

    rows = build_binary_threshold_prevalence([source], definitions)

    all_5pct = next(
        row
        for row in rows
        if row["split"] == "validation"
        and row["year"] == "2021"
        and row["label_source"] == "all"
        and row["threshold_label"] == "annual_max_ge_5pct"
    )
    station_gt0 = next(
        row
        for row in rows
        if row["label_source"] == "kelpwatch_station" and row["threshold_label"] == "annual_max_gt0"
    )
    assert all_5pct["positive_count"] == 1
    assert all_5pct["positive_rate"] == 1 / 3
    assert all_5pct["assumed_background_count"] == 1
    assert all_5pct["assumed_background_positive_count"] == 0
    assert station_gt0["positive_count"] == 2
    assert station_gt0["positive_rate"] == 1.0


def test_binary_threshold_recommendation_uses_validation_only() -> None:
    """Verify recommendation ranking ignores test rows."""
    rows = [
        comparison_row("validation", 2021, 0.0, 200, 0.20, 0.60),
        comparison_row("validation", 2021, 0.10, 150, 0.15, 0.50),
        comparison_row("test", 2022, 0.50, 400, 0.40, 0.95),
    ]

    recommendation = build_binary_threshold_recommendation(rows, 2021)

    selected = next(row for row in recommendation if row["selected_candidate"])
    assert selected["threshold_fraction"] == 0.10
    assert selected["recommended_threshold_label"] == "annual_max_ge_10pct"
    assert {row["selection_split"] for row in recommendation} == {"validation"}
    assert {row["selection_year"] for row in recommendation} == {2021}


def test_binary_threshold_recommendation_handles_no_positive_validation_rows() -> None:
    """Verify recommendation output remains explicit when validation has no positives."""
    rows = [
        comparison_row("validation", 2021, 0.0, 0, 0.0, float("nan")),
        comparison_row("validation", 2021, 0.01, 0, 0.0, float("nan")),
    ]

    recommendation = build_binary_threshold_recommendation(rows, 2021)

    assert recommendation
    assert not any(row["selected_candidate"] for row in recommendation)
    assert {row["recommendation_status"] for row in recommendation} == {
        "no_positive_validation_rows"
    }


def balance_row(
    split: str, year: int, label_source: str, station_id: int | None, fraction: float
) -> dict[str, object]:
    """Build one tiny target-balance test row."""
    return {
        "split": split,
        "year": year,
        "label_source": label_source,
        "kelpwatch_station_id": station_id,
        "kelp_fraction_y": fraction,
        "kelp_max_y": fraction * 900.0,
    }


def comparison_row(
    split: str,
    year: int,
    threshold: float,
    positive_count: int,
    positive_rate: float,
    f1: float,
) -> dict[str, object]:
    """Build one binary-threshold comparison row for recommendation tests."""
    label = "annual_max_gt0" if threshold == 0 else f"annual_max_ge_{int(threshold * 100)}pct"
    return {
        "data_scope": "sample_predictions",
        "mask_status": "plausible_kelp_domain",
        "evaluation_scope": "sample_predictions",
        "split": split,
        "year": year,
        "label_source": "all",
        "model_name": "ridge_regression",
        "threshold_fraction": threshold,
        "threshold_area": threshold * 900.0,
        "threshold_label": label,
        "threshold_operator": ">" if threshold == 0 else ">=",
        "threshold_role": "selection_candidate",
        "target_count": 1000,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "predicted_positive_rate": positive_rate,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "false_positive_rate": 0.1,
        "false_positive_area": 90.0,
        "false_negative_area": 45.0,
        "assumed_background_false_positive_rate": 0.05,
        "assumed_background_false_positive_area": 30.0,
    }


def write_phase2_training_regime_rows(path: Path) -> None:
    """Write compact Phase 2 training-regime comparison fixture rows."""
    rows = []
    for evaluation_region in ("big_sur", "monterey"):
        for training_regime, origin in (
            ("monterey_only", "monterey"),
            ("big_sur_only", "big_sur"),
            ("pooled_monterey_big_sur", "monterey_big_sur"),
        ):
            for model_name, model_family, composition_policy, bias, f1 in (
                ("no_skill_train_mean", "reference_baseline", "", 1.50, 0.05),
                ("previous_year_annual_max", "reference_baseline", "", -0.15, 0.80),
                ("grid_cell_climatology", "reference_baseline", "", -0.10, 0.70),
                ("geographic_ridge_lon_lat_year", "geographic_reference", "", 0.05, 0.0),
                ("ridge_regression", "aef_ridge", "", 0.30, 0.60),
                (
                    "calibrated_probability_x_conditional_canopy",
                    "hurdle",
                    "expected_value",
                    -0.05,
                    0.84,
                ),
                (
                    "calibrated_hard_gate_conditional_canopy",
                    "hurdle",
                    "hard_gate",
                    -0.02,
                    0.85,
                ),
            ):
                rows.append(
                    {
                        "training_regime": training_regime,
                        "model_origin_region": origin,
                        "evaluation_region": evaluation_region,
                        "model_name": model_name,
                        "model_family": model_family,
                        "composition_policy": composition_policy,
                        "split": "test",
                        "year": 2022,
                        "mask_status": "plausible_kelp_domain",
                        "evaluation_scope": "full_grid_masked",
                        "label_source": "all",
                        "row_count": 4,
                        "mae": 0.01,
                        "rmse": 0.04,
                        "r2": 0.8,
                        "f1_ge_10pct": f1,
                        "observed_canopy_area": 1800.0,
                        "predicted_canopy_area": 1800.0 * (1.0 + bias),
                        "area_pct_bias": bias,
                        "source_table": str(path),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_phase2_binary_predictions(path: Path) -> None:
    """Write tiny binary full-grid predictions for Phase 2 support tests."""
    rows = [
        phase2_binary_prediction_row(1, True, 0.90, "kelpwatch_station"),
        phase2_binary_prediction_row(2, True, 0.60, "kelpwatch_station"),
        phase2_binary_prediction_row(3, False, 0.70, "assumed_background"),
        phase2_binary_prediction_row(4, False, 0.20, "assumed_background"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def phase2_binary_prediction_row(
    cell_id: int,
    observed: bool,
    probability: float,
    label_source: str,
) -> dict[str, object]:
    """Build one tiny Phase 2 binary prediction row."""
    return {
        "split": "test",
        "year": 2022,
        "longitude": -122.0 + cell_id * 0.001,
        "latitude": 36.0 + cell_id * 0.001,
        "aef_grid_cell_id": cell_id,
        "aef_grid_row": cell_id,
        "aef_grid_col": cell_id,
        "label_source": label_source,
        "is_plausible_kelp_domain": True,
        "binary_observed_y": observed,
        "pred_binary_probability": probability,
        "target_label": "annual_max_ge_10pct",
    }


def write_phase2_calibration_payload(path: Path) -> None:
    """Write a raw-threshold calibration payload for Phase 2 support tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": "logistic_annual_max_ge_10pct",
            "policy_thresholds": {
                "validation_max_f1_calibrated": ("raw_logistic", 0.5),
            },
        },
        path,
    )


def append_phase2_config(
    config_path: Path,
    *,
    phase2_primary: Path,
    phase2_model_comparison: Path,
    phase2_manifest: Path,
    binary_summary: Path,
    binary_predictions: Path,
    calibration_model: Path,
) -> None:
    """Append a Phase 2 comparison block to a synthetic model-analysis config."""
    entries = "\n".join(
        phase2_binary_config_entry(
            name,
            training_regime=training_regime,
            model_origin_region=model_origin_region,
            evaluation_region=evaluation_region,
            binary_predictions=binary_predictions,
            calibration_model=calibration_model,
        )
        for name, training_regime, model_origin_region, evaluation_region in (
            ("big_sur_monterey_only_transfer", "monterey_only", "monterey", "big_sur"),
            ("big_sur_big_sur_only", "big_sur_only", "big_sur", "big_sur"),
            (
                "big_sur_pooled_monterey_big_sur",
                "pooled_monterey_big_sur",
                "monterey_big_sur",
                "big_sur",
            ),
            ("monterey_big_sur_only_transfer", "big_sur_only", "big_sur", "monterey"),
            ("monterey_monterey_only", "monterey_only", "monterey", "monterey"),
            (
                "monterey_pooled_monterey_big_sur",
                "pooled_monterey_big_sur",
                "monterey_big_sur",
                "monterey",
            ),
        )
    )
    with config_path.open("a") as file:
        file.write(
            f"""
training_regime_comparison:
  primary_split: test
  primary_year: 2022
  primary_mask_status: plausible_kelp_domain
  primary_evaluation_scope: full_grid_masked
  primary_label_source: all
  model_comparison: {phase2_model_comparison}
  primary_summary: {phase2_primary}
  manifest: {phase2_manifest}
  binary_support:
    primary_summary: {binary_summary}
    inputs:
{entries}
"""
        )


def phase2_binary_config_entry(
    name: str,
    *,
    training_regime: str,
    model_origin_region: str,
    evaluation_region: str,
    binary_predictions: Path,
    calibration_model: Path,
) -> str:
    """Build one YAML entry for a Phase 2 binary-support fixture."""
    return f"""      {name}:
        full_grid_predictions: {binary_predictions}
        calibration_model: {calibration_model}
        training_regime: {training_regime}
        model_origin_region: {model_origin_region}
        evaluation_region: {evaluation_region}
        threshold_policy: validation_max_f1_calibrated"""


def append_component_failure_config(
    config_path: Path,
    *,
    paths: dict[str, Path],
    hurdle_predictions: Path,
    label_path: Path,
) -> None:
    """Append a component-failure diagnostic block to a synthetic config."""
    with config_path.open("a") as file:
        file.write(
            f"""
  component_failure:
    summary: {paths["component_failure_summary"]}
    by_label_context: {paths["component_failure_by_label"]}
    by_domain_context: {paths["component_failure_by_domain"]}
    by_spatial_context: {paths["component_failure_by_spatial"]}
    by_model_context: {paths["component_failure_by_model"]}
    edge_effect_diagnostics: {paths["component_failure_edge"]}
    temporal_label_context: {paths["component_failure_temporal"]}
    manifest: {paths["component_failure_manifest"]}
    tolerance_m2: 90.0
    grid_cell_size_m: 30.0
    inputs:
      fixture_component_context:
        hurdle_predictions: {hurdle_predictions}
        label_path: {label_path}
        training_regime: big_sur_only
        model_origin_region: big_sur
        evaluation_region: big_sur
        required: true
"""
        )


def append_pooled_context_config(
    config_path: Path,
    *,
    paths: dict[str, Path],
    baseline_predictions: Path,
    binary_predictions: Path,
    calibration_model: Path,
    hurdle_predictions: Path,
    label_path: Path,
) -> None:
    """Append a pooled-context diagnostic block to a synthetic config."""
    with config_path.open("a") as file:
        file.write(
            f"""
  pooled_context:
    performance: {paths["pooled_context_performance"]}
    binary_context: {paths["pooled_binary_context"]}
    amount_context: {paths["pooled_amount_context"]}
    prediction_distribution: {paths["pooled_prediction_distribution"]}
    manifest: {paths["pooled_context_manifest"]}
    tolerance_m2: 90.0
    grid_cell_size_m: 30.0
    inputs:
      pooled_fixture_context:
        baseline_predictions: {baseline_predictions}
        binary_predictions: {binary_predictions}
        binary_calibration_model: {calibration_model}
        hurdle_predictions: {hurdle_predictions}
        label_path: {label_path}
        training_regime: pooled_monterey_big_sur
        model_origin_region: monterey_big_sur
        evaluation_region: big_sur
        threshold_policy: validation_max_f1_calibrated
        required: true
"""
        )


def write_phase2_diagnostics_cache_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write a synthetic Phase 2 diagnostics fixture with cache paths configured."""
    fixture = write_model_analysis_fixture(
        tmp_path,
        include_domain_mask=True,
        include_training_mask_columns=True,
        include_reference_area_calibration=True,
    )
    phase2_primary = tmp_path / "reports/tables/phase2_training_regime_primary.csv"
    phase2_model_comparison = tmp_path / "reports/tables/phase2_training_regime_all.csv"
    phase2_manifest = tmp_path / "interim/phase2_training_regime_manifest.json"
    binary_summary = tmp_path / "reports/tables/phase2_binary_support.csv"
    binary_predictions = tmp_path / "processed/phase2_binary_predictions.parquet"
    calibration_model = tmp_path / "models/phase2_binary_calibration.joblib"
    component_hurdle = tmp_path / "processed/component_hurdle_predictions.parquet"
    pooled_baseline = tmp_path / "processed/pooled_baseline_predictions.parquet"
    pooled_binary = tmp_path / "processed/pooled_binary_predictions.parquet"
    pooled_hurdle = tmp_path / "processed/pooled_hurdle_predictions.parquet"
    write_phase2_training_regime_rows(phase2_primary)
    write_phase2_training_regime_rows(phase2_model_comparison)
    phase2_manifest.parent.mkdir(parents=True, exist_ok=True)
    phase2_manifest.write_text(json.dumps({"command": "compare-training-regimes"}))
    write_phase2_binary_predictions(binary_predictions)
    write_phase2_calibration_payload(calibration_model)
    write_component_failure_hurdle_predictions(component_hurdle)
    write_pooled_context_baseline_predictions(pooled_baseline)
    write_pooled_context_binary_predictions(pooled_binary)
    write_component_failure_hurdle_predictions(pooled_hurdle)
    append_phase2_config(
        fixture["config_path"],
        phase2_primary=phase2_primary,
        phase2_model_comparison=phase2_model_comparison,
        phase2_manifest=phase2_manifest,
        binary_summary=binary_summary,
        binary_predictions=binary_predictions,
        calibration_model=calibration_model,
    )
    append_component_failure_config(
        fixture["config_path"],
        paths=fixture,
        hurdle_predictions=component_hurdle,
        label_path=tmp_path / "interim/labels_annual.parquet",
    )
    append_pooled_context_config(
        fixture["config_path"],
        paths=fixture,
        baseline_predictions=pooled_baseline,
        binary_predictions=pooled_binary,
        calibration_model=calibration_model,
        hurdle_predictions=pooled_hurdle,
        label_path=tmp_path / "interim/labels_annual.parquet",
    )
    return fixture


def append_binary_hex_map_config(
    config_path: Path,
    *,
    paths: dict[str, Path],
    binary_predictions: Path,
    calibration_model: Path,
) -> None:
    """Append a pooled binary hex-map diagnostic block to a synthetic config."""
    with config_path.open("a") as file:
        file.write(
            f"""
  pooled_binary_hex_map:
    figure: {paths["binary_hex_figure"]}
    table: {paths["binary_hex_table"]}
    manifest: {paths["binary_hex_manifest"]}
    source_crs: EPSG:4326
    target_crs: EPSG:32610
    hex_flat_diameter_m: 1000.0
    rate_clip_quantile: 0.98
    difference_clip_quantile: 0.98
    inputs:
      pooled_fixture_hex_context:
        binary_predictions: {binary_predictions}
        binary_calibration_model: {calibration_model}
        training_regime: pooled_monterey_big_sur
        model_origin_region: monterey_big_sur
        evaluation_region: big_sur
        threshold_policy: validation_max_f1_calibrated
        required: true
"""
        )


def write_pooled_context_baseline_predictions(path: Path) -> None:
    """Write tiny ridge predictions matching the pooled hurdle fixture cells."""
    rows = [
        pooled_context_baseline_row(cell_id=1, observed_area=900.0, predicted_area=500.0),
        pooled_context_baseline_row(cell_id=2, observed_area=450.0, predicted_area=100.0),
        pooled_context_baseline_row(cell_id=3, observed_area=0.0, predicted_area=300.0),
        pooled_context_baseline_row(cell_id=4, observed_area=0.0, predicted_area=30.0),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def pooled_context_baseline_row(
    *,
    cell_id: int,
    observed_area: float,
    predicted_area: float,
) -> dict[str, object]:
    """Build one tiny pooled ridge prediction row."""
    return {
        "year": 2022,
        "split": "test",
        "aef_grid_cell_id": cell_id,
        "model_name": "ridge_regression",
        "kelp_max_y": observed_area,
        "pred_kelp_max_y": predicted_area,
        "pred_kelp_fraction_y_clipped": predicted_area / 900.0,
    }


def write_pooled_context_binary_predictions(path: Path) -> None:
    """Write tiny binary predictions matching the pooled hurdle fixture cells."""
    rows = [
        pooled_context_binary_row(cell_id=1, observed=True, probability=0.80),
        pooled_context_binary_row(cell_id=2, observed=True, probability=0.20),
        pooled_context_binary_row(cell_id=3, observed=False, probability=0.90),
        pooled_context_binary_row(cell_id=4, observed=False, probability=0.10),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def pooled_context_binary_row(
    *,
    cell_id: int,
    observed: bool,
    probability: float,
) -> dict[str, object]:
    """Build one tiny pooled binary prediction row."""
    return {
        "year": 2022,
        "split": "test",
        "aef_grid_cell_id": cell_id,
        "binary_observed_y": observed,
        "pred_binary_probability": probability,
        "target_label": "annual_max_ge_10pct",
    }


def write_component_failure_hurdle_predictions(path: Path) -> None:
    """Write small expected-value hurdle rows for component-failure tests."""
    rows = []
    for year, split in ((2021, "validation"), (2022, "test")):
        rows.extend(
            [
                component_hurdle_row(
                    year,
                    split,
                    cell_id=1,
                    row=0,
                    col=0,
                    station_id=1,
                    observed_area=900.0,
                    probability=0.80,
                    threshold=0.50,
                    conditional_area=900.0,
                    expected_area=720.0,
                    hard_gate_area=900.0,
                    label_source="kelpwatch_station",
                ),
                component_hurdle_row(
                    year,
                    split,
                    cell_id=2,
                    row=0,
                    col=1,
                    station_id=2,
                    observed_area=450.0,
                    probability=0.20,
                    threshold=0.50,
                    conditional_area=600.0,
                    expected_area=120.0,
                    hard_gate_area=0.0,
                    label_source="kelpwatch_station",
                ),
                component_hurdle_row(
                    year,
                    split,
                    cell_id=3,
                    row=1,
                    col=0,
                    station_id=3,
                    observed_area=0.0,
                    probability=0.90,
                    threshold=0.50,
                    conditional_area=450.0,
                    expected_area=405.0,
                    hard_gate_area=450.0,
                    label_source="kelpwatch_station",
                ),
                component_hurdle_row(
                    year,
                    split,
                    cell_id=4,
                    row=1,
                    col=1,
                    station_id=None,
                    observed_area=0.0,
                    probability=0.10,
                    threshold=0.50,
                    conditional_area=300.0,
                    expected_area=30.0,
                    hard_gate_area=0.0,
                    label_source="assumed_background",
                ),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def component_hurdle_row(
    year: int,
    split: str,
    *,
    cell_id: int,
    row: int,
    col: int,
    station_id: int | None,
    observed_area: float,
    probability: float,
    threshold: float,
    conditional_area: float,
    expected_area: float,
    hard_gate_area: float,
    label_source: str,
) -> dict[str, object]:
    """Build one expected-value hurdle prediction fixture row."""
    return {
        "year": year,
        "split": split,
        "kelpwatch_station_id": station_id,
        "longitude": -122.0 + col * 0.001,
        "latitude": 36.0 + row * 0.001,
        "kelp_fraction_y": observed_area / 900.0,
        "kelp_max_y": observed_area,
        "aef_grid_cell_id": cell_id,
        "aef_grid_row": row,
        "aef_grid_col": col,
        "label_source": label_source,
        "is_kelpwatch_observed": label_source == "kelpwatch_station",
        "kelpwatch_station_count": 1 if label_source == "kelpwatch_station" else 0,
        "is_plausible_kelp_domain": True,
        "domain_mask_reason": "retained_ambiguous_coast" if row == 0 else "retained_depth_0_60m",
        "domain_mask_detail": "fixture",
        "domain_mask_version": "test_mask_v1",
        "crm_elevation_m": 0.0 if row == 0 else -12.0,
        "crm_depth_m": 0.0 if row == 0 else 12.0,
        "depth_bin": "ambiguous_coast" if row == 0 else "0_40m",
        "elevation_bin": "nearshore" if row == 0 else "subtidal",
        "mask_status": "plausible_kelp_domain",
        "evaluation_scope": "full_grid_masked",
        "model_name": "calibrated_probability_x_conditional_canopy",
        "composition_policy": "expected_value",
        "presence_probability_threshold": threshold,
        "calibrated_presence_probability": probability,
        "pred_presence_class": probability >= threshold,
        "pred_conditional_area_m2": conditional_area,
        "pred_expected_value_area_m2": expected_area,
        "pred_hard_gate_area_m2": hard_gate_area,
        "pred_hurdle_area_m2": expected_area,
        "pred_kelp_max_y": expected_area,
        "residual_kelp_max_y": observed_area - expected_area,
    }


def write_model_analysis_fixture(
    tmp_path: Path,
    *,
    include_domain_mask: bool = False,
    include_training_mask_columns: bool = False,
    include_reference_area_calibration: bool = False,
) -> dict[str, Path]:
    """Write synthetic model-analysis inputs and configured output paths."""
    labels = tmp_path / "interim/labels_annual.parquet"
    label_manifest = tmp_path / "interim/labels_annual_manifest.json"
    aligned = tmp_path / "interim/aligned_training_table.parquet"
    split_manifest = tmp_path / "interim/split_manifest.parquet"
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    paths = output_paths(tmp_path)
    domain_mask = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    domain_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    write_labels(labels)
    write_aligned(aligned, include_training_mask_columns=include_training_mask_columns)
    write_split_manifest(split_manifest)
    write_predictions(predictions)
    write_metrics(metrics)
    write_binary_presence_outputs(paths)
    if include_domain_mask:
        write_model_analysis_domain_mask(domain_mask, domain_manifest)
    label_manifest.parent.mkdir(parents=True, exist_ok=True)
    label_manifest.write_text(json.dumps({"spatial": {"crs": "EPSG:4326"}}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        config_text(
            tmp_path,
            labels,
            label_manifest,
            aligned,
            split_manifest,
            predictions,
            metrics,
            paths,
            domain_mask=domain_mask if include_domain_mask else None,
            domain_manifest=domain_manifest if include_domain_mask else None,
        )
    )
    if include_reference_area_calibration:
        write_reference_area_calibration(paths["reference_area_calibration"])
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
        "all_model_sampling_policy_comparison": tmp_path
        / "reports/tables/model_analysis_crm_stratified_all_models_comparison.csv",
        "reference_area_calibration": tmp_path
        / "reports/tables/reference_baseline_area_calibration.csv",
        "data_health": tmp_path / "reports/tables/model_analysis_data_health.csv",
        "class_balance_by_split": tmp_path
        / "reports/tables/model_analysis_class_balance_by_split.csv",
        "target_balance_by_label_source": tmp_path
        / "reports/tables/model_analysis_target_balance_by_label_source.csv",
        "background_rate_summary": tmp_path
        / "reports/tables/model_analysis_background_rate_summary.csv",
        "binary_threshold_prevalence": tmp_path
        / "reports/tables/model_analysis_binary_threshold_prevalence.csv",
        "binary_threshold_comparison": tmp_path
        / "reports/tables/model_analysis_binary_threshold_comparison.csv",
        "binary_threshold_recommendation": tmp_path
        / "reports/tables/model_analysis_binary_threshold_recommendation.csv",
        "binary_presence_metrics": tmp_path / "reports/tables/binary_presence_metrics.csv",
        "binary_presence_threshold_selection": tmp_path
        / "reports/tables/binary_presence_threshold_selection.csv",
        "binary_presence_full_grid_area_summary": tmp_path
        / "reports/tables/binary_presence_full_grid_area_summary.csv",
        "binary_presence_thresholded_model_comparison": tmp_path
        / "reports/tables/binary_presence_thresholded_model_comparison.csv",
        "binary_presence_precision_recall_figure": tmp_path
        / "reports/figures/binary_presence_precision_recall.png",
        "binary_presence_map_figure": tmp_path / "reports/figures/binary_presence_2022_map.png",
        "binary_presence_calibration_metrics": tmp_path
        / "reports/tables/binary_presence_calibration_metrics.csv",
        "binary_presence_calibrated_threshold_selection": tmp_path
        / "reports/tables/binary_presence_calibrated_threshold_selection.csv",
        "binary_presence_calibrated_full_grid_area_summary": tmp_path
        / "reports/tables/binary_presence_calibrated_full_grid_area_summary.csv",
        "binary_presence_calibration_manifest": tmp_path
        / "interim/binary_presence_calibration_manifest.json",
        "binary_presence_calibration_curve_figure": tmp_path
        / "reports/figures/binary_presence_calibration_curve.png",
        "binary_presence_calibrated_threshold_figure": tmp_path
        / "reports/figures/binary_presence_calibrated_thresholds.png",
        "residual_domain_context": tmp_path
        / "reports/tables/model_analysis_residual_by_domain_context.csv",
        "residual_by_mask_reason": tmp_path
        / "reports/tables/model_analysis_residual_by_mask_reason.csv",
        "residual_by_depth_bin": tmp_path
        / "reports/tables/model_analysis_residual_by_depth_bin.csv",
        "top_residual_context": tmp_path
        / "reports/tables/top_residual_stations.domain_context.csv",
        "quarter_mapping": tmp_path / "reports/tables/model_analysis_quarter_mapping.csv",
        "label_distribution_figure": tmp_path
        / "reports/figures/model_analysis_label_distribution_by_stage.png",
        "observed_predicted_figure": tmp_path
        / "reports/figures/model_analysis_observed_vs_predicted_distribution.png",
        "pixel_skill_area_calibration_figure": tmp_path
        / "reports/figures/model_analysis_pixel_skill_area_calibration.png",
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
        "class_balance_figure": tmp_path / "reports/figures/model_analysis_class_balance.png",
        "binary_threshold_comparison_figure": tmp_path
        / "reports/figures/model_analysis_binary_threshold_comparison.png",
        "residual_domain_context_figure": tmp_path
        / "reports/figures/model_analysis_residual_by_domain_context.png",
        "pooled_context_metric_breakdown_figure": tmp_path
        / "reports/figures/monterey_big_sur_pooled_context_metric_breakdown.png",
        "pooled_mean_max_binary_f1_figure": tmp_path
        / "reports/figures/monterey_big_sur_pooled_mean_max_binary_f1.png",
        "pooled_prediction_distribution_figure": tmp_path
        / "reports/figures/monterey_big_sur_pooled_prediction_distribution.png",
        "phase2_baseline_grounding_figure": tmp_path
        / "reports/figures/monterey_big_sur_compact_baseline_grounding.png",
        "component_failure_summary": tmp_path
        / "reports/tables/monterey_big_sur_component_failure_summary.csv",
        "component_failure_by_label": tmp_path
        / "reports/tables/monterey_big_sur_component_failure_by_label_context.csv",
        "component_failure_by_domain": tmp_path
        / "reports/tables/monterey_big_sur_component_failure_by_domain_context.csv",
        "component_failure_by_spatial": tmp_path
        / "reports/tables/monterey_big_sur_component_failure_by_spatial_context.csv",
        "component_failure_by_model": tmp_path
        / "reports/tables/monterey_big_sur_component_failure_by_model_context.csv",
        "component_failure_edge": tmp_path
        / "reports/tables/monterey_big_sur_edge_effect_diagnostics.csv",
        "component_failure_temporal": tmp_path
        / "reports/tables/monterey_big_sur_temporal_label_context.csv",
        "component_failure_manifest": tmp_path
        / "interim/monterey_big_sur_component_failure_manifest.json",
        "pooled_context_performance": tmp_path
        / "reports/tables/monterey_big_sur_pooled_context_model_performance.csv",
        "pooled_binary_context": tmp_path
        / "reports/tables/monterey_big_sur_pooled_binary_context_diagnostics.csv",
        "pooled_amount_context": tmp_path
        / "reports/tables/monterey_big_sur_pooled_amount_context_diagnostics.csv",
        "pooled_prediction_distribution": tmp_path
        / "reports/tables/monterey_big_sur_pooled_prediction_distribution_by_context.csv",
        "pooled_context_manifest": tmp_path
        / "interim/monterey_big_sur_pooled_context_diagnostics_manifest.json",
        "phase2_component_frame_cache": tmp_path
        / "interim/monterey_big_sur_phase2_component_failure_frames",
        "phase2_pooled_context_frame_cache": tmp_path
        / "interim/monterey_big_sur_phase2_pooled_context_frames",
        "phase2_diagnostics_cache_manifest": tmp_path
        / "interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json",
        "binary_hex_figure": tmp_path
        / "reports/figures/monterey_big_sur_pooled_binary_presence_hex_1km.png",
        "binary_hex_table": tmp_path
        / "reports/tables/monterey_big_sur_pooled_binary_presence_hex_1km.csv",
        "binary_hex_manifest": tmp_path
        / "interim/monterey_big_sur_pooled_binary_presence_hex_manifest.json",
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
    domain_mask: Path | None = None,
    domain_manifest: Path | None = None,
) -> str:
    """Build a minimal workflow config for model analysis."""
    domain_mask_config = (
        f"""
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
"""
        if domain_mask is not None and domain_manifest is not None
        else ""
    )
    masked_outputs = (
        f"    reference_baseline_area_calibration_masked: {paths['reference_area_calibration']}\n"
        if domain_mask is not None
        else ""
    )
    pixel_skill_area_figure = paths["pixel_skill_area_calibration_figure"]
    all_model_comparison = paths["all_model_sampling_policy_comparison"]
    pooled_prediction_distribution_figure = paths["pooled_prediction_distribution_figure"]
    pooled_context_metric_breakdown_output = (
        "    model_analysis_phase2_pooled_context_metric_breakdown_figure: "
        f"{paths['pooled_context_metric_breakdown_figure']}\n"
    )
    pooled_mean_max_binary_f1_output = (
        "    model_analysis_phase2_pooled_mean_max_binary_f1_figure: "
        f"{paths['pooled_mean_max_binary_f1_figure']}\n"
    )
    pooled_prediction_distribution_output = (
        "    model_analysis_phase2_pooled_prediction_distribution_figure: "
        f"{pooled_prediction_distribution_figure}\n"
    )
    phase2_baseline_grounding_output = (
        "    model_analysis_phase2_baseline_grounding_figure: "
        f"{paths['phase2_baseline_grounding_figure']}\n"
    )
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
    sample_policy: crm_stratified_mask_first_sample
    features: A00-A01
    sample_predictions: {predictions}
    predictions: {predictions}
    metrics: {metrics}
  binary_presence:
    metrics: {paths["binary_presence_metrics"]}
    threshold_selection: {paths["binary_presence_threshold_selection"]}
    full_grid_area_summary: {paths["binary_presence_full_grid_area_summary"]}
    thresholded_model_comparison: {paths["binary_presence_thresholded_model_comparison"]}
    precision_recall_figure: {paths["binary_presence_precision_recall_figure"]}
    map_figure: {paths["binary_presence_map_figure"]}
    calibration:
      metrics: {paths["binary_presence_calibration_metrics"]}
      threshold_selection: {paths["binary_presence_calibrated_threshold_selection"]}
      full_grid_area_summary: {paths["binary_presence_calibrated_full_grid_area_summary"]}
      manifest: {paths["binary_presence_calibration_manifest"]}
      calibration_curve_figure: {paths["binary_presence_calibration_curve_figure"]}
      threshold_figure: {paths["binary_presence_calibrated_threshold_figure"]}
reports:
  figures_dir: {tmp_path / "reports/figures"}
  tables_dir: {tmp_path / "reports/tables"}
{domain_mask_config}
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
    phase2_component_failure_frame_cache: {paths["phase2_component_frame_cache"]}
    phase2_pooled_context_frame_cache: {paths["phase2_pooled_context_frame_cache"]}
    phase2_diagnostics_cache_manifest: {paths["phase2_diagnostics_cache_manifest"]}
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
    model_analysis_crm_stratified_all_models_comparison: {all_model_comparison}
    reference_baseline_area_calibration: {paths["reference_area_calibration"]}
{masked_outputs}
    model_analysis_data_health: {paths["data_health"]}
    model_analysis_class_balance_by_split: {paths["class_balance_by_split"]}
    model_analysis_target_balance_by_label_source: {paths["target_balance_by_label_source"]}
    model_analysis_background_rate_summary: {paths["background_rate_summary"]}
    model_analysis_binary_threshold_prevalence: {paths["binary_threshold_prevalence"]}
    model_analysis_binary_threshold_comparison: {paths["binary_threshold_comparison"]}
    model_analysis_binary_threshold_recommendation: {paths["binary_threshold_recommendation"]}
    model_analysis_residual_by_domain_context: {paths["residual_domain_context"]}
    model_analysis_residual_by_mask_reason: {paths["residual_by_mask_reason"]}
    model_analysis_residual_by_depth_bin: {paths["residual_by_depth_bin"]}
    top_residual_stations_domain_context: {paths["top_residual_context"]}
    model_analysis_quarter_mapping: {paths["quarter_mapping"]}
    model_analysis_label_distribution_figure: {paths["label_distribution_figure"]}
    model_analysis_observed_predicted_figure: {paths["observed_predicted_figure"]}
    model_analysis_pixel_skill_area_calibration_figure: {pixel_skill_area_figure}
    model_analysis_residual_by_bin_figure: {paths["residual_by_bin_figure"]}
    model_analysis_observed_900_figure: {paths["observed_900_figure"]}
    model_analysis_residual_by_persistence_figure: {paths["residual_by_persistence_figure"]}
    model_analysis_alternative_targets_figure: {paths["alternative_targets_figure"]}
    model_analysis_feature_projection_figure: {paths["feature_projection_figure"]}
    model_analysis_spatial_readiness_figure: {paths["spatial_readiness_figure"]}
    model_analysis_class_balance_figure: {paths["class_balance_figure"]}
    model_analysis_binary_threshold_comparison_figure: {paths["binary_threshold_comparison_figure"]}
    model_analysis_residual_by_domain_context_figure: {paths["residual_domain_context_figure"]}
{pooled_context_metric_breakdown_output}{pooled_mean_max_binary_f1_output}{pooled_prediction_distribution_output}{phase2_baseline_grounding_output}
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


def write_aligned(path: Path, *, include_training_mask_columns: bool = False) -> None:
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
    if include_training_mask_columns:
        labels["is_plausible_kelp_domain"] = True
        labels["domain_mask_reason"] = "retained"
        labels["domain_mask_detail"] = "fixture"
        labels["domain_mask_version"] = "test_mask_v1"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)


def write_binary_presence_outputs(paths: dict[str, Path]) -> None:
    """Write compact binary-presence outputs consumed by model analysis."""
    paths["binary_presence_metrics"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "split": "validation",
                "year": 2021,
                "label_source": "all",
                "row_count": 3,
                "probability_threshold": 0.42,
                "auroc": 0.9,
                "auprc": 0.85,
                "precision": 0.8,
                "recall": 1.0,
                "f1": 0.889,
                "predicted_positive_rate": 2 / 3,
            },
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "row_count": 3,
                "probability_threshold": 0.42,
                "auroc": 0.8,
                "auprc": 0.75,
                "precision": 0.75,
                "recall": 1.0,
                "f1": 0.857,
                "predicted_positive_rate": 2 / 3,
            },
        ]
    ).to_csv(paths["binary_presence_metrics"], index=False)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "selection_status": "selected_from_validation_max_f1",
                "selected_threshold": True,
                "probability_threshold": 0.42,
                "f1": 0.889,
            }
        ]
    ).to_csv(paths["binary_presence_threshold_selection"], index=False)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "evaluation_scope": "full_grid_prediction",
                "row_count": 3,
                "predicted_positive_rate": 2 / 3,
                "predicted_positive_area_m2": 1800.0,
                "assumed_background_predicted_positive_rate": 0.5,
            }
        ]
    ).to_csv(paths["binary_presence_full_grid_area_summary"], index=False)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "model_family": "balanced_binary",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "row_count": 3,
                "auroc": 0.8,
                "auprc": 0.75,
                "precision": 0.75,
                "recall": 1.0,
                "f1": 0.857,
                "predicted_positive_rate": 2 / 3,
                "assumed_background_false_positive_rate": 0.5,
            },
            {
                "model_name": "ridge_regression",
                "model_family": "thresholded_continuous_baseline",
                "split": "test",
                "year": 2022,
                "label_source": "all",
                "row_count": 3,
                "auroc": 0.75,
                "auprc": 0.7,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.667,
                "predicted_positive_rate": 1 / 3,
                "assumed_background_false_positive_rate": 0.0,
            },
        ]
    ).to_csv(paths["binary_presence_thresholded_model_comparison"], index=False)
    write_tiny_png(paths["binary_presence_map_figure"])
    pd.DataFrame(
        [
            calibration_metric_row("validation", 2021, "raw_logistic", 0.20, 0.80, 0.10),
            calibration_metric_row("validation", 2021, "platt_calibrated", 0.18, 0.78, 0.08),
            calibration_metric_row("test", 2022, "raw_logistic", 0.24, 0.75, 0.12),
            calibration_metric_row("test", 2022, "platt_calibrated", 0.21, 0.73, 0.09),
        ]
    ).to_csv(paths["binary_presence_calibration_metrics"], index=False)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "calibration_method": "platt",
                "probability_source": "platt_calibrated",
                "calibration_split": "validation",
                "calibration_year": 2021,
                "threshold_policy": "validation_max_f1_calibrated",
                "selection_status": "selected_from_validation_max_f1",
                "recommended_policy": True,
                "selected_threshold": True,
                "probability_threshold": 0.37,
                "f1": 0.88,
            }
        ]
    ).to_csv(paths["binary_presence_calibrated_threshold_selection"], index=False)
    pd.DataFrame(
        [
            calibrated_full_grid_row("raw_logistic", "p1_18_validation_raw_threshold", 2 / 3),
            calibrated_full_grid_row(
                "platt_calibrated",
                "validation_max_f1_calibrated",
                1 / 3,
            ),
        ]
    ).to_csv(paths["binary_presence_calibrated_full_grid_area_summary"], index=False)
    paths["binary_presence_calibration_manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["binary_presence_calibration_manifest"].write_text(
        json.dumps({"command": "calibrate-binary-presence"})
    )
    write_tiny_png(paths["binary_presence_calibration_curve_figure"])
    write_tiny_png(paths["binary_presence_calibrated_threshold_figure"])


def calibration_metric_row(
    split: str,
    year: int,
    probability_source: str,
    brier: float,
    auprc: float,
    ece: float,
) -> dict[str, object]:
    """Build one binary calibration metric fixture row."""
    return {
        "model_name": "logistic_annual_max_ge_10pct",
        "target_label": "annual_max_ge_10pct",
        "calibration_method": "platt" if probability_source == "platt_calibrated" else "none",
        "probability_source": probability_source,
        "threshold_policy": "validation_max_f1_calibrated"
        if probability_source == "platt_calibrated"
        else "p1_18_validation_raw_threshold",
        "split": split,
        "year": year,
        "label_source": "all",
        "row_count": 3,
        "auroc": 0.8,
        "auprc": auprc,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "precision": 0.75,
        "recall": 1.0,
        "f1": 0.857,
        "predicted_positive_rate": 2 / 3,
    }


def calibrated_full_grid_row(
    probability_source: str,
    threshold_policy: str,
    predicted_positive_rate: float,
) -> dict[str, object]:
    """Build one calibrated full-grid summary fixture row."""
    return {
        "model_name": "logistic_annual_max_ge_10pct",
        "target_label": "annual_max_ge_10pct",
        "calibration_method": "platt" if probability_source == "platt_calibrated" else "none",
        "probability_source": probability_source,
        "threshold_policy": threshold_policy,
        "split": "test",
        "year": 2022,
        "label_source": "all",
        "evaluation_scope": "full_grid_prediction",
        "row_count": 3,
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_positive_area_m2": predicted_positive_rate * 2700.0,
        "assumed_background_predicted_positive_rate": 0.5,
    }


def write_tiny_png(path: Path) -> None:
    """Write a valid one-pixel PNG fixture for report image linking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
    )


def write_reference_area_calibration(path: Path) -> None:
    """Write cached masked full-grid calibration rows for duplicate checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "ridge_regression",
                "split": "test",
                "year": 2022,
                "mask_status": "plausible_kelp_domain",
                "evaluation_scope": "full_grid_masked",
                "label_source": "all",
                "row_count": 2,
                "observed_canopy_area": 900.0,
                "predicted_canopy_area": 540.0,
                "area_bias": -360.0,
                "area_pct_bias": -0.4,
                "mae": 0.25,
                "rmse": 0.32,
                "r2": 0.5,
                "f1_ge_10pct": 0.8,
            },
            {
                "model_name": "no_skill_train_mean",
                "split": "test",
                "year": 2022,
                "mask_status": "plausible_kelp_domain",
                "evaluation_scope": "full_grid_masked",
                "label_source": "all",
                "row_count": 2,
                "observed_canopy_area": 900.0,
                "predicted_canopy_area": 450.0,
                "area_bias": -450.0,
                "area_pct_bias": -0.5,
                "mae": 0.35,
                "rmse": 0.4,
                "r2": 0.0,
                "f1_ge_10pct": 0.5,
            },
        ]
    ).to_csv(path, index=False)


def write_split_manifest(path: Path) -> None:
    """Write a split manifest with one dropped missing-feature row."""
    rows = []
    for year, split in [(2021, "validation"), (2022, "test")]:
        for station_id, fraction in [(1, 0.0), (2, 0.5), (3, 1.0)]:
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": station_id,
                    "aef_grid_cell_id": station_id,
                    "aef_grid_row": station_id,
                    "aef_grid_col": station_id,
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


def write_model_analysis_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a tiny domain mask that drops the middle fixture station."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [1, 2, 3],
            "is_plausible_kelp_domain": [True, False, True],
            "domain_mask_reason": [
                "retained_ambiguous_coast",
                "dropped_too_deep",
                "retained_shallow_depth",
            ],
            "domain_mask_detail": ["fixture"] * 3,
            "domain_mask_version": ["test_mask_v1"] * 3,
            "crm_elevation_m": [0.2, -100.0, -6.0],
            "crm_depth_m": [0.0, 100.0, 6.0],
            "depth_bin": ["ambiguous_coast", "deep_water", "shallow_depth"],
            "elevation_bin": ["nearshore", "deep_subtidal", "subtidal"],
        }
    ).to_parquet(mask_path, index=False)
    manifest_path.write_text(json.dumps({"mask_version": "test_mask_v1"}))


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
                    "aef_grid_cell_id": station_id,
                    "aef_grid_row": station_id,
                    "aef_grid_col": station_id,
                    "label_source": "kelpwatch_station",
                    "is_kelpwatch_observed": True,
                    "kelpwatch_station_count": 1,
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
