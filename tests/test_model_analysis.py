import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main
from kelp_aef.evaluation.model_analysis import (
    BalanceSource,
    binary_threshold_definitions,
    build_binary_threshold_prevalence,
    build_binary_threshold_recommendation,
    build_class_balance_by_split,
    conditional_likely_positive_sampling_rows,
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
    assert "Decision / Next Modeling Step" in report
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
    assert "<h2>Decision / Next Modeling Step</h2>" in html_report
    assert 'src="data:image/png;base64,' in html_report


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
