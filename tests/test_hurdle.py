import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef import main


def test_compose_hurdle_model_writes_predictions_and_diagnostics(tmp_path: Path) -> None:
    """Verify compose-hurdle-model writes expected-value and hard-gated artifacts."""
    fixture = write_hurdle_fixture(tmp_path)

    assert main(["compose-hurdle-model", "--config", str(fixture["config_path"])]) == 0

    predictions = pd.read_parquet(fixture["predictions"])
    metrics = pd.read_csv(fixture["metrics"])
    area = pd.read_csv(fixture["area_calibration"])
    comparison = pd.read_csv(fixture["model_comparison"])
    leakage = pd.read_csv(fixture["assumed_background_leakage"])
    residuals = pd.read_csv(fixture["residual_by_observed_bin"])
    manifest = json.loads(fixture["manifest"].read_text())

    assert fixture["map_figure"].is_file()
    height, width = png_dimensions(fixture["map_figure"])
    assert height > width
    assert set(predictions["composition_policy"]) == {"expected_value", "hard_gate"}
    assert {
        "calibrated_probability_x_conditional_canopy",
        "calibrated_hard_gate_conditional_canopy",
    } == set(predictions["model_name"])
    assert set(predictions["presence_probability_source"]) == {"platt_calibrated"}
    assert set(predictions["presence_probability_threshold"]) == {0.4}

    expected_row = predictions.query(
        "composition_policy == 'expected_value' and year == 2022 and aef_grid_cell_id == 2"
    ).iloc[0]
    expected_product = (
        expected_row["calibrated_presence_probability"]
        * expected_row["pred_conditional_fraction_clipped"]
    )
    assert np.isclose(expected_row["pred_hurdle_fraction"], expected_product)

    hard_zero = predictions.query(
        "composition_policy == 'hard_gate' and year == 2022 and aef_grid_cell_id == 1"
    ).iloc[0]
    assert hard_zero["calibrated_presence_probability"] < 0.4
    assert hard_zero["pred_hurdle_fraction"] == 0.0

    assert {"expected_value", "hard_gate"} <= set(metrics["composition_policy"])
    assert {"all", "assumed_background", "kelpwatch_station"} <= set(area["label_source"])
    assert {"ridge_regression", "calibrated_probability_x_conditional_canopy"} <= set(
        comparison["model_name"]
    )
    assert set(leakage["label_source"]) == {"assumed_background"}
    assert {"000_zero", "(90, 225]", "(810, 900]"} <= set(residuals["observed_bin"])
    assert manifest["command"] == "compose-hurdle-model"
    assert not manifest["refit_binary_presence_model"]
    assert not manifest["refit_binary_calibrator"]
    assert not manifest["refit_conditional_canopy_model"]
    assert manifest["row_counts"]["prediction_rows"] == 40


def test_compose_hurdle_model_writes_crm_stratified_sidecar(tmp_path: Path) -> None:
    """Verify compose-hurdle-model writes path-distinct CRM-stratified artifacts."""
    fixture = write_hurdle_fixture(tmp_path, include_sidecar=True)

    assert main(["compose-hurdle-model", "--config", str(fixture["config_path"])]) == 0

    predictions = pd.read_parquet(fixture["sidecar_predictions"])
    area = pd.read_csv(fixture["sidecar_area_calibration"])
    manifest = json.loads(fixture["sidecar_manifest"].read_text())

    assert set(predictions["composition_policy"]) == {"expected_value", "hard_gate"}
    assert set(predictions["presence_probability_threshold"]) == {0.35}
    assert {"all", "assumed_background", "kelpwatch_station"} <= set(area["label_source"])
    assert manifest["sample_policy"] == "crm_stratified_background_sample"
    assert manifest["presence_threshold"] == 0.35
    assert manifest["inputs"]["binary_full_grid_predictions"] == str(
        fixture["sidecar_binary_predictions"]
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    """Return PNG image height and width in pixels."""
    image = mpimg.imread(path)
    return int(image.shape[0]), int(image.shape[1])


def write_hurdle_fixture(tmp_path: Path, *, include_sidecar: bool = False) -> dict[str, Path]:
    """Write synthetic inputs and config for hurdle composition."""
    full_grid = tmp_path / "interim/aligned_full_grid_training_table.parquet"
    binary_predictions = tmp_path / "processed/binary_presence_full_grid_predictions.parquet"
    sidecar_binary_predictions = (
        tmp_path / "processed/binary_presence_full_grid_predictions.crm_stratified.parquet"
    )
    calibration_model = (
        tmp_path / "models/binary_presence/logistic_annual_max_ge_10pct_calibration.joblib"
    )
    sidecar_calibration_model = (
        tmp_path
        / "models/binary_presence/logistic_annual_max_ge_10pct_calibration.crm_stratified.joblib"
    )
    threshold_selection = (
        tmp_path / "reports/tables/binary_presence_calibrated_threshold_selection.csv"
    )
    sidecar_threshold_selection = (
        tmp_path
        / "reports/tables/binary_presence_calibrated_threshold_selection.crm_stratified.csv"
    )
    conditional_model = tmp_path / "models/conditional_canopy/ridge_positive_annual_max.joblib"
    reference_area = tmp_path / "reports/tables/reference_baseline_area_calibration.masked.csv"
    domain_mask = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    domain_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    predictions = tmp_path / "processed/hurdle_full_grid_predictions.parquet"
    sidecar_predictions = tmp_path / "processed/hurdle_full_grid_predictions.crm_stratified.parquet"
    manifest = tmp_path / "interim/hurdle_prediction_manifest.json"
    sidecar_manifest = tmp_path / "interim/hurdle_prediction_manifest.crm_stratified.json"
    metrics = tmp_path / "reports/tables/hurdle_metrics.csv"
    sidecar_metrics = tmp_path / "reports/tables/hurdle_metrics.crm_stratified.csv"
    area_calibration = tmp_path / "reports/tables/hurdle_area_calibration.csv"
    sidecar_area_calibration = (
        tmp_path / "reports/tables/hurdle_area_calibration.crm_stratified.csv"
    )
    model_comparison = tmp_path / "reports/tables/hurdle_model_comparison.csv"
    sidecar_model_comparison = (
        tmp_path / "reports/tables/hurdle_model_comparison.crm_stratified.csv"
    )
    residual_by_observed_bin = tmp_path / "reports/tables/hurdle_residual_by_observed_bin.csv"
    sidecar_residual_by_observed_bin = (
        tmp_path / "reports/tables/hurdle_residual_by_observed_bin.crm_stratified.csv"
    )
    assumed_background_leakage = tmp_path / "reports/tables/hurdle_assumed_background_leakage.csv"
    sidecar_assumed_background_leakage = (
        tmp_path / "reports/tables/hurdle_assumed_background_leakage.crm_stratified.csv"
    )
    map_figure = tmp_path / "reports/figures/hurdle_2022_observed_predicted_residual.png"
    sidecar_map_figure = (
        tmp_path / "reports/figures/hurdle_2022_observed_predicted_residual.crm_stratified.png"
    )
    config_path = tmp_path / "config.yaml"

    rows = hurdle_rows()
    full_grid.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(full_grid, index=False)
    write_binary_predictions(binary_predictions, rows)
    if include_sidecar:
        write_binary_predictions(sidecar_binary_predictions, rows)
    write_calibration_payload(calibration_model)
    if include_sidecar:
        write_calibration_payload(sidecar_calibration_model)
    write_threshold_selection(threshold_selection)
    if include_sidecar:
        write_threshold_selection(sidecar_threshold_selection, probability_threshold=0.35)
    write_conditional_payload(conditional_model)
    write_reference_area(reference_area)
    write_domain_mask(domain_mask, domain_manifest)
    sidecar_config = (
        f"""
    sidecars:
      crm_stratified:
        sample_policy: crm_stratified_background_sample
        binary_full_grid_predictions: {sidecar_binary_predictions}
        binary_calibration_model: {sidecar_calibration_model}
        binary_calibrated_threshold_selection: {sidecar_threshold_selection}
        reference_area_calibration: {reference_area}
        predictions: {sidecar_predictions}
        prediction_manifest: {sidecar_manifest}
        metrics: {sidecar_metrics}
        area_calibration: {sidecar_area_calibration}
        model_comparison: {sidecar_model_comparison}
        residual_by_observed_bin: {sidecar_residual_by_observed_bin}
        assumed_background_leakage: {sidecar_assumed_background_leakage}
        map_figure: {sidecar_map_figure}
"""
        if include_sidecar
        else ""
    )
    config_path.write_text(
        f"""
features:
  bands: A00-A01
splits:
  train_years: [2018, 2019, 2020]
  validation_years: [2021]
  test_years: [2022]
models:
  hurdle:
    inference_table: {full_grid}
    binary_full_grid_predictions: {binary_predictions}
    binary_calibration_model: {calibration_model}
    binary_calibrated_threshold_selection: {threshold_selection}
    conditional_model: {conditional_model}
    reference_area_calibration: {reference_area}
    model_name: calibrated_probability_x_conditional_canopy
    hard_gate_model_name: calibrated_hard_gate_conditional_canopy
    presence_model_name: logistic_annual_max_ge_10pct
    presence_probability_source: platt_calibrated
    presence_threshold_policy: validation_max_f1_calibrated
    presence_threshold: 0.40
    presence_target_label: annual_max_ge_10pct
    presence_target_threshold_fraction: 0.10
    conditional_model_name: ridge_positive_annual_max
    target: kelp_fraction_y
    target_area_column: kelp_max_y
    cell_area_m2: 900.0
    composition_policies: [expected_value, hard_gate]
    features: A00-A01
    batch_size: 8
    predictions: {predictions}
    prediction_manifest: {manifest}
    metrics: {metrics}
    area_calibration: {area_calibration}
    model_comparison: {model_comparison}
    residual_by_observed_bin: {residual_by_observed_bin}
    assumed_background_leakage: {assumed_background_leakage}
    map_figure: {map_figure}
{sidecar_config}
reports:
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
  model_analysis:
    year: 2022
""".lstrip()
    )
    return {
        "config_path": config_path,
        "predictions": predictions,
        "manifest": manifest,
        "metrics": metrics,
        "area_calibration": area_calibration,
        "model_comparison": model_comparison,
        "residual_by_observed_bin": residual_by_observed_bin,
        "assumed_background_leakage": assumed_background_leakage,
        "map_figure": map_figure,
        "sidecar_binary_predictions": sidecar_binary_predictions,
        "sidecar_predictions": sidecar_predictions,
        "sidecar_manifest": sidecar_manifest,
        "sidecar_area_calibration": sidecar_area_calibration,
    }


def hurdle_rows() -> list[dict[str, object]]:
    """Return tiny retained-domain full-grid rows for all configured years."""
    rows: list[dict[str, object]] = []
    fractions = (0.0, 0.04, 0.20, 0.95)
    for year in (2018, 2019, 2020, 2021, 2022):
        for cell_id, fraction in enumerate(fractions):
            rows.append(
                {
                    "year": year,
                    "aef_grid_cell_id": cell_id,
                    "aef_grid_row": cell_id,
                    "aef_grid_col": cell_id,
                    "kelpwatch_station_id": cell_id if cell_id else np.nan,
                    "longitude": -122.0 + cell_id * 0.01,
                    "latitude": 36.0 + cell_id * 0.01,
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "A00": fraction,
                    "A01": fraction * 0.5 + (year - 2018) * 0.001,
                    "label_source": "kelpwatch_station" if cell_id else "assumed_background",
                    "is_kelpwatch_observed": cell_id > 0,
                    "kelpwatch_station_count": 1 if cell_id else 0,
                }
            )
    return rows


def write_binary_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    """Write raw binary full-grid probabilities matching the synthetic rows."""
    probabilities = {0: 0.05, 1: 0.20, 2: 0.80, 3: 0.90}
    output = []
    for row in rows:
        probability = probabilities[int(row["aef_grid_cell_id"])]
        output.append(
            {
                "year": row["year"],
                "aef_grid_cell_id": row["aef_grid_cell_id"],
                "pred_binary_probability": probability,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_parquet(path, index=False)


def write_calibration_payload(path: Path) -> None:
    """Write a no-op Platt calibration payload for deterministic tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "calibrator": None,
            "model_name": "logistic_annual_max_ge_10pct",
            "calibration_method": "platt",
            "calibration_status": "test_noop",
            "coefficient": 1.0,
            "intercept": 0.0,
        },
        path,
    )


def write_threshold_selection(path: Path, *, probability_threshold: float = 0.4) -> None:
    """Write a selected calibrated threshold row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "probability_source": "platt_calibrated",
                "threshold_policy": "validation_max_f1_calibrated",
                "selected_threshold": True,
                "recommended_policy": True,
                "probability_threshold": probability_threshold,
            }
        ]
    ).to_csv(path, index=False)


def write_conditional_payload(path: Path) -> None:
    """Fit and write a tiny conditional ridge payload."""
    train = pd.DataFrame(hurdle_rows())
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=0.1)),
        ]
    )
    model.fit(train[["A00", "A01"]].to_numpy(dtype=float), train["kelp_fraction_y"].to_numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": "ridge_positive_annual_max",
            "feature_columns": ["A00", "A01"],
            "target_column": "kelp_fraction_y",
            "target_area_column": "kelp_max_y",
        },
        path,
    )


def write_reference_area(path: Path) -> None:
    """Write one ridge reference row for hurdle comparison output."""
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
                "row_count": 4,
                "mae": 0.1,
                "rmse": 0.2,
                "r2": 0.0,
                "f1_ge_10pct": 0.5,
                "observed_canopy_area": 1071.0,
                "predicted_canopy_area": 1200.0,
                "area_pct_bias": 0.12,
            }
        ]
    ).to_csv(path, index=False)


def write_domain_mask(path: Path, manifest_path: Path) -> None:
    """Write a retained-domain mask for all synthetic cells."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "aef_grid_cell_id": cell_id,
                "is_plausible_kelp_domain": True,
                "domain_mask_reason": "retained_shallow_depth",
                "domain_mask_detail": "test",
                "domain_mask_version": "test_v1",
            }
            for cell_id in range(4)
        ]
    ).to_parquet(path, index=False)
    manifest_path.write_text(json.dumps({"row_count": 4}))
