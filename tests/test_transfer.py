import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from kelp_aef import main


def test_evaluate_transfer_writes_monterey_sidecars(tmp_path: Path) -> None:
    """Verify evaluate-transfer scores target rows with frozen source artifacts."""
    fixture = write_transfer_fixture(tmp_path)

    assert (
        main(
            [
                "evaluate-transfer",
                "--config",
                str(fixture["target_config"]),
                "--source-config",
                str(fixture["source_config"]),
            ]
        )
        == 0
    )

    baseline_predictions = pd.read_parquet(fixture["baseline_predictions"])
    binary_metrics = pd.read_csv(fixture["binary_metrics"])
    binary_area = pd.read_csv(fixture["binary_area"])
    hurdle_predictions = pd.read_parquet(fixture["hurdle_predictions"])
    comparison = pd.read_csv(fixture["model_comparison"])
    primary = pd.read_csv(fixture["primary_summary"])
    manifest = json.loads(fixture["manifest"].read_text())
    binary_manifest = json.loads(fixture["binary_manifest"].read_text())

    assert set(baseline_predictions["model_name"]) == {"ridge_regression"}
    assert set(binary_metrics["evaluation_scope"]) == {"full_grid_masked"}
    assert set(binary_area["probability_source"]) == {"platt_calibrated"}
    assert set(hurdle_predictions["composition_policy"]) == {"expected_value", "hard_gate"}
    assert set(comparison["training_regime"]) == {"monterey_transfer"}
    assert set(comparison["model_origin_region"]) == {"monterey"}
    assert {
        "ridge_regression",
        "previous_year_annual_max",
        "grid_cell_climatology",
        "calibrated_probability_x_conditional_canopy",
        "calibrated_hard_gate_conditional_canopy",
    } <= set(primary["model_name"])
    assert manifest["refit_aef_ridge_model"] is False
    assert manifest["refit_binary_presence_model"] is False
    assert manifest["refit_binary_calibrator"] is False
    assert manifest["refit_conditional_canopy_model"] is False
    assert binary_manifest["calibrated_probability_threshold"] == 0.4
    assert not fixture["target_ridge_model"].exists()
    assert not fixture["target_binary_model"].exists()
    assert not fixture["target_conditional_model"].exists()


def write_transfer_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write minimal source and target configs for transfer evaluation."""
    source_config = tmp_path / "source_config.yaml"
    target_config = tmp_path / "target_config.yaml"
    full_grid = tmp_path / "target/interim/big_sur_full_grid.parquet"
    target_sample = tmp_path / "target/interim/big_sur_sample.parquet"
    split_manifest = tmp_path / "target/interim/split_manifest.parquet"
    domain_mask = tmp_path / "target/interim/domain_mask.parquet"
    domain_manifest = tmp_path / "target/interim/domain_mask_manifest.json"
    source_ridge = tmp_path / "source/models/ridge.joblib"
    source_geographic = tmp_path / "source/models/geographic.joblib"
    source_binary = tmp_path / "source/models/binary.joblib"
    source_calibration = tmp_path / "source/models/binary_calibration.joblib"
    source_conditional = tmp_path / "source/models/conditional.joblib"
    source_thresholds = tmp_path / "source/reports/binary_thresholds.csv"
    target_ridge = tmp_path / "target/models/big_sur_ridge.joblib"
    target_binary = tmp_path / "target/models/big_sur_binary.joblib"
    target_conditional = tmp_path / "target/models/big_sur_conditional.joblib"
    outputs = transfer_output_paths(tmp_path)

    rows = target_rows()
    full_grid.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(full_grid, index=False)
    pd.DataFrame(rows).to_parquet(target_sample, index=False)
    write_split_manifest(split_manifest, rows)
    write_domain_mask(domain_mask, domain_manifest, rows)
    write_source_payloads(
        source_ridge,
        source_geographic,
        source_binary,
        source_calibration,
        source_conditional,
    )
    write_threshold_selection(source_thresholds)
    write_source_config(
        source_config,
        source_ridge,
        source_geographic,
        source_binary,
        source_calibration,
        source_conditional,
        source_thresholds,
    )
    write_target_config(
        target_config,
        full_grid,
        target_sample,
        split_manifest,
        domain_mask,
        domain_manifest,
        target_ridge,
        target_binary,
        target_conditional,
        outputs,
    )
    return {
        "source_config": source_config,
        "target_config": target_config,
        "target_ridge_model": target_ridge,
        "target_binary_model": target_binary,
        "target_conditional_model": target_conditional,
        **outputs,
    }


def target_rows() -> list[dict[str, object]]:
    """Return tiny full-grid target rows with stable cell history."""
    rows: list[dict[str, object]] = []
    fractions = {
        1: (0.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.05, 0.10, 0.20, 0.15, 0.25),
        3: (0.0, 0.02, 0.05, 0.12, 0.18),
        4: (0.80, 0.75, 0.70, 0.65, 0.60),
    }
    years = (2018, 2019, 2020, 2021, 2022)
    for cell_id, values in fractions.items():
        for index, year in enumerate(years):
            fraction = values[index]
            rows.append(
                {
                    "year": year,
                    "kelpwatch_station_id": cell_id,
                    "longitude": -121.9 + cell_id * 0.01,
                    "latitude": 36.0 + cell_id * 0.01,
                    "kelp_fraction_y": fraction,
                    "kelp_max_y": fraction * 900.0,
                    "A00": float(cell_id) / 4.0,
                    "A01": float(index) / 5.0,
                    "aef_grid_cell_id": cell_id,
                    "aef_grid_row": cell_id,
                    "aef_grid_col": cell_id + 10,
                    "label_source": (
                        "kelpwatch_station" if cell_id in {2, 4} else "assumed_background"
                    ),
                    "is_kelpwatch_observed": cell_id in {2, 4},
                    "kelpwatch_station_count": 1 if cell_id in {2, 4} else 0,
                }
            )
    return rows


def write_split_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a split manifest matching the target rows."""
    split_by_year = {2018: "train", 2019: "train", 2020: "train", 2021: "validation", 2022: "test"}
    frame = pd.DataFrame(rows)
    manifest = frame[
        [
            "year",
            "aef_grid_cell_id",
            "aef_grid_row",
            "aef_grid_col",
            "kelpwatch_station_id",
            "longitude",
            "latitude",
            "kelp_fraction_y",
            "label_source",
        ]
    ].copy()
    manifest["split"] = manifest["year"].map(split_by_year)
    manifest["has_complete_features"] = True
    manifest["has_target"] = True
    manifest["used_for_training_eval"] = True
    manifest["drop_reason"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)


def write_domain_mask(path: Path, manifest_path: Path, rows: list[dict[str, object]]) -> None:
    """Write a retained-domain mask for all target cells."""
    cells = sorted({int(str(row["aef_grid_cell_id"])) for row in rows})
    frame = pd.DataFrame(
        {
            "aef_grid_cell_id": cells,
            "is_plausible_kelp_domain": [True for _ in cells],
            "domain_mask_reason": ["retained_depth_0_60m" for _ in cells],
            "domain_mask_detail": ["test" for _ in cells],
            "domain_mask_version": ["fixture" for _ in cells],
            "crm_elevation_m": [-5.0 for _ in cells],
            "crm_depth_m": [5.0 for _ in cells],
            "depth_bin": ["0_40m" for _ in cells],
            "elevation_bin": ["below_msl" for _ in cells],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manifest_path.write_text('{"mask": "fixture"}\n')


def write_source_payloads(
    ridge_path: Path,
    geographic_path: Path,
    binary_path: Path,
    calibration_path: Path,
    conditional_path: Path,
) -> None:
    """Write frozen source model payloads used by the transfer command."""
    features = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.5, 0.5],
            [0.8, 0.9],
            [1.0, 1.0],
            [0.9, 0.4],
        ]
    )
    target = np.array([0.0, 0.05, 0.2, 0.7, 0.9, 0.6])
    binary_target = target >= 0.10
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.1))])
    ridge.fit(features, target)
    geographic = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.1))])
    geographic.fit(
        np.array(
            [
                [-121.9, 36.0, 2018],
                [-121.8, 36.1, 2019],
                [-121.7, 36.2, 2020],
                [-121.6, 36.3, 2021],
                [-121.5, 36.4, 2022],
                [-121.4, 36.5, 2022],
            ]
        ),
        target,
    )
    binary = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ]
    )
    binary.fit(features, binary_target)
    conditional = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=0.1))])
    conditional.fit(features[binary_target], target[binary_target])
    ridge_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": ridge,
            "model_name": "ridge_regression",
            "feature_columns": ["A00", "A01"],
            "target_column": "kelp_fraction_y",
            "selected_alpha": 0.1,
            "train_mean": float(target.mean()),
        },
        ridge_path,
    )
    joblib.dump(
        {
            "model": geographic,
            "model_name": "geographic_ridge_lon_lat_year",
            "feature_columns": ["longitude", "latitude", "year"],
            "target_column": "kelp_fraction_y",
            "selected_alpha": 0.1,
        },
        geographic_path,
    )
    joblib.dump(
        {
            "model": binary,
            "model_name": "logistic_annual_max_ge_10pct",
            "feature_columns": ["A00", "A01"],
            "target_label": "annual_max_ge_10pct",
            "target_column": "kelp_fraction_y",
            "target_threshold_fraction": 0.10,
            "target_threshold_area": 90.0,
            "class_weight": "balanced",
            "selected_c": 1.0,
            "probability_threshold": 0.3,
        },
        binary_path,
    )
    joblib.dump(
        {
            "calibrator": None,
            "model_name": "logistic_annual_max_ge_10pct",
            "calibration_method": "platt",
            "calibration_status": "identity_fixture",
            "recommended_threshold": 0.4,
            "recommended_policy": "validation_max_f1_calibrated",
            "coefficient": np.nan,
            "intercept": np.nan,
        },
        calibration_path,
    )
    joblib.dump(
        {
            "model": conditional,
            "model_name": "ridge_positive_annual_max",
            "model_family": "conditional_canopy",
            "selected_alpha": 0.1,
            "feature_columns": ["A00", "A01"],
            "target_column": "kelp_fraction_y",
            "target_area_column": "kelp_max_y",
        },
        conditional_path,
    )


def write_threshold_selection(path: Path) -> None:
    """Write the frozen source calibrated threshold table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "logistic_annual_max_ge_10pct",
                "target_label": "annual_max_ge_10pct",
                "target_threshold_fraction": 0.10,
                "target_threshold_area": 90.0,
                "calibration_method": "platt",
                "probability_source": "platt_calibrated",
                "calibration_split": "validation",
                "calibration_year": 2021,
                "threshold_policy": "validation_max_f1_calibrated",
                "selection_status": "fixture",
                "recommended_policy": True,
                "selected_threshold": True,
                "probability_threshold": 0.4,
            }
        ]
    ).to_csv(path, index=False)


def transfer_output_paths(tmp_path: Path) -> dict[str, Path]:
    """Return transfer output paths keyed by test assertions."""
    return {
        "baseline_predictions": tmp_path / "target/processed/transfer_baseline.parquet",
        "baseline_manifest": tmp_path / "target/interim/transfer_baseline_manifest.json",
        "binary_predictions": tmp_path / "target/processed/transfer_binary.parquet",
        "binary_manifest": tmp_path / "target/interim/transfer_binary_manifest.json",
        "binary_metrics": tmp_path / "target/reports/transfer_binary_metrics.csv",
        "binary_area": tmp_path / "target/reports/transfer_binary_area.csv",
        "hurdle_predictions": tmp_path / "target/processed/transfer_hurdle.parquet",
        "hurdle_manifest": tmp_path / "target/interim/transfer_hurdle_manifest.json",
        "hurdle_metrics": tmp_path / "target/reports/transfer_hurdle_metrics.csv",
        "hurdle_area": tmp_path / "target/reports/transfer_hurdle_area.csv",
        "hurdle_comparison": tmp_path / "target/reports/transfer_hurdle_comparison.csv",
        "hurdle_residuals": tmp_path / "target/reports/transfer_hurdle_residuals.csv",
        "hurdle_leakage": tmp_path / "target/reports/transfer_hurdle_leakage.csv",
        "hurdle_map": tmp_path / "target/reports/transfer_hurdle_map.png",
        "reference_area": tmp_path / "target/reports/transfer_reference_area.csv",
        "model_comparison": tmp_path / "target/reports/transfer_model_comparison.csv",
        "primary_summary": tmp_path / "target/reports/transfer_primary_summary.csv",
        "manifest": tmp_path / "target/interim/transfer_manifest.json",
    }


def write_source_config(
    path: Path,
    ridge: Path,
    geographic: Path,
    binary: Path,
    calibration: Path,
    conditional: Path,
    thresholds: Path,
) -> None:
    """Write a minimal Monterey-source config with frozen model paths."""
    path.write_text(
        f"""
region:
  name: monterey_peninsula
features:
  bands: A00-A01
alignment:
  output_table: {path.parent / "unused_source_sample.parquet"}
splits:
  train_years: [2018, 2019, 2020]
  validation_years: [2021]
  test_years: [2022]
  output_manifest: {path.parent / "source_split.parquet"}
models:
  baselines:
    input_table: {path.parent / "unused_source_sample.parquet"}
    inference_table: {path.parent / "unused_source_full_grid.parquet"}
    target: kelp_fraction_y
    features: A00-A01
    alpha_grid: [0.1]
    ridge_model: {ridge}
    geographic_model: {geographic}
    sample_predictions: {path.parent / "unused_source_baseline_sample.parquet"}
    predictions: {path.parent / "unused_source_baseline_full_grid.parquet"}
    prediction_manifest: {path.parent / "unused_source_baseline_manifest.json"}
    metrics: {path.parent / "unused_source_baseline_metrics.csv"}
    manifest: {path.parent / "unused_source_baseline_eval.json"}
  binary_presence:
    input_table: {path.parent / "unused_source_sample.parquet"}
    inference_table: {path.parent / "unused_source_full_grid.parquet"}
    target_label: annual_max_ge_10pct
    target_column: kelp_fraction_y
    target_threshold_fraction: 0.10
    target_threshold_area: 90.0
    features: A00-A01
    class_weight: balanced
    c_grid: [1.0]
    model: {binary}
    sample_predictions: {path.parent / "unused_source_binary_sample.parquet"}
    full_grid_predictions: {path.parent / "unused_source_binary_full_grid.parquet"}
    metrics: {path.parent / "unused_source_binary_metrics.csv"}
    threshold_selection: {path.parent / "unused_source_binary_thresholds.csv"}
    full_grid_area_summary: {path.parent / "unused_source_binary_area.csv"}
    thresholded_model_comparison: {path.parent / "unused_source_binary_comparison.csv"}
    prediction_manifest: {path.parent / "unused_source_binary_manifest.json"}
    precision_recall_figure: {path.parent / "unused_source_binary_pr.png"}
    map_figure: {path.parent / "unused_source_binary_map.png"}
    calibration:
      method: platt
      calibration_split: validation
      calibration_year: 2021
      evaluation_split: test
      evaluation_year: 2022
      input_sample_predictions: {path.parent / "unused_source_binary_sample.parquet"}
      input_full_grid_predictions: {path.parent / "unused_source_binary_full_grid.parquet"}
      model: {calibration}
      calibrated_sample_predictions: {path.parent / "unused_source_binary_cal_sample.parquet"}
      metrics: {path.parent / "unused_source_binary_cal_metrics.csv"}
      threshold_selection: {thresholds}
      full_grid_area_summary: {path.parent / "unused_source_binary_cal_area.csv"}
      calibration_curve_figure: {path.parent / "unused_source_binary_cal_curve.png"}
      threshold_figure: {path.parent / "unused_source_binary_cal_thresholds.png"}
      manifest: {path.parent / "unused_source_binary_cal_manifest.json"}
  conditional_canopy:
    input_table: {path.parent / "unused_source_sample.parquet"}
    target: kelp_fraction_y
    target_area_column: kelp_max_y
    features: A00-A01
    model: {conditional}
    sample_predictions: {path.parent / "unused_source_conditional_sample.parquet"}
    metrics: {path.parent / "unused_source_conditional_metrics.csv"}
    positive_residuals: {path.parent / "unused_source_conditional_residuals.csv"}
    model_comparison: {path.parent / "unused_source_conditional_comparison.csv"}
    full_grid_likely_positive_summary: {path.parent / "unused_source_conditional_likely.csv"}
    residual_figure: {path.parent / "unused_source_conditional_residual.png"}
    manifest: {path.parent / "unused_source_conditional_manifest.json"}
  hurdle:
    inference_table: {path.parent / "unused_source_full_grid.parquet"}
    binary_full_grid_predictions: {path.parent / "unused_source_binary_full_grid.parquet"}
    binary_calibration_model: {calibration}
    binary_calibrated_threshold_selection: {thresholds}
    conditional_model: {conditional}
    reference_area_calibration: {path.parent / "unused_source_reference_area.csv"}
    presence_threshold_policy: validation_max_f1_calibrated
    presence_threshold: 0.4
    features: A00-A01
    predictions: {path.parent / "unused_source_hurdle.parquet"}
    prediction_manifest: {path.parent / "unused_source_hurdle_manifest.json"}
    metrics: {path.parent / "unused_source_hurdle_metrics.csv"}
    area_calibration: {path.parent / "unused_source_hurdle_area.csv"}
    model_comparison: {path.parent / "unused_source_hurdle_comparison.csv"}
    residual_by_observed_bin: {path.parent / "unused_source_hurdle_residuals.csv"}
    assumed_background_leakage: {path.parent / "unused_source_hurdle_leakage.csv"}
reports:
  outputs:
    reference_baseline_fallback_summary: {path.parent / "unused_source_fallback.csv"}
    reference_baseline_area_calibration: {path.parent / "unused_source_reference_area.csv"}
""".lstrip()
    )


def write_target_config(
    path: Path,
    full_grid: Path,
    sample: Path,
    split_manifest: Path,
    domain_mask: Path,
    domain_manifest: Path,
    target_ridge: Path,
    target_binary: Path,
    target_conditional: Path,
    outputs: dict[str, Path],
) -> None:
    """Write a minimal Big Sur target config with transfer sidecar paths."""
    path.write_text(
        f"""
region:
  name: big_sur
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
    input_table: {sample}
    inference_table: {full_grid}
    target: kelp_fraction_y
    features: A00-A01
    alpha_grid: [0.1]
    ridge_model: {target_ridge}
    geographic_model: {path.parent / "target_geographic.joblib"}
    sample_predictions: {path.parent / "target_baseline_sample.parquet"}
    predictions: {path.parent / "target_baseline_full_grid.parquet"}
    prediction_manifest: {path.parent / "target_baseline_manifest.json"}
    metrics: {path.parent / "target_baseline_metrics.csv"}
    manifest: {path.parent / "target_baseline_eval.json"}
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
    model: {target_binary}
    sample_predictions: {path.parent / "target_binary_sample.parquet"}
    full_grid_predictions: {path.parent / "target_binary_full_grid.parquet"}
    metrics: {path.parent / "target_binary_metrics.csv"}
    threshold_selection: {path.parent / "target_binary_thresholds.csv"}
    full_grid_area_summary: {path.parent / "target_binary_area.csv"}
    thresholded_model_comparison: {path.parent / "target_binary_comparison.csv"}
    prediction_manifest: {path.parent / "target_binary_manifest.json"}
    precision_recall_figure: {path.parent / "target_binary_pr.png"}
    map_figure: {path.parent / "target_binary_map.png"}
    calibration:
      method: platt
      calibration_split: validation
      calibration_year: 2021
      evaluation_split: test
      evaluation_year: 2022
      input_sample_predictions: {path.parent / "target_binary_sample.parquet"}
      input_full_grid_predictions: {path.parent / "target_binary_full_grid.parquet"}
      model: {path.parent / "target_binary_calibration.joblib"}
      calibrated_sample_predictions: {path.parent / "target_binary_cal_sample.parquet"}
      metrics: {path.parent / "target_binary_cal_metrics.csv"}
      threshold_selection: {path.parent / "target_binary_cal_thresholds.csv"}
      full_grid_area_summary: {path.parent / "target_binary_cal_area.csv"}
      calibration_curve_figure: {path.parent / "target_binary_cal_curve.png"}
      threshold_figure: {path.parent / "target_binary_cal_thresholds.png"}
      manifest: {path.parent / "target_binary_cal_manifest.json"}
  conditional_canopy:
    input_table: {sample}
    target: kelp_fraction_y
    target_area_column: kelp_max_y
    features: A00-A01
    model: {target_conditional}
    sample_predictions: {path.parent / "target_conditional_sample.parquet"}
    metrics: {path.parent / "target_conditional_metrics.csv"}
    positive_residuals: {path.parent / "target_conditional_residuals.csv"}
    model_comparison: {path.parent / "target_conditional_comparison.csv"}
    full_grid_likely_positive_summary: {path.parent / "target_conditional_likely.csv"}
    residual_figure: {path.parent / "target_conditional_residual.png"}
    manifest: {path.parent / "target_conditional_manifest.json"}
  hurdle:
    inference_table: {full_grid}
    binary_full_grid_predictions: {path.parent / "target_binary_full_grid.parquet"}
    binary_calibration_model: {path.parent / "target_binary_calibration.joblib"}
    binary_calibrated_threshold_selection: {path.parent / "target_binary_cal_thresholds.csv"}
    conditional_model: {target_conditional}
    reference_area_calibration: {outputs["reference_area"]}
    presence_threshold_policy: validation_max_f1_calibrated
    presence_threshold: 0.4
    features: A00-A01
    predictions: {path.parent / "target_hurdle.parquet"}
    prediction_manifest: {path.parent / "target_hurdle_manifest.json"}
    metrics: {path.parent / "target_hurdle_metrics.csv"}
    area_calibration: {path.parent / "target_hurdle_area.csv"}
    model_comparison: {path.parent / "target_hurdle_comparison.csv"}
    residual_by_observed_bin: {path.parent / "target_hurdle_residuals.csv"}
    assumed_background_leakage: {path.parent / "target_hurdle_leakage.csv"}
  transfer:
    monterey:
      training_regime: monterey_transfer
      model_origin_region: monterey
      baseline_full_grid_predictions: {outputs["baseline_predictions"]}
      baseline_prediction_manifest: {outputs["baseline_manifest"]}
      binary_full_grid_predictions: {outputs["binary_predictions"]}
      binary_prediction_manifest: {outputs["binary_manifest"]}
      binary_metrics: {outputs["binary_metrics"]}
      binary_full_grid_area_summary: {outputs["binary_area"]}
      hurdle_full_grid_predictions: {outputs["hurdle_predictions"]}
      hurdle_prediction_manifest: {outputs["hurdle_manifest"]}
      hurdle_metrics: {outputs["hurdle_metrics"]}
      hurdle_area_calibration: {outputs["hurdle_area"]}
      hurdle_model_comparison: {outputs["hurdle_comparison"]}
      hurdle_residual_by_observed_bin: {outputs["hurdle_residuals"]}
      hurdle_assumed_background_leakage: {outputs["hurdle_leakage"]}
      hurdle_map_figure: {outputs["hurdle_map"]}
      reference_area_calibration: {outputs["reference_area"]}
      model_comparison: {outputs["model_comparison"]}
      primary_summary: {outputs["primary_summary"]}
      manifest: {outputs["manifest"]}
reports:
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
  outputs:
    reference_baseline_fallback_summary: {path.parent / "target_fallback.csv"}
    reference_baseline_area_calibration: {path.parent / "target_reference_area.csv"}
    reference_baseline_area_calibration_masked: {outputs["reference_area"]}
  model_analysis:
    year: 2022
""".lstrip()
    )
