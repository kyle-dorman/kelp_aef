import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main


def test_map_residuals_writes_exploration_artifacts(tmp_path: Path) -> None:
    """Verify map-residuals writes maps, HTML, summary tables, and a manifest."""
    fixture = write_residual_map_fixture(tmp_path)

    assert main(["map-residuals", "--config", str(fixture["config_path"])]) == 0

    for key in (
        "static_map",
        "scatter",
        "interactive_html",
        "area_by_year",
        "area_by_latitude",
        "top_residuals",
        "manifest",
    ):
        assert fixture[key].is_file()

    area_by_year = pd.read_csv(fixture["area_by_year"])
    test_row = area_by_year.query(
        "model_name == 'ridge_regression' and split == 'test' and year == 2022"
    ).iloc[0]
    assert int(test_row["row_count"]) == 3
    assert np.isclose(float(test_row["observed_canopy_area"]), 630.0)
    assert np.isclose(float(test_row["predicted_canopy_area"]), 630.0)
    assert np.isclose(float(test_row["area_bias"]), 0.0)

    area_by_latitude = pd.read_csv(fixture["area_by_latitude"], dtype={"latitude_band": str})
    test_bands = area_by_latitude.query("split == 'test' and year == 2022")
    assert set(test_bands["latitude_band"]) == {"00", "01"}

    top_residuals = pd.read_csv(fixture["top_residuals"])
    assert set(top_residuals["residual_type"]) == {"underprediction", "overprediction"}
    underprediction = top_residuals.query("residual_type == 'underprediction'").iloc[0]
    overprediction = top_residuals.query("residual_type == 'overprediction'").iloc[0]
    assert np.isclose(float(underprediction["residual_kelp_max_y"]), 90.0)
    assert np.isclose(float(overprediction["residual_kelp_max_y"]), -90.0)

    html = fixture["interactive_html"].read_text()
    manifest = json.loads(fixture["manifest"].read_text())
    assert "Kelpwatch labels, ridge predictions, and residuals" in html
    assert "residual = observed - predicted" in html
    assert manifest["command"] == "map-residuals"
    assert manifest["map_row_count"] == 3
    assert manifest["residual_sign"] == "observed minus predicted"


def test_map_residuals_filters_full_grid_rows_to_domain_mask(tmp_path: Path) -> None:
    """Verify map-residuals uses the plausible-kelp domain as the reporting scope."""
    fixture = write_residual_map_fixture(tmp_path, include_domain_mask=True)

    assert main(["map-residuals", "--config", str(fixture["config_path"])]) == 0

    area_by_year = pd.read_csv(fixture["area_by_year"])
    test_row = area_by_year.query(
        "model_name == 'ridge_regression' and split == 'test' and year == 2022"
    ).iloc[0]
    assert int(test_row["row_count"]) == 2
    assert set(area_by_year["mask_status"]) == {"plausible_kelp_domain"}
    assert set(area_by_year["evaluation_scope"]) == {"full_grid_masked"}

    manifest = json.loads(fixture["manifest"].read_text())
    audit = pd.read_csv(fixture["off_domain_audit"])
    assert manifest["map_row_count"] == 2
    assert manifest["mask_status"] == "plausible_kelp_domain"
    assert set(audit["domain_mask_reason"]) == {"dropped_too_deep"}


def write_residual_map_fixture(
    tmp_path: Path, *, include_domain_mask: bool = False
) -> dict[str, Path]:
    """Write synthetic residual-map inputs and return configured artifact paths."""
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    footprint = tmp_path / "geos/footprint.geojson"
    suffix = ".masked" if include_domain_mask else ""
    static_map = tmp_path / f"reports/figures/ridge_2022_observed_predicted_residual{suffix}.png"
    scatter = tmp_path / f"reports/figures/ridge_observed_vs_predicted{suffix}.png"
    interactive_html = tmp_path / f"reports/figures/ridge_2022_residual_interactive{suffix}.html"
    area_by_year = tmp_path / f"reports/tables/area_bias_by_year{suffix}.csv"
    area_by_latitude = tmp_path / f"reports/tables/area_bias_by_latitude_band{suffix}.csv"
    top_residuals = tmp_path / "reports/tables/top_residual_stations.csv"
    domain_mask = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    domain_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    off_domain_audit = tmp_path / "reports/tables/off_domain_prediction_leakage_audit.csv"
    manifest = tmp_path / "interim/map_residuals_manifest.json"
    config_path = tmp_path / "config.yaml"
    write_prediction_table(predictions)
    write_metrics_table(metrics)
    write_footprint(footprint)
    if include_domain_mask:
        write_domain_mask(domain_mask, domain_manifest)
    domain_mask_config = (
        f"""
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {domain_mask}
    mask_manifest: {domain_manifest}
    off_domain_audit_table: {off_domain_audit}
"""
        if include_domain_mask
        else ""
    )
    masked_outputs = (
        f"""
    ridge_observed_predicted_residual_map_masked: {static_map}
    ridge_observed_vs_predicted_masked: {scatter}
    ridge_residual_interactive_masked: {interactive_html}
    area_bias_by_year_masked: {area_by_year}
    area_bias_by_latitude_band_masked: {area_by_latitude}
"""
        if include_domain_mask
        else ""
    )
    config_path.write_text(
        f"""
data_root: {tmp_path}
region:
  geometry:
    path: {footprint}
models:
  baselines:
    predictions: {predictions}
    metrics: {metrics}
reports:
  figures_dir: {tmp_path / "reports/figures"}
  tables_dir: {tmp_path / "reports/tables"}
{domain_mask_config}
  map_residuals:
    model_name: ridge_regression
    split: test
    year: 2022
    latitude_band_count: 2
    top_residual_count: 1
  outputs:
    ridge_observed_predicted_residual_map: {static_map}
    ridge_observed_vs_predicted: {scatter}
    ridge_residual_interactive: {interactive_html}
    area_bias_by_year: {area_by_year}
    area_bias_by_latitude_band: {area_by_latitude}
{masked_outputs}
    top_residual_stations: {top_residuals}
    map_residuals_manifest: {manifest}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "static_map": static_map,
        "scatter": scatter,
        "interactive_html": interactive_html,
        "area_by_year": area_by_year,
        "area_by_latitude": area_by_latitude,
        "top_residuals": top_residuals,
        "off_domain_audit": off_domain_audit,
        "manifest": manifest,
    }


def write_prediction_table(path: Path) -> None:
    """Write a tiny baseline prediction table with both residual signs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        prediction_row(2020, "train", 1, 0.2, 0.2, -122.004, 36.000),
        prediction_row(2020, "train", 2, 0.4, 0.35, -122.003, 36.001),
        prediction_row(2021, "validation", 3, 0.3, 0.35, -122.002, 36.002),
        prediction_row(2021, "validation", 4, 0.5, 0.45, -122.001, 36.003),
        prediction_row(2022, "test", 5, 0.5, 0.4, -122.000, 36.000),
        prediction_row(2022, "test", 6, 0.2, 0.3, -121.999, 36.005),
        prediction_row(2022, "test", 7, 0.0, 0.0, -121.998, 36.006),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def prediction_row(
    year: int,
    split: str,
    station_id: int,
    observed_fraction: float,
    predicted_fraction: float,
    longitude: float,
    latitude: float,
) -> dict[str, object]:
    """Build one prediction row with area columns derived from 900 square meters."""
    observed_area = observed_fraction * 900.0
    predicted_area = predicted_fraction * 900.0
    return {
        "year": year,
        "split": split,
        "kelpwatch_station_id": station_id,
        "aef_grid_cell_id": station_id,
        "aef_grid_row": station_id,
        "aef_grid_col": station_id,
        "longitude": longitude,
        "latitude": latitude,
        "kelp_fraction_y": observed_fraction,
        "kelp_max_y": observed_area,
        "model_name": "ridge_regression",
        "alpha": 1.0,
        "pred_kelp_fraction_y": predicted_fraction,
        "pred_kelp_fraction_y_clipped": predicted_fraction,
        "pred_kelp_max_y": predicted_area,
        "residual_kelp_fraction_y": observed_fraction - predicted_fraction,
        "residual_kelp_fraction_y_clipped": observed_fraction - predicted_fraction,
        "residual_kelp_max_y": observed_area - predicted_area,
    }


def write_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a tiny mask that drops one selected test prediction row."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [1, 2, 3, 4, 5, 6, 7],
            "is_plausible_kelp_domain": [True, False, True, False, True, False, True],
            "domain_mask_reason": [
                "retained",
                "dropped_too_deep",
                "retained",
                "dropped_too_deep",
                "retained",
                "dropped_too_deep",
                "retained",
            ],
            "domain_mask_detail": ["fixture"] * 7,
            "domain_mask_version": ["test_mask_v1"] * 7,
        }
    ).to_parquet(mask_path, index=False)
    manifest_path.write_text(json.dumps({"mask_version": "test_mask_v1"}))


def write_metrics_table(path: Path) -> None:
    """Write a placeholder metrics CSV because the manifest records its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model_name": "ridge_regression",
                "split": "test",
                "year": 2022,
                "rmse": 0.0,
            }
        ]
    ).to_csv(path, index=False)


def write_footprint(path: Path) -> None:
    """Write a small WGS84 polygon footprint for map outlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-122.01, 35.99],
                            [-121.99, 35.99],
                            [-121.99, 36.01],
                            [-122.01, 36.01],
                            [-122.01, 35.99],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload))
