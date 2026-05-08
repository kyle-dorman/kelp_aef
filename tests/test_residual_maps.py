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


def write_residual_map_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write synthetic residual-map inputs and return configured artifact paths."""
    predictions = tmp_path / "processed/baseline_predictions.parquet"
    metrics = tmp_path / "reports/tables/baseline_metrics.csv"
    footprint = tmp_path / "geos/footprint.geojson"
    static_map = tmp_path / "reports/figures/ridge_2022_observed_predicted_residual.png"
    scatter = tmp_path / "reports/figures/ridge_observed_vs_predicted.png"
    interactive_html = tmp_path / "reports/figures/ridge_2022_residual_interactive.html"
    area_by_year = tmp_path / "reports/tables/area_bias_by_year.csv"
    area_by_latitude = tmp_path / "reports/tables/area_bias_by_latitude_band.csv"
    top_residuals = tmp_path / "reports/tables/top_residual_stations.csv"
    manifest = tmp_path / "interim/map_residuals_manifest.json"
    config_path = tmp_path / "config.yaml"
    write_prediction_table(predictions)
    write_metrics_table(metrics)
    write_footprint(footprint)
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
