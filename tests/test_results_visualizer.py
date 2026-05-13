import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main


def test_visualize_results_writes_leaflet_viewer_assets(tmp_path: Path) -> None:
    """Verify visualize-results writes HTML, overlays, inspection points, and manifest."""
    fixture = write_results_visualizer_fixture(tmp_path)

    assert main(["visualize-results", "--config", str(fixture["config_path"])]) == 0

    html_path = fixture["html"]
    asset_dir = fixture["asset_dir"]
    manifest_path = fixture["manifest"]
    inspection_csv = fixture["inspection_csv"]
    assert html_path.is_file()
    assert manifest_path.is_file()
    assert inspection_csv.is_file()
    for filename in ("inspection_points.geojson", "inspection_points.js"):
        assert (asset_dir / filename).is_file()
    assert not list(asset_dir.glob("*.png"))
    assert not list(asset_dir.glob("*.tif"))

    html = html_path.read_text()
    assert "L.control.layers" in html
    assert 'position: "topleft"' in html
    assert "OpenStreetMap" in html
    assert "Expected-value hurdle prediction" in html
    assert "Expected-value hurdle residual" in html
    assert "Conditional ridge prediction" in html
    assert "Binary presence probability" in html
    assert "Binary outcome TP/FP/FN/TN" in html
    assert "Data layer" in html
    assert 'input.type = "radio"' in html
    assert "Label source" not in html
    assert "Mask reason" not in html
    assert "Hurdle pred m2" in html
    assert "Hurdle resid m2" in html
    assert "Cond ridge m2" in html
    assert "Binary prob" in html
    assert "Binary outcome" in html
    assert "L.imageOverlay" not in html
    assert "pointLayer.propertyName" in html
    assert "opacity-controls" not in html
    assert "coordinate-based hurdle" in html
    assert "Copy lat/lon" in html

    inspection = pd.read_csv(inspection_csv)
    manifest = json.loads(manifest_path.read_text())
    assert len(inspection) == 2
    assert {
        "longitude",
        "latitude",
        "expected_value_hurdle_prediction_m2",
        "conditional_ridge_prediction_m2",
        "binary_presence_probability",
        "binary_outcome_outcome",
    } <= set(inspection.columns)
    assert set(inspection["binary_outcome_outcome"]) == {"TP", "TN"}
    assert manifest["command"] == "visualize-results"
    assert manifest["mask_status"] == "plausible_kelp_domain"
    assert manifest["inspection"]["row_count"] == 2
    assert manifest["basemap"]["runtime_dependency_only"] is True
    assert {row["layer_id"] for row in manifest["layers"]} == {
        "expected_value_hurdle",
        "conditional_ridge",
        "binary_presence",
        "binary_outcome",
    }
    assert {row["layer_id"] for row in manifest["point_layers"]} == {
        "expected_value_hurdle_prediction",
        "expected_value_hurdle_residual",
        "conditional_ridge_prediction",
        "binary_presence_probability",
        "binary_outcome_outcome",
    }
    assert all(row["coordinate_based"] for row in manifest["point_layers"])


def write_results_visualizer_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write tiny synthetic visualizer inputs and return expected artifact paths."""
    hurdle_predictions = tmp_path / "processed/hurdle_full_grid_predictions.parquet"
    mask_path = tmp_path / "interim/plausible_kelp_domain_mask.parquet"
    mask_manifest = tmp_path / "interim/plausible_kelp_domain_mask_manifest.json"
    footprint = tmp_path / "geos/footprint.geojson"
    html_path = tmp_path / "reports/interactive/monterey_results_visualizer.html"
    asset_dir = tmp_path / "reports/interactive/monterey_results_visualizer"
    manifest_path = tmp_path / "interim/results_visualizer_manifest.json"
    inspection_csv = tmp_path / "reports/tables/results_visualizer_inspection_points.csv"
    config_path = tmp_path / "config.yaml"
    write_continuous_predictions(
        hurdle_predictions,
        model_name="calibrated_probability_x_conditional_canopy",
        predictions=(360.0, 360.0, 0.0),
    )
    write_domain_mask(mask_path, mask_manifest)
    write_footprint(footprint)
    config_path.write_text(
        f"""
data_root: {tmp_path}
region:
  geometry:
    path: {footprint}
models: {{}}
reports:
  figures_dir: {tmp_path / "reports/figures"}
  tables_dir: {tmp_path / "reports/tables"}
  domain_mask:
    primary_full_grid_domain: plausible_kelp_domain
    mask_status: plausible_kelp_domain
    evaluation_scope: full_grid_masked
    mask_table: {mask_path}
    mask_manifest: {mask_manifest}
  results_visualizer:
    split: test
    year: 2022
    max_inspection_points: 2
    basemap:
      enabled: true
      name: OpenStreetMap
      url_template: https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png
      attribution: OSM fixture
      max_zoom: 19
    layers:
      - id: expected_value_hurdle
        display_name: Expected-value hurdle
        type: continuous
        path: {hurdle_predictions}
        model_name: calibrated_probability_x_conditional_canopy
        prediction_column: pred_kelp_max_y
        residual_column: residual_kelp_max_y
        default_visible: true
      - id: conditional_ridge
        display_name: Conditional ridge
        type: continuous_prediction
        path: {hurdle_predictions}
        model_name: calibrated_probability_x_conditional_canopy
        prediction_column: pred_conditional_area_m2
        default_visible: false
      - id: binary_presence
        display_name: Binary presence
        type: binary_probability
        path: {hurdle_predictions}
        model_name: calibrated_probability_x_conditional_canopy
        probability_column: calibrated_presence_probability
        default_visible: false
      - id: binary_outcome
        display_name: Binary outcome
        type: binary_outcome
        path: {hurdle_predictions}
        model_name: calibrated_probability_x_conditional_canopy
        prediction_column: pred_presence_class
        probability_column: calibrated_presence_probability
        default_visible: false
  outputs:
    results_visualizer_html: {html_path}
    results_visualizer_asset_dir: {asset_dir}
    results_visualizer_manifest: {manifest_path}
    results_visualizer_inspection_points: {inspection_csv}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "html": html_path,
        "asset_dir": asset_dir,
        "manifest": manifest_path,
        "inspection_csv": inspection_csv,
    }


def write_continuous_predictions(
    path: Path,
    *,
    model_name: str,
    predictions: tuple[float, float, float],
) -> None:
    """Write a tiny continuous prediction table for visualizer tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        continuous_row(5, 10, 20, 450.0, predictions[0], -122.000, 36.000, model_name),
        continuous_row(6, 11, 21, 180.0, predictions[1], -121.999, 36.001, model_name),
        continuous_row(7, 12, 22, 0.0, predictions[2], -121.998, 36.002, model_name),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def continuous_row(
    cell_id: int,
    row: int,
    col: int,
    observed_area: float,
    predicted_area: float,
    longitude: float,
    latitude: float,
    model_name: str,
) -> dict[str, object]:
    """Return one continuous model prediction row."""
    return {
        "year": 2022,
        "split": "test",
        "kelpwatch_station_id": np.nan,
        "longitude": longitude,
        "latitude": latitude,
        "kelp_fraction_y": observed_area / 900.0,
        "kelp_max_y": observed_area,
        "aef_grid_cell_id": cell_id,
        "aef_grid_row": row,
        "aef_grid_col": col,
        "label_source": "kelpwatch_station" if observed_area > 0 else "assumed_background",
        "is_kelpwatch_observed": observed_area > 0,
        "kelpwatch_station_count": 1 if observed_area > 0 else 0,
        "model_name": model_name,
        "pred_kelp_max_y": predicted_area,
        "residual_kelp_max_y": observed_area - predicted_area,
        "pred_conditional_area_m2": 900.0 if predicted_area > 0 else 0.0,
        "calibrated_presence_probability": min(predicted_area / 900.0, 1.0),
        "pred_presence_class": predicted_area >= 90.0,
        "presence_target_threshold_fraction": 0.10,
    }


def write_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a tiny retained-domain mask that drops one test cell."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [5, 6, 7],
            "is_plausible_kelp_domain": [True, False, True],
            "domain_mask_reason": [
                "retained_depth_0_60m",
                "dropped_too_deep",
                "retained_ambiguous_coast",
            ],
            "domain_mask_detail": ["fixture"] * 3,
            "domain_mask_version": ["test_mask_v1"] * 3,
            "crm_elevation_m": [-4.0, -90.0, 0.5],
            "crm_depth_m": [4.0, 90.0, 0.0],
            "depth_bin": ["0_40m", "deep_water", "ambiguous_coast"],
            "elevation_bin": ["subtidal", "deep_water", "ambiguous_coast"],
        }
    ).to_parquet(mask_path, index=False)
    manifest_path.write_text(json.dumps({"mask_version": "test_mask_v1"}) + "\n")


def write_footprint(path: Path) -> None:
    """Write a tiny WGS84 polygon footprint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
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
        )
        + "\n"
    )
