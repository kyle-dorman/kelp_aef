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
    assert "Kelpwatch observed label" in html
    assert "Expected-value hurdle prediction" in html
    assert "Expected-value hurdle residual" in html
    assert "Conditional ridge prediction" in html
    assert "Binary presence probability" in html
    assert "Binary outcome TP/FP/FN/TN" in html
    assert "Binary outcome TN only" not in html
    assert "Data layer" in html
    assert "Layer filter" in html
    assert "Active legend" in html
    assert 'input.type = "radio"' in html
    assert 'input.type = "number"' in html
    assert 'input.type = "checkbox"' in html
    assert "Label source" not in html
    assert "Mask reason" not in html
    assert "Hurdle pred m2" in html
    assert "Hurdle resid m2" in html
    assert "Cond ridge m2" in html
    assert "Binary prob" in html
    assert "Binary outcome" in html
    assert "L.imageOverlay" not in html
    assert "pointLayer.propertyName" in html
    assert "pointLayer.allowedValues.includes" in html
    assert "filterState" in html
    assert "legend-ramp" in html
    assert "binary-outcome-filter" in html
    assert "Minimum area m2" in html
    assert "Minimum label area m2" in html
    assert "Minimum abs residual m2" in html
    assert "opacity-controls" not in html
    assert "Predicted canopy area" in html
    assert "Copy lat/lon" in html

    inspection = pd.read_csv(inspection_csv)
    manifest = json.loads(manifest_path.read_text())
    assert len(inspection) == 4
    assert {
        "longitude",
        "latitude",
        "expected_value_hurdle_prediction_m2",
        "conditional_ridge_prediction_m2",
        "binary_presence_probability",
        "binary_outcome_outcome",
        "selection_reasons",
    } <= set(inspection.columns)
    assert set(inspection["binary_outcome_outcome"]) == {"TP", "FP", "FN", "TN"}
    assert int(inspection["selection_binary_non_true_negative"].sum()) == 3
    assert int(inspection["selection_kelpwatch_positive"].sum()) == 2
    assert int(inspection["selection_binary_false_negative"].sum()) == 1
    assert int(inspection["selection_large_hurdle_residual"].sum()) == 4
    assert int(inspection["selection_true_negative"].sum()) == 1
    assert manifest["command"] == "visualize-results"
    assert manifest["mask_status"] == "plausible_kelp_domain"
    assert manifest["inspection"]["row_count"] == 4
    assert manifest["inspection"]["max_points"] == 4
    assert manifest["inspection"]["cap_was_enough_for_priority_buckets"] is True
    assert manifest["inspection"]["selection_buckets"]["kelpwatch_positive"] == {
        "candidate_count": 2,
        "included_count": 2,
        "omitted_count": 0,
    }
    assert manifest["inspection"]["selection_buckets"]["binary_false_negative"] == {
        "candidate_count": 1,
        "included_count": 1,
        "omitted_count": 0,
    }
    assert manifest["inspection"]["selection_buckets"]["binary_non_true_negative"] == {
        "candidate_count": 3,
        "included_count": 3,
        "omitted_count": 0,
    }
    assert manifest["inspection"]["selection_buckets"]["large_hurdle_residual"] == {
        "candidate_count": 4,
        "included_count": 4,
        "omitted_count": 0,
    }
    assert manifest["inspection"]["selection_buckets"]["true_negative"] == {
        "candidate_count": 2,
        "included_count": 1,
        "omitted_count": 1,
    }
    assert manifest["basemap"]["runtime_dependency_only"] is True
    assert manifest["filter_defaults"]["continuous_min_area_m2"] == 90.0
    assert manifest["filter_defaults"]["label_min_area_m2"] == 1.0
    assert manifest["filter_defaults"]["residual_min_abs_area_m2"] == 45.0
    assert manifest["filter_defaults"]["probability_min"] == 0.2
    assert manifest["filter_defaults"]["binary_outcomes_visible"] == ["TP", "FN"]
    assert (
        manifest["filter_defaults"]["layer_overrides"]["conditional_ridge"][
            "continuous_min_area_m2"
        ]
        == 450.0
    )
    assert {row["layer_id"] for row in manifest["layers"]} == {
        "expected_value_hurdle",
        "conditional_ridge",
        "binary_presence",
        "binary_outcome",
    }
    assert {row["layer_id"] for row in manifest["point_layers"]} == {
        "kelpwatch_observed_label",
        "expected_value_hurdle_prediction",
        "expected_value_hurdle_residual",
        "conditional_ridge_prediction",
        "binary_presence_probability",
        "binary_outcome_outcome",
    }
    binary_layers = {
        row["layer_id"]: row["allowed_values"]
        for row in manifest["point_layers"]
        if row["type"] == "binary_outcome"
    }
    assert binary_layers == {
        "binary_outcome_outcome": ["TP", "FP", "FN", "TN"],
    }
    filters = {row["layer_id"]: row["filter"] for row in manifest["point_layers"]}
    legends = {row["layer_id"]: row["legend"] for row in manifest["point_layers"]}
    assert filters["kelpwatch_observed_label"]["defaultMinValue"] == 1.0
    assert filters["expected_value_hurdle_prediction"]["defaultMinValue"] == 90.0
    assert filters["expected_value_hurdle_residual"]["defaultMinValue"] == 45.0
    assert filters["conditional_ridge_prediction"]["defaultMinValue"] == 450.0
    assert filters["binary_presence_probability"]["defaultMinValue"] == 0.2
    assert filters["binary_outcome_outcome"]["defaultValues"] == ["TP", "FN"]
    assert legends["kelpwatch_observed_label"]["description"].startswith("Observed Kelpwatch")
    assert legends["expected_value_hurdle_prediction"]["unit"] == "m2"
    assert legends["expected_value_hurdle_residual"]["diverging"] is True
    assert all(row["coordinate_based"] for row in manifest["point_layers"])


def test_visualize_results_writes_configured_contexts(tmp_path: Path) -> None:
    """Verify multi-context visualizers keep region and regime labels attached."""
    base_fixture = write_results_visualizer_fixture(tmp_path)
    coordinator = write_results_visualizer_context_fixture(tmp_path, base_fixture["config_path"])

    assert main(["visualize-results", "--config", str(coordinator["config_path"])]) == 0

    assert coordinator["index_html"].is_file()
    assert coordinator["collection_manifest"].is_file()
    for context_id, paths in coordinator["contexts"].items():
        assert paths["html"].is_file()
        assert paths["manifest"].is_file()
        assert paths["inspection_csv"].is_file()
        html = paths["html"].read_text()
        assert paths["display_name"] in html
        assert paths["evaluation_region"] in html
        assert paths["training_regime"] in html
        assert "Data layer" in html
        assert "Binary outcome TP/FP/FN/TN" in html
        manifest = json.loads(paths["manifest"].read_text())
        assert manifest["context"]["context_id"] == context_id
        assert manifest["context"]["display_name"] == paths["display_name"]
        assert manifest["context"]["evaluation_region"] == paths["evaluation_region"]
        assert manifest["context"]["training_regime"] == paths["training_regime"]
        assert manifest["context"]["model_origin_region"] == paths["model_origin_region"]
        assert "full_grid_prediction_path" in manifest["context"]["input_paths"]
        inspection = pd.read_csv(paths["inspection_csv"])
        assert set(inspection["context_id"]) == {context_id}
        assert set(inspection["display_name"]) == {paths["display_name"]}
        assert set(inspection["evaluation_region"]) == {paths["evaluation_region"]}
        assert set(inspection["training_regime"]) == {paths["training_regime"]}
        assert set(inspection["model_origin_region"]) == {paths["model_origin_region"]}

    index_html = coordinator["index_html"].read_text()
    assert "Big Sur Local Results Visualizer" in index_html
    assert "Pooled Monterey+Big Sur On Monterey Results Visualizer" in index_html
    collection_manifest = json.loads(coordinator["collection_manifest"].read_text())
    assert collection_manifest["mode"] == "multi_context"
    assert collection_manifest["context_count"] == 2
    assert {row["context_id"] for row in collection_manifest["contexts"]} == {
        "big_sur_local",
        "pooled_monterey_big_sur_on_monterey",
    }


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
        predictions=(360.0, 0.0, 0.0, 500.0, 120.0),
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
    max_inspection_points: 4
    filter_defaults:
      continuous_min_area_m2: 90.0
      label_min_area_m2: 1.0
      residual_min_abs_area_m2: 45.0
      probability_min: 0.20
      binary_outcomes_visible: [TP, FN]
    layer_filter_defaults:
      conditional_ridge:
        continuous_min_area_m2: 450.0
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


def write_results_visualizer_context_fixture(
    tmp_path: Path,
    base_config_path: Path,
) -> dict[str, object]:
    """Write a coordinator config for multi-context visualizer tests."""
    config_path = tmp_path / "context_config.yaml"
    index_html = tmp_path / "reports/interactive/monterey_big_sur_pooled_results_visualizer.html"
    collection_manifest = tmp_path / "interim/monterey_big_sur_results_visualizer_manifest.json"
    hurdle_predictions = tmp_path / "processed/hurdle_full_grid_predictions.parquet"
    context_paths = {
        "big_sur_local": {
            "display_name": "Big Sur Local Results Visualizer",
            "evaluation_region": "big_sur",
            "training_regime": "big_sur_only",
            "model_origin_region": "big_sur",
            "html": tmp_path / "reports/interactive/big_sur_results_visualizer.html",
            "asset_dir": tmp_path / "reports/interactive/big_sur_results_visualizer",
            "manifest": tmp_path / "interim/big_sur_results_visualizer_manifest.json",
            "inspection_csv": tmp_path
            / "reports/tables/big_sur_results_visualizer_inspection_points.csv",
        },
        "pooled_monterey_big_sur_on_monterey": {
            "display_name": "Pooled Monterey+Big Sur On Monterey Results Visualizer",
            "evaluation_region": "monterey",
            "training_regime": "pooled_monterey_big_sur",
            "model_origin_region": "monterey_big_sur",
            "html": tmp_path
            / "reports/interactive/monterey_pooled_monterey_big_sur_results_visualizer.html",
            "asset_dir": tmp_path
            / "reports/interactive/monterey_pooled_monterey_big_sur_results_visualizer",
            "manifest": tmp_path
            / "interim/monterey_pooled_monterey_big_sur_results_visualizer_manifest.json",
            "inspection_csv": tmp_path
            / "reports/tables"
            / "monterey_pooled_monterey_big_sur_results_visualizer_inspection_points.csv",
        },
    }
    big_sur_context = context_paths["big_sur_local"]
    pooled_context = context_paths["pooled_monterey_big_sur_on_monterey"]
    config_path.write_text(
        f"""
reports:
  results_visualizer:
    collection_display_name: Monterey And Big Sur Results Visualizers
    context_index_html: {index_html}
    context_manifest: {collection_manifest}
    contexts:
      - context_id: big_sur_local
        display_name: Big Sur Local Results Visualizer
        evaluation_region: big_sur
        training_regime: big_sur_only
        model_origin_region: big_sur
        config_path: {base_config_path}
        full_grid_prediction_path: {hurdle_predictions}
        binary_prediction_path: {hurdle_predictions}
        outputs:
          html: {big_sur_context["html"]}
          asset_dir: {big_sur_context["asset_dir"]}
          manifest: {big_sur_context["manifest"]}
          inspection_points: {big_sur_context["inspection_csv"]}
      - context_id: pooled_monterey_big_sur_on_monterey
        display_name: Pooled Monterey+Big Sur On Monterey Results Visualizer
        evaluation_region: monterey
        training_regime: pooled_monterey_big_sur
        model_origin_region: monterey_big_sur
        config_path: {base_config_path}
        full_grid_prediction_path: {hurdle_predictions}
        binary_prediction_path: {hurdle_predictions}
        outputs:
          html: {pooled_context["html"]}
          asset_dir: {pooled_context["asset_dir"]}
          manifest: {pooled_context["manifest"]}
          inspection_points: {pooled_context["inspection_csv"]}
""".lstrip()
    )
    return {
        "config_path": config_path,
        "index_html": index_html,
        "collection_manifest": collection_manifest,
        "contexts": context_paths,
    }


def write_continuous_predictions(
    path: Path,
    *,
    model_name: str,
    predictions: tuple[float, ...],
) -> None:
    """Write a tiny continuous prediction table for visualizer tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        continuous_row(5, 10, 20, 450.0, predictions[0], -122.000, 36.000, model_name),
        continuous_row(6, 11, 21, 180.0, predictions[1], -121.999, 36.001, model_name),
        continuous_row(7, 12, 22, 0.0, predictions[2], -121.998, 36.002, model_name),
        continuous_row(8, 13, 23, 0.0, predictions[3], -121.997, 36.003, model_name),
        continuous_row(
            9,
            14,
            24,
            0.0,
            predictions[4],
            -121.996,
            36.004,
            model_name,
            predicted_class=False,
        ),
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
    predicted_class: bool | None = None,
) -> dict[str, object]:
    """Return one continuous model prediction row."""
    presence_class = predicted_area >= 90.0 if predicted_class is None else predicted_class
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
        "pred_presence_class": presence_class,
        "presence_target_threshold_fraction": 0.10,
    }


def write_domain_mask(mask_path: Path, manifest_path: Path) -> None:
    """Write a tiny retained-domain mask for visualizer tests."""
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aef_grid_cell_id": [5, 6, 7, 8, 9],
            "is_plausible_kelp_domain": [True, True, True, True, True],
            "domain_mask_reason": [
                "retained_depth_0_60m",
                "retained_ambiguous_coast",
                "retained_ambiguous_coast",
                "retained_depth_0_60m",
                "retained_depth_0_60m",
            ],
            "domain_mask_detail": ["fixture"] * 5,
            "domain_mask_version": ["test_mask_v1"] * 5,
            "crm_elevation_m": [-4.0, -8.0, 0.5, -12.0, -14.0],
            "crm_depth_m": [4.0, 8.0, 0.0, 12.0, 14.0],
            "depth_bin": ["0_40m", "0_40m", "ambiguous_coast", "0_40m", "0_40m"],
            "elevation_bin": [
                "subtidal",
                "subtidal",
                "ambiguous_coast",
                "subtidal",
                "subtidal",
            ],
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
