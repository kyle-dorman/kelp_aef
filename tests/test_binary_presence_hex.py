"""Tests for pooled binary-presence hex aggregation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef.evaluation.binary_presence_hex import (
    BinaryPresenceHexInput,
    BinaryPresenceHexMapConfig,
    aggregate_binary_presence_hex_frame,
    hex_centers,
    hex_indices_for_projected_points,
)


def test_hex_indices_are_deterministic_for_known_centers() -> None:
    """Verify flat-top center coordinates round back to their axial ids."""
    flat_diameter = 1000.0
    xs, ys = hex_centers(
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([0, 0, -1], dtype=np.int64),
        flat_diameter,
    )

    q_values, r_values = hex_indices_for_projected_points(xs, ys, flat_diameter)

    assert q_values.tolist() == [0, 1, 1]
    assert r_values.tolist() == [0, 0, -1]


def test_hex_aggregation_counts_rates_and_zero_denominators(tmp_path: Path) -> None:
    """Verify each cell belongs to one hex and FP/FN rates handle zero denominators."""
    config = binary_hex_config(tmp_path)
    input_config = binary_hex_input(tmp_path)
    q0_x, q0_y = hex_centers(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        config.hex_flat_diameter_m,
    )
    q1_x, q1_y = hex_centers(
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        config.hex_flat_diameter_m,
    )
    frame = pd.DataFrame(
        [
            hex_fixture_row(float(q0_x[0]), float(q0_y[0]), True, True, 0.90),
            hex_fixture_row(float(q0_x[0]) + 10.0, float(q0_y[0]), True, False, 0.20),
            hex_fixture_row(float(q1_x[0]), float(q1_y[0]), False, True, 0.80),
            hex_fixture_row(float(q1_x[0]) + 10.0, float(q1_y[0]), False, False, 0.10),
        ]
    )

    rows = aggregate_binary_presence_hex_frame(frame, input_config, config)

    assert len(rows) == 2
    assert sum(int(row["n_cells"]) for row in rows) == 4
    observed_hex = next(row for row in rows if int(row["observed_positive_count"]) == 2)
    background_hex = next(row for row in rows if int(row["observed_positive_count"]) == 0)
    assert observed_hex["hex_id"] == "big_sur:q0:r0"
    assert observed_hex["observed_positive_rate"] == 1.0
    assert observed_hex["predicted_positive_rate"] == 0.5
    assert int(observed_hex["true_positive_count"]) == 1
    assert int(observed_hex["false_negative_count"]) == 1
    assert math.isnan(float(observed_hex["false_positive_rate"]))
    assert background_hex["predicted_positive_rate"] == 0.5
    assert float(background_hex["false_positive_rate"]) == 0.5
    assert math.isnan(float(background_hex["false_negative_rate"]))


def binary_hex_config(tmp_path: Path) -> BinaryPresenceHexMapConfig:
    """Build a test config that treats fixture x/y values as projected coordinates."""
    return BinaryPresenceHexMapConfig(
        inputs=(),
        figure_path=tmp_path / "figure.png",
        table_path=tmp_path / "table.csv",
        manifest_path=tmp_path / "manifest.json",
        primary_split="test",
        primary_year=2022,
        primary_mask_status="plausible_kelp_domain",
        primary_evaluation_scope="full_grid_masked",
        primary_label_source="all",
        source_crs="EPSG:32610",
        target_crs="EPSG:32610",
        hex_flat_diameter_m=1000.0,
        observed_column="binary_observed_y",
        probability_column="pred_binary_probability",
        rate_clip_quantile=0.98,
        difference_clip_quantile=0.98,
        background_color="#dceff7",
        hex_edge_color="#f8fafc",
        coastline_enabled=False,
        coastline_source_manifest_paths=(),
        coastline_color="#111827",
        coastline_linewidth=0.6,
    )


def binary_hex_input(tmp_path: Path) -> BinaryPresenceHexInput:
    """Build a test input descriptor for aggregation-only tests."""
    return BinaryPresenceHexInput(
        context_id="pooled_fixture",
        binary_predictions_path=tmp_path / "predictions.parquet",
        binary_calibration_model_path=tmp_path / "calibration.joblib",
        training_regime="pooled_monterey_big_sur",
        model_origin_region="monterey_big_sur",
        evaluation_region="big_sur",
        threshold_policy="validation_max_f1_calibrated",
        required=True,
    )


def hex_fixture_row(
    x_value: float,
    y_value: float,
    observed: bool,
    predicted: bool,
    probability: float,
) -> dict[str, object]:
    """Build one already-calibrated projected-coordinate hex fixture row."""
    return {
        "split": "test",
        "year": 2022,
        "longitude": x_value,
        "latitude": y_value,
        "binary_observed_y": observed,
        "pred_binary_probability": probability,
        "calibrated_binary_probability": probability,
        "calibrated_pred_binary_class": predicted,
        "probability_source": "raw_logistic",
        "probability_threshold": 0.5,
        "target_label": "annual_max_ge_10pct",
    }
