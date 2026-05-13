import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kelp_aef import main
from kelp_aef.domain.domain_mask import (
    DROPPED_DEEP_WATER,
    DROPPED_DEFINITE_LAND,
    QA_MISSING_CRM,
    RETAINED_AMBIGUOUS_COAST,
    RETAINED_DEPTH_0_60M,
    DomainMaskThresholds,
    classify_domain_mask_frame,
    load_domain_mask_config,
)


def test_load_domain_mask_config_reads_fast_outputs(tmp_path: Path) -> None:
    """Load domain-mask paths, thresholds, and inherited fast windows."""
    fixture = write_domain_mask_fixture(tmp_path)

    config = load_domain_mask_config(fixture["config_path"], fast=True)

    assert config.aligned_crm_table == fixture["fast_crm"]
    assert config.output_table == fixture["fast_output"]
    assert config.thresholds.max_depth_m == 60.0
    assert config.row_window == (0, 4)
    assert config.col_window == (0, 4)


def test_classify_domain_mask_frame_uses_explicit_reason_precedence() -> None:
    """Classify missing CRM, ambiguous coast, land, deep water, and retained cells."""
    frame = domain_mask_input_frame()
    thresholds = DomainMaskThresholds(
        max_depth_m=60.0,
        definite_land_elevation_m=5.0,
        ambiguous_coast_elevation_min_m=-5.0,
        ambiguous_coast_elevation_max_m=5.0,
        nearshore_shallow_depth_m=40.0,
        intermediate_depth_m=60.0,
    )

    classified = classify_domain_mask_frame(frame, thresholds)

    assert classified["domain_mask_reason"].to_list() == [
        QA_MISSING_CRM,
        RETAINED_AMBIGUOUS_COAST,
        DROPPED_DEFINITE_LAND,
        DROPPED_DEEP_WATER,
        RETAINED_DEPTH_0_60M,
        RETAINED_DEPTH_0_60M,
    ]
    assert classified["is_plausible_kelp_domain"].to_list() == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert classified["depth_bin"].to_list() == [
        QA_MISSING_CRM,
        "ambiguous_coast",
        "land_positive",
        "60m_plus",
        "0_40m",
        "40_60m",
    ]


def test_build_domain_mask_fast_writes_mask_and_qa_outputs(tmp_path: Path) -> None:
    """Run the package CLI fast path and verify mask artifacts and QA summaries."""
    fixture = write_domain_mask_fixture(tmp_path)

    assert main(["build-domain-mask", "--config", str(fixture["config_path"]), "--fast"]) == 0

    mask = pd.read_parquet(fixture["fast_output"]).sort_values("aef_grid_cell_id")
    manifest = json.loads(fixture["fast_manifest"].read_text())
    coverage = pd.read_csv(fixture["fast_summary"])
    retention = pd.read_csv(fixture["fast_retention"])
    depth_bins = pd.read_csv(fixture["fast_depth_bins"])

    assert mask["aef_grid_cell_id"].to_list() == [0, 1, 2, 3]
    assert mask["domain_mask_reason"].to_list() == [
        QA_MISSING_CRM,
        RETAINED_AMBIGUOUS_COAST,
        DROPPED_DEFINITE_LAND,
        DROPPED_DEEP_WATER,
    ]
    assert manifest["fast"] is True
    assert manifest["row_counts"]["mask_cells"] == 4
    assert manifest["row_counts"]["dropped_cells"] == 2
    assert manifest["missing_crm_policy"] == "retain_for_qa"
    assert fixture["fast_figure"].is_file()
    dropped_land = coverage.query("domain_mask_reason == @DROPPED_DEFINITE_LAND")
    assert int(dropped_land["cell_count"].iloc[0]) == 1
    all_retention = retention.query("year == 2018 and domain_mask_reason == 'all'")
    assert int(all_retention["positive_cell_year_rows"].iloc[0]) == 3
    assert int(all_retention["retained_positive_cell_year_rows"].iloc[0]) == 1
    assert "60m_plus" in set(depth_bins["depth_bin"])


def write_domain_mask_fixture(tmp_path: Path) -> dict[str, Path]:
    """Write a complete tiny domain-mask fixture under a temp directory."""
    config_path = tmp_path / "config.yaml"
    full_grid = tmp_path / "interim/full_grid.parquet"
    full_grid_manifest = tmp_path / "interim/full_grid_manifest.json"
    crm = tmp_path / "interim/aligned_noaa_crm.parquet"
    crm_manifest = tmp_path / "interim/aligned_noaa_crm_manifest.json"
    fast_crm = tmp_path / "interim/aligned_noaa_crm.fast.parquet"
    fast_manifest = tmp_path / "interim/plausible_kelp_domain_mask.fast_manifest.json"
    fast_output = tmp_path / "interim/plausible_kelp_domain_mask.fast.parquet"
    fast_summary = tmp_path / "tables/plausible_kelp_domain_mask.fast_summary.csv"
    fast_retention = tmp_path / "tables/plausible_kelp_domain_mask_kelpwatch_retention.fast.csv"
    fast_depth_bins = tmp_path / "tables/plausible_kelp_domain_mask_depth_bins.fast.csv"
    fast_figure = tmp_path / "figures/plausible_kelp_domain_mask.fast_qa.png"
    cusp_manifest = tmp_path / "interim/noaa_cusp_source_manifest.json"

    write_crm_table(crm)
    write_crm_table(fast_crm)
    write_full_grid_table(full_grid)
    write_json_manifest(crm_manifest, {"command": "align-noaa-crm", "fast": False})
    write_json_manifest(full_grid_manifest, {"command": "align-full-grid", "fast": False})
    write_json_manifest(cusp_manifest, {"command": "download-noaa-cusp", "record_count": 1})
    write_domain_mask_config(
        config_path=config_path,
        crm=crm,
        crm_manifest=crm_manifest,
        fast_crm=fast_crm,
        full_grid=full_grid,
        full_grid_manifest=full_grid_manifest,
        cusp_manifest=cusp_manifest,
        fast_output=fast_output,
        fast_manifest=fast_manifest,
        fast_summary=fast_summary,
        fast_retention=fast_retention,
        fast_depth_bins=fast_depth_bins,
        fast_figure=fast_figure,
    )
    return {
        "config_path": config_path,
        "fast_crm": fast_crm,
        "fast_output": fast_output,
        "fast_manifest": fast_manifest,
        "fast_summary": fast_summary,
        "fast_retention": fast_retention,
        "fast_depth_bins": fast_depth_bins,
        "fast_figure": fast_figure,
    }


def domain_mask_input_frame() -> pd.DataFrame:
    """Return tiny aligned CRM rows covering every first-pass mask reason."""
    return pd.DataFrame(
        {
            "aef_grid_row": [0, 1, 2, 3, 4, 5],
            "aef_grid_col": [0, 1, 2, 3, 4, 5],
            "aef_grid_cell_id": [0, 1, 2, 3, 4, 5],
            "longitude": [-122.0, -122.0, -122.0, -122.0, -122.0, -122.0],
            "latitude": [36.9, 36.91, 36.92, 36.93, 36.94, 36.95],
            "crm_elevation_m": [np.nan, 2.0, 8.0, -150.0, -20.0, -45.0],
            "crm_depth_m": [np.nan, 0.0, 0.0, 150.0, 20.0, 45.0],
            "crm_source_product_id": [
                None,
                "crm_socal_v2_1as",
                "crm_socal_v2_1as",
                "crm_socal_v2_1as",
                "crm_socal_v2_1as",
                "crm_socal_v2_1as",
            ],
            "crm_vertical_datum": ["mean sea level"] * 6,
            "crm_value_status": ["no_data", "valid", "valid", "valid", "valid", "valid"],
            "cudem_value_status": ["valid"] * 6,
            "usgs_3dep_value_status": ["valid"] * 6,
        }
    )


def write_crm_table(path: Path) -> None:
    """Write a tiny aligned CRM input table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    domain_mask_input_frame().to_parquet(path, index=False)


def write_full_grid_table(path: Path) -> None:
    """Write a tiny full-grid table with positive labels across years."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in (2018, 2019):
        for cell_id in range(6):
            rows.append(
                {
                    "year": year,
                    "aef_grid_row": cell_id,
                    "aef_grid_col": cell_id,
                    "aef_grid_cell_id": cell_id,
                    "label_source": "kelpwatch_station",
                    "kelp_max_y": 1.0 if cell_id in {1, 2, 3, 4, 5} else 0.0,
                    "kelp_fraction_y": 0.1 if cell_id in {1, 2, 3, 4, 5} else 0.0,
                    "kelp_present_gt0_y": cell_id in {1, 2, 3, 4, 5},
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_json_manifest(path: Path, payload: dict[str, object]) -> None:
    """Write a small JSON manifest fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_domain_mask_config(
    *,
    config_path: Path,
    crm: Path,
    crm_manifest: Path,
    fast_crm: Path,
    full_grid: Path,
    full_grid_manifest: Path,
    cusp_manifest: Path,
    fast_output: Path,
    fast_manifest: Path,
    fast_summary: Path,
    fast_retention: Path,
    fast_depth_bins: Path,
    fast_figure: Path,
) -> None:
    """Write the minimal YAML config needed by domain-mask tests."""
    output_table = config_path.parent / "interim/plausible_kelp_domain_mask.parquet"
    output_manifest = config_path.parent / "interim/plausible_kelp_domain_mask_manifest.json"
    coverage_table = config_path.parent / "tables/plausible_kelp_domain_mask_summary.csv"
    retention_table = (
        config_path.parent / "tables/plausible_kelp_domain_mask_kelpwatch_retention.csv"
    )
    depth_bins_table = config_path.parent / "tables/plausible_kelp_domain_mask_depth_bins.csv"
    visual_figure = config_path.parent / "figures/plausible_kelp_domain_mask_qa.png"
    config_path.write_text(
        f"""
years:
  smoke: [2018, 2019]
region:
  name: monterey_peninsula
alignment:
  full_grid:
    output_table: {full_grid}
    fast:
      years: [2018]
      row_window: [0, 4]
      col_window: [0, 4]
domain:
  noaa_crm:
    alignment:
      cusp_source_manifest: {cusp_manifest}
  plausible_kelp_mask:
    aligned_crm_table: {crm}
    aligned_crm_manifest: {crm_manifest}
    full_grid_table: {full_grid}
    full_grid_manifest: {full_grid_manifest}
    cusp_source_manifest: {cusp_manifest}
    output_table: {output_table}
    output_manifest: {output_manifest}
    coverage_summary_table: {coverage_table}
    kelpwatch_retention_table: {retention_table}
    depth_bin_summary_table: {depth_bins_table}
    visual_qa_figure: {visual_figure}
    max_depth_m: 60.0
    definite_land_elevation_m: 5.0
    ambiguous_coast_elevation_band_m: [-5.0, 5.0]
    nearshore_shallow_depth_m: 40.0
    intermediate_depth_m: 60.0
    row_chunk_size: 2
    fast:
      aligned_crm_table: {fast_crm}
      aligned_crm_manifest: {crm_manifest}
      output_table: {fast_output}
      output_manifest: {fast_manifest}
      coverage_summary_table: {fast_summary}
      kelpwatch_retention_table: {fast_retention}
      depth_bin_summary_table: {fast_depth_bins}
      visual_qa_figure: {fast_figure}
""".lstrip()
    )
