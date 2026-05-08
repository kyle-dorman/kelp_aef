# Task 09: Map Residuals And Area Bias

## Goal

Create the first visual and tabular QA outputs for the ridge baseline. This task
should answer whether the apparently strong tabular metrics are spatially sane:
does the model reproduce the observed Kelpwatch-style canopy pattern, and where
does it underpredict or overpredict?

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Baseline predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_predictions.parquet`.
- Baseline metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.

## Outputs

- 2022 observed/predicted/residual map:
  `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png`.
- Observed-vs-predicted scatter or hexbin:
  `/Volumes/x10pro/kelp_aef/reports/figures/ridge_observed_vs_predicted.png`.
- Area bias by split/year:
  `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`.
- Area bias by latitude band:
  `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_latitude_band.csv`.
- Top residual stations table:
  `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.csv`.
- QA manifest:
  `/Volumes/x10pro/kelp_aef/interim/map_residuals_manifest.json`.

Optional if straightforward:

- Self-contained interactive HTML map:
  `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.html`.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions should cover:

- Prediction input path.
- Model name to map, initially `ridge_regression`.
- Primary map split/year, initially test year 2022.
- Output figure/table/manifest paths.
- Latitude-band size or count.
- Residual ranking count.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should state
files to change, plotting method, output schemas, map year/split, color scale
choices, and whether an interactive HTML preview is included.

## Proposed CLI

```bash
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
```

## Implementation Notes

- Use the ridge rows from `baseline_predictions.parquet`.
- Focus the first map on the 2022 test split.
- Use clipped predicted fraction for mapped canopy area:
  `pred_kelp_fraction_y_clipped` and `pred_kelp_max_y`.
- Map observed canopy area, predicted canopy area, and residual canopy area in
  common units of `m2 canopy / 900m2 pixel`.
- Use a shared observed/predicted color scale so the two maps are visually
  comparable.
- Use a diverging residual scale centered on zero.
- Keep the maps point-based at Kelpwatch station centers for this task; do not
  interpolate to a raster unless that becomes necessary for readability.
- Overlay or outline the Monterey smoke footprint when useful.
- Use structured logging for input paths, selected model/year, row counts,
  output paths, and missing-data counts.

## Figure Requirements

The static 2022 map figure should contain:

- Observed Kelpwatch annual max canopy.
- Ridge predicted canopy.
- Residual: observed minus predicted.
- Clear titles with model name, split, and year.
- A note or caption in the figure or manifest defining residual sign.

The observed-vs-predicted figure should contain:

- All ridge predictions by split, or separate panels for train, validation, and
  test.
- A 1:1 reference line.
- Enough transparency or binning to handle dense zero/low-kelp rows.

## Table Schemas

Area bias by year:

- `model_name`
- `split`
- `year`
- `row_count`
- `observed_canopy_area`
- `predicted_canopy_area`
- `area_bias`
- `area_pct_bias`
- `mae_fraction`
- `rmse_fraction`
- `r2_fraction`

Area bias by latitude band:

- `model_name`
- `split`
- `year`
- `latitude_band`
- `latitude_min`
- `latitude_max`
- `row_count`
- `observed_canopy_area`
- `predicted_canopy_area`
- `area_bias`
- `area_pct_bias`
- `mae_fraction`
- `rmse_fraction`

Top residual stations:

- `model_name`
- `split`
- `year`
- `kelpwatch_station_id`
- `longitude`
- `latitude`
- `kelp_max_y`
- `pred_kelp_max_y`
- `residual_kelp_max_y`
- `abs_residual_kelp_max_y`

## Validation Command

```bash
make check
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
```

Add focused tests with tiny synthetic prediction tables to verify:

- The command filters to the configured model and year.
- Area-bias tables compute expected observed, predicted, bias, and percent bias.
- Latitude-band grouping is deterministic.
- Top residual rows are sorted by absolute residual.
- Static figure and manifest files are written.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Primary QA year: 2022 test split.
- Include train/validation/test in summary tables where useful.

## Acceptance Criteria

- The command writes the configured static map, scatter/hexbin figure, bias
  tables, top residual table, and manifest.
- The 2022 map shows observed, predicted, and residual values with sensible
  color scales and no unreadable overlap.
- The area-bias tables include ridge rows for all relevant splits/years.
- The top residual table identifies the largest overpredictions and
  underpredictions.
- `make check` passes.

## Known Constraints And Non-Goals

- Do not train new models in this task.
- Do not choose or finalize a binary kelp threshold in this task.
- Do not interpret model skill as independent field truth; these are
  Kelpwatch-style labels.
- Do not build a polished report or manuscript figure yet. This is first-pass
  model QA.
- Do not create full West Coast maps.
