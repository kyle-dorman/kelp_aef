# Task 11: Correct Full-Grid Background Alignment

## Goal

Correct the Phase 0 aligned training data contract so the Monterey smoke test
includes land and open-ocean background examples, not only Kelpwatch station
centers. Then rerun the baseline, map, and model-analysis steps against the
corrected artifact before closing Phase 0.

The current `aligned_training_table.parquet` is useful as a station-sample QA
artifact, but it is not the intended grid-cell-by-year training table. It has
one row per retained Kelpwatch station/year:

```text
30,232 Kelpwatch stations x 5 years = 151,160 rows
```

The selected AEF chip is much larger:

```text
10 m source pixels per year: 8192 x 8192 = 67,108,864
30 m AEF-aligned cells per year: ceil(8192 / 3)^2 = 7,458,361
30 m AEF-aligned cells across 2018-2022: 37,291,805
```

Phase 0 should not be considered complete until the report makes this correction
and the model has seen a defensible background set.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Annual labels:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Annual label manifest:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual_manifest.json`.
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`.
- AEF read assets declared in the tile manifest, using each asset's preferred
  read path.
- Smoke-test years: 2018-2022.
- AEF bands: `A00` through `A63`.
- Existing station-sample alignment output for QA comparison:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`.

## Outputs

Use explicit artifact names so it is clear which products are full-grid versus
station-sample products.

- Corrected full-grid aligned table, partitioned if needed:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Corrected full-grid alignment manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json`.
- Corrected full-grid alignment summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_full_grid_training_table_summary.csv`.
- Optional background-inclusive sampled training table, if the full grid is too
  large for first-pass scikit-learn training:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`.
- Optional background-sample manifest and summary:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table_manifest.json`.
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table_summary.csv`.
- Rerun baseline artifacts from Task 08, using the corrected configured training
  table.
- Rerun residual-map and area-bias artifacts from Task 09, using corrected
  predictions.
- Rerun model-analysis report artifacts from Task 10, with explicit discussion
  of the station-only error and corrected background coverage.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config changes should cover:

- Full-grid alignment output table, manifest, and summary paths.
- Optional background-sample output paths.
- Label policy for cells that do not match a Kelpwatch station.
- Fast-path window or row/column bounds for local testing.
- Downstream model input path, so Task 08 no longer silently consumes the
  station-only table.

## Plan/Spec Requirement

Write a short implementation spec before editing code. The spec must state:

- Whether the corrected default model input is the full-grid table or a
  documented background-inclusive sample derived from it.
- How Kelpwatch station labels are mapped onto the AEF-aligned 30 m grid.
- How cells outside Kelpwatch station support are labeled and flagged.
- Expected row counts for the full and fast paths.
- Memory/runtime strategy for writing and reading tens of millions of rows.
- Exact downstream commands to rerun Tasks 08, 09, and 10.

Do not start implementation until this spec is reviewed.

## Proposed CLI

The exact command shape can change in the implementation spec, but it should be
package-backed and restartable. A reasonable starting point is:

```bash
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Implementation Notes

- Keep the existing station-sample alignment code as a QA/reference path unless
  the implementation spec decides to rename or split it first.
- Build the corrected product on the AEF-aligned 30 m grid, not the raw 10 m
  grid, for the first Phase 0 correction.
- Use Rasterio/GDAL resampling or windowed reads to aggregate 10 m AEF pixels to
  30 m cells. Do not materialize all 64 full-resolution bands in memory at once.
- Enumerate target-grid cells by year in row chunks or parquet partitions.
- Store target-grid identifiers such as `aef_grid_row`, `aef_grid_col`, and a
  stable `aef_grid_cell_id` so reruns and joins are deterministic.
- Map Kelpwatch labels onto the target grid with maintained geospatial/raster
  tools. Do not hand-roll projection or geometry logic.
- Preserve Kelpwatch station labels exactly where they match the target grid.
- Add explicit label provenance columns so downstream metrics can distinguish
  observed Kelpwatch support from assumed background.
- Keep a fast path that exercises the same logic on a small spatial window that
  contains positive Kelpwatch labels, Kelpwatch zero labels, and outside-support
  background cells.
- Use structured logging for row chunks, years, raster paths, row counts,
  label-source counts, output paths, and downstream rerun commands.

## Label Policy

The corrected table should make label provenance explicit. Minimum policy:

- `label_source = "kelpwatch_station"` for cells with a Kelpwatch station label.
- `label_source = "assumed_background"` for full-grid cells with no matching
  Kelpwatch station.
- `kelp_max_y = 0.0` and `kelp_fraction_y = 0.0` for assumed-background cells in
  this smoke-test correction.
- `is_kelpwatch_observed = true` for Kelpwatch-station cells.
- `is_kelpwatch_observed = false` for assumed-background cells.

This is a weak-label assumption for Phase 0, not independent truth. The report
must evaluate metrics both overall and on the `kelpwatch_station` subset so we
can tell whether background examples dominate the headline metrics.

## Expected Output Schema

Minimum columns for the corrected full-grid table:

- `year`
- `aef_grid_cell_id`
- `aef_grid_row`
- `aef_grid_col`
- `longitude`
- `latitude`
- `label_source`
- `is_kelpwatch_observed`
- `kelpwatch_station_id` when available
- `kelp_max_y`
- `kelp_fraction_y`
- annual label diagnostic columns where available
- `A00` through `A63`
- `aef_expected_pixel_count`
- `aef_valid_pixel_count`
- `aef_missing_pixel_count`
- `aef_alignment_method`
- `aef_source_path`
- `aef_source_href`

Additional provenance columns are acceptable if they are documented in the
manifest and summary table.

## Runtime And Sampling Strategy

The full-grid table may be too large for the first scikit-learn smoke model in a
single in-memory pandas frame. If that is true, keep both products:

- A full-grid aligned artifact for prediction, mapping, and future modeling.
- A documented background-inclusive sample for Task 08 training/evaluation.

The sample must include:

- All or a documented sample of positive Kelpwatch-station rows.
- Kelpwatch-station zero rows.
- Assumed-background land/open-ocean rows.
- Stable random seed and sampling fractions/counts in the manifest.

Do not silently train only on Kelpwatch-station rows.

## Validation Command

```bash
make check
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast
```

After the full run is implemented and reviewed:

```bash
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Add focused unit tests with tiny synthetic rasters and labels to verify:

- A tiny 10 m raster produces the expected number of 30 m target-grid rows.
- Kelpwatch labels are assigned to the correct target-grid cells.
- Non-Kelpwatch cells are retained as `assumed_background`.
- The fast path includes observed positives, observed zeros, and assumed
  background rows.
- The manifest reports full-grid row counts, label-source counts, and sampling
  policy.

## Smoke-Test Region And Years

- Region: Monterey Peninsula footprint GeoJSON from the selected AEF tile.
- Years: 2018-2022.

## Acceptance Criteria

- The task produces a corrected full-grid or explicitly background-inclusive
  aligned artifact that includes cells outside Kelpwatch station support.
- The manifest reports the expected order of magnitude:
  roughly 7.46 million 30 m target cells per year before any documented
  filtering or sampling.
- The summary table reports row counts by year and `label_source`.
- The old station-only row count of 30,232 rows/year is no longer presented as
  the complete training universe.
- Task 08 baselines are rerun from the corrected configured model input.
- Task 09 maps and area-bias summaries are rerun from corrected predictions.
- Task 10 report is regenerated and explicitly states:
  - the original Phase 0 table was station-only;
  - the corrected artifact includes assumed-background rows;
  - metrics are reported overall and on Kelpwatch-observed rows separately.
- `make check` passes.

## Known Constraints And Non-Goals

- Do not expand beyond the selected Monterey AEF tile in this task.
- Do not bulk-download broader AEF coverage.
- Do not move to 10 m labels, CNN chips, tensors, Zarr, or TFRecords in this
  task.
- Do not treat assumed-background labels as independent field truth.
- Do not hide runtime shortcuts. If sampling is needed, record the sampling
  policy in config, manifests, summaries, and the report.
- Do not hand-roll geometry, projection, raster-window, or model logic when a
  maintained geospatial/scientific library can do it clearly.
