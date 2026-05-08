# Task 11 Implementation Spec: Full-Grid Background Alignment

Status: draft for review.

## Decision

Implement Task 11 as three related paths:

1. A full-grid AEF-aligned 30 m Parquet dataset for the Monterey tile and
   2018-2022.
2. A background-inclusive sampled modeling table derived from that full-grid
   dataset.
3. A streamed full-grid prediction dataset produced by applying the trained
   model to the full-grid rows in chunks.

The sampled table becomes the training input for the first corrected Task 08
rerun. The full-grid dataset becomes the inference universe. The first ridge
baseline should not try to fit roughly 37 million rows in pandas, but inference
can stream those rows chunk by chunk because ridge predictions are independent
per row.

This preserves the Phase 0 goal: the model trains with land/open-ocean/background
examples and the maps/area summaries evaluate full-grid behavior rather than
only the sampled training universe.

## Current Problem

The current alignment command builds rows only at Kelpwatch station centers.
That artifact has:

```text
30,232 Kelpwatch stations/year
151,160 rows across 2018-2022
```

The selected AEF tile has:

```text
8192 x 8192 = 67,108,864 source pixels/year at 10 m
ceil(8192 / 3)^2 = 7,458,361 target cells/year at 30 m
37,291,805 target cells across 2018-2022
```

So the existing Phase 0 model is a Kelpwatch-station sample model, not a
full-background model.

## Config Changes

Add explicit full-grid and background-sample paths under `alignment`:

```yaml
alignment:
  station_sample:
    output_table: /Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet
    output_manifest: /Volumes/x10pro/kelp_aef/interim/aligned_training_table_manifest.json
    summary_table: /Volumes/x10pro/kelp_aef/reports/tables/aligned_training_table_summary.csv
  full_grid:
    output_table: /Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet
    output_manifest: /Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json
    summary_table: /Volumes/x10pro/kelp_aef/reports/tables/aligned_full_grid_training_table_summary.csv
    target_row_chunk_size: 128
    fast:
      years: [2022]
      row_window: [900, 1156]
      col_window: [900, 1156]
      output_table: /Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.fast.parquet
      output_manifest: /Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.fast_manifest.json
      summary_table: /Volumes/x10pro/kelp_aef/reports/tables/aligned_full_grid_training_table.fast_summary.csv
  background_sample:
    output_table: /Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet
    output_manifest: /Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table_manifest.json
    summary_table: /Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table_summary.csv
    random_seed: 13
    background_rows_per_year: 250000
    include_all_kelpwatch_observed: true
    sample_weight_column: sample_weight
```

Add explicit baseline training and full-grid inference paths so Task 08 no
longer silently uses the station-only table:

```yaml
models:
  baselines:
    input_table: /Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet
    inference_table: /Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet
    sample_predictions: /Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet
    predictions: /Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet
    prediction_manifest: /Volumes/x10pro/kelp_aef/interim/baseline_full_grid_prediction_manifest.json
    use_sample_weight: true
    sample_weight_column: sample_weight
```

Keep `alignment.output_table` temporarily for backward compatibility during the
migration, but treat it as the legacy station-sample artifact until we clean up
the config shape.

## Command Changes

Add a new package-backed command:

```bash
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml --fast
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
```

The full Task 11 rerun sequence is:

```bash
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Full-Grid Alignment Method

For each selected year:

1. Open the preferred AEF read path from the AEF tile manifest.
2. Build the 30 m target grid from the AEF transform using the existing
   `aef_average_target_grid(...)` logic.
3. Transform Kelpwatch annual-label lon/lat points into the AEF CRS with
   GeoPandas.
4. Map each Kelpwatch station point to a target-grid row and column with
   Rasterio's `rowcol(...)`.
5. Build a per-year label lookup keyed by `(aef_grid_row, aef_grid_col)`.
6. Read AEF bands through a `WarpedVRT` with `Resampling.average` in row
   chunks.
7. Write one Parquet partition or part file per year/chunk.

The full-grid output path is a Parquet dataset directory even if the configured
path ends in `.parquet`.

## Label Mapping

The first correction uses target-grid-cell assignment, not polygon overlap.
This is appropriate for the smoke correction because both products are on
regular 30 m-ish supports after AEF aggregation, and it matches the current
station-centered alignment logic closely enough for Phase 0.

For Kelpwatch station cells:

- `label_source = "kelpwatch_station"`
- `is_kelpwatch_observed = true`
- Keep `kelpwatch_station_id` when the cell maps to exactly one station.
- Keep `kelpwatch_station_count`.
- Copy annual label columns from `labels_annual.parquet`.

For cells without a Kelpwatch station:

- `label_source = "assumed_background"`
- `is_kelpwatch_observed = false`
- `kelpwatch_station_id = null`
- `kelpwatch_station_count = 0`
- `kelp_max_y = 0.0`
- `kelp_fraction_y = 0.0`

If multiple Kelpwatch stations map to the same AEF target cell:

- Set `kelpwatch_station_count` to the number of matched stations.
- Set `kelpwatch_station_id = null`.
- Aggregate continuous area/fraction labels with the mean.
- Aggregate presence flags with logical OR.
- Record duplicate-cell counts in the manifest and summary.

This duplicate policy should be reported in the manifest even if duplicate
counts are zero or near zero.

## Output Schema

Minimum full-grid and sampled-table columns:

```text
year
aef_grid_cell_id
aef_grid_row
aef_grid_col
longitude
latitude
label_source
is_kelpwatch_observed
kelpwatch_station_id
kelpwatch_station_count
kelp_max_y
kelp_fraction_y
area_q1
area_q2
area_q3
area_q4
max_area_quarter
valid_quarter_count
nonzero_quarter_count
kelp_present_gt0_y
kelp_present_ge_1pct_y
kelp_present_ge_5pct_y
kelp_present_ge_10pct_y
A00-A63
aef_expected_pixel_count
aef_valid_pixel_count
aef_missing_pixel_count
aef_alignment_method
aef_source_path
aef_source_href
```

For assumed-background rows, quarterly label columns can be `0.0` where they
represent area and null where they represent observed Kelpwatch metadata. The
implementation should choose one convention and write it to the manifest.

## Sampling Policy For Task 08

The sampled modeling table should include:

- All Kelpwatch-observed rows by default.
- A deterministic random sample of assumed-background rows per year.
- At least one explicit count of sampled rows by `year`, `split`, and
  `label_source`.
- A `sample_weight` column that records each row's expansion factor back to the
  full-grid label-source/year population.

Default sample:

```text
include_all_kelpwatch_observed = true
background_rows_per_year = 250,000
random_seed = 13
```

Expected sampled-table size:

```text
151,160 Kelpwatch-observed rows
1,250,000 assumed-background rows
about 1.4 million rows total across 2018-2022
```

This is large enough to expose the model to land/open-ocean/background, but
small enough for the first ridge baseline to remain a pandas/scikit-learn smoke
test.

Default weighting:

- Kelpwatch-observed rows get `sample_weight = 1.0`.
- Sampled background rows get
  `sample_weight = full_background_count_for_year / sampled_background_count_for_year`.
- Ridge training uses `sample_weight` by default.

The manifest must record both the sampled row counts and the full-grid
population counts so weighted and unweighted interpretations are auditable. If
weighted ridge training behaves poorly, add an unweighted comparison row rather
than silently changing the weighting policy.

## Downstream Code Changes

### Baselines

Update Task 08 code so `models.baselines.input_table` overrides
`alignment.output_table`.

The baseline code should preserve new identity/provenance columns in the split
manifest, sample prediction table, and streamed full-grid prediction table:

```text
aef_grid_cell_id
aef_grid_row
aef_grid_col
label_source
is_kelpwatch_observed
kelpwatch_station_id
kelpwatch_station_count
sample_weight
```

Metrics should be reported:

- Overall.
- For `label_source = "kelpwatch_station"`.
- For `label_source = "assumed_background"`.
- Weighted and unweighted when `sample_weight` is available.

The headline report should not use only overall metrics, because assumed-zero
background rows can dominate the error distribution.

### Full-Grid Prediction

Add a `predict-full-grid` command that:

1. Loads the trained model artifact from Task 08.
2. Opens the full-grid Parquet dataset.
3. Reads row groups or file partitions in chunks.
4. Applies the model to `A00-A63`.
5. Writes prediction chunks to `models.baselines.predictions`.
6. Writes a prediction manifest with input row counts, output row counts, chunk
   counts, model metadata, and label-source counts.

The prediction schema should match the current Task 08 prediction schema plus
the full-grid identity/provenance columns. It should be a Parquet dataset
directory if chunked output is used.

### Maps

Task 09 should consume `models.baselines.predictions`, which now points to the
streamed full-grid predictions. For the selected map year/split, render grid
rasters from `aef_grid_row` and `aef_grid_col` rather than plotting millions of
points with scatter.

The required first full-grid maps are:

- Observed target: Kelpwatch station cells plus assumed-background zeros.
- Predicted canopy area.
- Residual canopy area.

The map should still include a Kelpwatch-observed-only inset or companion table
so the report can separate weak assumed-background behavior from observed
Kelpwatch-support behavior.

### Model Analysis Report

Task 10 report generation should add a correction section:

- The original Phase 0 table was Kelpwatch-station-only.
- Task 11 adds assumed-background rows from the full AEF tile.
- The model trains on a sampled background-inclusive table.
- The model is applied to the full-grid dataset through streamed inference.
- Metrics are split by `label_source`.
- Overall metrics are interpreted cautiously because background rows are
  assumed labels and can dominate counts.

## Runtime Strategy

- Do not materialize a full year of 64-band AEF data in memory.
- Use row chunks on the 30 m target grid.
- Prefer `pyarrow.parquet.ParquetWriter` or partitioned Parquet writes through
  pandas/pyarrow for chunk output.
- Keep feature columns as `float32` in the full-grid and sample tables.
- Keep row/col ids as integer columns.
- Write the summary and manifest incrementally from chunk-level counters.
- During inference, read full-grid Parquet chunks, predict, write prediction
  chunks, and discard each chunk before reading the next one.

Expected target-grid dimensions for the current tile:

```text
target height = ceil(8192 / 3) = 2731
target width = ceil(8192 / 3) = 2731
target cells/year = 7,458,361
```

A `target_row_chunk_size` of 128 produces roughly:

```text
128 x 2731 = 349,568 cells/chunk
```

At 64 float32 feature bands, that is roughly 90 MB for the feature matrix before
DataFrame overhead, which is acceptable for a first local implementation.

## Fast Path

The fast path should use the same code path with a small target-grid row/column
window.

Default fast window:

```text
year = 2022
row_window = [900, 1156]
col_window = [900, 1156]
```

Before finalizing this window, the implementation should verify that it contains
at least:

- one Kelpwatch-observed positive row;
- one Kelpwatch-observed zero row;
- one assumed-background row.

If the configured window does not satisfy those conditions, the command should
raise a clear error and print candidate windows derived from observed label
locations.

## Validation

Before full data runs:

```bash
make check
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast
```

After implementation and fast-path review:

```bash
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Unit tests should use tiny synthetic rasters and labels to verify:

- 10 m input cells aggregate to the expected 30 m target-grid count.
- Kelpwatch points are assigned to the expected target-grid cells.
- Unmatched target-grid cells are retained as `assumed_background`.
- Duplicate station mappings are handled deterministically.
- Fast output includes observed positive, observed zero, and assumed-background
  rows.
- Sample weights are computed from full-grid and sampled background counts.
- Baseline metrics are emitted overall and by `label_source`.
- Full-grid prediction writes the same row count it reads.
- Map generation consumes full-grid predictions, not sample predictions.

## Acceptance Checkpoints

The task is complete when:

- The full-grid manifest reports about 7.46 million 30 m target cells per year
  before any documented filtering.
- The sampled modeling table is no longer station-only.
- Task 08 trains from `models.baselines.input_table`.
- The model is applied to the full-grid artifact with streamed inference.
- Task 09 maps corrected predictions from full-grid streamed predictions.
- Task 10 report is regenerated with the correction, streamed full-grid
  prediction results, and stratified metrics.
- `make check` passes.

## Deferred Follow-Ups

These are not required for Task 11:

- Land/ocean/coastline masks for stratified background sampling.
- 10 m weak-label experiments.
- CNN/chip/tensor datasets.
- Out-of-core model training.
