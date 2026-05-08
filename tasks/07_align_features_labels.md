# Task 07: Align Features And Labels Into First Table

## Goal

Build the first modeling table for the Monterey smoke test by aligning
AlphaEarth annual embedding features with the derived annual Kelpwatch labels.
Use a flat parquet table for this first end-to-end path because the immediate
models are tabular baselines.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Annual labels:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Annual label manifest:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual_manifest.json`.
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`.
- AEF read assets declared in the tile manifest, using the preferred read path
  for each year.
- Smoke-test years: 2018-2022.
- AEF bands: `A00` through `A63`.

## Outputs

- Aligned table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`.
- Alignment summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_training_table_summary.csv`.
- Alignment manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table_manifest.json`.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config entries should cover:

- Label input path.
- AEF tile manifest path.
- Output table, manifest, and summary paths.
- Years to align.
- AEF bands to retain.
- Alignment method name, initially `mean_10m_to_kelpwatch_30m`.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should state the
files to change, exact output schema, validation commands, and any assumptions
about Kelpwatch station coordinates and the 30 m support window.

## Proposed CLI

```bash
uv run kelp-aef align --config configs/monterey_smoke.yaml
```

## Implementation Notes

- Read labels from the annual label parquet produced by Task 06.
- Read AEF tile paths from the AEF tile manifest instead of reconstructing paths
  from strings.
- Prefer each manifest asset's local VRT or other preferred read path for raster
  reads, because raw upstream TIFFs may have orientation and codec issues.
- Transform Kelpwatch station coordinates into the AEF raster CRS with standard
  geospatial libraries such as GeoPandas, pyproj, rasterio, or rioxarray.
- Aggregate the 10 m AEF pixels that fall within each Kelpwatch 30 m station
  support to one feature vector per station-year.
- Start with the mean of valid AEF values for each band, ignoring the AEF NoData
  value.
- Keep diagnostic columns for the number of expected, valid, and missing AEF
  pixels used per station-year.
- Preserve the annual label columns needed by the first tabular baselines,
  including `kelp_max_y`, `kelp_fraction_y`, threshold diagnostics, year,
  station id, longitude, and latitude.
- Use structured logging for year-level progress, raster paths, row counts,
  missing-feature counts, output paths, and recoverable diagnostics.

## Expected Output Schema

Minimum columns:

- `year`
- `kelpwatch_station_id`
- `longitude`
- `latitude`
- `kelp_max_y`
- `kelp_fraction_y`
- annual label diagnostic columns from Task 06
- `A00` through `A63`
- `aef_expected_pixel_count`
- `aef_valid_pixel_count`
- `aef_missing_pixel_count`
- `aef_alignment_method`
- `aef_source_path`

The exact schema can include additional provenance columns, but the first
baseline task must be able to read this parquet table without joining hidden
state from another artifact.

## Validation Command

```bash
make check
uv run kelp-aef align --config configs/monterey_smoke.yaml
```

Add a focused unit test with a tiny synthetic multiband raster and synthetic
labels to verify:

- The 10 m to 30 m aggregation returns the expected band means.
- AEF NoData values are excluded from means and counted.
- Output parquet columns and manifest fields are stable.
- Missing or out-of-footprint stations are reported clearly.

## Smoke-Test Region And Years

- Region: Monterey Peninsula footprint GeoJSON from the selected AEF tile.
- Years: 2018-2022.

## Acceptance Criteria

- The command writes the aligned parquet table, summary CSV, and manifest to the
  configured artifact paths.
- The aligned table has one row per retained Kelpwatch station-year. For the
  current Monterey annual labels, the expected full-row count is 151,160 unless
  the implementation reports and justifies dropped rows.
- AEF feature columns `A00` through `A63` are present and numeric.
- The summary reports row count, missing AEF feature count, valid pixel count
  distribution, years, label class balance diagnostics, and output paths.
- Logs clearly identify each year processed, each raster path read, and final
  artifact paths.
- `make check` passes.

## Known Constraints And Non-Goals

- Do not train or evaluate models in this task.
- Do not create 10 m labels, replicated child labels, or cubic-resampled labels
  in this task.
- Do not create CNN chips, tensors, Zarr stores, TFRecords, or spatial model
  inputs in this task. The parquet table is the first tabular artifact only.
- Do not expand beyond the Monterey smoke footprint or bulk-download broader AEF
  coverage.
- Do not hand-roll geometry, reprojection, raster-window, or table logic when a
  maintained geospatial or scientific Python library can do it clearly.
