# Task: Build Initial Annual Kelpwatch Labels

## Goal

Create the first production annual Kelpwatch label artifact for the Monterey
smoke workflow:

```text
/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet
```

This step should turn the raw quarterly Kelpwatch NetCDF into a clean annual
station/pixel-by-year label table. It should not sample AlphaEarth features or
perform alignment.

The active first-pass alignment policy remains Option A: keep labels at their
native Kelpwatch 30 m support and later aggregate AEF 10 m embeddings to that
30 m label support. This task should preserve the fields needed for that
alignment without resampling labels to the AEF grid.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json`.
- Downloaded Kelpwatch NetCDF path from the source manifest.
- Region geometry:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Years: 2018-2022.
- Label variable: Kelpwatch `area`.

## Outputs

- Annual label table:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Recommended QA summary table:
  `/Volumes/x10pro/kelp_aef/reports/tables/labels_annual_summary.csv`.
- Recommended metadata supplement:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual_manifest.json`.

## Config File

- `configs/monterey_smoke.yaml`.
- Use `labels.paths.annual_labels` for the Parquet output path.
- Use `labels.paths.source_manifest` for the Kelpwatch source manifest path.
- Use `region.geometry.path` for the footprint.
- Use `reports.tables_dir` for the optional summary table.

## Plan/Spec Requirement

No separate spec is required after this task file unless implementation reveals
an unexpected NetCDF layout or missing-data convention. Before editing code,
write a short implementation note covering files to change, output schema, and
validation commands.

## Proposed CLI

Use the existing scaffolded command:

```bash
kelp-aef build-labels --config configs/monterey_smoke.yaml
```

The command should be restartable and overwrite only its declared outputs.

## Implementation Approach

- Add a package-backed label derivation module, likely under
  `src/kelp_aef/labels/`.
- Keep functions small and documented with docstrings.
- Use external libraries for geospatial and tabular work:
  - `xarray` / `h5netcdf` for NetCDF reads.
  - `geopandas` / `shapely` for footprint filtering.
  - `pandas` / `pyarrow` for Parquet output.
- Reuse or factor shared Kelpwatch source helpers from the visualization step
  rather than duplicating station-layout and footprint-selection logic.
- Read the local NetCDF path and selected label variable from the Kelpwatch
  source manifest. Validate that the selected variable is `area`, not `area_se`.
- Select Kelpwatch stations whose `longitude` / `latitude` points intersect the
  configured Monterey footprint.
- Restrict to configured years using Kelpwatch `year` / `quarter` variables when
  present. Fall back to decoded `time` only if those variables are absent.
- Treat Kelpwatch fill values and negative area values as missing before
  aggregation.
- Build one row per selected Kelpwatch station and year.
- Compute the primary continuous target:

```text
kelp_max_y = max(area across available quarters for station/year)
```

- Preserve enough seasonal detail to support later label-aggregation research
  without rereading the raw NetCDF:
  - quarter-level area columns such as `area_q1`, `area_q2`, `area_q3`,
    `area_q4`.
  - `max_area_quarter`.
  - `valid_quarter_count`.
  - `nonzero_quarter_count`.
- Do not make binary labels the primary target in this task. If a convenience
  `kelp_present_gt0_y` field is added, document it as a derived diagnostic
  using `kelp_max_y > 0`, not as the final binary-label contract.
- Record binary-threshold diagnostics without choosing the final threshold.
  Bell et al. describe Kelpwatch-style canopy area as emergent canopy detected
  by Landsat/MESMA, note that submerged canopy may fall below the detection
  limit, and report that low-biomass patches are more sensitive to tidal
  correction error. This supports evaluating thresholds above zero before using
  a binary target.

## Proposed Output Schema

Required columns:

```text
year
kelpwatch_station_id
longitude
latitude
kelp_max_y
area_q1
area_q2
area_q3
area_q4
max_area_quarter
valid_quarter_count
nonzero_quarter_count
source_variable
source_package_id
source_revision
```

Optional diagnostic columns:

```text
kelp_present_gt0_y
kelp_present_ge_1pct_y
kelp_present_ge_5pct_y
kelp_present_ge_10pct_y
source_units
label_aggregation
region_name
kelp_fraction_y
```

`kelp_fraction_y` should be `kelp_max_y / 900` when included. It is useful for
future 10 m label experiments, but the primary first-pass target remains
`kelp_max_y` at Kelpwatch 30 m support.

Candidate binary diagnostics should be treated as threshold exploration aids:

```text
kelp_present_gt0_y    = kelp_fraction_y > 0
kelp_present_ge_1pct_y = kelp_fraction_y >= 0.01
kelp_present_ge_5pct_y = kelp_fraction_y >= 0.05
kelp_present_ge_10pct_y = kelp_fraction_y >= 0.10
```

These are not accepted final binary thresholds until model behavior and label
noise near low canopy values are evaluated.

## Validation Command

```bash
make check
kelp-aef build-labels --config configs/monterey_smoke.yaml
```

Add a fast unit test using a tiny station-layout NetCDF fixture. The test should
verify:

- footprint filtering keeps only expected stations.
- annual max is computed from quarter values.
- missing/fill values do not become valid label values.
- output schema includes the required columns.
- Parquet output can be read back.

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

Expected real-data scale from task 05:

```text
30,232 selected Kelpwatch stations x 5 years = 151,160 annual rows
```

## Acceptance Criteria

- `kelp-aef build-labels --config configs/monterey_smoke.yaml` writes
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- The output has one row per selected station/year for 2018-2022.
- The output includes `kelp_max_y` derived from Kelpwatch `area`, not `area_se`.
- The output includes station coordinates and stable station identifiers needed
  by the alignment step.
- The output keeps Kelpwatch-native 30 m label support and does not create
  replicated or resampled 10 m labels.
- The output preserves quarter-level area values or equivalent seasonal
  diagnostics needed to compare annual max, fall-only, winter-only, and
  fall-to-winter-loss labels later.
- The summary table records row counts, station counts, year counts, missingness,
  zero/nonzero counts, and value ranges.
- The summary table includes counts for candidate binary thresholds, at minimum
  `>0`, `>=1%`, `>=5%`, and `>=10%` canopy fraction.
- The command logs source variable, selected station count, selected years, and
  output paths.
- `make check` passes after implementation.

## Known Constraints Or Non-Goals

- Do not sample AEF rasters in this task.
- Do not build `aligned_training_table.parquet` in this task.
- Do not train or evaluate models in this task.
- Do not decide the final binary-label threshold in this task.
- Do not broaden beyond the Monterey footprint or configured 2018-2022 years.
