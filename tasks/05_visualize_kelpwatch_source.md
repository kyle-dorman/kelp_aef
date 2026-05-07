# Task: Visualize Kelpwatch Source QA

## Goal

Create the first Monterey-focused visual QA outputs for the downloaded
Kelpwatch NetCDF before deriving `labels_annual.parquet`.

This task should answer basic source questions:

- Does the NetCDF cover the Monterey smoke footprint?
- Which variable is the likely canopy-area label source for `kelp_max_y`?
- Do quarterly and annual-max values have plausible ranges and missingness?
- Are the 2018-2022 Kelpwatch values spatially located where expected?
- Can we inspect the Monterey subset interactively or in an external GIS tool?

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json`.
- Downloaded Kelpwatch NetCDF path from the source manifest.
- Region geometry:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Years: 2018-2022.

## Outputs

- Annual-max map contact sheet:
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_monterey_annual_max_qa.png`.
- Quarterly aggregate time-series figure:
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_monterey_quarterly_timeseries_qa.png`.
- Value and missingness summary table:
  `/Volumes/x10pro/kelp_aef/reports/tables/kelpwatch_monterey_source_qa.csv`.
- GIS-friendly annual-max raster export for external inspection:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_monterey_annual_max_2018_2022.tif`.
- Interactive HTML QA viewer:
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_monterey_interactive_qa.html`.
- Optional small metadata supplement:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_monterey_source_qa.json`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include the selected NetCDF
variable, spatial subsetting approach, temporal subsetting approach, and the
exact figure/table outputs.

## Implementation Approach

- Add a package-backed CLI command, likely:

```bash
kelp-aef visualize-kelpwatch --config configs/monterey_smoke.yaml
```

- Read the NetCDF path and candidate label variable from the Kelpwatch source
  manifest. Do not hard-code the revision-specific NetCDF filename.
- Use external libraries for geospatial and NetCDF work:
  - `xarray` / `h5netcdf` for lazy NetCDF reads.
  - `geopandas` / `shapely` for the footprint geometry.
  - `rioxarray` or other maintained geospatial helpers if CRS-aware clipping is
    needed.
  - `matplotlib` for static QA figures.
- First inspect the NetCDF coordinate structure. If it is a regular rectilinear
  grid, subset by coordinate windows before loading data. If it is stored as
  coordinate vectors or sparse-like arrays, use the appropriate xarray/geospatial
  indexing path rather than hand-rolled geometry logic.
- Restrict the loaded data to the Monterey footprint and 2018-2022 quarters.
- For visualization only, compute annual max canopy area in memory:
  `max(quarterly canopy area per pixel, by year)`.
- Use a shared color scale across yearly map panels. Use a robust upper bound
  such as p99 or the physical maximum if confirmed by metadata, and record the
  chosen scale in the QA JSON/table.
- Plot zeros, missing values, and nonzero canopy distinctly enough that a blank
  or all-zero subset is obvious.
- Export the Monterey annual-max subset as a small GIS-friendly GeoTIFF, ideally
  one band per year with band descriptions `2018` through `2022`. This is the
  easiest path for inspection in QGIS or other external tools.
- Also create a lightweight interactive local HTML viewer if the subset size is
  reasonable. Prefer a static self-contained HTML artifact with year toggles,
  footprint overlay, and a simple value legend. If a full-resolution interactive
  grid is too heavy, use a decimated preview for HTML while keeping the GeoTIFF
  full resolution.

## Validation Command

```bash
make check
kelp-aef visualize-kelpwatch --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- CLI command is package-backed and accepts `--config`.
- Command reads the Kelpwatch source manifest and fails clearly if the NetCDF is
  missing or if the label source variable is ambiguous.
- Annual-max contact sheet shows one Monterey panel per year from 2018 through
  2022 with a shared legend/color scale and footprint context.
- GIS export opens as a georeferenced raster with one band per year and no
  dependency on the original NetCDF.
- Interactive HTML opens locally and allows visually switching between the
  2018-2022 annual-max layers, or explicitly documents that the full-resolution
  subset was too large and uses a downsampled preview.
- Quarterly time-series figure shows aggregate canopy area inside the smoke
  footprint for 2018-2022 and makes missing quarters visible.
- Summary table includes, per year, pixel count, valid count, missing count,
  zero count, nonzero count, min, median, p95, p99, max, and aggregate canopy
  area.
- Outputs are written under `/Volumes/x10pro/kelp_aef/reports` or
  `/Volumes/x10pro/kelp_aef/interim`, not tracked in git.

## Known Constraints Or Non-Goals

- Do not build `labels_annual.parquet` in this task.
- Do not align Kelpwatch to AEF embeddings in this task.
- Do not define binary `kelp_present_y` thresholds.
- Do not interpret the figures as independent biomass validation.
- Avoid full West Coast plotting; subset before loading or plotting whenever the
  NetCDF coordinate structure allows it.
