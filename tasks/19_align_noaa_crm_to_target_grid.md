# Task 19: Align NOAA CRM And Validate Domain Sources To The 30 m Target Grid

## Goal

Align the selected NOAA Coastal Relief Model California topo-bathy source(s) to
the existing Monterey 30 m AEF/Kelpwatch target grid, then validate the aligned
CRM values against the other downloaded domain-support sources.

This task should produce the static depth/elevation support layer needed by the
first broad domain-mask task. It should also produce cross-source QA summaries
so P1-12 can decide whether CRM, CUDEM, 3DEP, and CUSP are behaving as expected.
It should not build the mask itself.

The first mask can use a permissive depth threshold, such as approximately
100 m, so continuity and coverage are more important than CUDEM-level local
resolution. Preserve CRM product provenance so later QA can separate older
Southern California CRM cells from newer Central Pacific CRM cells.

Post-download source check on 2026-05-11:

- NOAA CRM downloaded two valid NetCDF rasters: SoCal v2 and Volume 7.
- NOAA CUDEM downloaded nine valid GeoTIFF topo-bathy tiles, but they only cover
  part of the current grid and should be treated as higher-resolution QA where
  available.
- USGS 3DEP downloaded four valid GeoTIFF land DEM tiles. It is land-side QA or
  fallback only, not offshore bathymetry.
- NOAA CUSP downloaded `West.zip`, which opens as 82,942 `LineString` shoreline
  features in EPSG:4269. It is not a polygon product and should not be
  raster-aligned as depth/elevation in this task. It can be validated for
  source coverage/provenance here, while shoreline-side classification belongs
  in P1-12 or a later mask task.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note:
  `docs/phase1_bathymetry_dem_source_decision.md`.
- NOAA CRM source manifest:
  `/Volumes/x10pro/kelp_aef/interim/noaa_crm_source_manifest.json`.
- NOAA CRM query manifest, for product/subset provenance:
  `/Volumes/x10pro/kelp_aef/interim/noaa_crm_query_manifest.json`.
- NOAA CUDEM tile manifest, for higher-resolution topo-bathy QA:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cudem_tile_manifest.json`.
- USGS 3DEP source manifest, for land-side DEM QA:
  `/Volumes/x10pro/kelp_aef/interim/usgs_3dep_source_manifest.json`.
- NOAA CUSP source manifest, for shoreline-vector source coverage QA only:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json`.
- Existing full-grid target table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Existing full-grid target manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json`.
- Monterey region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.

## Outputs

- Package-backed CRM alignment module, for example:
  `src/kelp_aef/domain/crm_alignment.py`.
- CLI command wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef align-noaa-crm`.
- Static aligned domain support table under:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- Domain-source alignment manifest under:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm_manifest.json`.
- Domain-source alignment QA table under:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_noaa_crm_summary.csv`.
- Cross-source comparison QA table under:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_domain_source_comparison.csv`.
- Unit tests for config loading, target-grid extraction, raster sampling or
  resampling, product-boundary handling, optional QA source handling, manifest
  construction, and fast-path behavior.

## Config File

Use `configs/monterey_smoke.yaml`.

Add concrete CRM alignment paths and optional QA-source paths needed by this
task, preferably under `domain.noaa_crm.alignment`. Keep CRM as the primary
output source and make CUDEM/3DEP/CUSP validation explicitly optional or
recoverable when those manifests are missing.

Expected config fields:

- Source manifest path.
- Optional CUDEM tile manifest path.
- Optional USGS 3DEP source manifest path.
- Optional CUSP source manifest path.
- Target grid table or manifest path.
- Output table path.
- Output manifest path.
- QA summary table path.
- Cross-source comparison table path.
- Resampling method.
- Product priority or boundary rule.
- Fast-path row/column window for agent-safe validation.

Do not add GEBCO, Copernicus, or domain-mask output paths in this task.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include:

- How the command derives the static target grid from the existing full-grid
  artifact without duplicating rows across years.
- Whether CRM values are sampled at target-cell centers or averaged across
  target-cell footprints, and why.
- How Southern California v2 and Central Pacific products are split or
  prioritized near 37 degrees north.
- Which downloaded raster sources are sampled for QA (`noaa_cudem` and
  `usgs_3dep`) and how missing coverage is represented.
- How CUSP is validated as a `LineString` shoreline source without treating it
  as a raster depth/elevation source.
- Expected output schema.
- Fast-path behavior.
- Validation and QA summary fields.
- Idempotency rules for existing outputs.

## Implementation Plan

- Add a small package-backed alignment module under `src/kelp_aef/domain/`.
- Add one CLI command that reads the CRM source manifest, opens local CRM
  raster(s), derives the unique target grid from the full-grid artifact, and
  writes one static row per target cell.
- Use CRM as the primary aligned depth/elevation source. Derive
  `crm_depth_m = max(0, -crm_elevation_m)` using the sign convention in the CRM
  manifest/config.
- Also sample downloaded raster support sources for validation where local files
  and manifests exist:
  - NOAA CUDEM, as higher-resolution topo-bathy QA where its tiles cover the
    target cell.
  - USGS 3DEP, as land-side DEM QA or fallback where its tiles cover the target
    cell.
- Use maintained geospatial libraries such as Rasterio, xarray, rioxarray,
  PyArrow, Polars, or Pandas. Do not hand-roll CRS transforms, raster windows,
  or table readers.
- For NOAA CRM NetCDFs, use xarray coordinate variables as the source of truth
  for geographic extent and sampling coordinates. A post-download check on
  2026-05-11 found that Rasterio can report misleading bounds for
  `crm_vol7_2025.nc` from the embedded GeoTransform even though the NetCDF
  `lat` and `lon` coordinate arrays are correct.
- Preserve grid identifiers from the existing full-grid artifact:
  `aef_grid_row`, `aef_grid_col`, `aef_grid_cell_id`, `longitude`, and
  `latitude`.
- Write CRM columns such as:
  `crm_elevation_m`, `crm_depth_m`, `crm_source_product_id`,
  `crm_source_product_name`, `crm_source_path`, `crm_vertical_datum`,
  `crm_alignment_method`, and `crm_value_status`.
- Write optional CUDEM QA columns such as:
  `cudem_elevation_m`, `cudem_depth_m`, `cudem_source_tile_id`,
  `cudem_source_path`, and `cudem_value_status`.
- Write optional 3DEP QA columns such as:
  `usgs_3dep_elevation_m`, `usgs_3dep_source_id`, `usgs_3dep_source_path`, and
  `usgs_3dep_value_status`.
- Do not write CUSP depth/elevation columns. Instead, validate the CUSP manifest
  and local source metadata in the alignment manifest or QA summary: geometry
  type, CRS, bounds, feature count, and whether the shoreline package covers the
  target grid bounds.
- Use an explicit product-boundary rule. A reasonable first rule is SoCal v2
  south of 37 degrees north and CRM Volume 7 north of 37 degrees north, with no
  blending across products unless a later QA pass justifies it.
- Write a manifest with source manifest paths, selected product paths, CRS,
  vertical datum, sign convention, target-grid source, row counts, coverage
  counts, CUSP vector validation, output schema, and QA summary paths.
- Write a QA summary with total cells, valid CRM cells, missing CRM cells,
  source-product counts, min/max elevation, min/max depth, broad depth bins,
  and optional CUDEM/3DEP valid/missing counts.
- Write a cross-source comparison table over cells where sources overlap:
  CRM-vs-CUDEM elevation/depth differences, CRM-vs-3DEP land elevation
  differences, source coverage percentages, and counts of cells where source
  signs disagree near the coast.
- If labels are cheap to join, include a small count of Kelpwatch-positive cells
  by CRM depth bin and by whether CUDEM/3DEP coverage is present.
- Add a fast mode that runs on the configured full-grid fast row/column window
  and writes fast output paths.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_crm_alignment.py
uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml --fast
```

Full validation if code/config changes are made:

```bash
make check
```

Real data command:

```bash
uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: static support layer aligned to the existing full-grid target grid.
  The output should not repeat identical CRM values for each model year.
- Fast path: use the existing `alignment.full_grid.fast` row/column window.

## Acceptance Criteria

- Adds one package-backed CRM alignment command with cross-source validation.
- Reads the existing NOAA CRM source manifest and fails clearly if local primary
  CRM source files or subsets are missing.
- Reads CUDEM, 3DEP, and CUSP manifests when configured and available. Missing
  QA-source manifests should be recorded as skipped validation inputs rather
  than blocking CRM alignment.
- Produces one static row per unique target-grid cell in the selected scope.
- Output table keeps target-grid identifiers and includes CRM elevation,
  derived depth, source product, alignment method, vertical datum, and
  value-status fields.
- Output table includes optional CUDEM and 3DEP QA fields where those rasters
  overlap the target grid.
- Manifest records source provenance, target-grid source, CRS, vertical datum,
  sign convention, row counts, coverage counts, CUSP vector validation, and
  output schema.
- Cross-source comparison table makes CRM/CUDEM/3DEP agreement and coverage
  visible before P1-12 uses the aligned support layer.
- CUSP is validated as shoreline `LineString` data, not aligned as a
  depth/elevation source.
- Fast-path validation writes a small output and QA summary without requiring a
  full-grid run.
- Full run row count matches the unique target-grid cell count from the
  configured full-grid artifact.
- Re-running the command overwrites or skips outputs only according to an
  explicit, logged policy.

## Known Constraints Or Non-Goals

- Do not build the plausible-kelp domain mask; that starts in P1-12.
- Do not classify shoreline side with CUSP in this task; only validate the CUSP
  source package and record its coverage/provenance.
- Do not promote 3DEP or CUDEM to the primary bathymetry source in this task.
- Do not align GEBCO or Copernicus products.
- Do not use CRM depth/elevation as model predictors in Phase 1.
- Do not rerun model training, prediction, maps, or reports.
