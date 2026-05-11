# Task 17: Query And Download USGS 3DEP Land DEM Fallback

## Goal

Create package-backed query and download scripts for the USGS 3DEP
1/3 arc-second DEM as the U.S. land-side fallback for the Monterey domain
filter.

This task should create exactly one dataset-specific query/download pair. It
should not download CUDEM, CUSP, GEBCO, or Copernicus data.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note from P1-09.
- USGS 3DEP 1/3 arc-second DEM source page:
  <https://data.usgs.gov/datacatalog/data/USGS%3A3a81321b-c153-416f-98b7-cc8e5f0e17c3>.
- Monterey region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Current full-grid artifact, for bounds context only:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.

## Outputs

- New package module for the 3DEP query/download workflow, for example:
  `src/kelp_aef/domain/usgs_3dep.py`.
- CLI commands wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef query-usgs-3dep`.
  `kelp-aef download-usgs-3dep`.
- Raw 3DEP artifact under:
  `/Volumes/x10pro/kelp_aef/raw/domain/usgs_3dep/`.
- 3DEP query manifest under:
  `/Volumes/x10pro/kelp_aef/interim/usgs_3dep_query_manifest.json`.
- 3DEP source manifest under:
  `/Volumes/x10pro/kelp_aef/interim/usgs_3dep_source_manifest.json`.
- Unit tests for manifest construction, dry-run behavior, and local path
  derivation.

## Config File

Use `configs/monterey_smoke.yaml`.

Add only concrete 3DEP local paths and manifest paths needed by later tasks. Do
not add CUDEM or CUSP paths in this task.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include:

- Exact USGS 3DEP artifact selected for Monterey.
- How the command queries the available 3DEP source index, package listing,
  service, or metadata endpoint with the configured region geometry.
- Expected query manifest path, selected source URI fields, and local mirror
  path policy.
- Whether the query task downloads only a small source index or uses a
  pre-registered local source index.
- Whether the download task downloads selected DEM artifact(s) or registers
  already-downloaded files.
- Dry-run behavior.
- Idempotency and checksum/file-size validation.

## Implementation Plan

- Add a small package-backed query/download module. Avoid notebook-only
  workflow.
- Add one CLI command to query 3DEP source metadata using the configured region
  geometry and write a selected-artifact manifest for review.
- Add one CLI command to download or register only the 3DEP artifact(s) selected
  by the query manifest.
- Both commands should include `--dry-run`, `--manifest-output`, `--force`, and
  timeout options where relevant.
- Use structured logging for progress, selected source, local path decisions,
  and validation status.
- Write transfers to a temporary sibling path such as `*.part`, then replace
  the final path only after validation succeeds.
- Build query/download manifests even in dry-run mode.
- Inspect local raster metadata with maintained geospatial libraries, not ad
  hoc parsing.
- Mark 3DEP manifest entries as land-side fallback, not preferred topo-bathy,
  because 3DEP does not solve the offshore bathymetry side.
- Add docstrings to every new function, including private helpers and test
  helpers.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_usgs_3dep.py
kelp-aef query-usgs-3dep --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/usgs_3dep_query_manifest_dry_run.json
kelp-aef download-usgs-3dep --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/usgs_3dep_query_manifest_dry_run.json --manifest-output /private/tmp/usgs_3dep_source_manifest_dry_run.json
```

Full validation if code/config changes are made:

```bash
make check
```

Real data command, to run only after reviewing the dry-run plan:

```bash
kelp-aef query-usgs-3dep --config configs/monterey_smoke.yaml
kelp-aef download-usgs-3dep --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022 for downstream model context.
- 3DEP DEM is a static support layer.

## Acceptance Criteria

- Adds one package-backed 3DEP query command and one package-backed 3DEP
  download command.
- Dry-run writes manifests without downloading large data.
- Query output selects 3DEP artifact(s) by intersection or coverage with the
  configured Monterey geometry.
- Real download/register run downloads or registers only the selected Monterey
  3DEP artifact(s).
- Query and download manifests record source name, source URI, local path, date
  accessed, CRS, vertical datum when available, bounds, units, elevation sign
  convention, raster resolution, file size or checksum, and license/access
  notes.
- Manifests include a Monterey coverage check.
- Manifests mark the source as a land-only fallback, not the preferred
  topo-bathy source.
- Re-running the command skips already valid local files unless `--force` is
  passed.

## Known Constraints Or Non-Goals

- Do not download NOAA CUDEM/Coastal DEM data.
- Do not download NOAA CUSP shoreline data.
- Do not use 3DEP as the offshore bathymetry source.
- Do not align 3DEP to the 30 m grid.
- Do not build depth bins or domain masks.
- Do not change model training, prediction, maps, or reports.
