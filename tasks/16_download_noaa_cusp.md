# Task 16: Query And Download NOAA CUSP Shoreline

## Goal

Create package-backed query and download scripts for NOAA CUSP shoreline data
for Monterey landward/oceanward side classification.

This task should create exactly one dataset-specific query/download pair. It
should not download CUDEM, 3DEP, GEBCO, or Copernicus data.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note from P1-09.
- NOAA CUSP source page:
  <https://coast.noaa.gov/digitalcoast/data/cusp.html>.
- Monterey region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Current full-grid artifact, for bounds context only:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.

## Outputs

- New package module for the CUSP query/download workflow, for example:
  `src/kelp_aef/domain/noaa_cusp.py`.
- CLI commands wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef query-noaa-cusp`.
  `kelp-aef download-noaa-cusp`.
- Raw CUSP shoreline artifact under:
  `/Volumes/x10pro/kelp_aef/raw/domain/noaa_cusp/`.
- CUSP query manifest under:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_query_manifest.json`.
- CUSP source manifest under:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json`.
- Unit tests for manifest construction, dry-run behavior, and local path
  derivation.

## Config File

Use `configs/monterey_smoke.yaml`.

Add only concrete CUSP local paths and manifest paths needed by later tasks. Do
not add CUDEM or 3DEP paths in this task.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include:

- Exact NOAA CUSP source artifact selected for Monterey.
- How the command queries the available CUSP source index, package listing, or
  metadata endpoint with the configured region geometry.
- Expected query manifest path, selected source URI fields, and local mirror
  path policy.
- Whether the query task downloads only a small source index or uses a
  pre-registered local source index.
- Whether the download task downloads the selected shoreline artifact or
  registers an already-downloaded file.
- Dry-run behavior.
- Idempotency and checksum/file-size validation.

## Implementation Plan

- Add a small package-backed query/download module. Avoid notebook-only
  workflow.
- Add one CLI command to query CUSP source metadata using the configured region
  geometry and write a selected-artifact manifest for review.
- Add one CLI command to download or register only the CUSP artifacts selected
  by the query manifest.
- Both commands should include `--dry-run`, `--manifest-output`, `--force`, and
  timeout options where relevant.
- Use structured logging for progress, selected source, local path decisions,
  and validation status.
- Write transfers to a temporary sibling path such as `*.part`, then replace
  the final path only after validation succeeds.
- Build query/download manifests even in dry-run mode.
- Inspect local vector metadata with maintained geospatial libraries, not ad
  hoc parsing.
- Record the fields needed by P1-11/P1-12 to determine shoreline side, but do
  not perform side classification in this task.
- Add docstrings to every new function, including private helpers and test
  helpers.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_noaa_cusp.py
kelp-aef query-noaa-cusp --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/noaa_cusp_query_manifest_dry_run.json
kelp-aef download-noaa-cusp --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/noaa_cusp_query_manifest_dry_run.json --manifest-output /private/tmp/noaa_cusp_source_manifest_dry_run.json
```

Full validation if code/config changes are made:

```bash
make check
```

Real data command, to run only after reviewing the dry-run plan:

```bash
kelp-aef query-noaa-cusp --config configs/monterey_smoke.yaml
kelp-aef download-noaa-cusp --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022 for downstream model context.
- CUSP shoreline is a static support layer.

## Acceptance Criteria

- Adds one package-backed CUSP query command and one package-backed CUSP
  download command.
- Dry-run writes manifests without downloading large data.
- Query output selects CUSP shoreline artifact(s) by intersection or coverage
  with the configured Monterey geometry.
- Real download/register run downloads or registers only the selected Monterey
  CUSP shoreline artifact(s).
- Query and download manifests record source name, source URI, local path, date
  accessed, CRS, bounds, vector geometry type, scale/resolution notes, file size
  or checksum, and license/access notes.
- Manifests include a Monterey coverage check.
- Manifests clearly distinguish shoreline-side data from topo-bathy/elevation
  data.
- Re-running the command skips already valid local files unless `--force` is
  passed.

## Known Constraints Or Non-Goals

- Do not download NOAA CUDEM/Coastal DEM data.
- Do not download USGS 3DEP fallback data.
- Do not infer coastline from elevation in this task.
- Do not classify grid cells as landward or oceanward.
- Do not build the domain mask.
- Do not change model training, prediction, maps, or reports.
