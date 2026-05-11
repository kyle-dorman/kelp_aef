# Task 15: Download NOAA CUDEM / Coastal DEM

## Goal

Create the first package-backed script for downloading or registering the
preferred Monterey topo-bathy source from NOAA Coastal DEMs / CUDEM.

This task should create exactly one dataset-specific downloader. It should not
download CUSP, 3DEP, GEBCO, or Copernicus data.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note from P1-09.
- NOAA Coastal DEMs / CUDEM source page:
  <https://www.ncei.noaa.gov/products/coastal-elevation-models>.
- Monterey region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Current full-grid artifact, for bounds context only:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.

## Outputs

- New package module for the CUDEM downloader, for example:
  `src/kelp_aef/domain/noaa_cudem.py`.
- CLI command wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef download-noaa-cudem`.
- Raw CUDEM/Coastal DEM artifact under:
  `/Volumes/x10pro/kelp_aef/raw/domain/noaa_cudem/`.
- CUDEM source manifest under:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cudem_source_manifest.json`.
- Unit tests for manifest construction, dry-run behavior, and local path
  derivation.

## Config File

Use `configs/monterey_smoke.yaml`.

Add only concrete CUDEM local paths and manifest paths needed by later tasks.
Do not add CUSP or 3DEP paths in this task.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include:

- Exact NOAA CUDEM/Coastal DEM product or artifact selected for Monterey.
- Expected source URI and local mirror path.
- Whether the task downloads the source file or registers an already-downloaded
  file.
- Dry-run behavior.
- Idempotency and checksum/file-size validation.

## Implementation Plan

- Add a small package-backed downloader module. Avoid notebook-only workflow.
- Add one CLI command for this source with `--dry-run`, `--manifest-output`,
  `--force`, and timeout options if remote downloads are used.
- Use structured logging for progress, selected source, local path decisions,
  and validation status.
- Write transfers to a temporary sibling path such as `*.part`, then replace
  the final path only after validation succeeds.
- Build a source manifest even in dry-run mode.
- Inspect local raster metadata with maintained geospatial libraries, not ad
  hoc parsing.
- Add docstrings to every new function, including private helpers and test
  helpers.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_noaa_cudem.py
kelp-aef download-noaa-cudem --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/noaa_cudem_source_manifest_dry_run.json
```

Full validation if code/config changes are made:

```bash
make check
```

Real data command, to run only after reviewing the dry-run plan:

```bash
kelp-aef download-noaa-cudem --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022 for downstream model context.
- CUDEM/Coastal DEM is a static support layer.

## Acceptance Criteria

- Adds one package-backed CUDEM downloader command.
- Dry-run writes a manifest without downloading large data.
- Real run downloads or registers only the selected Monterey CUDEM/Coastal DEM
  source.
- Manifest records source name, source URI, local path, date accessed, CRS,
  vertical datum when available, bounds, units, elevation/depth sign convention,
  raster resolution, file size or checksum, and license/access notes.
- Manifest includes a Monterey coverage check.
- Re-running the command skips already valid local files unless `--force` is
  passed.

## Known Constraints Or Non-Goals

- Do not download NOAA CUSP shoreline data.
- Do not download USGS 3DEP fallback data.
- Do not align CUDEM to the 30 m grid.
- Do not build depth bins or domain masks.
- Do not change model training, prediction, maps, or reports.
- Do not bulk-download all NOAA coastal DEM products.
