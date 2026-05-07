# Task: Stage And Inspect AEF Tiles

## Goal

Stage the matching single-grid AEF GeoTIFFs for 2018-2022, mirror the S3 key
layout locally, and write a manifest that downstream feature/alignment steps can
consume.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- AEF provider: Source Cooperative bucket
  `us-west-2.opendata.source.coop`.
- AEF prefix: `tge-labs/aef/v1/annual`.
- Grid: `10N`.
- Footprint GeoJSON:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Known source examples for 2018 and 2022 from the config.

## Outputs

- Raw TIFF mirror paths under:
  `/Volumes/x10pro/kelp_aef/raw/aef/v1/annual/{year}/10N/`.
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`.
- AEF source metadata contribution to:
  `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include how the task identifies
the matching 2019-2021 object names and how it avoids broad AEF downloads.

## Validation Command

```bash
make check
kelp-aef fetch-aef-chip --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- Manifest contains exactly one local TIFF path for each configured year.
- Local paths mirror the S3 key layout below `/Volumes/x10pro/kelp_aef/raw/aef`.
- Manifest records source URI, local path, year, grid, bounds, CRS, raster shape,
  transform, band count, band names if available, nodata, and file size.
- TIFF inspections confirm the expected annual AEF embedding product and bands.
- The implementation does not bulk-download unrelated AEF tiles or years.

## Known Constraints Or Non-Goals

- The exact matching 2019-2021 object names are not known yet.
- Do not aggregate embeddings or build the feature table in this task.
- Do not introduce a STAC catalog until the manifest becomes insufficient.
