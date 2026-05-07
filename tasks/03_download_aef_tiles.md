# Task: Download AEF Tiles From Catalog Query

## Goal

Write the first AEF download step that consumes the Monterey catalog query
artifact, downloads only the selected annual tile assets, mirrors the Source
Cooperative key layout locally, and writes a tile manifest.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- Catalog query result:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`.
- Catalog query summary:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`.
- AEF provider: Source Cooperative bucket
  `us-west-2.opendata.source.coop`.
- AEF prefix: `tge-labs/aef/v1/annual`.
- Grid: `10N`.
- Region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.

## Outputs

- Raw AEF mirror paths under:
  `/Volumes/x10pro/kelp_aef/raw/aef/v1/annual/{year}/10N/`.
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`.
- AEF source metadata contribution to:
  `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include download idempotency,
resume behavior, checksum or size validation if available, and how local paths
are derived from catalog asset hrefs.

## Validation Command

```bash
make check
kelp-aef download-aef --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- Adds a package-backed CLI command, `kelp-aef download-aef`.
- Downloads only the assets selected by `aef_monterey_catalog_query.parquet`.
- Mirrors the Source Cooperative key layout below
  `/Volumes/x10pro/kelp_aef/raw/aef`.
- Downloads or records the matching VRT assets when available, and records which
  local asset should be used for analysis reads.
- Manifest contains exactly one selected tile record per configured year unless
  the catalog query explicitly shows multiple intersecting tiles are required.
- Manifest records source href, local path, preferred read path, year, grid,
  bounds, CRS, raster shape, transform, band count, band names if available,
  nodata, file size, and validation status.
- Re-running the command skips already valid local files.

## Known Constraints Or Non-Goals

- Do not query the STAC catalog in this task except to validate that the query
  artifact exists and has the expected fields.
- Do not aggregate embeddings or build the feature table in this task.
- Do not bulk-download unrelated AEF tiles or years.
