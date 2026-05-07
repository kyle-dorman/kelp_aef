# Task: Query AEF STAC GeoParquet Catalog

## Goal

Write the first package-backed catalog query step that uses the AEF STAC
GeoParquet index to find the annual AEF assets intersecting the Monterey smoke
footprint and years.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- AEF STAC GeoParquet URL:
  `https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet`.
- Visual QA URL for manual catalog inspection:
  `https://developmentseed.org/stac-map/?href=https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet`.
- Region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Years: 2018-2022.

## Outputs

- Catalog query result:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`.
- Human-readable query summary:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include the expected catalog
columns, geometry predicate, year filtering, and how the script avoids reading
or downloading pixel data.

## Validation Command

```bash
make check
kelp-aef query-aef-catalog --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- Adds a package-backed CLI command, `kelp-aef query-aef-catalog`.
- Reads the STAC GeoParquet catalog from the configured Source Cooperative URL.
- Filters by the configured years and intersection with the footprint GeoJSON.
- Writes one query result artifact with the selected catalog rows.
- Writes a JSON summary with row counts by year, selected asset hrefs, CRS or
  projection metadata if present, and geometry bounds.
- Identifies the relevant TIFF and VRT assets for each selected tile when both
  are available.
- Does not download or read AEF pixel arrays.

## Known Constraints Or Non-Goals

- Do not download AEF COGs in this task.
- Do not build the feature table in this task.
- Prefer the provided VRT asset for later raster reads when the catalog exposes
  it, because the upstream README warns that the TIFFs are bottom-up.
- The exact catalog schema should be discovered during implementation rather
  than assumed from file names.
