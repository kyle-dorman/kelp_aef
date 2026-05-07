# Phase 0 Monterey Smoke Decision

Status: draft for review.

## Goal

Lock the first feasibility smoke test tightly enough that implementation agents
can work from `configs/monterey_smoke.yaml` without expanding into the full West
Coast or bulk-downloading AlphaEarth/AEF.

## Selected Scope

- Region: Monterey Peninsula.
- Geometry: one footprint GeoJSON extracted from a single AEF `10N` tile.
- Geometry artifact:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Years: 2018-2022.
- Split: train 2018-2020, validate 2021, test 2022.
- Label target: Kelpwatch annual max canopy, `kelp_max_y`.
- Feature product: Source Cooperative AlphaEarth/AEF v1 annual GeoTIFFs.
- Feature grid: one `10N` tile covering the Monterey smoke region.
- Alignment policy: aggregate 10 m AEF embeddings to the Kelpwatch 30 m grid.

## AEF Access Decision

Use the Source Cooperative AEF STAC GeoParquet catalog as the discovery path for
future AEF asset selection:

```text
https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet
```

The catalog can be inspected visually with:

```text
https://developmentseed.org/stac-map/?href=https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet
```

Catalog query output:

```text
/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet
/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json
```

Mirror selected assets from the catalog under the external raw artifact root:

```text
/Volumes/x10pro/kelp_aef/raw/aef/v1/annual/{year}/10N/{tile}.tiff
```

Known source tile examples:

- 2018:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/xdz8z3a9znk5b1j75-0000008192-0000008192.tiff`
- 2022:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000008192-0000008192.tiff`

The smoke config records those known examples and a manifest output path:

```text
/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json
```

The manifest should be filled by the AEF download task using the catalog query
result rather than hard-coded tile names.

## Kelpwatch Decision

Kelpwatch remains the weak-label source. The first implementation work should
write a downloader/reader rather than assume a manually staged Kelpwatch path.
The downloader should record source files in:

```text
/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json
```

Do not implement annual label derivation until the source format, variables,
units, seasons, CRS, and missing-data conventions are inspected.

## Artifact Paths

- AEF footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`
- AEF raw mirror root: `/Volumes/x10pro/kelp_aef/raw/aef`
- AEF catalog query result:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`
- AEF catalog query summary:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`
- Kelpwatch raw root: `/Volumes/x10pro/kelp_aef/raw/kelpwatch`
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json`
- Metadata summary: `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`
- Annual labels: `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`
- AEF sample table: `/Volumes/x10pro/kelp_aef/interim/aef_samples.parquet`
- Aligned training table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`
- Split manifest: `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`

## Non-Goals

- Do not use provisional config bounds as the active smoke geometry.
- Do not create a per-cell grid GeoJSON for the smoke config; use a footprint.
- Do not bulk-download the full AEF collection.
- Do not start full West Coast processing.
- Do not frame Kelpwatch-style label reproduction as independent biomass truth.

## Validation Plan

- Docs/config validation: inspect diffs and run `make check`.
- AEF footprint task: verify the GeoJSON has one EPSG:4326 polygon feature and
  overlaps the configured local AEF tile bounds.
- AEF catalog query task: verify the query result contains the selected
  intersecting Monterey assets for 2018-2022 and records TIFF/VRT hrefs when
  available.
- AEF download task: verify the tile manifest contains one selected local asset
  per year for 2018-2022 and that each raster exposes the expected embedding
  bands.
- Kelpwatch downloader task: verify the source manifest and metadata summary
  include CRS, bounds, variables, seasons, years, units, and missing-data notes.

## Open Questions

- The exact matching AEF object names for 2019-2021 should come from the STAC
  GeoParquet catalog query.
- The Kelpwatch download endpoint, file format, and local naming convention still
  need source inspection.
- The binary label target `kelp_present_y` and any threshold should remain
  deferred until Kelpwatch value ranges are inspected.
