# Phase 0 Monterey Smoke Decision

Status: closed for now as of 2026-05-08.

## Goal

Record the selected Monterey feasibility smoke test and its closeout results.
This document is now a Phase 0 decision and outcome note. It should not be read
as a Phase 1 plan.

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
- Final model policy: train the simple ridge baseline on the
  background-inclusive sample without background expansion weights, then stream
  predictions over the full-grid artifact.

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

The manifest is filled by the AEF download task using the catalog query result
rather than hard-coded tile names.

## Kelpwatch Decision

Kelpwatch remains the weak-label source. Phase 0 implemented a
downloader/reader and records source files in:

```text
/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json
```

Annual label derivation was implemented after source format, variables, units,
seasons, CRS, and missing-data conventions were inspected.

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
- Full-grid aligned table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`
- Background-inclusive training sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`
- Split manifest: `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`
- Sample predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
- Full-grid predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- Baseline metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`
- Full-grid area-bias summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`
- Phase 0 report snapshot:
  `docs/report_snapshots/monterey_phase0_model_analysis.md`

## Non-Goals

- Do not use provisional config bounds as the active smoke geometry.
- Do not create a per-cell grid GeoJSON for the smoke config; use a footprint.
- Do not bulk-download the full AEF collection.
- Do not start full West Coast processing.
- Do not frame Kelpwatch-style label reproduction as independent biomass truth.
- Do not assume the next phase before reviewing the Phase 0 report.

## Implementation Results

Phase 0 produced the Monterey smoke pipeline end to end:

- Kelpwatch source download, metadata inspection, and annual label derivation.
- AEF STAC catalog query, tile download, and local VRT/TIFF handling.
- AEF 10 m to 30 m aggregation for the selected Monterey tile.
- Station-centered alignment retained as a QA/reference artifact.
- Full-grid 30 m alignment with `kelpwatch_station` and `assumed_background`
  label provenance.
- Background-inclusive sampled model input.
- Year-holdout split manifest.
- No-skill and ridge baseline metrics.
- Streamed full-grid ridge predictions.
- Static and interactive observed/predicted/residual maps.
- Phase 0 model-analysis report.

Key final row counts:

- Annual labels: 151,160 rows across 2018-2022.
- Full-grid aligned table: 37,291,805 rows.
- Background-inclusive model input sample: 1,400,809 rows.
- Retained model rows after missing-feature drops: 1,342,631 rows.
- Full-grid predictions: 37,291,805 rows in 430 Parquet parts.
- Primary report/map split: 2022 test year with 7,458,361 full-grid prediction
  rows.

Final simple baseline behavior:

- Ridge is trained without background expansion weights.
- Kelpwatch-station 2022 test rows: R2 around 0.356 and area bias around
  -36.7%.
- Background-inclusive sample 2022 test rows: area bias around +16.6%.
- Full-grid 2022 area bias remains very poor because small positive predictions
  over many assumed-background cells accumulate into large area overprediction.
- High-canopy rows remain underpredicted.

These are good enough to close Phase 0 as a feasibility spike, but not good
enough to call the model solved.

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

Final closeout validation:

```bash
make check
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Deferred Questions

- Which question should define Phase 1: sampling/calibration, target framing,
  stronger baselines, spatial holdouts, ingestion hardening, or scale-up.
- Whether and how to calibrate small positive full-grid predictions that
  accumulate over assumed-background cells.
- Whether alternative Kelpwatch target framings, such as fall-only,
  winter-only, annual mean, persistence, or binary thresholds, are more
  appropriate than annual max.
- Whether to add a second region before broader scale-up.
