# Task: Download And Inspect Kelpwatch

## Goal

Write the first Kelpwatch downloader/reader for the Monterey smoke config and
record enough metadata to define annual label derivation safely.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- Region geometry:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Years: 2018-2022.
- Label source: Kelpwatch.
- Source package: EDI/SBC LTER package `knb-lter-sbc.74`.
- Latest-revision discovery endpoint:
  `https://pasta.lternet.edu/package/eml/knb-lter-sbc/74?filter=newest`.

## Outputs

- Downloaded or staged Kelpwatch source files under:
  `/Volumes/x10pro/kelp_aef/raw/kelpwatch`.
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json`.
- Metadata summary:
  `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include the chosen Kelpwatch
source endpoint or file source, expected file format, and how downloads remain
limited to the Monterey smoke geometry and 2018-2022 years.

Current source decision from exploration:

- Use the EDI/SBC LTER NetCDF package as the canonical Kelpwatch label source:
  `knb-lter-sbc.74`.
- Discover the current revision dynamically instead of hard-coding it. On
  2026-05-07, the newest revision endpoint resolved to revision `32`.
- Revision `32` metadata:
  - Package id: `knb-lter-sbc.74.32`.
  - DOI: `doi:10.6073/pasta/9e67d5f471f448a966f3d6bd81bc5825`.
  - Object name: `LandsatKelpBiomass_2025_Q4_withmetadata.nc`.
  - Entity id: `c2bea785267fa434c40a22e2239bb337`.
  - Data URL:
    `https://pasta.lternet.edu/package/data/eml/knb-lter-sbc/74/32/c2bea785267fa434c40a22e2239bb337`.
  - Size: `2598167830` bytes.
  - EML MD5: `c9ff4edaa5ba5bb1c893b722bde5474d`.
  - Temporal coverage: `1984-03-23` through `2025-12-31`.
  - Spatial coverage: west `-124.77`, east `-114.04`, north `48.40`,
    south `27.01`.
  - Format: NetCDF.
- Package revisions are cumulative quarterly snapshots, not one file per
  quarter. For example:
  - `knb-lter-sbc.74.22` is `LandsatKelpBiomass_2023_Q3_withmetadata.nc`
    covering `1984-03-23` through `2023-09-30`.
  - `knb-lter-sbc.74.23` is `LandsatKelpBiomass_2023_Q4_withmetadata.nc`
    covering `1984-03-23` through `2023-12-31`.
  - `knb-lter-sbc.74.32` is `LandsatKelpBiomass_2025_Q4_withmetadata.nc`
    covering `1984-03-23` through `2025-12-31`.
- Download only the latest cumulative NetCDF unless a future task explicitly
  needs to reproduce an older release.
- The PASTA data endpoint did not honor HTTP range requests during exploration,
  so the initial download cannot be restricted to Monterey or 2018-2022 at the
  source. Limit spatial and temporal scope during the local inspect/subset step.
- Kelpwatch web-app endpoints are useful for QA but should not be the primary
  pixel-level label source:
  - Region API:
    `https://kelp-production-agg.kelpwatch.org/db/region`.
  - Aggregate CSV pattern:
    `https://data-production.kelpwatch.org/aggregates/aggregate-{regionID}-{subregionID}.csv`.
  - Central California aggregate example: region id `2`, subregion id `7`.
  - Vector tile metadata:
    `https://data-production.kelpwatch.org/california/latest.tiles/metadata.json`.
  - These expose aggregate time series and PBF vector-tile map products, not the
    original 30 m NetCDF grid needed for training labels.

## Validation Command

```bash
make check
kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml --dry-run --skip-checksum --manifest-output /private/tmp/kelpwatch_source_manifest_dry_run.json
```

Real download/inspection command:

```bash
kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml
```

The real command downloads the current cumulative NetCDF if it is not already
present under `/Volumes/x10pro/kelp_aef/raw/kelpwatch`, validates MD5 unless
`--skip-checksum` is passed, inspects the local NetCDF metadata, writes the
configured manifest, and updates the shared metadata summary.

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- Downloader/reader is package-backed and runnable from the `kelp-aef` CLI.
- Source manifest records the latest-revision discovery endpoint, resolved
  package id, entity id, source URL, local paths, checksum, years, seasons or
  dates, variables, units, CRS, bounds, and file sizes.
- Metadata summary includes CRS, bounds, variables, seasons, years, units, and
  missing-data notes.
- The downloaded NetCDF is inspected locally and the manifest records that
  spatial/temporal filtering is applied locally because the source endpoint is a
  single cumulative NetCDF.
- The task identifies the value field needed for `kelp_max_y`.

## Known Constraints Or Non-Goals

- The selected Kelpwatch source is a single large cumulative NetCDF, not a
  Monterey-specific file.
- The fast validation path should use `--dry-run`; the default command may
  download a multi-GB NetCDF.
- Do not download all historical package revisions.
- Do not build `labels_annual.parquet` in this task.
- Do not define `kelp_present_y` thresholds before inspecting value ranges.
- Do not claim Kelpwatch labels are independent field truth.
