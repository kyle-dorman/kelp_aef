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

## Validation Command

```bash
make check
kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula via the configured AEF footprint GeoJSON.
- Years: 2018, 2019, 2020, 2021, 2022.

## Acceptance Criteria

- Downloader/reader is package-backed and runnable from the `kelp-aef` CLI.
- Source manifest records source URL or endpoint, local paths, years, seasons or
  dates, variables, units, CRS, bounds, and file sizes.
- Metadata summary includes CRS, bounds, variables, seasons, years, units, and
  missing-data notes.
- Downloads are limited to the configured smoke scope where the source allows
  spatial/temporal filtering.
- The task identifies the value field needed for `kelp_max_y`.

## Known Constraints Or Non-Goals

- The Kelpwatch source endpoint and file format are not decided yet.
- Do not build `labels_annual.parquet` in this task.
- Do not define `kelp_present_y` thresholds before inspecting value ranges.
- Do not claim Kelpwatch labels are independent field truth.
