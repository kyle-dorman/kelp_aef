# Task: Extract AEF Footprint GeoJSON

## Goal

Create the smoke-test region geometry by extracting one footprint polygon from
the configured Monterey AEF tile.

## Inputs

- Config file: `configs/monterey_smoke.yaml`.
- Source tile URI recorded in config:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000008192-0000008192.tiff`.
- Local raw tile path after manual staging:
  `/Volumes/x10pro/kelp_aef/raw/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000008192-0000008192.tiff`.

## Outputs

- `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.

## Config File

- `configs/monterey_smoke.yaml`.

## Plan/Spec Requirement

No separate spec required. Keep the implementation narrow and artifact-focused.

## Validation Command

```bash
make check
ogrinfo -so /Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Geometry: footprint of the configured AEF `10N` tile.
- Years: geometry is year-independent for the 2018-2022 smoke window.

## Acceptance Criteria

- GeoJSON exists at the configured path.
- GeoJSON contains exactly one polygon or multipolygon feature.
- Geometry CRS is EPSG:4326.
- Geometry is a tile footprint, not a per-cell grid.
- The footprint overlaps the local 2022 AEF raster bounds.

## Known Constraints Or Non-Goals

- Do not download additional AEF years in this task.
- Do not create a named region registry yet.
- Do not write generated GeoJSON into the git checkout.
