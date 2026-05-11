# Task 18: Query And Download NOAA CRM California Mosaic

## Goal

Create package-backed query and download scripts for a NOAA Coastal Relief Model
(CRM) California topo-bathy mosaic that can support broad kelp-domain filtering.

The first implementation should be query-first. It should select only NOAA CRM
products or product subsets that intersect the configured target-grid footprint
and write a manifest for inspection. Do not download full regional CRM NetCDF
files or extracted subsets until the query manifest has been reviewed.

This CRM workflow is the primary broad topo-bathy source for the first
approximately 100 m depth filter. NOAA CUDEM can remain a higher-resolution QA
source where it has coverage, but it should not be the primary mask input while
its selected Monterey tiles leave large gaps.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note:
  `docs/phase1_bathymetry_dem_source_decision.md`.
- NOAA Coastal Relief Model product page:
  <https://www.ncei.noaa.gov/products/coastal-relief-model>.
- NOAA Southern California CRM version 2 metadata:
  <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem%3A4970>.
- NOAA CRM THREDDS catalog:
  <https://www.ngdc.noaa.gov/thredds/catalog/crm/cudem/catalog.html>.
- Monterey region footprint:
  `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
- Current full-grid artifact, for target-grid bounds context:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.

## Outputs

- New package module for the CRM query/download workflow, for example:
  `src/kelp_aef/domain/noaa_crm.py`.
- CLI commands wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef query-noaa-crm`.
  `kelp-aef download-noaa-crm`.
- Raw CRM artifact or extracted subset directory under:
  `/Volumes/x10pro/kelp_aef/raw/domain/noaa_crm/`.
- CRM query manifest under:
  `/Volumes/x10pro/kelp_aef/interim/noaa_crm_query_manifest.json`.
- CRM source/download manifest under:
  `/Volumes/x10pro/kelp_aef/interim/noaa_crm_source_manifest.json`.
- Unit tests for product-registry loading, geometry intersection, manifest
  construction, dry-run behavior, and local path derivation.

## Config File

Use `configs/monterey_smoke.yaml`.

Add only concrete CRM local paths, source registry entries, and manifest paths
needed by this task, preferably under a narrow `domain.noaa_crm` block.

Expected config fields:

- Product registry or product-registry path.
- Query manifest path.
- Source/download manifest path.
- Raw CRM output directory.
- Download mode, defaulting to clipped subset when the service supports it.
- Optional query padding in degrees or meters.
- Remote timeout and chunk size.

Do not add domain-mask output paths in this task.

## Plan/Spec Requirement

Brief implementation plan before editing code. Include:

- Exact NOAA CRM products in the California registry.
- How the command derives the current target-grid footprint or bounds.
- How product intersection is computed in EPSG:4326.
- Query manifest schema, including skipped products and reasons.
- Whether later download uses full NetCDF files or clipped service extracts.
- Dry-run behavior.
- Idempotency and checksum/file-size validation.

## Product Registry

Start with this California-focused registry:

- `crm_socal_v2_1as`: NOAA Southern California CRM version 2, 1 arc-second,
  available from `crm_socal_1as_vers2.nc`. Use it for California grid area
  south of about 37 degrees north.
- `crm_vol7_2025`: NOAA CRM Volume 7 Central Pacific, 1 arc-second, available
  from `crm_vol7_2025.nc`. Use it for California grid area north of about
  37 degrees north.
- `crm_vol8_2025`: NOAA CRM Volume 8 Northwest Pacific, 1 arc-second. Keep in
  the registry only so future Oregon/Washington scale-up can exclude or include
  it by intersection; it should not be selected for the current Monterey grid
  unless the configured footprint actually intersects it.

The query should read service metadata when available and not rely only on
hard-coded product names. Record both configured/known product bounds and any
actual coordinate bounds discovered from THREDDS or OPeNDAP metadata.

## Implementation Plan

- Add a small package-backed query/download module under
  `src/kelp_aef/domain/`.
- Add one CLI command that builds a CRM query manifest by intersecting the
  configured target-grid footprint with the CRM product registry.
- For the current Monterey target grid, expect the query to select the SoCal
  v2 product and, if the grid crosses 37 degrees north, a small Volume 7
  Central Pacific intersection.
- Query mode should write a manifest only. It should not download regional
  NetCDF files or extracted raster subsets.
- The query manifest should include selected products, skipped products,
  intersecting bounds, recommended subset bounds, source URLs, service URLs,
  CRS, vertical datum, units, sign convention, nominal resolution, and source
  role.
- Add one CLI command that consumes the reviewed query manifest and downloads
  or registers only selected CRM products or subsets.
- Prefer clipped downloads to the current grid bounds if NOAA THREDDS, WCS, or
  OPeNDAP access makes that clear and reliable. Fall back to full product
  download only if explicitly configured.
- Write transfers to a temporary sibling path such as `*.part`, then replace
  the final path only after validation succeeds.
- Build query/download manifests even in dry-run mode.
- Inspect local raster metadata with maintained geospatial libraries, not ad
  hoc parsing.
- Add docstrings to every new function, including private helpers and test
  helpers.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_noaa_crm.py
uv run kelp-aef query-noaa-crm --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/noaa_crm_query_manifest_dry_run.json
uv run kelp-aef download-noaa-crm --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/noaa_crm_query_manifest_dry_run.json --manifest-output /private/tmp/noaa_crm_source_manifest_dry_run.json
```

Full validation if code/config changes are made:

```bash
make check
```

Real query command, safe to run before download review:

```bash
uv run kelp-aef query-noaa-crm --config configs/monterey_smoke.yaml
```

Real download command, to run only after reviewing the query manifest:

```bash
uv run kelp-aef download-noaa-crm --config configs/monterey_smoke.yaml
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022 for downstream model context.
- CRM is a static support layer; the source manifest should not repeat data by
  model year.

## Acceptance Criteria

- Adds one package-backed CRM query command and one package-backed CRM download
  command.
- Query command writes a manifest of CRM products or subsets intersecting the
  configured target-grid footprint.
- Query command records non-selected registry products and why they were
  skipped.
- Query command performs no large data download.
- Download command consumes a reviewed query manifest and downloads or
  registers only selected CRM products or subsets.
- Manifests record source name, source URI, service URI, selected product ID,
  local path, date accessed, CRS, vertical datum, bounds, units,
  elevation/depth sign convention, raster resolution, file size or checksum,
  and license/access notes.
- Manifest includes a Monterey coverage check and an all-California coverage
  note for the SoCal v2 plus Volume 7 pairing.
- Re-running the command skips already valid local files unless `--force` is
  passed.

## Known Constraints Or Non-Goals

- Do not build the plausible-kelp domain mask; that starts in P1-12.
- Do not align CRM to the 30 m grid; that starts in P1-11.
- Do not use CRM depth/elevation as model predictors in Phase 1.
- Do not download NOAA CUDEM, CUSP, 3DEP, GEBCO, or Copernicus data.
- Do not bulk-download every NOAA CRM volume.
- Do not run the real CRM download until the query manifest has been inspected.
- Do not rerun model training, prediction, maps, or reports.
