# Task 14: Bathymetry And DEM Source Plan

## Goal

Write the source and threshold decision note for the Monterey bathymetry/DEM
domain filter before implementing any downloaders or masks.

The goal is to define the domain-filter data contract, not to download data or
change model behavior.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Phase 1 plan: `docs/phase1_model_domain_hardening.md`.
- Active Phase 1 checklist: `docs/todo.md`.
- Current full-grid artifact:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Preferred U.S. coastal topo-bathy source: NOAA Coastal DEMs / CUDEM,
  <https://www.ncei.noaa.gov/products/coastal-elevation-models>.
- Shoreline side source: NOAA CUSP,
  <https://coast.noaa.gov/digitalcoast/data/cusp.html>.
- U.S. land-only fallback: USGS 3DEP 1/3 arc-second DEM,
  <https://data.usgs.gov/datacatalog/data/USGS%3A3a81321b-c153-416f-98b7-cc8e5f0e17c3>.
- Later global bathymetry fallback: GEBCO_2026,
  <https://www.gebco.net/data-products/gridded-bathymetry-data/>.
- Later global land-side fallback: Copernicus DEM GLO-30,
  <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>.

## Outputs

- Decision note under `docs/` defining the Monterey domain-filter source
  priority, script sequence, artifact paths, metadata fields, threshold
  assumptions, and non-goals.
- Updated `docs/todo.md` only if the source decision changes task order or
  adds/removes a dataset-specific P1-10 downloader task.
- No downloaded data, no config path changes, and no model/report behavior
  changes in this task.

## Config File

Use `configs/monterey_smoke.yaml` as context only.

Do not add source paths to the config until a P1-10 downloader task has a real
local artifact and manifest path.

## Plan/Spec Requirement

This task is the required decision note before implementation. No separate
pre-plan is needed.

The note should be short enough to review before coding, but explicit enough
that each downloader task can create one script without re-deciding the data
contract.

## Source Priority

Use this priority for the first Monterey implementation:

1. NOAA CUDEM / Coastal DEM plus NOAA CUSP shoreline for U.S. coastal work.
2. USGS 3DEP only as a U.S. land-side fallback.
3. GEBCO_2026 only as a later global bathymetry fallback.
4. Copernicus DEM GLO-30 only as a later global land-side fallback.

For Phase 1 Monterey, the implementation tasks should start with one downloader
script per near-term source:

- NOAA CUDEM / Coastal DEM.
- NOAA CUSP shoreline.
- USGS 3DEP fallback.

Do not create GEBCO or Copernicus downloader tasks for Monterey unless the
decision note explicitly finds NOAA coverage unavailable or unsuitable.

## Threshold Contract

Assume topo-bathy elevation values where positive is land elevation and
negative is ocean depth, unless selected source metadata proves otherwise.

Use these first-pass derived fields and categories:

```text
depth_m = max(0, -elevation_m)

strict kelp candidate:
  oceanward of shoreline
  depth_m between 0 and 40 m

QA/permissive candidate:
  oceanward of shoreline
  depth_m between 0 and 50 m

definite land:
  landward of shoreline
  elevation_m > 5 m

ambiguous coast:
  elevation_m between -5 m and +5 m
```

Do not use `elevation_m > 0` alone as a hard coastline rule. Datums, tides,
beaches, marshes, cliffs, and mixed pixels make the immediate shoreline noisy.

After the mask exists, validate known Kelpwatch-positive labels by depth. Any
positives deeper than 40 m should be inspected before tightening the cutoff.

## Validation Command

Docs-only validation:

```bash
git diff --check -- docs/todo.md docs/
```

If the decision note updates task files, include those files in the diff check.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split policy remains train 2018-2020, validation 2021, test 2022.
- Bathymetry, DEM, and shoreline inputs are static support layers.

## Acceptance Criteria

- The decision note selects the Monterey source priority and explains why NOAA
  coastal sources come before global products.
- The note identifies one downloader script task per selected near-term source.
- The note records expected manifest fields: source URI, local path, CRS,
  vertical datum when available, bounds, units, resolution, sign convention,
  checksum or file size, date accessed, and license/access notes.
- The note preserves the strict 40 m and permissive 50 m depth thresholds.
- The note states that NOAA CUSP, not elevation alone, is the source for
  landward/oceanward side classification.
- The note keeps bathymetry/DEM out of the model predictors for Phase 1 unless
  a later decision explicitly changes that scope.

## Known Constraints Or Non-Goals

- Do not download data in this task.
- Do not add config paths that point to nonexistent artifacts.
- Do not align sources to the 30 m target grid.
- Do not build the domain mask.
- Do not change model training, prediction, residual maps, or reports.
- Do not start full West Coast or global source work.
