# Phase 1 Bathymetry And DEM Source Decision

Status: accepted for P1-09 on 2026-05-11.

## Decision

For the Monterey Phase 1 domain filter, use regional U.S. coastal sources
before global products, but prioritize continuous coverage over maximum local
resolution:

1. NOAA Coastal Relief Model (CRM) Southern California v2 plus CRM Volume 7
   Central Pacific as the first broad California topo-bathy source.
2. NOAA CUSP shoreline for shoreline-side classification when the first mask
   needs landward versus oceanward context.
3. NOAA CUDEM / Coastal DEM only as a higher-resolution QA or refinement source
   where the selected tiles actually cover the target grid.
4. USGS 3DEP 1/3 arc-second DEM only as a U.S. land-side fallback.
5. GEBCO_2026 only as a later global bathymetry fallback.
6. Copernicus DEM GLO-30 only as a later global land-side fallback.

This filter is a conservative candidate-area filter for Kelpwatch-style kelp
mapping. It is not ecological truth, and it should not become a model predictor
in Phase 1 unless a later decision explicitly changes that scope.

Coverage update on 2026-05-11: the selected NOAA CUDEM tiles cover only about
43% of the current AEF footprint and miss many Kelpwatch-positive cells south
of Monterey. Treat those gaps as source coverage gaps, not as off-domain ocean.
Do not use CUDEM as the primary Phase 1 mask input unless a later source query
finds continuous coverage for the configured grid.

Implementation update on 2026-05-11: add a query-first NOAA CRM California
mosaic workflow. For now, query the configured target-grid footprint against
the CRM product registry and write a reviewable manifest. Do not download CRM
source data until the manifest has been inspected. P1-11 should align the CRM
mosaic, not CUDEM, to the 30 m target grid.

## Why This Source Order

NOAA CRM is the preferred first mask source because the current filtering need
is coarse and coverage-sensitive. A permissive depth cutoff such as
approximately 100 m should remove obviously deep offshore cells while retaining
plausible kelp habitat. For that use, a continuous 1 arc-second coastal
topo-bathy mosaic is more useful than a finer CUDEM source with large local
coverage gaps.

NOAA Coastal DEMs / CUDEM remain useful where available because the NCEI
Coastal DEM catalog is explicitly coastal and includes tiled CUDEM
bathymetric-topographic products at 1/9 and 1/3 arc-second access paths.
However, the current selected CUDEM tile set is not continuous across the
configured Monterey/Big Sur target grid, so CUDEM should be QA/refinement rather
than the first primary mask input.

NOAA CUSP is the shoreline-side source because it is a U.S. coastal shoreline
product, distributed as ESRI shapefile, with national coastal coverage. It
should determine landward versus oceanward side; the model should not infer the
coastline from elevation alone.

USGS 3DEP 1/3 arc-second DEM is retained only as a U.S. land-side fallback. It
is approximately 10 m, bare-earth, and NAVD88-referenced over the continental
United States, but it is a topographic DEM and does not solve offshore
bathymetry.

GEBCO_2026 is deferred to global fallback status. It is current, global, free,
and includes elevation in meters at 15 arc-second spacing plus a Type
Identifier grid, but that spacing is too coarse for the first Monterey
nearshore filter.

Copernicus DEM GLO-30 is deferred to global land-side fallback status. It has
global 30 m coverage, but it is a DSM that includes buildings, infrastructure,
and vegetation, so it is less appropriate than bare-earth U.S. elevation data
where 3DEP is available.

## Downloader Task Sequence

Implement one package-backed query/download workflow per selected near-term
source when the source has a catalog, index, package listing, or national
archive that should be filtered to the configured Monterey geometry before
download:

- `P1-10a`: NOAA CUDEM / Coastal DEM query/download pair,
  `tasks/15_download_noaa_cudem.md`.
- `P1-10b`: NOAA CUSP shoreline query/download pair,
  `tasks/16_download_noaa_cusp.md`.
- `P1-10c`: USGS 3DEP fallback query/download pair,
  `tasks/17_download_usgs_3dep.md`.
- `P1-10d`: NOAA CRM California mosaic query/download pair,
  `tasks/18_query_download_noaa_crm.md`.

Do not add GEBCO or Copernicus downloader tasks for Monterey unless NOAA CRM
coverage is unavailable or unsuitable after source inspection. Do not run a real
CRM download until the P1-10d query manifest has been inspected.

## Expected Artifacts

Raw source files should remain outside the code repo:

```text
/Volumes/x10pro/kelp_aef/raw/domain/noaa_cudem/
/Volumes/x10pro/kelp_aef/raw/domain/noaa_cusp/
/Volumes/x10pro/kelp_aef/raw/domain/usgs_3dep/
/Volumes/x10pro/kelp_aef/raw/domain/noaa_crm/
```

CUSP, 3DEP, and CRM source manifests should be written under:

```text
/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json
/Volumes/x10pro/kelp_aef/interim/usgs_3dep_source_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_crm_source_manifest.json
```

The NOAA CUDEM, CUSP, 3DEP, and CRM tasks use separate query and download
manifests because selected artifacts should come from the configured Monterey
geometry:

```text
/Volumes/x10pro/kelp_aef/interim/noaa_cudem_tile_query_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_cudem_tile_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_cusp_query_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json
/Volumes/x10pro/kelp_aef/interim/usgs_3dep_query_manifest.json
/Volumes/x10pro/kelp_aef/interim/usgs_3dep_source_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_crm_query_manifest.json
/Volumes/x10pro/kelp_aef/interim/noaa_crm_source_manifest.json
```

Small Monterey coverage footprints or indexes, if needed, may be written under:

```text
/Volumes/x10pro/kelp_aef/geos/
```

Do not add config paths for these sources until the corresponding P1-10 task
creates or registers a concrete local artifact.

## Manifest Fields

Every P1-10 source manifest should record:

- Source name and source URI.
- Selected product/artifact identifier.
- Local path.
- Download or registration date.
- CRS and vertical datum when available.
- Bounds and Monterey coverage check.
- Units.
- Elevation/depth sign convention.
- Raster resolution or vector scale/resolution notes.
- File size or checksum.
- License/access notes.
- Source role: primary broad topo-bathy, higher-resolution QA topo-bathy,
  shoreline-side source, land-side fallback, global bathymetry fallback, or
  global land-side fallback.

The CUSP manifest should clearly distinguish shoreline-side data from
topo-bathy/elevation data.

## Threshold Contract

Assume topo-bathy elevation values where positive is land elevation and
negative is ocean depth, unless selected source metadata proves otherwise.

```text
depth_m = max(0, -elevation_m)

broad first-pass kelp candidate:
  oceanward of shoreline
  depth_m between 0 and 100 m

strict kelp candidate for later QA:
  oceanward of shoreline
  depth_m between 0 and 40 m

intermediate QA candidate:
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

After the aligned mask exists, validate known Kelpwatch-positive labels by
depth. The first CRM-backed mask may use approximately 100 m as a permissive
cutoff. Any positives deeper than 40-50 m should be inspected before tightening
the cutoff, not dropped automatically during the first broad filter.

## Non-Goals

- Do not download source data in P1-09.
- Do not align bathymetry, DEM, or shoreline data to the 30 m target grid in
  P1-09 or P1-10.
- Do not build the domain mask before P1-12.
- Do not use bathymetry/DEM as model predictors in Phase 1.
- Do not start full West Coast or global source work in the Monterey Phase 1
  domain-filter tasks.
- Do not bulk-download full NOAA CRM regional products before inspecting the
  CRM query manifest for the configured target-grid intersection.
- Do not bulk-download global GEBCO or Copernicus products for Monterey unless
  the source decision is revised.

## Source Checks

Checked on 2026-05-11:

- NOAA Coastal DEMs / CUDEM:
  <https://www.ncei.noaa.gov/products/coastal-elevation-models>.
- NOAA Coastal Relief Model:
  <https://www.ncei.noaa.gov/products/coastal-relief-model>.
- NOAA Southern California CRM version 2:
  <https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem%3A4970>.
- NOAA CRM THREDDS catalog:
  <https://www.ngdc.noaa.gov/thredds/catalog/crm/cudem/catalog.html>.
- NOAA CUSP:
  <https://coast.noaa.gov/digitalcoast/data/cusp.html>.
- USGS 3DEP 1/3 arc-second DEM:
  <https://data.usgs.gov/datacatalog/data/USGS%3A3a81321b-c153-416f-98b7-cc8e5f0e17c3>.
- GEBCO gridded bathymetry:
  <https://www.gebco.net/data-products/gridded-bathymetry-data/>.
- Copernicus DEM:
  <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>.
