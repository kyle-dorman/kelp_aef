# Data And Artifact Contract

## Purpose

This document defines where data and generated artifacts live, which outputs are
expected from the Monterey feasibility spike, and what can be tracked in git.

The repo should remain focused on source code, configs, tests, and docs. Raw
data, generated tables, rasters, figures, and models should live outside the
repo unless explicitly promoted as tiny fixtures.

## Canonical Root

Canonical data and generated artifact root:

```text
/Volumes/x10pro/kelp_aef
```

Expected directory layout:

```text
/Volumes/x10pro/kelp_aef/
  raw/
    kelpwatch/
    aef/
    aef_samples/
  geos/
  interim/
  processed/
  models/
  reports/
    figures/
    tables/
```

Create the directories with:

```bash
mkdir -p /Volumes/x10pro/kelp_aef/raw/kelpwatch \
  /Volumes/x10pro/kelp_aef/raw/aef \
  /Volumes/x10pro/kelp_aef/raw/aef_samples \
  /Volumes/x10pro/kelp_aef/geos \
  /Volumes/x10pro/kelp_aef/interim \
  /Volumes/x10pro/kelp_aef/processed \
  /Volumes/x10pro/kelp_aef/models \
  /Volumes/x10pro/kelp_aef/reports/figures \
  /Volumes/x10pro/kelp_aef/reports/tables
```

## Git Policy

Tracked in git:

- Source code under `src/`.
- Tests under `tests/`.
- Configs under `configs/`.
- Documentation under `docs/`.
- Small hand-authored fixtures only when needed for deterministic tests.

Not tracked in git:

- Raw Kelpwatch data.
- AlphaEarth samples, chips, rasters, Zarr stores, or Earth Engine exports.
- AlphaEarth tile footprint GeoJSONs under the external artifact root.
- Derived Parquet/CSV/JSON artifacts from normal pipeline runs.
- Generated maps, figures, and report tables from normal pipeline runs.
- Trained model files and serialized estimators.

The `.gitignore` file ignores repo-local `data/`, `reports/`, `models/`,
Earth Engine export folders, common raster/Zarr outputs, and common model
serialization formats. This is defensive: the intended artifact location is the
external root above, not the checkout.

## Visual QA Samples

No visual QA outputs are intentionally tracked yet.

If a tiny output needs to become a tracked reference sample, put it under a
purpose-specific fixture directory such as:

```text
tests/fixtures/
docs/qa_samples/
```

Tracked QA samples should be small, manually selected, and documented with:

- Source command or producer.
- Input config.
- Date generated.
- Reason it is useful to keep.

Do not track large rasters, Zarr stores, full-resolution maps, or model outputs
as visual QA samples.

## Monterey Feasibility Outputs

The first feasibility milestone should produce these outputs under the external
artifact root.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| AEF tile footprint | `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson` | Single-footprint smoke geometry extracted from the configured AlphaEarth tile. | Not tracked |
| AEF catalog query result | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet` | STAC GeoParquet rows selected for the Monterey footprint and 2018-2022 years. | Not tracked |
| AEF catalog query summary | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json` | Human-readable counts, bounds, and selected asset hrefs from the catalog query. | Not tracked |
| AEF tile manifest | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json` | Local and source URIs for the matching 2018-2022 AEF tiles. | Not tracked |
| Kelpwatch source manifest | `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json` | Downloaded/source Kelpwatch files used by the smoke config. | Not tracked |
| Metadata summary | `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json` | CRS, bounds, variables, seasons, years, units, and missing-data notes for source inputs. | Not tracked |
| Annual labels | `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet` | Kelpwatch seasonal labels collapsed to the configured annual target. | Not tracked |
| AEF samples | `/Volumes/x10pro/kelp_aef/interim/aef_samples.parquet` | AlphaEarth features staged for the configured region and years. | Not tracked |
| Aligned training table | `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet` | Grid-cell-by-year modeling table with A00-A63, location/year fields, and labels. | Not tracked |
| Split manifest | `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet` | Train/validation/test assignments for year-holdout evaluation. | Not tracked |
| Sample QA figure | `/Volumes/x10pro/kelp_aef/reports/figures/sample_kelpwatch_vs_aef.png` | Visual check that labels and embeddings cover the same region. | Not tracked by default |
| Predicted map | `/Volumes/x10pro/kelp_aef/reports/figures/predicted_map.png` | Map of model predictions for the held-out year or region. | Not tracked by default |
| Residual map | `/Volumes/x10pro/kelp_aef/reports/figures/residual_map.png` | Spatial residual check against Kelpwatch-style labels. | Not tracked by default |
| Area-bias summary | `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_summary.csv` | Region/year area bias and aggregate error summary. | Not tracked |

These paths are the current Monterey smoke-test contracts. The Kelpwatch source
format still needs source inspection before data-heavy implementation. AEF asset
selection should come from the Source Cooperative STAC GeoParquet catalog query.
