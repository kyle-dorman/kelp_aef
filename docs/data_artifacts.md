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
    domain/
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

The first feasibility milestone produced these outputs under the external
artifact root. They are generated artifacts and should not be tracked in git.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| AEF tile footprint | `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson` | Single-footprint smoke geometry extracted from the configured AlphaEarth tile. | Not tracked |
| AEF catalog query result | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet` | STAC GeoParquet rows selected for the Monterey footprint and 2018-2022 years. | Not tracked |
| AEF catalog query summary | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json` | Human-readable counts, bounds, and selected asset hrefs from the catalog query. | Not tracked |
| AEF tile manifest | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json` | Local and source URIs for the matching 2018-2022 AEF tiles. | Not tracked |
| Kelpwatch source manifest | `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json` | Downloaded/source Kelpwatch files used by the smoke config. | Not tracked |
| Metadata summary | `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json` | CRS, bounds, variables, seasons, years, units, and missing-data notes for source inputs. | Not tracked |
| Annual labels | `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet` | Kelpwatch seasonal labels collapsed to the configured annual target. | Not tracked |
| Station aligned table | `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet` | Station-centered alignment artifact retained as QA/reference; not the primary Phase 0 model input after Task 11. | Not tracked |
| Full-grid aligned table | `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet` | Full AEF-aligned 30 m target grid with Kelpwatch-station and assumed-background rows. | Not tracked |
| Full-grid alignment manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json` | Row counts, label-source counts, assets, and alignment settings for the full-grid artifact. | Not tracked |
| Background sample table | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet` | Background-inclusive sampled model input used by the Phase 0 ridge baseline. | Not tracked |
| Background sample manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table_manifest.json` | Sampling policy, schema, and sample count metadata. | Not tracked |
| Split manifest | `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet` | Train/validation/test assignments for year-holdout evaluation. | Not tracked |
| Baseline model | `/Volumes/x10pro/kelp_aef/models/baselines/ridge_kelp_fraction.joblib` | Serialized Phase 0 ridge model payload. | Not tracked |
| Sample predictions | `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet` | No-skill and ridge predictions on the sampled model frame. | Not tracked |
| Full-grid predictions | `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet` | Streamed ridge predictions for the full grid across 2018-2022. | Not tracked |
| Full-grid prediction manifest | `/Volumes/x10pro/kelp_aef/interim/baseline_full_grid_prediction_manifest.json` | Row count, part count, and label-source counts for full-grid predictions. | Not tracked |
| Baseline metrics | `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv` | No-skill and ridge metrics by split and label source. | Not tracked |
| Residual map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png` | Three-panel observed, predicted, and residual map for the held-out year. | Not tracked by default |
| Interactive residual map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.html` | Local HTML map for exploring observed, predicted, and residual values. | Not tracked by default |
| Area bias by year | `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv` | Full-grid year-level area-bias summary. | Not tracked |
| Phase 0 report snapshot | `docs/report_snapshots/monterey_phase0_model_analysis.md` | Repo copy of the closed Phase 0 report. | Tracked |
| Phase 1 Markdown report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md` | Active model-hardening report with model comparison and data-health sections. | Not tracked |
| Phase 1 HTML report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html` | Rendered Phase 1 report with embedded figures and model comparison. | Not tracked |
| Phase 1 PDF report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf` | PDF Phase 1 report with the same core model-comparison content. | Not tracked |

## Phase 0 Artifact Counts

The final Phase 0 run used the Monterey smoke config with 2018-2022:

- Full-grid aligned table: 37,291,805 rows.
- Per year full-grid rows: 7,428,207 `assumed_background` rows and 30,154
  `kelpwatch_station` rows.
- Background-inclusive model input sample: 1,400,809 rows.
- Baseline retained model rows after missing-feature drops: 1,342,631 rows.
- Full-grid prediction dataset: 37,291,805 rows in 430 Parquet parts.
- Phase 0 report primary map/report year: 2022 test split with 7,458,361
  full-grid prediction rows.

## Phase 0 Interpretation Notes

The current Phase 0 ridge baseline is trained on the background-inclusive sample
without sample expansion weights. This avoids the near-zero collapse observed
with population-expanded background weights, but it is still not calibrated on
the full grid. In the final report, station-supported Kelpwatch rows and
assumed-background rows are interpreted separately.

Phase 1 has been selected as Monterey annual-max model and domain hardening.
The high-level artifact categories are defined in
`docs/phase1_model_domain_hardening.md`. Add exact artifact rows here only when
the corresponding implementation task defines concrete paths in
`configs/monterey_smoke.yaml`.

## Phase 1 Domain Source Outputs

These outputs support the Monterey domain-filter tasks. They are generated
artifacts and should not be tracked in git.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| NOAA CUSP query manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_query_manifest.json` | Selected CUSP regional package, local mirror path, source metadata, and Monterey coverage check. | Not tracked |
| NOAA CUSP source package | `/Volumes/x10pro/kelp_aef/raw/domain/noaa_cusp/West.zip` | NOAA NSDE West regional CUSP shoreline shapefile ZIP for shoreline-side support. | Not tracked |
| NOAA CUSP source manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json` | Transfer status, remote/local file size, vector metadata, and source provenance for the CUSP package. | Not tracked |
