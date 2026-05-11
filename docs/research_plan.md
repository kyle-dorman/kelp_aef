# AlphaEarth Kelp Mapping Project Plan

## Project Framing

Use AlphaEarth/Satellite Embedding features to learn kelp canopy products from Kelpwatch-derived weak labels across the U.S. West Coast.

Important caveat: Kelpwatch is not independent field truth. It is a Landsat-derived kelp canopy product. The first version should be framed as learning, improving, or reproducing Kelpwatch-style labels from AlphaEarth embeddings, not proving true kelp biomass.

## Phase 0: Monterey Feasibility Spike

Status: complete for now as of 2026-05-08.

The Phase 0 region is Monterey Peninsula, using 2018-2022. Do not start with
the whole West Coast until a later phase explicitly chooses scale-up.

Goal:

```text
Kelpwatch label tile + AlphaEarth embedding tile -> aligned training table -> simple model -> map
```

Success criteria:

- AlphaEarth embeddings load correctly.
- Kelpwatch labels load correctly.
- Coordinate alignment works.
- Full-grid and background-inclusive artifacts are produced.
- A simple model and diagnostics run end to end.
- Outputs expose whether the first model is useful or broken.

AlphaEarth public embeddings are 10 m, 64-dimensional, annual products available in Earth Engine. The Geo-Embeddings xarray/Zarr workflow is useful, but the full collection is enormous, so pull only coastal chips rather than bulk-downloading everything.

Phase 0 outcome:

- Kelpwatch source download/inspection works for the smoke config.
- AEF STAC catalog query and selected tile download work for the Monterey
  footprint and 2018-2022.
- Annual Kelpwatch max labels are built.
- AEF features are aggregated to a 30 m target grid.
- The first station-centered alignment mistake was corrected with a full-grid
  artifact containing `kelpwatch_station` and `assumed_background` rows.
- A ridge baseline trained without background expansion weights has some
  Kelpwatch-station signal but underpredicts high canopy and is poorly
  calibrated on the full grid.
- The closed Phase 0 report snapshot is tracked at
  `docs/report_snapshots/monterey_phase0_model_analysis.md`.
- Current model interpretation now belongs in the active Phase 1 report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.

## Candidate Next Work: Data Ingestion Hardening

This is not the main Phase 1 direction, except for narrow bathymetry/DEM
manifesting needed by the Monterey domain filter.

Inputs:

```text
Kelpwatch:
- seasonal/quarterly 30 m kelp canopy data
- West Coast U.S. subset: Oregon + California, maybe Washington if available

AlphaEarth:
- annual 10 m embeddings
- 64 bands: A00-A63
- years overlapping Kelpwatch and embeddings
```

Codex tasks:

- Create downloader/reader scripts.
- Inspect metadata, CRS, bounds, variables, seasons, and units.
- Write a small manifest of available years and regions.
- Save tiny sample outputs for visual QA.

Expected artifacts:

```text
/Volumes/x10pro/kelp_aef/raw/kelpwatch/
/Volumes/x10pro/kelp_aef/raw/aef_samples/
/Volumes/x10pro/kelp_aef/interim/metadata_summary.json
/Volumes/x10pro/kelp_aef/reports/figures/sample_kelpwatch_vs_aef.png
```

## Candidate Next Work: Derived Labels

Alternative temporal labels are deferred until after Phase 1. Binary labels
derived by thresholding annual max are in scope for Phase 1.

Turn seasonal Kelpwatch into annual or multi-year targets compatible with annual AlphaEarth embeddings.

Candidate targets:

```text
kelp_max_y          = annual max seasonal canopy value
kelp_mean_y         = annual mean seasonal canopy value
fall_kelp_y         = fall kelp value
fall_change_2yr_y   = fall_kelp_y - fall_kelp_y_minus_2
fall_persistent_y   = fall_kelp_y > threshold and fall_kelp_y_minus_1 > threshold
kelp_present_y      = kelp_max_y > threshold
```

Keep both continuous and binary versions:

```text
regression: fractional/canopy amount
classification: present/absent or persistent/not persistent
```

Generate label diagnostics:

- Positive pixels by year, state, and latitude band.
- Missing-data counts.
- Seasonal coverage.
- Maps for a few known kelp regions.

## Candidate Next Work: Alignment

The main AEF/Kelpwatch alignment path is already implemented for Monterey.
Phase 1 alignment work should be limited to domain-mask QA and bathymetry/DEM
alignment to the existing 30 m grid.

Align into the AlphaEarth coordinate frame, then decide whether labels are upsampled from 30 m to 10 m or AlphaEarth embeddings are aggregated from 10 m to 30 m.

Start by aggregating AlphaEarth to the Kelpwatch 30 m grid because Kelpwatch labels are 30 m. The Earth Engine documentation says embeddings are linearly composable and can be aggregated to coarser resolutions, so this is a defensible first pass.

Phase 0 already implements this first pass for Monterey. Future alignment work
should focus on QA and alternative model artifacts, not on redoing the same
station-centered path.

First table product:

```text
one row = grid cell x year
columns = A00-A63 + lat + lon + year + state + label columns
```

Later chip product:

```text
one sample = spatial crop x year
X = 64-channel embedding chip
Y = kelp label chip
```

## Candidate Next Work: Splits

Expanded split families are deferred until after Phase 1 hardens the Monterey
annual-max pipeline. The current Phase 1 default remains the year holdout.

Use multiple split families, not one split.

```text
random/block split:
- sanity check only, likely optimistic

year split:
- train 2017-2021, validate 2022, test 2023-2024

latitude split:
- hold out coastal bands to test spatial generalization

state split:
- train CA, test OR
- or train south/central CA, test north CA/OR

region/site split:
- hold out named kelp regions if Kelpwatch has those geometries
```

Main results should emphasize year and space holdouts, not random pixels, because coastal pixels are autocorrelated.

## Candidate Next Work: Models And Calibration

A subset of this section is selected for Phase 1: reference baselines,
imbalance-aware annual-max models, and calibration. Deep spatial models and
alternative temporal labels remain deferred.

Start simple.

Baselines:

```text
no-skill baseline: previous-year kelp / climatology
geographic baseline: lat/lon/year only
spectral-product baseline if available
```

Embedding models:

```text
logistic/ridge regression
random forest or XGBoost/LightGBM
small MLP on 64 embeddings
spatial CNN/UNet over embedding chips
temporal model using embeddings from y, y-1, y-2
```

Do not jump to deep spatial models first. A tree model or MLP on 64-dimensional embeddings will quickly show whether AlphaEarth contains useful kelp signal.

Phase 0-specific model lesson:

- Training the continuous ridge model with full population-expanded background
  weights collapsed predictions toward zero.
- The final Phase 0 baseline therefore trains unweighted on the
  background-inclusive sample.
- This recovers useful Kelpwatch-station signal, but it still underpredicts
  canopy magnitude and greatly overpredicts full-grid area because tiny positive
  predictions accumulate over many assumed-background cells.
- Sampling, objective weighting, and calibration should be treated as explicit
  research questions if selected next.

## Phase 1: Model And Domain Hardening

Status: selected for planning as of 2026-05-08.

Phase 1 keeps the Monterey annual-max label input fixed and hardens the data
domain, reference baselines, model objective, and report loop before scale-up.
The plan is:

```text
docs/phase1_model_domain_hardening.md
```

Selected work:

- Add previous-year, station-climatology, and lat/lon/year-only baselines.
- Use bathymetry/DEM inputs to build a plausible-kelp domain mask.
- Evaluate binary annual-max and hurdle-style imbalance-aware models.
- Compare pixel skill, background leakage, and full-grid area calibration in
  every report rerun.

Explicitly deferred work:

- Alternative temporal label inputs such as annual mean, fall-only, winter-only,
  or multi-season persistence targets.
- Full West Coast scale-up.
- Deep spatial models.

## Candidate Next Work: Evaluation

Metrics should match the ecology use case.

Classification:

```text
AUROC / AUPRC
precision-recall at useful thresholds
false positives on shoreline/water
```

Regression:

```text
RMSE/MAE
area bias
region-year total canopy error
correlation by region/year
```

Ecological products:

```text
annual max area error
fall persistence agreement
2-year fall change error
collapse/recovery classification
```

Maps:

- Predicted vs. Kelpwatch.
- Residuals.
- Held-out region maps.
- Uncertainty maps if using ensembles or quantile models.

## Codex and Agent Workflow

The project should be agent-friendly from the start. Keep modules narrow, artifacts explicit, and every step runnable from the command line.

Current package shape and command strategy:

```text
src/kelp_aef/
  io/
  labels/
  alignment/
  features/
  models/
  evaluation/
  viz/

kelp-aef commands:
  query-aef-catalog
  download-aef
  inspect-kelpwatch
  visualize-kelpwatch
  build-labels
  align
  align-full-grid
  train-baselines
  predict-full-grid
  map-residuals
  analyze-model

configs/
  monterey_smoke.yaml
  # future configs should be added only when a phase task needs them

external artifact root: /Volumes/x10pro/kelp_aef/
  raw/
  interim/
  processed/
  models/
  reports/
    figures/
    tables/
```

Phase 1 agent-sized work is outlined in `docs/todo.md`. Future agent-sized work
should use this contract shape:

```text
Goal:
Inputs:
Outputs:
Config file:
Plan/spec requirement:
Validation command:
Smoke-test region and years:
Acceptance criteria:
Known constraints or non-goals:
```

Each agent should get a narrow contract:

- Input files.
- Output files.
- Validation command.
- Tiny smoke-test region.
- Clear acceptance criteria.

## First Concrete Milestone

Completed for now:

```text
Region: Monterey Peninsula
Years: 2018-2022
Label: Kelpwatch annual max
Features: AlphaEarth embeddings aggregated to 30 m
Models: no-skill + ridge regression
Split: year holdout
Output: predicted map, residual map, and area-bias summary
```

Next active milestone:

```text
Phase 1: harden Monterey annual-max baselines, domain filtering, imbalance-aware
models, and report calibration before scale-up.
```

Do not assume the next step is full U.S. West Coast scale-up.

## Sources

- AlphaEarth blog: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/
- Earth Engine Satellite Embedding V1: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Geo-Embeddings xarray tutorial: https://geoembeddings.org/tutorials/xarray_geospatial_embeddings_intro.html
- Kelpwatch: https://kelpwatch.org/
