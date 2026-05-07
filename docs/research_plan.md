# AlphaEarth Kelp Mapping Project Plan

## Project Framing

Use AlphaEarth/Satellite Embedding features to learn kelp canopy products from Kelpwatch-derived weak labels across the U.S. West Coast.

Important caveat: Kelpwatch is not independent field truth. It is a Landsat-derived kelp canopy product. The first version should be framed as learning, improving, or reproducing Kelpwatch-style labels from AlphaEarth embeddings, not proving true kelp biomass.

## Phase 0: Tiny Feasibility Spike

Do not start with the whole West Coast.

Pick one region, such as Monterey or Santa Barbara, and 2-3 years in the AlphaEarth overlap window.

Goal:

```text
Kelpwatch label tile + AlphaEarth embedding tile -> aligned training table -> simple model -> map
```

Success criteria:

- AlphaEarth embeddings load correctly.
- Kelpwatch labels load correctly.
- Coordinate alignment works.
- A simple model beats dumb baselines.
- Outputs look spatially plausible.

AlphaEarth public embeddings are 10 m, 64-dimensional, annual products available in Earth Engine. The Geo-Embeddings xarray/Zarr workflow is useful, but the full collection is enormous, so pull only coastal chips rather than bulk-downloading everything.

## Phase 1: Data Ingestion

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

## Phase 2: Derived Labels

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

## Phase 3: Alignment

Align into the AlphaEarth coordinate frame, then decide whether labels are upsampled from 30 m to 10 m or AlphaEarth embeddings are aggregated from 10 m to 30 m.

Start by aggregating AlphaEarth to the Kelpwatch 30 m grid because Kelpwatch labels are 30 m. The Earth Engine documentation says embeddings are linearly composable and can be aggregated to coarser resolutions, so this is a defensible first pass.

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

## Phase 4: Splits

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

## Phase 5: Models

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

## Phase 6: Evaluation

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

## Phase 7: Codex and Agent Workflow

The project should be agent-friendly from the start. Keep modules narrow, artifacts explicit, and every step runnable from the command line.

Suggested structure:

```text
src/kelp_aef/
  io/
  labels/
  alignment/
  features/
  models/
  evaluation/
  viz/

scripts/
  download_kelpwatch.py
  fetch_aef_chips.py
  build_labels.py
  align_features_labels.py
  train_model.py
  evaluate_model.py
  make_maps.py

configs/
  monterey_smoke.yaml
  west_coast_full.yaml

external artifact root: /Volumes/x10pro/kelp_aef/
  raw/
  interim/
  processed/
  models/
  reports/
    figures/
    tables/
```

Agent-style task breakdown:

```text
Agent 1: inspect Kelpwatch data format and write reader
Agent 2: implement AlphaEarth xarray/Zarr chip fetcher
Agent 3: design derived label builder
Agent 4: implement alignment and split manifests
Agent 5: train/evaluate baseline models
Agent 6: make maps and write results summary
```

Each agent should get a narrow contract:

- Input files.
- Output files.
- Validation command.
- Tiny smoke-test region.
- Clear acceptance criteria.

## First Concrete Milestone

Start with:

```text
Region: Monterey Peninsula
Years: 2018-2024, depending on available embedding coverage
Label: Kelpwatch annual max
Features: AlphaEarth embeddings aggregated to 30 m
Models: logistic regression + random forest
Split: year holdout
Output: predicted map, residual map, and area-bias summary
```

If that works, scale to the full U.S. West Coast.

## Sources

- AlphaEarth blog: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/
- Earth Engine Satellite Embedding V1: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Geo-Embeddings xarray tutorial: https://geoembeddings.org/tutorials/xarray_geospatial_embeddings_intro.html
- Kelpwatch: https://kelpwatch.org/
