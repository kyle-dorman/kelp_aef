# Product

## Goal

Build a reproducible pipeline that uses AlphaEarth/Satellite Embedding features
to learn Kelpwatch-style kelp canopy labels for the U.S. West Coast.

The first version should be framed as learning, improving, or reproducing
Kelpwatch-derived labels from AlphaEarth embeddings. Kelpwatch is a
Landsat-derived product, not independent field truth, so this project should not
claim to prove true kelp biomass.

## Phase 0 Status

Phase 0, the Monterey Peninsula feasibility spike, is complete for now as of
2026-05-08. The spike produced an end-to-end weak-label pipeline, not a final
modeling answer.

```text
Kelpwatch label tile + AlphaEarth embedding tile -> aligned training table -> simple model -> map
```

Implemented scope:

- Region: Monterey Peninsula.
- Years: 2018-2022 for the first Monterey smoke test, using a year holdout.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings aggregated from 10 m to the
  30 m target grid.
- Model: simple ridge baseline trained on a background-inclusive sample without
  background expansion weights.
- Split: train 2018-2020, validate 2021, test 2022.
- Outputs: full-grid predictions, residual maps, area-bias tables, and a Phase 0
  report.

Phase 0 closed with a useful but imperfect smoke result:

- The end-to-end data path works from source discovery through report.
- The original station-center-only alignment mistake was corrected with a
  full-grid artifact that includes assumed-background rows.
- The ridge baseline has some station-row signal, but it still underpredicts
  canopy magnitude on Kelpwatch-supported rows.
- Full-grid calibration remains poor because small positive predictions over a
  very large assumed-background area accumulate into large area overprediction.
- Phase 1 has been selected as model and domain hardening for the Monterey
  annual-max pipeline.

## Phase 1 Status

Phase 1 is active planning as of 2026-05-08. The selected theme is:

```text
Harden the Monterey annual-max pipeline before scale-up.
```

Phase 1 keeps the Phase 0 label input fixed as Kelpwatch annual max canopy. The
main work is to add meaningful reference baselines, filter the prediction domain
with bathymetry/DEM data, and evaluate imbalance-aware models that are less
likely to leak small positives across large background areas.

Primary Phase 1 questions:

- Does AlphaEarth beat previous-year, station-climatology, and lat/lon/year-only
  baselines?
- Does a physically plausible kelp-domain mask improve full-grid area
  calibration without discarding real Kelpwatch-supported canopy?
- Can a binary or hurdle-style annual-max model handle imbalance better than the
  Phase 0 ridge baseline?
- Can each pipeline rerun update the report in a way that makes the latest
  improvement or regression visible?

Phase 1 non-goals:

- Do not evaluate alternative temporal label inputs beyond annual max.
- Do not use fall-only, winter-only, annual mean, or multi-season persistence
  labels as Phase 1 targets.
- Do not start full West Coast scale-up.
- Do not use bathymetry/DEM as model predictors unless that is explicitly
  approved later; use them first for filtering and diagnostics.

## Success Criteria

The feasibility spike is considered useful enough to pause when:

- AlphaEarth embeddings load for the selected region and years.
- Kelpwatch labels load for the same spatial/temporal scope.
- Coordinate alignment produces both Kelpwatch-supported rows and
  assumed-background grid rows.
- A simple embedding model and no-skill baseline run end to end.
- Maps, diagnostics, and a report make model limitations visible.

Phase 0 does not establish that the current model is production-ready. It only
establishes that the project has enough plumbing and diagnostics to make the
next research decision deliberately.

## Primary Data Products

Canonical data and generated artifact root:

```text
/Volumes/x10pro/kelp_aef
```

Expected first table product:

```text
one row = grid cell x year
columns = A00-A63 + lat + lon + year + state + label columns
```

Key Phase 0 artifacts:

- `/Volumes/x10pro/kelp_aef/raw/kelpwatch/`
- `/Volumes/x10pro/kelp_aef/raw/aef/`
- `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase0_model_analysis.md`

## Non-Goals

- Do not start with the whole West Coast.
- Do not bulk-download the full AlphaEarth embedding collection.
- Do not claim independent validation of real kelp biomass from Kelpwatch alone.
- Do not jump to spatial deep learning before simple tabular baselines work.
- Do not let one random-pixel split stand in for temporal or spatial
  generalization.
- Do not treat the current Phase 0 ridge model as a production target.
- Do not tune on the 2022 test split.

## Evaluation Priorities

Primary results should emphasize temporal and spatial holdouts because coastal
pixels are spatially autocorrelated.

Useful split families:

- Year holdout.
- Latitude band holdout.
- State or regional holdout.
- Random or block split only as a sanity check.

Useful metrics:

- Classification: AUROC, AUPRC, precision-recall at useful thresholds, and
  false positives near shoreline or water boundaries.
- Regression: RMSE, MAE, area bias, region-year total canopy error, and
  region/year correlations.
- Ecological products: annual max area error, fall persistence agreement,
  two-year fall change error, and collapse/recovery classification.

Phase 1 evaluation priorities are reference-baseline comparison, domain-mask
effects, imbalance-aware model behavior, and area calibration. Alternative
temporal target framing remains deferred.
