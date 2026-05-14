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

Phase 1, Monterey annual-max model and domain hardening, is closed as of
2026-05-13. The selected theme was:

```text
Harden the Monterey annual-max pipeline before scale-up.
```

Phase 1 kept the Phase 0 label input fixed as Kelpwatch annual max canopy. It
added meaningful reference baselines, a CRM-derived plausible-kelp retained
domain, CRM-stratified mask-first sampling, binary calibration, positive-only
conditional canopy modeling, and hurdle composition.

Final Phase 1 policy:

- Default data policy: `crm_stratified_mask_first_sample` inside
  `plausible_kelp_domain`.
- Selected AEF policy: expected-value hurdle,
  `calibrated_probability_x_conditional_canopy`.
- Diagnostic support policy: hard-gated hurdle,
  `calibrated_hard_gate_conditional_canopy`.
- Decision note: `docs/phase1_closeout_model_policy_decision.md`.
- Final report snapshot:
  `docs/report_snapshots/monterey_phase1_closeout_model_analysis.md`.

Phase 1 proved that the AlphaEarth/Kelpwatch annual-max pipeline can be run end
to end with a physically filtered reporting domain and a model policy that beats
the one-stage AEF ridge baseline. It did not prove independent ecological
biomass truth, and it did not solve high-canopy amount underprediction.

Phase 1 non-goals:

- Do not evaluate alternative temporal label inputs beyond annual max.
- Do not use fall-only, winter-only, annual mean, or multi-season persistence
  labels as Phase 1 targets.
- Do not start full West Coast scale-up.
- Do not use bathymetry/DEM as model predictors unless that is explicitly
  approved later; use them first for filtering and diagnostics.

## Phase 2 Status

Phase 2 has been selected as a Big Sur generalization gate as of 2026-05-14.
The plan is tracked in:

```text
docs/phase2_big_sur_generalization.md
```

Phase 2 asks whether the closed Monterey Phase 1 policy generalizes to the
neighboring Big Sur region before the project chooses a broader Phase 3
direction.

Selected Phase 2 region:

- Region: Big Sur.
- Region shorthand: `big_sur`.
- AEF STAC item id: `8957`.
- AEF CRS: `EPSG:32610`.
- AEF bbox:
  `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`.
- Example 2022 AEF asset:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`.

The working assumption is that AEF and Kelpwatch coverage exist for Big Sur.
Domain-source coverage is likely but must be verified quickly before model
metrics are interpreted.

Phase 2 should stay short. It should start with the Phase 1 annual-max policy
as the Monterey-trained transfer baseline, run quick visual QA for Big Sur
labels, AEF coverage, and domain context, compare Big Sur-only and pooled
Monterey+Big Sur training regimes, generate a Big Sur visualizer, adapt the
report and visualizer for region/year review where needed, and close with a
Phase 3 recommendation.

Phase 2 non-goals:

- Do not start full West Coast scale-up.
- Do not change the label target away from Kelpwatch annual max unless Phase 2
  closes with a Phase 3 temporal-label recommendation.
- Do not choose final thresholds, sample quotas, or model policy by tuning on
  held-out Big Sur test rows. Validation-driven Big Sur-only and pooled
  training comparisons are in scope.
- Do not use bathymetry/DEM as predictors unless a later decision explicitly
  changes the feature scope.

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
  (historical Phase 0 sample)
- `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
  (active retained-domain model-input sample)
- `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`
- `docs/report_snapshots/monterey_phase0_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`

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

The closed Phase 1 evaluation emphasized reference-baseline comparison,
domain-mask effects, imbalance-aware model behavior, and area calibration.
Phase 2 adds a neighboring-region generalization check before any broader
scale-up or temporal-target pivot.
