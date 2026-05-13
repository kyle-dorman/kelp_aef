# Task 37: Test Stratified-Background Continuous Model

## Goal

Test one direct continuous AlphaEarth model whose training objective balances
assumed-background contribution by retained-domain strata.

The target remains the Monterey Phase 1 annual-max weak label:

```text
kelp_fraction_y = kelp_max_y / 900 m2
kelp_max_y      = Kelpwatch-style annual max canopy area in one 30 m cell
```

This task should answer whether a single continuous model can reduce retained
background leakage and preserve station skill by controlling the loss across
background strata, without needing a two-stage hurdle composition.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current default masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current masked sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current retained-domain comparison anchors:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`

Current anchors to preserve:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label input: Kelpwatch-style annual max, `kelp_fraction_y` / `kelp_max_y`.
- Features: `A00-A63`.
- Primary reporting scope: `full_grid_masked` inside `plausible_kelp_domain`.
- Default sample policy: `crm_stratified_mask_first_sample`.
- CRM depth, depth bins, and mask reasons are sampling/objective context only,
  not model predictors.

## Outputs

- Package-backed command or sidecar mode, for example:
  `kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment stratified-background`.
- Config block for the stratified-background continuous experiment.
- Serialized model, for example:
  `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_stratified_background.joblib`.
- Row-level retained-domain full-grid predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_full_grid_predictions.parquet`.
- Sample prediction and metric artifacts, for example:
  - `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/interim/continuous_objective_stratified_background_manifest.json`
- Updated Phase 1 report and model-comparison rows that compare the
  stratified-background continuous model against ridge, capped-weight continuous
  if available, expected-value hurdle, hard-gated hurdle, and reference
  baselines.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Add an explicit experiment block rather than silently changing
  `models.baselines`.
- Keep the active default sample path:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Define the training strata from retained-domain context, for example:
  `label_source`, `domain_mask_reason`, `depth_bin`, and `year`.
- Use stratum context only to compute fit weights or losses. Do not append CRM
  depth, depth bin, elevation, or mask reason to the model feature matrix.
- Keep evaluation metrics unweighted unless a table explicitly labels a weighted
  diagnostic row. Full-grid area calibration should be computed from row-level
  full-grid predictions.
- Preserve `features: A00-A63` and `target: kelp_fraction_y`.

## Plan/Spec Requirement

This is a new model objective and artifact family. Before implementation, write a
brief implementation plan that confirms:

- Whether the implementation extends baseline sidecars or creates a dedicated
  continuous-objective command.
- The exact stratum definition used for fit-weight construction.
- Whether Kelpwatch-supported rows are one stratum or are always kept at
  baseline weight `1.0`.
- How assumed-background strata are balanced, for example:

```text
each retained background stratum contributes the same total training weight per year
```

- Whether the objective combines stratum balancing with a global background cap.
  If yes, keep that cap fixed from the plan or select it on validation only.
- How stratum weights are kept separate from unweighted station metrics and
  full-grid area calibration.
- How predictions are clipped to `[0, 1]` before canopy-area summaries.
- Which comparison table becomes authoritative for P1-22b.
- How the task avoids selecting strata, weights, thresholds, or report choices on
  2022 test rows.

## Implementation Plan

- Load the current masked sample, split assignments, and configured AEF feature
  columns.
- Build training strata from retained-domain metadata already present in the
  sample.
- Derive a fit-weight column that balances assumed-background contribution across
  the configured strata while retaining Kelpwatch-supported rows.
- Fit a ridge regression on train rows with the stratified fit weights.
- Select any hyperparameters using validation rows only.
- Predict on sample rows and retained-domain full-grid rows for all configured
  years.
- Write row-level predictions with enough metadata to distinguish:
  `model_family`, `model_name`, `objective_policy`, `fit_weight_policy`,
  `stratum_columns`, `sample_policy`, `split`, `year`, `label_source`,
  `mask_status`, and `evaluation_scope`.
- Report station skill separately from background-inclusive sample diagnostics.
- Report assumed-background leakage for 2022 retained-domain rows.
- Report full-grid area calibration for the same scope as the current
  scoreboard.
- Update the Phase 1 report so P1-22b answers:
  - Does stratified-background weighting reduce retained-domain false positives?
  - Does it preserve Kelpwatch-station skill and high-canopy magnitude better
    than ridge?
  - Does it approach or beat the expected-value hurdle on RMSE, F1, and area
    calibration?

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment stratified-background
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

If the implementation reuses `train-baselines` instead of a new command, update
the command name in this task when the plan is accepted.

Full validation:

```bash
make check
```

Manual inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_area_calibration.csv'; print(pd.read_csv(p).to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_assumed_background_leakage.csv'; print(pd.read_csv(p).to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings, `A00-A63`.
- Reporting scope: retained plausible-kelp domain with
  `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- The stratified-background model is a direct continuous model, not a binary or
  hurdle model.
- The active annual-max target, mask, split, feature set, and sample policy are
  unchanged.
- Stratum definitions and fit weights are documented in config and the output
  manifest.
- CRM-derived context affects training weights only; it is not used as a
  predictor.
- Fit weights do not silently replace unweighted evaluation metrics.
- The report includes a stratified-background comparison row for 2022
  `full_grid_masked`.
- The report includes Kelpwatch-station skill, assumed-background leakage, and
  full-grid area calibration.
- The comparison clearly states whether the stratified-background model beats
  ridge, competes with the capped-weight continuous model, competes with the
  expected-value hurdle, or fails.
- No stratum policy, weight, threshold, or report choice is tuned on 2022 test
  rows.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not use CRM depth, elevation, depth bin, or mask reason as predictors.
- Do not change the plausible-kelp mask.
- Do not change the annual-max label input.
- Do not retrain binary presence, probability calibration, conditional canopy, or
  hurdle artifacts unless a later task explicitly asks for that.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
- Do not start full West Coast scale-up.
