# Task 31: CRM-Stratified Background Sampling For All Model Families

## Goal

Extend the CRM-stratified background-sampling experiment from Task 30 across
the current Phase 1 model families so the report can make fair, apples-to-apples
comparisons under a shared sampling policy.

Task 30 only trained a binary-presence sidecar from:

```text
/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet
```

The existing continuous baselines, calibrated binary model, conditional canopy
stage, and first hurdle composition still primarily use the current masked
sample and current binary artifacts. This task should make the sampling-policy
comparison explicit:

- Current policy:
  `aligned_background_sample_training_table.masked.parquet`
- CRM-stratified policy:
  `aligned_background_sample_training_table.crm_stratified.masked.parquet`

The purpose is to compare model families under the same target, features, split,
inference grid, and domain mask, while changing only the training sample policy
where a stage actually consumes assumed-background rows.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- CRM-stratified masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`.
- CRM-stratified sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked_manifest.json`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current split policy from `configs/monterey_smoke.yaml` and split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current model artifacts and reports from:
  - `kelp-aef train-baselines`
  - `kelp-aef train-binary-presence`
  - `kelp-aef calibrate-binary-presence`
  - `kelp-aef train-conditional-canopy`
  - `kelp-aef compose-hurdle-model`
  - `kelp-aef analyze-model`

Current anchors to preserve:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label input: Kelpwatch-style annual max, `kelp_fraction_y` / `kelp_max_y`.
- Binary target: `annual_max_ge_10pct`.
- Features: `A00-A63`.
- Primary reporting scope: retained plausible-kelp domain.

## Outputs

- Config sidecar blocks, or equivalent reusable config structure, for every
  model stage that consumes the model-input sample.
- CRM-stratified continuous baseline outputs, for example:
  - `/Volumes/x10pro/kelp_aef/models/baselines/ridge_kelp_fraction.crm_stratified.joblib`
  - `/Volumes/x10pro/kelp_aef/models/baselines/geographic_ridge_lon_lat_year.crm_stratified.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.crm_stratified.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.crm_stratified.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/interim/baseline_eval_manifest.crm_stratified.json`
- CRM-stratified binary calibration outputs, if calibration is still part of the
  primary hurdle path, for example:
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/logistic_annual_max_ge_10pct_calibration.crm_stratified.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_calibrated_sample_predictions.crm_stratified.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/interim/binary_presence_calibration_manifest.crm_stratified.json`
- Conditional-canopy decision artifact:
  - Either sidecar conditional outputs if the fitted rows differ under the new
    policy, or a manifest/report note proving the conditional model is unchanged
    because the configured support policy uses only observed-positive rows and
    Task 30 retained all Kelpwatch-observed rows.
- CRM-stratified hurdle outputs, for example:
  - `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.crm_stratified.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/hurdle_prediction_manifest.crm_stratified.json`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.crm_stratified.csv`
  - `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.crm_stratified.png`
- All-model sampling-policy comparison table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`.
- Updated Phase 1 report section that compares current versus CRM-stratified
  sample policy for every relevant model family.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config shape:

- Keep existing current-policy model paths intact.
- Keep `alignment.background_sample.crm_stratified` as the producer of the
  sidecar sample.
- Add explicit `sample_policy` metadata to sidecar outputs.
- Add sidecar output paths for:
  - `models.baselines`;
  - `models.binary_presence.calibration`, if calibration is needed for hurdle
    composition;
  - `models.hurdle`.
- Reuse or explicitly mark the conditional canopy model as shared when the row
  support is identical.
- Do not add CRM depth, elevation, depth bin, or mask reason to any model
  feature matrix.
- Do not silently replace current default outputs. Sidecar artifacts must be
  path-distinct until a later decision task promotes a policy.

## Plan/Spec Requirement

This is a multi-stage model-comparison task. Before implementation, write a
brief implementation plan that confirms:

- Which existing commands read the model-input sample and therefore need a
  CRM-stratified sidecar.
- Which commands use only full-grid inference, saved predictions, or
  observed-positive rows and therefore can reuse existing artifacts.
- Whether sidecars are implemented as nested config blocks, a common
  sample-policy abstraction, or a separate comparison command.
- How split assignments are handled for CRM-stratified rows that are absent
  from the current split manifest.
- Which artifacts are overwritten versus written to sidecar paths.
- How binary calibration is paired with the CRM-stratified binary model.
- How the hurdle sidecar chooses between reused conditional canopy artifacts
  and refit conditional artifacts.
- Which report table is the authoritative all-model comparison.
- Which metrics decide whether CRM-stratified sampling should be promoted,
  while avoiding any tuning on 2022 test rows.

## Implementation Plan

- Audit the current model commands and config loaders:
  - `src/kelp_aef/evaluation/baselines.py`
  - `src/kelp_aef/evaluation/binary_presence.py`
  - `src/kelp_aef/evaluation/conditional_canopy.py`
  - `src/kelp_aef/evaluation/hurdle.py`
  - `src/kelp_aef/evaluation/model_analysis.py`
- Define a consistent sidecar contract with:
  - `sample_policy = current_masked_sample`
  - `sample_policy = crm_stratified_background_sample`
- Add deterministic split assignment for CRM-stratified sidecar rows by the
  configured year split, rather than relying only on the current split manifest.
- Extend continuous baseline training and prediction to run the current and
  CRM-stratified sample policies without changing targets, features, alpha
  grids, inference table, or domain mask.
- Extend binary calibration so the CRM-stratified binary model has its own
  validation-fit calibration artifact and threshold table.
- Check conditional canopy support:
  - If training and evaluation rows are unchanged because all
    Kelpwatch-observed rows are retained, write a clear manifest/report note
    and reuse the conditional model.
  - If any configured likely-positive diagnostics depend on the binary model,
    regenerate only those diagnostics as CRM-stratified sidecars.
- Compose a CRM-stratified hurdle using:
  - CRM-stratified binary probabilities and calibration;
  - the reused or sidecar conditional canopy model;
  - the same retained full-grid inference scope.
- Update model analysis so every comparison row carries enough metadata to
  distinguish:
  - model family;
  - model name;
  - sample policy;
  - split/year;
  - label source;
  - mask status;
  - evaluation scope.
- Add focused tests for sidecar config loading, split fallback by year, and
  report comparison rows.
- Regenerate the Phase 1 report after all sidecar artifacts are written.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_binary_presence.py tests/test_conditional_canopy.py tests/test_hurdle.py tests/test_model_analysis.py
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection should include:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv'; print(pd.read_csv(p).to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.crm_stratified.csv'; print(pd.read_csv(p).to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`.
- Features: `A00-A63`.
- Mask: current retained plausible-kelp domain.
- Sampling context: CRM-derived `domain_mask_reason` and `depth_bin` define the
  CRM-stratified sample only. They are not model predictors.

## Acceptance Criteria

- The current masked sample and CRM-stratified masked sample both remain
  available.
- Every model family that consumes assumed-background training rows has current
  and CRM-stratified sidecar outputs with path-distinct artifacts.
- Stages that are unchanged by background sampling, especially the configured
  observed-positive conditional canopy model, are explicitly proven unchanged
  or documented as reused.
- Binary calibration and hurdle composition use CRM-stratified-compatible
  upstream artifacts rather than mixing current-policy and CRM-stratified
  binary outputs.
- All model comparisons use the same target, split, feature set, inference
  grid, and retained-domain reporting scope.
- Comparison tables include a `sample_policy` or equivalent field so current
  versus CRM-stratified rows cannot be confused.
- The Phase 1 report shows whether CRM-stratified sampling improves background
  leakage and full-grid area behavior across baselines, binary, and hurdle
  outputs, and also shows any loss in Kelpwatch-station recall or positive-cell
  canopy skill.
- The implementation does not tune quotas, model settings, calibration
  thresholds, or policy selection on the 2022 test split.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not use CRM depth, elevation, depth bin, or mask reason as model
  predictors.
- Do not change the P1-12 plausible-kelp domain mask thresholds.
- Do not change the annual-max label input or the `annual_max_ge_10pct`
  threshold.
- Do not replace current default model artifacts without an explicit follow-up
  promotion decision.
- Do not tune on the 2022 test split.
- Do not start full West Coast scale-up.
- Do not add a new model family beyond what is needed to compare the existing
  Phase 1 model families under the CRM-stratified sample policy.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
