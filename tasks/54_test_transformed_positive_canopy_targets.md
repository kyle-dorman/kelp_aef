# Task 54: Test Transformed Positive-Canopy Targets

## Goal

Test whether a small transformed-target positive-canopy model reduces
high-canopy shrinkage before Phase 2 recommends a more complex tabular model
such as random forest or gradient boosting.

This task is separate from the deep component failure analysis. It should be a
controlled model-capacity diagnostic that changes only the conditional
positive-canopy target scale, then composes the result through the existing
calibrated binary support gate.

The main question is:

```text
Does a log or logit-style positive-canopy target transformation fix enough of
the high-canopy amount shrinkage to avoid escalating immediately to
non-linear tabular models?
```

Frame results as Kelpwatch-style annual maximum reproduction, not independent
field-truth biomass validation.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- Existing positive-only conditional canopy artifacts:
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/big_sur_ridge_positive_annual_max.joblib`
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_annual_max.joblib`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_conditional_canopy_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv`
- Existing binary calibration artifacts for composing hurdle outputs:
  - Big Sur local binary full-grid predictions and calibrator.
  - Monterey local binary full-grid predictions and calibrator.
  - Pooled binary full-grid predictions and calibrators if pooled transformed
    sidecars are included.
- Existing retained-domain model-input samples and full-grid inference tables.
- Task 53 component-failure outputs if already completed, especially
  high-canopy, edge/interior, and domain-context bins.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

Do not overwrite current default conditional canopy or hurdle artifacts. Write
sidecars with explicit transformed-target names, for example:

- Transformed conditional model payloads:
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/big_sur_ridge_positive_log1p_area.joblib`
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_log1p_area.joblib`
- Transformed conditional sample predictions:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_conditional_canopy_log1p_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/conditional_canopy_log1p_sample_predictions.parquet`
- Transformed hurdle full-grid predictions:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_hurdle_log1p_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/hurdle_log1p_full_grid_predictions.parquet`
- Report-visible tables:
  - `transformed_positive_canopy_metrics.csv`
  - `transformed_positive_canopy_model_comparison.csv`
  - `transformed_hurdle_area_calibration.csv`
  - `transformed_hurdle_residual_by_observed_bin.csv`
  - `transformed_hurdle_by_component_failure_context.csv`, if Task 53 outputs
    exist.
- Updated Phase 2 report section or compact decision note that compares
  transformed-target results against the current raw positive-only ridge.
- Manifest recording transforms, inverse transforms, clipping rules, model
  paths, source prediction paths, and primary filters.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add
path-explicit sidecar blocks for transformed conditional canopy and transformed
hurdle outputs.

If Monterey-local transformed sidecars need paths, add them to
`configs/monterey_smoke.yaml` without changing current default model paths.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- which target transformations are included;
- the exact inverse transform and clipping rules;
- whether the task runs only local Monterey/Big Sur models or also pooled
  transformed sidecars;
- how validation rows select alpha or transformation variants;
- how transformed conditional predictions are composed with the existing
  calibrated binary probability;
- how results are compared against the current raw conditional model without
  overwriting the current default policy.

## Required Analysis

Train and evaluate at least one transformed positive-canopy target:

- `log1p_area`: fit ridge on `log1p(kelp_max_y)` for positive support rows,
  inverse with `expm1`, and clip predicted area to `[0, 900] m2`.

Also consider a second simple transformed target if the implementation remains
small:

- `logit_fraction`: fit ridge on a clipped logit of `kelp_fraction_y`, inverse
  with sigmoid, and clip predicted area to `[0, 900] m2`.

Keep the current raw positive-only ridge as the control:

- raw target: `kelp_fraction_y`;
- positive support: `kelp_fraction_y >= 0.10`;
- same AEF embedding predictors;
- same train/validation/test split semantics.

For each transformed model, analyze:

- positive-row MAE and RMSE area on validation and test;
- mean residual and median residual on positive rows;
- residual by observed canopy bin:
  - `(90, 225]`;
  - `(225, 450]`;
  - `(450, 810]`;
  - `(810, 900]`;
- near-saturated rows with `kelp_max_y >= 810 m2`;
- overprediction on low positive rows;
- predicted area distribution and clipping rate at 0 or 900 m2;
- alpha selected on validation rows;
- whether validation improvement transfers to held-out 2022 rows.

Compose each transformed conditional model through the existing calibrated
binary gate:

- expected-value hurdle:
  `calibrated_probability * transformed_conditional_canopy`;
- hard-gated diagnostic:
  transformed conditional canopy only where calibrated support exceeds the
  validation-selected threshold.

For each composed hurdle output, compare against the current raw conditional
hurdle:

- F1 at `annual_max_ge_10pct`;
- RMSE and R2 on the retained full-grid scope;
- predicted area and area percent bias;
- background predicted area;
- assumed-background leakage;
- high-canopy underprediction;
- zero/low-canopy overprediction;
- edge/interior and depth/domain failure contexts if Task 53 outputs exist.

Minimum model contexts:

- Big Sur local transformed conditional model and transformed hurdle.
- Monterey local transformed conditional model and transformed hurdle.

Optional model contexts if the implementation is still narrow:

- pooled Monterey+Big Sur transformed conditional model with target-specific
  validation for Big Sur and Monterey;
- reciprocal transfer diagnostics, clearly labeled as exploratory.

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_conditional_canopy.py tests/test_hurdle.py
uv run kelp-aef train-conditional-canopy --config configs/big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

If new CLI flags or commands are added for transformed sidecars, include
focused tests and run the transformed commands explicitly.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Training years: 2018-2020.
- Validation year: 2021.
- Held-out test year: 2022.
- Label input: Kelpwatch-style annual max canopy.
- Positive support target: rows with `kelp_fraction_y >= 0.10`.
- Features: AEF annual embedding bands `A00-A63`.

## Acceptance Criteria

- Transformed models are written as sidecars and do not overwrite current raw
  conditional or hurdle artifacts.
- The raw conditional model remains the control row in every comparison.
- Validation selection is separated from held-out test evaluation.
- The report or decision note states whether transformed targets reduce
  high-canopy shrinkage, worsen low-canopy overprediction, change area bias, or
  leave the current failure mode unresolved.
- If the transformed target helps, the outcome records whether the next step is
  to promote the transformed conditional target or compare it against simple
  non-linear tabular models.
- If it does not help, the outcome records that target-scale transformation is
  insufficient evidence against random forest or gradient boosting.

## Known Constraints / Non-Goals

- Do not change the Phase 2 label input away from annual max.
- Do not add annual mean, seasonal, or deferred temporal labels as training
  targets in this task.
- Do not add random forest, gradient boosting, or other non-linear models here.
- Do not add bathymetry, edge-distance, or label-persistence fields as
  predictors.
- Do not tune thresholds, masks, or sample quotas on held-out 2022 rows.
- Do not remove or rename the current raw conditional canopy artifacts.
