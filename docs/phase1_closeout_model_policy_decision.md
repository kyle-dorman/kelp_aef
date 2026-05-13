# Phase 1 Closeout Model Policy Decision

Status: accepted for P1-23 on 2026-05-13.

## Closeout Plan

The closeout compares the current Monterey Phase 1 policies using the
authoritative generated tables from `configs/monterey_smoke.yaml`.

Candidate policies:

- Reference previous-year annual max.
- AEF ridge regression.
- Calibrated binary presence for `annual_max_ge_10pct`.
- Positive-only conditional canopy.
- Expected-value hurdle.
- Hard-gated hurdle diagnostic.

The decisive comparison row is `test` / `2022` / `full_grid_masked` /
`label_source = all` from
`/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`.
Component-only evidence comes from
`binary_presence_calibration_metrics.csv`,
`conditional_canopy_metrics.csv`,
`conditional_canopy_positive_residuals.csv`, and
`hurdle_assumed_background_leakage.csv`.

The generated report remains a final closeout report with the current scope,
policy, evidence table, improvements, and failure modes. It removes the live
Phase 1 task queue and does not add post-Phase-1 work items. Historical
planning and negative-result records remain tracked. The final Markdown report
is copied to `docs/report_snapshots/monterey_phase1_closeout_model_analysis.md`
after rerunning `kelp-aef analyze-model`.

## Decision

Select `calibrated_probability_x_conditional_canopy` as the best current AEF
Phase 1 model policy.

The selected policy composes the Platt-calibrated annual-max presence
probability with the positive-only conditional canopy amount model. It keeps the
current Monterey annual-max target, year split, `A00-A63` feature set,
plausible-kelp retained-domain mask, validation-selected binary threshold, and
`crm_stratified_mask_first_sample` default sample policy.

This is a Kelpwatch-style weak-label reproduction policy for the Monterey Phase
1 feasibility setting. It is not independent biomass validation and should not
be described as field-truth kelp biomass performance.

## Primary Evidence

Rows below use:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

| Policy | Role | RMSE | R2 | F1 >=10% | Predicted area | Area bias | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Previous-year annual max | non-AEF reference | 0.0341 | 0.765 | 0.797 | 3.50M m2 | -15.9% | Strong benchmark, not an AEF policy. |
| AEF ridge regression | one-stage AEF baseline | 0.0452 | 0.587 | 0.476 | 8.42M m2 | +102.1% | Reject as final policy because retained-domain area leakage remains too high. |
| Expected-value hurdle | selected AEF policy | 0.0322 | 0.790 | 0.812 | 3.50M m2 | -16.0% | Select as the best AEF Phase 1 candidate. |
| Hard-gated hurdle | diagnostic AEF policy | 0.0336 | 0.771 | 0.825 | 3.49M m2 | -16.1% | Keep as support/leakage diagnostic, not the selected continuous area policy. |

The expected-value hurdle is the selected AEF policy because it improves over
AEF ridge on retained-domain RMSE, R2, 10% annual-max F1, and total-area bias.
It also edges previous-year persistence on RMSE and F1 while landing at nearly
the same total-area underprediction.

## Component Evidence

The calibrated binary-presence model is strong as a support component, but it
does not predict canopy amount by itself. On `test` / `2022` / `all`
model-input rows, the Platt-calibrated `annual_max_ge_10pct` row has AUPRC
`0.943`, precision `0.908`, recall `0.804`, F1 `0.853`, and threshold `0.36`.

The positive-only conditional canopy model is useful as an amount component,
but it still underpredicts large observed-positive Kelpwatch-style annual-max
rows. On `test` / `2022` observed-positive rows, its area RMSE is `191.8 m2`
and total-area bias is `-16.7%`. On `annual_max_ge_50pct` rows, the mean
residual is `+185.4 m2`; on `near_saturated_ge_810m2` rows, the mean residual
is `+208.8 m2`.

The hard-gated hurdle has the strongest 10% support F1 and lower
assumed-background predicted area than the expected-value hurdle (`0.29M m2`
versus `0.46M m2` in 2022 retained assumed-background rows). It is still a
thresholded support policy, so it remains diagnostic rather than the selected
continuous expected-area policy.

## Failures Preserved

Phase 1 does not produce a production-ready biomass model. The selected policy
still underpredicts high-canopy Kelpwatch-style annual-max rows, and the result
is limited to Monterey 2018-2022.

The failed P1-22 one-stage direct-continuous experiments remain negative
evidence. Capped-weight and stratified-background variants were removed from
the active pipeline in Task 38 and are preserved in:

- `tasks/36_test_capped_weight_continuous_model.md`
- `tasks/37_test_stratified_background_continuous_model.md`
- `tasks/38_remove_failed_p1_22_continuous_paths.md`
- `docs/phase1_stratified_background_sweep_results.md`

## Active Closeout State

Active model policy:

- Data scope: Monterey Peninsula, 2018-2022.
- Label input: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Features: AlphaEarth annual embedding bands `A00-A63`.
- Split: train 2018-2020, validation 2021, test 2022.
- Retained-domain mask: `plausible_kelp_domain`.
- Sample policy: `crm_stratified_mask_first_sample`.
- Selected AEF model policy: `calibrated_probability_x_conditional_canopy`.
- Diagnostic model policy: `calibrated_hard_gate_conditional_canopy`.

Config/code cleanup:

- The active config no longer carries disabled historical CRM-stratified
  sidecar blocks for the promoted default policy.
- The optional sidecar implementation remains in code and tests because it is a
  general comparison mechanism, but it is not part of the selected closeout
  policy in `configs/monterey_smoke.yaml`.
- P1-22 direct-continuous active code paths were already removed by Task 38.

## Validation

Closeout validation should run:

```bash
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
uv run pytest tests/test_model_analysis.py tests/test_package.py
make check
```
