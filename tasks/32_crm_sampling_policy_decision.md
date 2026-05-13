# Task 32: CRM-Stratified Sampling Policy Decision

## Goal

Document the decision to promote CRM-stratified, mask-first background sampling
from a sidecar experiment into the default Monterey Phase 1 sampling policy.

Task 31 showed that the CRM-stratified sample improves the current model
families in the retained masked evaluation scope. Before changing default
pipeline behavior, write a design decision that records:

- why the CRM-stratified policy should replace the current sidecar approach;
- which metric improvements justify the decision;
- the exact config values used in the successful sidecar run;
- the planned 60 m maximum-depth domain change;
- how `background_rows_per_year` should be replaced or reinterpreted in
  mask-first sampling.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Active Phase 1 report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.
- All-model CRM comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`.
- Current CRM-stratified sidecar sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`.
- Current CRM-stratified sidecar manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked_manifest.json`.
- Current default masked sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current plausible-kelp mask manifest:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`.
- Existing source decision:
  `docs/phase1_bathymetry_dem_source_decision.md`.

Current sidecar config values to document:

```yaml
random_seed: 31
default_fraction: 0.0
default_min_rows_per_year: 0
strata:
  - domain_mask_reason: retained_ambiguous_coast
    depth_bin: ambiguous_coast
    fraction: 0.12
    min_rows_per_year: 5000
  - domain_mask_reason: retained_depth_0_100m
    depth_bin: 0_40m
    fraction: 0.04
    min_rows_per_year: 8000
  - domain_mask_reason: retained_depth_0_100m
    depth_bin: 40_50m
    fraction: 0.01
    min_rows_per_year: 500
  - domain_mask_reason: retained_depth_0_100m
    depth_bin: 50_100m
    fraction: 0.003
    min_rows_per_year: 500
```

Metric evidence to include from the 2022 retained-domain test/all scope:

- AEF ridge regression improved RMSE from `0.0438` to `0.0378`, F1 from
  `0.342` to `0.469`, predicted canopy area from `12.12M m2` to `10.31M m2`,
  and assumed-background predicted area from `8.66M m2` to `6.94M m2`.
- Calibrated binary presence improved AUPRC from `0.846` to `0.893`, F1 from
  `0.773` to `0.824`, precision from `0.733` to `0.842`, and reduced
  assumed-background false-positive rate from `0.20%` to `0.07%`. Recall moved
  from `0.818` to `0.807`.
- Expected-value hurdle improved RMSE from `0.0316` to `0.0262`, F1 from
  `0.756` to `0.811`, and moved predicted area from `4.32M m2` to `3.98M m2`
  against `4.16M m2` observed.
- Hard-gated hurdle improved RMSE from `0.0345` to `0.0275`, F1 from `0.774`
  to `0.825`, and reduced assumed-background predicted area from `0.88M m2` to
  `0.30M m2`, while underpredicting total area more strongly.

## Outputs

- Design decision note, for example:
  `docs/phase1_crm_stratified_sampling_policy_decision.md`.
- A short update to `docs/todo.md` marking this decision task complete when
  implemented.
- Optional update to `docs/backlog.md` if the decision closes or supersedes the
  existing mask-first sampling backlog item.

The decision note should be durable enough that later agents do not need the
temporary sidecar report section to understand why CRM-stratified sampling
became the default.

## Config File

Use `configs/monterey_smoke.yaml` as the source of current sidecar settings.

The decision should describe the intended next config state, but this task
should not change executable config behavior. It should state that the next
implementation task will:

- use the CRM-stratified sampling policy as the default masked sample producer;
- make sampling mask-first before any retained-background row budget is applied;
- replace the broad `background_rows_per_year` pre-mask cap with explicit
  retained-domain quotas or retained-domain budget fields;
- change the plausible-kelp mask maximum depth from `100.0` m to `60.0` m;
- remove the current `50_100m` retained stratum and replace the deep retained
  bin with `40_60m`;
- keep CRM depth, elevation, depth bins, and mask reasons out of the model
  feature matrix.

## Plan/Spec Requirement

This is the spec/decision task for the policy change. No separate plan is
required before editing this docs-only task, but the decision note should
explicitly answer:

- Is CRM-stratified sampling replacing the default masked sample or remaining a
  sidecar?
- Which metrics justify promotion?
- Which exact sidecar config values produced the evidence?
- Which config values are intentionally changing before promotion?
- Why is the mask changing to a 60 m maximum depth?
- How should `background_rows_per_year` change when sampling is mask-first?
- What report section or table becomes obsolete after promotion?
- Which claims are limited to Kelpwatch-style weak labels rather than
  independent ecological truth?

## Validation Command

Docs-only validation:

```bash
git diff -- docs/todo.md docs/backlog.md docs/phase1_crm_stratified_sampling_policy_decision.md
```

If the decision note includes refreshed metric snippets, validate those numbers
against:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv'; df=pd.read_csv(p); print(df.shape); print(df.head())"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`.
- Reporting scope: retained plausible-kelp domain.

## Acceptance Criteria

- The decision note exists and clearly promotes CRM-stratified sampling to the
  default policy for the Monterey Phase 1 pipeline.
- The note includes the side-by-side metric improvements from Task 31.
- The note records the successful sidecar quota values exactly enough to
  reproduce or audit them.
- The note states that the promoted policy will tighten the maximum depth to
  `60.0` m and use a final retained depth bin of `40_60m`.
- The note states that `background_rows_per_year` should not remain a broad
  pre-mask cap for the default masked workflow.
- The note says the active report can drop the temporary current-vs-CRM sidecar
  section after the policy is promoted.
- The note keeps CRM depth/elevation as sampling and diagnostics context only,
  not model predictors.

## Known Constraints Or Non-Goals

- Do not implement the sampling change in this task.
- Do not rerun model artifacts in this task.
- Do not tune the policy on the 2022 test split. The 2022 metrics justify a
  policy-selection decision after the sidecar experiment; they should not drive
  further threshold or quota tuning.
- Do not claim independent ecological truth; the comparison is against
  Kelpwatch-style labels.
- Do not start full West Coast scale-up.
