# Phase 1 CRM-Stratified Sampling Policy Decision

Status: accepted for P1-21b on 2026-05-13.

## Decision

Promote CRM-stratified, mask-first background sampling from a sidecar
experiment to the default Monterey Phase 1 masked model-input policy.

The next implementation task should replace the current post-hoc masked sample
path with a retained-domain sampler that filters to the plausible-kelp domain
first, then applies explicit retained-background quotas by CRM-derived mask
reason and depth bin. The default training sample should keep all retained
Kelpwatch-supported rows and sample assumed-background rows from the retained
domain by stratum.

CRM depth, elevation, depth bins, and mask reasons remain sampling and
diagnostic context only. They should not be added to the AlphaEarth model
feature matrix unless a later decision explicitly changes the feature scope.

This decision is about reproducing and improving Kelpwatch-style annual-max
weak labels for Monterey Phase 1. It does not claim independent ecological
truth or field-validated kelp biomass.

## Evidence

Task 31 compared current and CRM-stratified artifacts under the same annual-max
target, year split, AlphaEarth feature set, full-grid inference table, and
retained-domain reporting scope. Threshold and calibration choices remained
validation-driven; the 2022 test rows are audit evidence for selecting the next
pipeline policy, not a basis for further quota or threshold tuning.

The key 2022 `test` / `full_grid_masked` evidence is:

| Model | Current masked sample | CRM-stratified background | Decision signal |
| --- | ---: | ---: | --- |
| AEF ridge RMSE | 0.0438 | 0.0378 | Better pixel error under the same retained scope. |
| AEF ridge F1 at 10% annual max | 0.342 | 0.469 | Better binary support from the continuous output. |
| AEF ridge predicted canopy area | 12.12M m2 | 10.31M m2 | Less overprediction against 4.16M m2 observed. |
| AEF ridge assumed-background predicted area | 8.66M m2 | 6.94M m2 | Lower background leakage. |
| Calibrated binary AUPRC | 0.846 | 0.893 | Better ranking for the weak binary target. |
| Calibrated binary F1 | 0.773 | 0.824 | Better validation-selected operating behavior. |
| Calibrated binary precision | 0.733 | 0.842 | Fewer selected false positives. |
| Calibrated binary recall | 0.818 | 0.807 | Small recall tradeoff. |
| Calibrated binary assumed-background false-positive rate | 0.20% | 0.07% | Large leakage reduction. |
| Expected-value hurdle RMSE | 0.0316 | 0.0262 | Better composed continuous prediction. |
| Expected-value hurdle F1 | 0.756 | 0.811 | Better 10% annual-max support. |
| Expected-value hurdle predicted canopy area | 4.32M m2 | 3.98M m2 | Closer to 4.16M m2 observed. |
| Hard-gated hurdle RMSE | 0.0345 | 0.0275 | Better hard-gated pixel error. |
| Hard-gated hurdle F1 | 0.774 | 0.825 | Better hard-gated support. |
| Hard-gated hurdle assumed-background predicted area | 0.88M m2 | 0.30M m2 | Stronger background leakage control. |

The calibrated binary recall decrease is real but acceptable for this policy
decision because precision, AUPRC, F1, full-grid predicted-positive area, and
assumed-background leakage all move in the intended direction. The hard-gated
hurdle underpredicts total area more strongly than the current hard gate, so it
remains a diagnostic output; the expected-value hurdle is the better composed
area-calibration signal.

## Successful Sidecar Policy

The successful sidecar run used `configs/monterey_smoke.yaml` and wrote:

- sample table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`
- sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked_manifest.json`
- summary table:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.crm_stratified.masked_summary.csv`
- all-model comparison:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`

The sidecar quota values were:

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

The manifest records the quota type as
`per_year_crm_stratum_fraction_with_min_max_caps`, uses the deterministic key
`aef_grid_cell_id`, `year`, and `random_seed`, keeps all retained Kelpwatch rows,
and weights sampled assumed-background rows by retained stratum population
divided by sampled stratum rows. Kelpwatch rows keep sample weight `1.0`.

## Intended Promotion State

Task 33 should implement the promoted default with these changes:

- apply the plausible-kelp mask before any retained-background quota;
- replace the current sidecar path with the default masked sample producer;
- keep a deterministic, reproducible quota policy by retained CRM stratum;
- keep all retained Kelpwatch-supported rows;
- keep CRM-derived context out of the model predictors;
- regenerate downstream baseline, binary, calibration, conditional, hurdle, and
  report artifacts from the promoted sample.

The promoted policy should also tighten the plausible-kelp domain from the
current broad `100.0` m maximum-depth mask to `60.0` m. The retained depth bins
should no longer include `50_100m`; the final retained deep bin should become
`40_60m`.

This 60 m change is intentionally a next implementation step, not evidence
already reflected in the Task 31 metrics. The Task 31 evidence used the current
100 m broad mask. Task 33 must regenerate the mask and verify Kelpwatch-positive
retention before treating the 60 m default as accepted in artifacts.

## Background Row Budget

`background_rows_per_year` should not remain a broad pre-mask cap for the
default masked workflow.

In the current pipeline, `background_rows_per_year: 250000` mostly limits the
pre-mask background-inclusive file size. The plausible-kelp mask then performs
most of the real retained-training-set selection after the fact. That makes the
current masked sample and CRM-stratified sample comparable only because of
sidecar bookkeeping, not because the default data contract defines the same
sampling population.

For the promoted workflow, replace or reinterpret that field as explicit
retained-domain budget fields. A clearer contract is:

- mask first to the retained plausible-kelp domain;
- keep all retained Kelpwatch-supported rows;
- apply per-year, per-stratum quotas to retained assumed-background rows;
- record quota fractions, minimums, maximums, sampled counts, dropped counts,
  and sample-weight policy in the sample manifest.

The exact config names can be chosen in Task 33, but they should describe
retained-domain budgets rather than a broad pre-mask background cap.

## Report Cleanup

After Task 33 promotes the policy, the active Phase 1 report no longer needs a
temporary current-vs-CRM sidecar section as a recurring live comparison. Task 34
should retire that section from the main report and leave this decision note as
the durable historical explanation for why CRM-stratified sampling became the
default.

The all-model comparison CSV and current sidecar report section remain useful
audit evidence until promotion is implemented and validated.

## Non-Goals

- Do not rerun model artifacts in this decision task.
- Do not tune quotas or thresholds on the 2022 test split.
- Do not use CRM bathymetry, elevation, depth bins, or mask reasons as model
  features.
- Do not change the label input away from Kelpwatch annual max canopy.
- Do not start full West Coast scale-up.
