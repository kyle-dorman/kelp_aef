# Task 30: CRM-Stratified Background Sampling

## Goal

Create a sidecar masked model-input sample that uses CRM-derived domain context
to sample assumed-background rows more deliberately for the binary annual-max
model.

The current `aligned_background_sample_training_table.masked.parquet` keeps all
Kelpwatch-observed rows and samples assumed-background rows after applying the
P1-12 plausible-kelp mask. That sample is still mostly area-proportional inside
the retained domain. Current diagnostics suggest the binary model's remaining
assumed-background false positives are concentrated in hard nearshore context,
especially `retained_ambiguous_coast`, while the large retained `50_100m`
stratum has no Kelpwatch-positive support and appears to be an easier negative
stratum.

This task should test whether a CRM-stratified assumed-background sample gives
the binary model better nearshore negatives without throwing away the existing
baseline sample or changing the Phase 1 label target.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current full-grid aligned table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current masked sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current plausible-kelp depth-bin summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_depth_bins.csv`.
- Current binary model outputs:
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv`
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.

Current anchors to refresh during the task rather than hard-code:

- The existing masked sample has about 313,954 rows.
- The current retained full-grid domain has about 999,519 cells per year.
- The retained `50_100m` depth bin has no Kelpwatch-positive support in the
  current mask summary.
- The current 2022 binary full-grid assumed-background predicted positives are
  concentrated in `ambiguous_coast`.

## Outputs

- Package-backed sampling code, likely extending
  `src/kelp_aef/alignment/full_grid.py` or a narrow helper module under
  `src/kelp_aef/alignment/`.
- Tests covering CRM-stratified quota logic and all-observed-row retention.
- Config additions under `alignment.background_sample` or a nested sidecar
  block for CRM-stratified sampling.
- Sidecar CRM-stratified masked sample, for example:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`.
- Sidecar manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked_manifest.json`.
- Sidecar sample summary, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.crm_stratified.masked_summary.csv`.
- Binary model outputs trained from the sidecar sample, either as sidecar paths
  or as a clearly documented temporary comparison run.
- Comparison table or report section showing the current P1-18 sample versus
  the CRM-stratified sample.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config shape:

- Keep the existing masked sample path intact.
- Add a sidecar output path for the CRM-stratified sample and manifest.
- Add explicit assumed-background quotas or quota weights by CRM stratum.
- Define strata using existing mask/domain metadata such as
  `domain_mask_reason` and `depth_bin`.
- Start with a conservative policy that:
  - keeps all Kelpwatch-observed rows;
  - samples more assumed-background rows from `retained_ambiguous_coast`;
  - samples enough shallow `0_40m` rows to preserve normal nearshore context;
  - keeps a smaller floor for `40_50m` and `50_100m`;
  - avoids increasing the total background row count unless the plan justifies
    the runtime cost.

Do not add CRM depth, elevation, depth bin, or mask reason to the model feature
matrix in this task.

## Plan/Spec Requirement

This task changes model fitting data. Before implementation, write a brief
implementation plan that confirms:

- Whether the sidecar sample is derived from the full-grid table plus mask or
  from the existing masked sample.
- The exact strata and quota policy.
- Whether quotas are fixed row counts, fractions, or weights.
- How deterministic sampling is keyed so reruns are stable.
- How sample weights, if present, are recomputed or intentionally ignored by
  the binary model.
- Which binary model paths are overwritten versus written as sidecars.
- Which comparison metrics decide whether this sampling policy should become
  the default.

## Implementation Plan

- Resolved source: derive the CRM-stratified sidecar from the full-grid table
  plus the plausible-kelp domain mask, not from the existing masked sample. The
  current masked sample remains the comparison baseline.
- Resolved strata: use `domain_mask_reason` and `depth_bin` as sampling strata.
  All retained `kelpwatch_station` rows are kept; only
  `assumed_background` rows are sampled by stratum.
- Resolved quota policy: use configured per-stratum sample fractions with
  conservative defaults that emphasize `retained_ambiguous_coast` /
  `ambiguous_coast`, keep normal `0_40m` nearshore support, and reduce
  `50_100m` pressure. Missing per-stratum settings fall back to a configured
  default fraction.
- Resolved deterministic key: sample decisions are a pure function of
  `aef_grid_cell_id`, `year`, and the configured random seed, so reruns are
  stable without storing row order.
- Resolved weights: recompute `sample_weight` for assumed-background rows as
  retained stratum population divided by sampled stratum rows. Kelpwatch rows
  keep weight `1.0`. The current binary model uses class weighting and does
  not consume `sample_weight`; the column is retained for audit compatibility.
- Resolved binary paths: write CRM-stratified binary outputs to sidecar paths
  under the existing `models.binary_presence.sidecars.crm_stratified` block and
  do not overwrite the current P1-18 binary artifacts.
- Resolved comparison: write a compact comparison table covering validation and
  test sample metrics, 2022 full-grid area/leakage rows, and
  assumed-background predicted-positive behavior by CRM stratum.

- Read the current mask and full-grid domain metadata needed for
  stratification.
- Verify the current sample and current binary false-positive distribution by
  `split`, `year`, `label_source`, `domain_mask_reason`, and `depth_bin`.
- Build a deterministic CRM-stratified sampler for assumed-background rows.
- Keep all `kelpwatch_station` rows inside the retained plausible-kelp domain.
- Sample assumed-background rows by configured CRM strata, with higher coverage
  for ambiguous/coastal strata and lower coverage for easy deeper-water strata.
- Write a sidecar sample, manifest, and summary table that report source
  population counts, sampled counts, and effective sample fractions by year,
  label source, `domain_mask_reason`, and `depth_bin`.
- Retrain the binary presence model from the sidecar sample.
- Rerun binary prediction/report commands needed to compare against P1-18.
- Compare at least:
  - validation 2021 AUPRC, precision, recall, and F1;
  - held-out 2022 AUPRC, precision, recall, and F1;
  - masked full-grid 2022 predicted positive area;
  - assumed-background predicted-positive count/rate by `depth_bin` and
    `domain_mask_reason`;
  - Kelpwatch-station recall for `annual_max_ge_10pct`.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_full_grid_alignment.py tests/test_binary_presence.py tests/test_model_analysis.py
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection should include the sidecar sample summary and a compact
comparison of current versus CRM-stratified binary behavior by retained CRM
stratum.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`, defined by
  `kelp_fraction_y >= 0.10` / `kelp_max_y >= 90 m2`.
- Mask: current P1-12/P1-14 plausible-kelp domain mask.
- Sampling context: CRM-derived `domain_mask_reason` and `depth_bin` are
  sampling strata only, not model predictors.

## Acceptance Criteria

- The existing masked sample remains available for comparison.
- A sidecar CRM-stratified masked sample is produced from deterministic
  sampling rules.
- All retained Kelpwatch-observed rows are included in the sidecar sample.
- The sample manifest records population counts, sampled counts, sampling
  fractions, and dropped counts by year, label source, `domain_mask_reason`,
  and `depth_bin`.
- The sidecar sample increases assumed-background coverage in hard coastal
  strata relative to the current sample and reduces easy deeper-water sampling
  pressure.
- Binary model comparison shows whether the sampling change reduces
  ambiguous-coast assumed-background false positives without unacceptable loss
  of Kelpwatch-station recall.
- The implementation does not tune on the 2022 test split.
- The Phase 1 report or comparison table makes the sampling-policy comparison
  visible.

## Known Constraints Or Non-Goals

- Do not use CRM depth, elevation, depth bin, or mask reason as model
  predictors.
- Do not change the P1-12 plausible-kelp domain mask thresholds.
- Do not change the annual-max label input or binary target threshold.
- Do not replace the existing masked sample without a comparison run and an
  explicit follow-up decision.
- Do not tune the sampling policy on the 2022 test split.
- Do not start full West Coast scale-up.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
