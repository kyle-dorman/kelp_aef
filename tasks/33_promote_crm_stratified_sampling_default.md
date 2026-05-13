# Task 33: Promote CRM-Stratified Sampling As Default

## Goal

Change the Monterey Phase 1 default masked model-input sample from the current
post-hoc masked background sample to a CRM-stratified, mask-first retained-domain
sample.

This task should replace the sidecar experiment with default pipeline behavior:

- build or load the plausible-kelp mask first;
- filter to the retained mask area before applying background sampling budgets;
- use CRM `domain_mask_reason` and `depth_bin` only as sampling strata;
- keep all retained Kelpwatch-observed rows;
- sample retained assumed-background rows by explicit quota;
- tighten the plausible-kelp mask maximum depth to `60.0` m;
- remove the retained `50_100m` sampling stratum and use a final retained
  `40_60m` bin;
- replace or retire `background_rows_per_year` for the masked default workflow.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Design decision from Task 32:
  `docs/phase1_crm_stratified_sampling_policy_decision.md`.
- Current full-grid aligned table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current aligned CRM support table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- Current plausible-kelp mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current default masked sample and manifest:
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json`
- Current CRM-stratified sidecar sample and manifest:
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked_manifest.json`
- Current model artifacts from P1-21a, used only for comparison and migration
  checks.

## Outputs

- Updated domain-mask config and implementation for a `60.0` m broad retained
  depth cutoff.
- Updated depth-bin labeling so the retained deep bin becomes `40_60m`, not a
  mislabeled `40_50m` or empty `50_100m`.
- Updated mask reason naming or manifest metadata if the prior
  `retained_depth_0_100m` reason code becomes misleading under a 60 m mask.
- Updated background-sampling config where CRM-stratified mask-first sampling
  produces the default masked sample path:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Updated masked sample manifest that records:
  - retained-domain population counts;
  - sampled retained-background counts;
  - per-year quota/fraction settings;
  - all retained Kelpwatch-observed row counts;
  - dropped-row counts caused by the 60 m mask;
  - deterministic sampling seed.
- Regenerated default model artifacts for the current Phase 1 commands:
  - `kelp-aef build-domain-mask`
  - `kelp-aef align-full-grid`
  - `kelp-aef train-baselines`
  - `kelp-aef predict-full-grid`
  - `kelp-aef train-binary-presence`
  - `kelp-aef calibrate-binary-presence`
  - `kelp-aef train-conditional-canopy`
  - `kelp-aef compose-hurdle-model`
  - `kelp-aef analyze-model`
- Tests for the mask-first sampler, 60 m bins, quota logic, all-observed-row
  retention, manifest fields, and config migration.

Historical sidecar artifacts may remain on disk for audit, but the default
workflow should not require current-vs-CRM sidecar execution.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config changes:

- Set `domain.plausible_kelp_mask.max_depth_m: 60.0`.
- Keep `domain.plausible_kelp_mask.nearshore_shallow_depth_m: 40.0`.
- Change depth-bin support so retained ocean bins are `0_40m` and `40_60m`.
- Remove the default retained `50_100m` sampling stratum.
- Promote the successful CRM sidecar quota shape into the default masked
  background sample policy, updated for the 60 m mask:

```yaml
random_seed: 31
default_fraction: 0.0
default_min_rows_per_year: 0
strata:
  - domain_mask_reason: retained_ambiguous_coast
    depth_bin: ambiguous_coast
    fraction: 0.12
    min_rows_per_year: 5000
  - domain_mask_reason: retained_depth_0_60m
    depth_bin: 0_40m
    fraction: 0.04
    min_rows_per_year: 8000
  - domain_mask_reason: retained_depth_0_60m
    depth_bin: 40_60m
    fraction: 0.01
    min_rows_per_year: 500
```

If the implementation keeps the generic `retained_depth_within_max_m` reason
instead of `retained_depth_0_60m`, document that in the manifest and tests.

Replace or reinterpret `alignment.background_sample.background_rows_per_year`.
For masked workflows, the effective row budget should be applied after retained
mask filtering, or replaced by explicit retained-background quotas. Do not keep
a large pre-mask cap as the control that appears to define the training sample.

## Plan/Spec Requirement

This task changes the default data contract for model fitting. Before
implementation, write a short implementation plan that confirms:

- Whether Task 32's decision note has been accepted.
- The exact depth-bin labels and reason-code vocabulary for the 60 m mask.
- Whether existing mask artifact paths are overwritten in place or a v2
  migration path is used first.
- How the sampler ensures mask-first behavior before any background budget.
- What replaces `background_rows_per_year` in config and manifests.
- Which sidecar config blocks and paths remain temporarily for audit.
- Which default model artifacts will be overwritten.
- How the report will prove the default artifacts now use the promoted policy.
- How Kelpwatch-positive retention is checked before accepting the 60 m mask.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_domain_mask.py tests/test_full_grid_alignment.py tests/test_baselines.py tests/test_binary_presence.py tests/test_conditional_canopy.py tests/test_hurdle.py tests/test_model_analysis.py
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml --fast
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
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

Manual checks should inspect the regenerated mask depth-bin summary, masked
sample summary, masked sample manifest, and primary model-analysis report.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`.
- Mask: retained plausible-kelp domain, revised to a 60 m maximum depth.
- Sampling context: CRM-derived mask reason and depth bin are sampling strata
  only, not model predictors.

## Acceptance Criteria

- Default masked sample generation filters to the retained plausible-kelp mask
  before applying any background row budget or stratum quota.
- `background_rows_per_year` no longer controls the default masked workflow as a
  broad pre-mask file-size cap.
- The default sample keeps all retained Kelpwatch-observed rows.
- The 60 m mask drops no Kelpwatch-positive rows, or any dropped positives are
  explicitly reported and reviewed before the task is marked complete.
- The retained depth-bin summary includes `0_40m` and `40_60m`; retained
  `50_100m` is gone from the default policy.
- The default model artifacts use the promoted CRM-stratified sampling policy
  without needing sidecar config paths.
- The default report and manifests make the active sampling policy clear.
- CRM depth, elevation, depth bin, and mask reason remain excluded from model
  feature matrices.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not tune new quota values on the 2022 test split.
- Do not use CRM bathymetry or mask metadata as model predictors.
- Do not change the annual-max label input or the `annual_max_ge_10pct` binary
  target.
- Do not delete historical sidecar artifacts from `/Volumes/x10pro/kelp_aef`
  unless a later cleanup task explicitly chooses to remove generated outputs.
- Do not start full West Coast scale-up.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
