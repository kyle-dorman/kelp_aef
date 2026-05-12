# Task 22: Apply The Domain Mask To Training And Sampling

## Goal

Apply the P1-12 plausible-kelp domain mask to the model-input training sample
so the next baseline run trains and evaluates on physically plausible cells by
default.

This task follows P1-13, which applied the same mask to reporting only. P1-14
should make the model-input sample match the primary reporting domain before
interpreting model changes. The goal is not to solve or calibrate predictions
for cells outside the mask.

When this task says `full-grid calibration`, it means calibration over the full
retained plausible-kelp domain: all cells with
`is_plausible_kelp_domain == true`. It does not mean the full unmasked AEF tile.
Dropped land and deep-water cells may remain in a separate audit table, but
they should not be part of primary calibration, model comparison, or acceptance
criteria.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Plausible-kelp domain mask manifest:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`.
- Full-grid aligned training table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current unmasked background-inclusive sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`.
- Current baseline config under `models.baselines`.
- P1-13 reporting-domain config under `reports.domain_mask`.

Known current behavior to verify before editing: the current training sample is
drawn from the unmasked full tile. A previous inspection found 1,400,809 sample
rows, with 313,954 retained by the plausible-kelp mask and 1,086,855 dropped by
the mask. Treat those counts as a sanity target to refresh, not a hard-coded
contract.

## Outputs

- Updated package-backed sampling/training code, likely in:
  - `src/kelp_aef/alignment/full_grid.py`
  - `src/kelp_aef/evaluation/baselines.py`
  - tests under `tests/test_full_grid_alignment.py` and/or
    `tests/test_baselines.py`
- Config paths under a narrow training/sampling mask block in
  `configs/monterey_smoke.yaml`.
- Masked model-input sample, for example:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Masked sample manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json`.
- Masked sample summary, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.masked_summary.csv`.
- Retrained baseline artifacts using the masked sample, reusing or sidecar-ing
  current model/prediction paths according to the implementation plan.
- Updated masked-domain full-grid calibration/report outputs from P1-13 after
  retraining.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Input domain mask table path.
- Input domain mask manifest path.
- A training/sampling policy flag that makes `plausible_kelp_domain` the
  default model-input domain.
- Masked sample output table.
- Masked sample output manifest.
- Masked sample summary table.
- Either updated `models.baselines.input_table` pointing at the masked sample or
  explicit masked model-output sidecar paths if the unmasked model needs to be
  preserved for comparison.

Prefer reusing the existing `reports.domain_mask` input paths for the mask
itself instead of duplicating mask paths in multiple config sections. Add
sampling-specific output paths only where needed.

## Plan/Spec Requirement

This task changes model fitting semantics. Before implementation, write a brief
implementation plan that confirms:

- Whether the mask is applied during background-sample construction or by
  deriving a masked sample from the existing full-grid/sample artifact.
- How population counts and sample weights should be computed after masking.
- Whether Kelpwatch-observed zero rows outside the mask are dropped. Default:
  drop all off-domain rows, but report how many observed rows were dropped, if
  any.
- Whether output paths replace the old sample/model paths or write `.masked`
  sidecars first.
- Which report commands must rerun after retraining.
- How the unmasked run is preserved, if needed, for a one-time comparison.

## Implementation Plan

- Load the P1-12 mask keyed by `aef_grid_cell_id`.
- Verify the current unmasked sample composition by split, label source, and
  mask retention before changing code.
- Produce a masked model-input sample containing only rows where
  `is_plausible_kelp_domain == true`.
- Carry mask metadata columns into the sample and split manifest:
  `is_plausible_kelp_domain`, `domain_mask_reason`, `domain_mask_version`, and
  enough detail to audit the join.
- Recompute sample population counts and any `sample_weight` values using the
  masked-domain population, not the unmasked full tile.
- Keep Kelpwatch-station rows only if they are in the retained mask. If this
  drops any observed positive rows, stop and inspect before accepting the run,
  because P1-12 expected all positives to be retained.
- Point `train-baselines` at the masked sample and retrain the baseline models.
- Run `predict-full-grid` on the existing full-grid inference table. Prediction
  generation may remain unmasked because P1-13 filters reporting to the mask;
  do not make off-domain prediction generation a blocker unless it creates
  runtime or artifact-size problems.
- Rerun P1-13 reporting commands so area calibration and residual maps reflect
  the retrained masked-sample model on the masked full-grid domain.

## Expected Evaluation Semantics

Use these definitions unless the implementation plan revises them:

- `model_input_domain = plausible_kelp_domain`: rows available for training and
  sample-level evaluation after applying the P1-12 mask.
- `mask_status = plausible_kelp_domain`: full-grid report rows restricted to
  retained P1-12 mask cells.
- `evaluation_scope = full_grid_masked`: complete retained plausible-kelp
  domain, not the whole unmasked tile.
- `label_source = all`: all label-source groups within the retained
  plausible-kelp domain.
- Off-domain rows: diagnostic only, not primary model-comparison rows.

Avoid using unqualified `full_grid` or `all` in new outputs unless the table
also has `mask_status` and `evaluation_scope` columns that make the domain
explicit.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_full_grid_alignment.py tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet'; df=pd.read_parquet(p); print(len(df)); print(df[['year','label_source','is_plausible_kelp_domain']].value_counts().sort_index())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv'; df=pd.read_csv(p); print(df.query('split == \"test\" and year == 2022 and label_source == \"all\"')[['model_name','mask_status','evaluation_scope','row_count','area_pct_bias']])"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: year holdout, with 2018-2020 train, 2021 validation, and 2022 test.
- Mask: static P1-12 plausible-kelp domain mask.
- Reporting domain: full retained plausible-kelp domain only.

## Acceptance Criteria

- The model-input sample used by `train-baselines` contains only retained
  plausible-kelp domain rows by default.
- Sample manifests and summaries report retained/dropped counts by year and
  label source.
- Any Kelpwatch-observed rows dropped by the mask are counted and reviewed; no
  Kelpwatch-positive rows should be dropped without an explicit follow-up
  decision.
- Split manifest and prediction outputs retain enough mask metadata to trace the
  sample domain.
- Baselines are retrained from the masked sample.
- The refreshed model-analysis report uses masked-domain full-grid calibration
  as the primary area-calibration scope.
- Full-grid calibration means all retained mask cells, not the unmasked tile.
- Tests cover masked sample construction, sample-weight/population semantics,
  and report/model-comparison domain labels.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not use bathymetry, DEM, depth, elevation, or mask reason as model
  predictors in this task.
- Do not tighten or tune the P1-12 depth threshold in this task.
- Do not optimize for unmasked land/deep-water cells. They are outside the
  primary Phase 1 calibration domain after P1-13.
- Do not start binary, hurdle, or imbalance-aware model changes here; those are
  later Phase 1 tasks.
- Do not claim ecological truth. The mask is a physically plausible training
  and reporting domain for Kelpwatch-style labels.
