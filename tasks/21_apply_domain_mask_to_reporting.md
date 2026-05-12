# Task 21: Apply The Domain Mask To Full-Grid Reporting

## Goal

Apply the P1-12 plausible-kelp domain mask to full-grid inference summaries and
residual reporting before any retraining.

This task should make the masked plausible-kelp domain the default largest
reporting area for Phase 1 area calibration. The old unmasked full-grid
`all`/overall area should not remain a recurring headline scope, because it
mostly measures obvious off-domain land and deep-water leakage that the mask is
now intended to exclude.

P1-12 already measured static mask coverage and Kelpwatch-positive retention.
P1-13 should measure the model/reporting effect of applying that mask to
existing full-grid predictions and reference-baseline area summaries.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Plausible-kelp domain mask manifest:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`.
- Full-grid prediction artifact:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`.
- Full-grid prediction manifest:
  `/Volumes/x10pro/kelp_aef/interim/baseline_full_grid_prediction_manifest.json`.
- Reference-baseline area calibration table:
  `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.csv`.
- Existing residual-map and model-analysis report config under
  `reports.map_residuals`, `reports.model_analysis`, and `reports.outputs`.

P1-12 result to preserve: the mask has 7,458,361 unique cells, retains 999,519
cells, drops 6,458,842 cells, retains all 58,497 Kelpwatch-positive cell-year
rows, has five positive rows in the 40-50 m QA bin, and has no positives deeper
than 50 m.

## Outputs

- Updated package-backed reporting code, likely in:
  - `src/kelp_aef/evaluation/baselines.py`
  - `src/kelp_aef/viz/residual_maps.py`
  - `src/kelp_aef/evaluation/model_analysis.py`
- Config paths under a narrow reporting/domain-mask block in
  `configs/monterey_smoke.yaml`.
- Masked full-grid area-bias table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.masked.csv`.
- Masked latitude-band area-bias table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_latitude_band.masked.csv`.
- Masked reference-baseline area-calibration table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv`.
- Masked residual map figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.masked.png`.
- Optional one-time off-domain leakage audit table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/off_domain_prediction_leakage_audit.csv`.
- Updated model-analysis report that treats the masked domain as the primary
  full-grid reporting scope.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Input domain mask table path.
- Input domain mask manifest path.
- Masked area-bias output paths.
- Masked residual-map output path.
- Masked reference-baseline area-calibration output path.
- Optional off-domain leakage audit output path.
- A reporting policy flag or string that makes `plausible_kelp_domain` the
  primary full-grid reporting domain.

Do not add training-sample or retraining paths in this task. Those belong to
P1-14.

## Plan/Spec Requirement

This task changes report semantics and downstream interpretation. Before
implementation, write a brief implementation plan that confirms:

- Which tables will be filtered to `is_plausible_kelp_domain == true`.
- Whether outputs replace old unmasked paths or write `.masked` sidecar paths
  first.
- How `mask_status`, `evaluation_scope`, and `label_source` values will be
  named so `all` does not mean unmasked full-grid area.
- Whether a one-time off-domain leakage audit is kept and where it appears.
- How the model-analysis report will identify the masked domain as the primary
  full-grid area calibration scope.
- Which downstream training/sampling commands remain unchanged until P1-14.

## Implementation Plan

- Load the P1-12 mask as a static key table keyed by `aef_grid_cell_id`.
- Join the mask to full-grid prediction rows before area-bias aggregation and
  residual-map row selection.
- Filter primary full-grid reporting to `is_plausible_kelp_domain == true`.
- Keep Kelpwatch-station skill metrics unchanged unless they are explicitly
  full-grid area summaries. P1-12 showed all positive Kelpwatch rows are
  retained, but station/sample metrics should still state their evaluation
  source clearly.
- Update area-calibration rows so the primary full-grid rows have
  `mask_status = plausible_kelp_domain` or equivalent, and so any `all`
  grouping means all label sources within the masked domain, not unmasked area.
- Remove or de-emphasize recurring unmasked `all` full-grid rows from the main
  report. If useful, write a separate off-domain leakage audit table with
  dropped-domain predicted area by model/year/reason so the migration can be
  inspected once without becoming a headline metric.
- Update residual maps to plot only masked-domain cells by default. Optionally
  use the off-domain audit table for dropped-domain summaries rather than a
  second unmasked map.
- Update report prose and comparison tables so masked-domain area calibration is
  the default full-grid interpretation for Phase 1.
- Keep model training, sampling, and prediction generation unchanged in this
  task. Existing predictions are filtered in reporting only.

## Expected Reporting Semantics

Use these definitions unless the implementation plan revises them:

- `mask_status = plausible_kelp_domain`: rows restricted to retained P1-12 mask
  cells.
- `evaluation_scope = full_grid_masked`: full-grid prediction/reporting rows
  after applying the domain mask.
- `label_source = all`: all available label-source groups within the masked
  domain only.
- `mask_status = off_domain_audit`: optional diagnostic rows summarizing cells
  dropped by the mask. These are not primary model-comparison rows.

Avoid using `mask_status = unmasked` or `label_source = all` as a recurring
headline for the whole AEF tile after this task.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_residual_maps.py tests/test_model_analysis.py tests/test_baselines.py
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.masked.csv'; df=pd.read_csv(p); print(df.head()); print(df.filter(regex='mask|scope|label').drop_duplicates())"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: keep the configured 2018-2022 full-grid predictions and report year
  2022.
- Mask: static P1-12 plausible-kelp domain mask.
- Fast path: use existing fast prediction/report fixtures where available, or
  add small synthetic tests that join a tiny mask to prediction rows.

## Acceptance Criteria

- Full-grid area-bias and residual-map reporting filter to retained mask cells
  by default.
- The model-analysis report treats masked-domain area calibration as the
  primary full-grid area scope.
- `all` no longer means unmasked full-tile area in recurring headline tables.
- Optional unmasked/off-domain diagnostics are isolated as audit rows or an
  audit table, not mixed into the main model-comparison headline.
- Existing full-grid prediction generation remains unchanged.
- Existing training and sampling remain unchanged.
- Tests cover mask joins, masked aggregation row counts, `mask_status` /
  `evaluation_scope` naming, and report table behavior.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not retrain models in this task.
- Do not apply the mask to the training or sampling artifact in this task.
- Do not tighten the depth threshold based on reporting outcomes in this task.
- Do not remove the unmasked prediction artifact; keep it as the reusable source
  for masked reporting and audit summaries.
- Do not claim ecological truth. The mask is a physically plausible reporting
  domain for Kelpwatch-style labels.
