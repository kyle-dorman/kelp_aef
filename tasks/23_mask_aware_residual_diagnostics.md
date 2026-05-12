# Task 23: Add Mask-Aware Residual Diagnostics

## Goal

Add the final Bathymetry and DEM domain-filter diagnostic task for Phase 1:
explain where the masked-sample ridge model is still failing inside the retained
plausible-kelp domain.

P1-12 built the static domain mask, P1-13 applied it to full-grid reporting, and
P1-14 applied it to training and sampling. P1-15 should make the remaining
masked-domain residuals interpretable by mask reason, CRM depth/elevation bin,
label source, observed canopy bin, and top residual class. This should close the
Bathymetry and DEM domain-filter block before moving into imbalance-aware model
tasks.

This task is diagnostic only. It should not change the mask thresholds, model
objective, training sample, or fitted model.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current masked-sample baseline outputs from P1-14:
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`
- Plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Plausible-kelp domain mask manifest:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`.
- Existing masked-domain reporting tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.masked.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_latitude_band.masked.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/off_domain_prediction_leakage_audit.csv`
- Existing model-analysis report and table outputs under `reports.model_analysis`
  and `reports.outputs`.

Current P1-14 sanity anchors to preserve unless refreshed by implementation:
the masked model-input sample has 313,954 retained rows, 0 dropped
Kelpwatch-observed rows, and 0 dropped Kelpwatch-positive rows. The primary
2022 masked full-grid report scope has 999,519 retained plausible-domain cells.

## Outputs

- Updated package-backed diagnostic code, likely in:
  - `src/kelp_aef/evaluation/model_analysis.py`
  - `src/kelp_aef/viz/residual_maps.py`
  - tests under `tests/test_model_analysis.py` and/or
    `tests/test_residual_maps.py`
- Config output paths in `configs/monterey_smoke.yaml`, under
  `reports.outputs` or a narrow `reports.model_analysis` diagnostics block.
- Mask-aware residual taxonomy table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv`.
- Residual summary by retained mask reason, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_mask_reason.csv`.
- Residual summary by CRM depth/elevation bin, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_depth_bin.csv`.
- Top residual table with domain context, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.csv` extended
  or sidecar-ed as
  `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.domain_context.csv`.
- Optional compact figure showing residual behavior by retained depth/elevation
  bin, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_domain_context.png`.
- Updated Phase 1 model-analysis Markdown, HTML, and PDF report with a short
  "Mask-Aware Residual Diagnostics" section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Reuse `reports.domain_mask.mask_table` and `reports.domain_mask.mask_manifest`
  for the static P1-12 mask.
- Reuse existing model-analysis settings for model name, split, and year.
- Add output paths only for new diagnostic tables/figures.
- Do not duplicate mask input paths in a second config block unless the current
  loader pattern makes that unavoidable.

## Plan/Spec Requirement

This is a multi-file diagnostic/reporting task. Before implementation, write a
brief implementation plan that confirms:

- Which prediction table is the source of truth for each diagnostic:
  sample-level predictions, masked full-grid predictions, or both.
- Which mask columns will be joined into residual rows. Expected minimum:
  `is_plausible_kelp_domain`, `domain_mask_reason`, `depth_bin`,
  `elevation_bin`, `crm_elevation_m`, `crm_depth_m`, and
  `domain_mask_version` when available.
- Whether top residual output replaces the existing table or writes a sidecar
  with domain context first.
- How diagnostics separate retained-domain residuals from off-domain leakage.
- How observed canopy bins and residual classes are named.
- Which report section will summarize the diagnostic findings.

## Implementation Plan

- Load the P1-12 domain mask keyed by `aef_grid_cell_id`, including depth,
  elevation, reason, and version columns when present.
- Join domain context onto the current primary model residual rows for the
  configured report split/year/model.
- Keep the primary diagnostics restricted to
  `is_plausible_kelp_domain == true`, matching the P1-14 masked training and
  reporting domain.
- Preserve off-domain rows only through the existing
  `off_domain_prediction_leakage_audit.csv` or an explicitly labeled audit
  appendix table. Do not mix off-domain rows into primary residual rankings.
- Classify residuals into a small stable taxonomy, for example:
  `observed_zero_false_positive`, `low_canopy_overprediction`,
  `positive_underprediction`, `high_canopy_underprediction`,
  `near_correct`, and `missing_or_uncalculable`.
- Summarize residuals by:
  - `domain_mask_reason`
  - `depth_bin`
  - `elevation_bin`
  - `label_source`
  - observed canopy bin
  - residual taxonomy class
- For each group, report at least:
  `row_count`, `observed_canopy_area`, `predicted_canopy_area`, `area_bias`,
  `area_pct_bias`, `mae`, `rmse`, `mean_residual`, `median_residual`,
  `underprediction_count`, `overprediction_count`, and high-error counts.
- Add domain context to the top residual table so each high-error row can be
  inspected with mask reason, CRM depth/elevation, depth bin, elevation bin,
  label source, observed canopy, prediction, residual, and coordinates.
- Add a concise report section that answers:
  - Are retained ambiguous-coast cells behaving differently from retained
    depth-threshold cells?
  - Are the largest underpredictions concentrated in specific depth/elevation
    bins?
  - Are remaining false positives mostly background rows inside plausible
    habitat, rather than off-domain leakage?
  - Does the diagnostic point to an imbalance/objective problem, a domain-mask
    problem, or both?
- Keep this as explanation and triage. Do not tune thresholds, labels, model
  class, or training weights in this task.

## Expected Reporting Semantics

Use these definitions unless the implementation plan revises them:

- `mask_status = plausible_kelp_domain`: retained P1-12 cells only.
- `evaluation_scope = full_grid_masked`: complete retained plausible-kelp
  domain for the configured report split/year.
- `model_input_domain = plausible_kelp_domain`: masked sample produced by P1-14.
- `domain_context = retained_mask_context`: domain-mask reason and CRM
  depth/elevation metadata joined to residual rows.
- `off_domain_audit`: cells dropped by the mask; diagnostic only, not primary
  model-comparison rows.

Avoid unqualified `all` or `full_grid` labels in new outputs unless
`mask_status` and `evaluation_scope` make the domain explicit.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py tests/test_residual_maps.py
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv'; df=pd.read_csv(p); print(df.head()); print(df[['mask_status','evaluation_scope','domain_mask_reason']].drop_duplicates().head(20))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.domain_context.csv'; df=pd.read_csv(p); print(df[['residual_type','domain_mask_reason','depth_bin','crm_depth_m','observed_canopy_area','predicted_canopy_area','residual_kelp_max_y']].head(20).to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022 for model inputs and report tables.
- Report split/year: current `reports.model_analysis` settings, expected
  `test` / `2022`.
- Mask: static P1-12 plausible-kelp domain mask.
- Model: current masked-sample ridge baseline from P1-14.

## Acceptance Criteria

- The report includes a mask-aware residual diagnostics section that is
  generated from package-backed code, not manual notes.
- New diagnostic tables have explicit `mask_status`, `evaluation_scope`,
  `model_name`, `split`, and `year` columns.
- Primary residual diagnostics include only retained plausible-kelp domain rows.
- Off-domain leakage remains isolated as audit context.
- Top residual rows include domain mask reason, CRM depth/elevation, depth bin,
  and elevation bin.
- Residual summaries make clear whether remaining errors are concentrated in
  ambiguous-coast cells, specific depth/elevation bins, label-source groups, or
  observed-canopy bins.
- Tests cover mask-context joins, residual taxonomy aggregation, top-residual
  context, and report-section output.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not change the P1-12 mask thresholds.
- Do not rebuild the domain mask unless an input artifact is missing.
- Do not retrain models or change the training sample.
- Do not introduce binary, hurdle, balanced, capped-weight, or stratified
  background models. Those start in later Phase 1 tasks.
- Do not use bathymetry, DEM, depth, elevation, or mask reason as model
  predictors in this task.
- Do not tune decisions on the 2022 test split. Use diagnostics to plan the
  next validation-oriented tasks.
- Do not claim ecological truth. These diagnostics explain Kelpwatch-style
  residuals within a physically plausible domain.
