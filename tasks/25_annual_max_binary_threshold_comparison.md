# Task 25: Add Annual-Max Binary Threshold Comparison

## Goal

Add the validation-year threshold-comparison task for Phase 1. P1-16 made the
annual-max target imbalance explicit; P1-17 should now compare candidate binary
targets derived only from that same annual-max label before P1-18 trains a
balanced binary presence model.

The task should choose one or more candidate binary target thresholds for the
next modeling task using the validation year only. It should not select a final
production threshold, retrain any model, change sample weights, change the
annual-max label input, or tune decisions on the 2022 test split.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current baseline sample predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`.
- Current masked full-grid predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`.
- Current P1-16 class and target-balance diagnostics:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_balance_by_label_source.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_background_rate_summary.csv`
- Current model-analysis report outputs under `reports.outputs`.
- Current P1-12/P1-14 mask metadata carried on sample and prediction rows when
  available:
  `is_plausible_kelp_domain`, `domain_mask_reason`, `depth_bin`,
  `elevation_bin`, and `domain_mask_version`.

Current anchors to preserve unless an upstream artifact is intentionally
refreshed:

- Masked model-input sample has 313,954 retained rows.
- Primary validation year is 2021; primary held-out test year is 2022.
- The primary 2022 masked full-grid report scope has 999,519 retained
  plausible-domain prediction rows.
- P1-16 found the primary 2022 masked full-grid scope is 1.29% positive,
  0.41% high canopy, 0.06% saturated/near-saturated, 98.7% zero, and 97.0%
  assumed-background.
- The report frames results as learning Kelpwatch-style annual-max labels, not
  independent kelp biomass truth.

## Outputs

- Updated package-backed diagnostic code, likely in:
  - `src/kelp_aef/evaluation/model_analysis.py`
  - tests under `tests/test_model_analysis.py`
- Config output paths in `configs/monterey_smoke.yaml`, under
  `reports.outputs`.
- Validation-first threshold comparison table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_comparison.csv`.
- Candidate-threshold recommendation table or JSON, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_recommendation.csv`.
- Threshold prevalence table by split and label source, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_prevalence.csv`.
- Optional compact validation figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_binary_threshold_comparison.png`.
- Updated Phase 1 model-analysis Markdown, HTML, and PDF report with a short
  "Annual-Max Binary Threshold Comparison" section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Reuse `reports.model_analysis.threshold_fractions` for configured candidate
  thresholds when possible.
- Ensure the minimum candidate set includes 1%, 5%, and 10% annual max:
  `0.01`, `0.05`, and `0.10`.
- Retain diagnostic thresholds already used by the report when useful, such as
  `>0`, 50%, or 90%, but keep those clearly labeled as diagnostic thresholds.
- Reuse `reports.domain_mask.mask_status` and `reports.domain_mask.evaluation_scope`
  when the reporting mask is active.
- Add output paths only for new diagnostic tables/figures.
- Do not add a new training or model-objective config block in this task.

## Plan/Spec Requirement

This is a multi-output validation-oriented diagnostic task. Before
implementation, write a brief implementation plan that confirms:

- Which artifact is the source of truth for label prevalence and split
  membership: expected default is the current split manifest and masked
  model-input sample.
- Which prediction artifact is used to score threshold behavior for existing
  models: expected default is current sample predictions filtered to the
  validation split.
- Which candidate thresholds are compared. Minimum required:
  `>0`, `>=1%`, `>=5%`, and `>=10%` annual max; include retained diagnostic
  thresholds only when they are useful and clearly labeled.
- Which metrics are validation-selection metrics. Expected minimum:
  positive prevalence, positive count, precision, recall, F1, predicted positive
  rate, observed positive area, and false-positive area on assumed-background
  rows.
- How grouping keeps `split`, `year`, `label_source`, `mask_status`,
  `evaluation_scope`, `data_scope`, and `threshold_fraction` explicit.
- How the report text will state that the candidate choice is based on the
  2021 validation year and does not tune on the 2022 test split.

## Implementation Plan

- Load the current model-analysis inputs through the existing `analyze-model`
  path where possible.
- Build candidate annual-max binary target columns from `kelp_fraction_y`:
  - `annual_max_gt0`: `kelp_fraction_y > 0`
  - `annual_max_ge_1pct`: `kelp_fraction_y >= 0.01`
  - `annual_max_ge_5pct`: `kelp_fraction_y >= 0.05`
  - `annual_max_ge_10pct`: `kelp_fraction_y >= 0.10`
  - retained diagnostic thresholds from config, if present.
- Compare thresholds on validation rows only for the recommendation policy.
  Expected validation split/year: `validation` / `2021`.
- Report train and test rows only as context or locked-audit rows. Do not use
  2022 test rows to choose the candidate target threshold.
- For each threshold and scope, summarize at least:
  `row_count`, `station_count`, `positive_count`, `positive_rate`,
  `assumed_background_positive_count`, `assumed_background_positive_rate`,
  `observed_positive_area`, `mean_positive_area`, and label-source counts.
- For existing model predictions, compare threshold behavior with at least:
  `model_name`, `threshold_fraction`, `threshold_area`, `precision`, `recall`,
  `f1`, `predicted_positive_count`, `predicted_positive_rate`,
  `false_positive_count`, `false_positive_rate`, `false_positive_area`,
  `false_negative_count`, and `false_negative_area`.
- Rank candidate thresholds with a transparent validation-only policy. A
  reasonable default is to prefer thresholds that keep enough positives for
  training while reducing extreme background prevalence, then use F1 or
  precision-recall balance as a secondary diagnostic. Record the policy in the
  recommendation table.
- Add a concise report section that answers:
  - Which annual-max thresholds have enough validation positives to train a
    binary model?
  - How quickly do positives become rare as the threshold moves from `>0` to
    1%, 5%, and 10%?
  - Which threshold is recommended as the next candidate for P1-18, and why?
  - Which numbers are validation-selection evidence versus test/audit context?
- Keep the current annual-max label input. Do not add seasonal, persistence, or
  non-annual-max target inputs.

## Expected Reporting Semantics

Use these definitions unless the implementation plan revises them:

- `selection_split = validation`: threshold recommendation uses validation
  rows only.
- `selection_year = 2021`: current validation year in the Monterey smoke
  config.
- `test_split = test`: held-out 2022 rows are not used for choosing the
  candidate threshold.
- `mask_status = plausible_kelp_domain`: retained P1-12/P1-14 plausible-domain
  rows.
- `evaluation_scope = model_input_sample`: masked sampled rows used for model
  fitting/evaluation.
- `evaluation_scope = full_grid_masked`: complete retained plausible-kelp
  domain for configured full-grid report rows.
- `label_source = kelpwatch_station`: rows with Kelpwatch annual-max support.
- `label_source = assumed_background`: rows treated as zero canopy because they
  are not Kelpwatch-supported in the full-grid/background sample.

Avoid unqualified threshold labels such as `positive` or `present`. Use labels
that encode the annual-max rule, such as `annual_max_ge_5pct`.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_comparison.csv'; df=pd.read_csv(p); print(df.head()); print(df.query(\"split == 'validation'\")[['threshold_fraction','label_source','positive_rate','precision','recall','f1']].head(30).to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_recommendation.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "from pathlib import Path; p=Path('/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md'); text=p.read_text(); start=text.find('## Annual-Max Binary Threshold Comparison'); print(start >= 0); print(text[start:text.find('##', start + 3)])"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Threshold-selection split/year: validation 2021.
- Held-out test split/year: test 2022, not used for selection.
- Label input: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Mask: current P1-12/P1-14 plausible-kelp domain mask.
- Model context: current masked-sample ridge/reference predictions, but this
  task should not train a binary model or change fitted model artifacts.

## Acceptance Criteria

- The report includes a package-generated "Annual-Max Binary Threshold
  Comparison" section.
- New threshold tables have explicit `mask_status`, `evaluation_scope`,
  `data_scope`, `split`, `year`, `label_source`, `threshold_fraction`, and
  threshold-label columns.
- The minimum threshold set includes annual-max `>0`, `>=1%`, `>=5%`, and
  `>=10%`.
- The recommendation table names the validation split/year and states the
  selection policy used.
- The selected candidate threshold for P1-18 is derived from validation rows,
  not from the 2022 test split.
- Test-year rows, if reported, are explicitly labeled as audit/context rows and
  are not used in selection.
- The report explains how candidate thresholds trade off prevalence, background
  reduction, recall, and precision without claiming ecological truth.
- Tests cover threshold-label generation, validation-only recommendation,
  grouped prevalence/rate calculations, report-section output, and no-positive
  edge cases.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not train a binary, balanced, hurdle, conditional, or stratified model.
- Do not select the final production binary threshold.
- Do not tune decisions on the 2022 test split.
- Do not change the annual max label input or add alternative temporal targets.
- Do not change model sample weights, mask thresholds, training sample rows, or
  fitted model artifacts.
- Do not use bathymetry, DEM, depth, elevation, or mask reason as model
  predictors in this task.
- Do not claim ecological truth. These diagnostics compare Kelpwatch-style
  annual-max binary target definitions.
