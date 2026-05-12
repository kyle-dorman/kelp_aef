# Task 24: Add Class And Target-Balance Diagnostics

## Goal

Add the first imbalance-robust modeling diagnostic for Phase 1: make the annual
max target imbalance explicit before changing objectives, thresholds, sampling,
or model classes.

P1-15 closed the Bathymetry and DEM domain-filter block by explaining residuals
inside the retained plausible-kelp domain. P1-16 should now quantify the target
distribution that later binary, balanced, hurdle, or conditional models are
trying to fix. The core question is: how rare are positive, high-canopy, and
saturated annual-max rows relative to zero and assumed-background rows, after
the current mask-aware sample and reporting scope?

This task is diagnostic only. It should not train a binary model, select a
production threshold, change sample weights, change the annual-max label input,
or alter the fitted ridge/reference models.

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
- Current masked-domain reporting tables and model-analysis report outputs under
  `reports.outputs`.
- Current P1-12/P1-14 mask metadata carried on the sample and full-grid
  prediction rows when available:
  `is_plausible_kelp_domain`, `domain_mask_reason`, `depth_bin`,
  `elevation_bin`, and `domain_mask_version`.

Current anchors to preserve unless a prior artifact is intentionally refreshed:

- Masked model-input sample has 313,954 retained rows.
- The primary 2022 masked full-grid report scope has 999,519 retained
  plausible-domain prediction rows.
- The report frames results as learning Kelpwatch-style annual-max labels, not
  independent kelp biomass truth.

## Outputs

- Updated package-backed diagnostic code, likely in:
  - `src/kelp_aef/evaluation/model_analysis.py`
  - tests under `tests/test_model_analysis.py`
- Config output paths in `configs/monterey_smoke.yaml`, under
  `reports.outputs`.
- Class-balance summary table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv`.
- Target-balance summary table by label source and mask status, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_balance_by_label_source.csv`.
- Background-rate summary table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_background_rate_summary.csv`.
- Optional compact figure showing annual-max class imbalance, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_class_balance.png`.
- Updated Phase 1 model-analysis Markdown, HTML, and PDF report with a short
  "Class And Target Balance" section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Reuse existing model-analysis settings for split/year and threshold fractions.
- Reuse `reports.domain_mask.mask_status` and `reports.domain_mask.evaluation_scope`
  when the reporting mask is active.
- Add output paths only for new diagnostic tables/figures.
- Do not add a new training or modeling config block in this task.

## Plan/Spec Requirement

This is a multi-output diagnostic/reporting task. Before implementation, write a
brief implementation plan that confirms:

- Which artifacts are the source of truth for each diagnostic:
  masked model-input sample, split manifest, sample predictions, masked
  full-grid predictions, or compact cached report tables.
- Which scopes will be compared. Expected minimum:
  `model_input_sample`, `split_manifest_retained`, `sample_predictions`, and
  `full_grid_masked` when available.
- Which target classes will be reported. Expected minimum:
  zero annual max, positive annual max, configured threshold positives, high
  canopy, saturated canopy, and assumed-background rows.
- How rates will be grouped by `split`, `year`, `label_source`, `mask_status`,
  and `evaluation_scope`.
- How the report text will explain imbalance without choosing a model threshold
  or tuning on the 2022 test split.

## Implementation Plan

- Load the current model-analysis inputs through the existing
  `analyze-model` code path where possible.
- Build class-balance rows from the current annual-max target, using stable
  diagnostic classes:
  - `zero_canopy`: `kelp_max_y == 0`
  - `positive_canopy`: `kelp_max_y > 0`
  - `positive_ge_1pct`: `kelp_fraction_y >= 0.01`
  - `positive_ge_5pct`: `kelp_fraction_y >= 0.05`
  - `positive_ge_10pct`: `kelp_fraction_y >= 0.10`
  - `high_canopy_ge_450m2`: `kelp_max_y >= 450`
  - `very_high_canopy_ge_810m2`: `kelp_max_y >= 810`
  - `saturated_or_near_saturated`: `kelp_max_y >= 900`
- Keep these thresholds diagnostic. Do not select a production binary threshold
  in P1-16; P1-17 handles threshold comparison on the validation year.
- Summarize each group with at least:
  `row_count`, `station_count` when available, `zero_count`, `positive_count`,
  `positive_rate`, configured threshold counts/rates, `high_canopy_count`,
  `high_canopy_rate`, `saturated_count`, `saturated_rate`,
  `assumed_background_count`, `assumed_background_rate`,
  `observed_canopy_area`, and `mean_observed_canopy_area`.
- Include explicit metadata columns:
  `mask_status`, `evaluation_scope`, `data_scope`, `split`, `year`, and
  `label_source`.
- Add a concise report section that answers:
  - How dominant are zero and assumed-background rows in the current masked
    model-input sample?
  - How rare are positive, high-canopy, and saturated annual-max rows by split?
  - Does masking reduce off-domain background but leave a strong within-domain
    class imbalance?
  - Why should later binary, balanced, hurdle, or conditional models be
    evaluated against these rates rather than introduced blindly?
- Preserve the current annual-max label input. Do not add alternative seasonal
  targets or non-annual-max labels.

## Expected Reporting Semantics

Use these definitions unless the implementation plan revises them:

- `mask_status = plausible_kelp_domain`: retained P1-12/P1-14 plausible-domain
  rows.
- `evaluation_scope = model_input_sample`: masked sampled rows used for model
  fitting/evaluation.
- `evaluation_scope = full_grid_masked`: complete retained plausible-kelp domain
  for configured full-grid report rows.
- `label_source = kelpwatch_station`: rows with Kelpwatch annual-max support.
- `label_source = assumed_background`: rows treated as zero canopy because they
  are not Kelpwatch-supported in the full-grid/background sample.

Avoid labels like `all` unless paired with explicit `mask_status`,
`evaluation_scope`, and `data_scope` columns.

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
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv'; df=pd.read_csv(p); print(df.head()); print(df[['data_scope','mask_status','evaluation_scope','split','label_source','positive_rate','high_canopy_rate','assumed_background_rate']].head(20).to_string(index=False))"
uv run python -c "from pathlib import Path; p=Path('/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md'); text=p.read_text(); print('Class And Target Balance' in text); print(text[text.find('## Class And Target Balance'):text.find('##', text.find('## Class And Target Balance') + 3)])"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label input: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Mask: current P1-12/P1-14 plausible-kelp domain mask.
- Model context: current masked-sample ridge/reference outputs, but this task
  should not change or retrain any model.

## Acceptance Criteria

- The report includes a package-generated "Class And Target Balance" section.
- New diagnostic tables have explicit `mask_status`, `evaluation_scope`,
  `data_scope`, `split`, `year`, and `label_source` columns.
- Tables include positive-rate, high-canopy-rate, saturated-rate, and
  assumed-background-rate summaries by split, label source, and mask status.
- The report makes the current annual-max imbalance visible before P1-17 starts
  threshold comparison.
- The report reproduces the known Phase 0/Phase 1 failure mode: many zero or
  assumed-background rows, few high-canopy rows, and strong incentive for a
  continuous ridge model to shrink high canopy while leaking small positives.
- Tests cover grouped rate calculations, mask/status metadata, report-section
  output, and edge cases with no positive or no high-canopy rows.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not train a binary, balanced, hurdle, conditional, or stratified model.
- Do not select the production annual-max binary threshold.
- Do not tune decisions on the 2022 test split.
- Do not change the annual max label input or add alternative temporal targets.
- Do not change model sample weights, mask thresholds, training sample rows, or
  fitted model artifacts.
- Do not claim ecological truth. These diagnostics quantify imbalance in
  Kelpwatch-style annual-max labels.
