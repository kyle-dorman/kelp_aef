# Task 26: Train Balanced Binary Presence Model

## Goal

Train the first imbalance-aware binary annual-max model for Phase 1. P1-16 made
the masked annual-max class imbalance explicit, and P1-17 selected
`annual_max_ge_10pct` as the next validation-backed candidate target. P1-18
should now train a package-backed classifier that predicts the probability that
a row has Kelpwatch-style annual max canopy at or above 10% of a 30 m cell.

Use the P1-17 target definition as the modeling target for this task:

```text
annual_max_ge_10pct = kelp_fraction_y >= 0.10
annual_max_ge_10pct = kelp_max_y >= 90 m2
```

The task should reduce the current ridge failure mode where small positive
continuous predictions leak over a large assumed-background population. It
should not claim ecological presence truth; the binary target is still derived
from Kelpwatch annual-max weak labels.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current P1-17 threshold outputs:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_prevalence.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_recommendation.csv`
- Current ridge/reference baseline outputs for report comparison:
  - `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`
- Current model-analysis report outputs under `reports.outputs`.
- Current P1-12/P1-14 mask metadata carried on rows when available:
  `is_plausible_kelp_domain`, `domain_mask_reason`, `depth_bin`,
  `elevation_bin`, and `domain_mask_version`.

Current anchors to preserve unless an upstream artifact is intentionally
refreshed:

- Masked model-input sample has 313,954 retained rows.
- Primary training years are 2018-2020, validation year is 2021, and held-out
  test year is 2022.
- The primary 2022 masked full-grid report scope has 999,519 retained
  plausible-domain prediction rows.
- P1-17 recommended `annual_max_ge_10pct` from 2021 validation rows only.
- The report frames results as learning Kelpwatch-style annual-max labels, not
  independent kelp biomass truth.

## Outputs

- Package-backed binary model training and prediction code, likely in new or
  existing modules under `src/kelp_aef/`, for example:
  - `src/kelp_aef/evaluation/binary_presence.py`
  - `src/kelp_aef/cli.py`
  - `src/kelp_aef/evaluation/model_analysis.py`
  - tests under `tests/`
- Config model and output paths in `configs/monterey_smoke.yaml`.
- Serialized binary classifier, for example:
  `/Volumes/x10pro/kelp_aef/models/binary_presence/logistic_annual_max_ge_10pct.joblib`.
- Sample probability predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet`.
- Masked full-grid probability predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`.
- Binary model metrics table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv`.
- Validation probability-threshold selection table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv`.
- Full-grid positive-area summary, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv`.
- Prediction manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/binary_presence_prediction_manifest.json`.
- Compact precision-recall or threshold diagnostic figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_precision_recall.png`.
- Updated Phase 1 model-analysis Markdown, HTML, and PDF report with a short
  balanced binary model section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config fields:

- Add a `models.binary_presence` block rather than overloading
  `models.baselines`.
- Use the masked model-input sample as the training/evaluation input:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Use the current full-grid table plus the reporting domain mask for full-grid
  probability prediction.
- Set:
  - `target_label: annual_max_ge_10pct`
  - `target_column: kelp_fraction_y`
  - `target_threshold_fraction: 0.10`
  - `target_threshold_area: 90.0`
- Start with a class-weighted logistic regression probability baseline:
  `class_weight: balanced`.
- Keep `features: A00-A63`.
- Add a small regularization grid only if validation selection is implemented
  clearly and recorded. Do not tune anything on the 2022 test split.
- Add output paths for the model, predictions, metrics, selected probability
  threshold, full-grid area summary, manifest, and optional figure.

## Plan/Spec Requirement

This is a new model stage with new artifacts and report contracts. Before
implementation, write a brief implementation plan that confirms:

- The binary target source and threshold:
  `kelp_fraction_y >= 0.10`, equivalent to `kelp_max_y >= 90 m2`.
- The primary classifier and imbalance strategy. Expected default:
  class-weighted logistic regression with `class_weight = balanced`.
- Whether a second balanced-sampling model is included. Keep it out of scope
  unless it can be implemented without broadening the task.
- Which artifacts are source-of-truth for split membership and mask status.
- Which probability threshold policy is used for the report. Expected default:
  select a diagnostic operating threshold on 2021 validation rows only, such as
  max-F1 or a documented precision/recall policy. This is not the final
  calibrated production threshold; P1-19 handles calibration and threshold
  refinement.
- How sample metrics and full-grid area behavior are grouped by `split`, `year`,
  `label_source`, `mask_status`, `evaluation_scope`, and `threshold_label`.
- How the report will compare the binary model against current ridge/reference
  behavior without claiming independent ecological truth.

## Implementation Plan

- Add a package-backed CLI command, for example:
  `kelp-aef train-binary-presence --config configs/monterey_smoke.yaml`.
- Load the masked model-input sample and derive:
  `target_binary_y = kelp_fraction_y >= 0.10`.
- Reuse the existing train/validation/test split assignments:
  train 2018-2020, validation 2021, test 2022.
- Train a class-weighted logistic regression model on the training split using
  AEF bands `A00-A63`.
- If using a regularization grid, select the regularization parameter on the
  2021 validation split using a declared metric such as AUPRC or F1 at the
  validation-selected probability threshold. Record all grid rows.
- Write sample predictions for all retained sample rows with at least:
  `model_name`, `target_label`, `target_threshold_fraction`,
  `target_threshold_area`, `binary_observed_y`, `pred_binary_probability`,
  `pred_binary_class`, `probability_threshold`, split/year/label-source fields,
  and mask metadata when available.
- Stream full-grid probability predictions in chunks, applying or preserving
  the plausible-kelp domain mask so the primary full-grid report scope is
  `full_grid_masked`.
- Write full-grid prediction rows with the same target/model metadata plus
  probability and selected-class outputs. Do not use full-grid rows to fit or
  tune the model.
- Compute binary sample metrics by split and label source:
  AUROC, AUPRC, precision, recall, F1, true/false positives, true/false
  negatives, predicted positive rate, observed positive rate, and selected
  probability threshold.
- Compute full-grid behavior by split/year and label source:
  predicted positive count, predicted positive rate, predicted positive area
  in 30 m cell units, assumed-background predicted-positive count/rate, and
  comparison to observed `annual_max_ge_10pct` area where labels are available.
- Add the binary model rows to the Phase 1 report. The report should answer:
  - Does the binary model improve ranking skill on validation/test relative to
    the current ridge threshold behavior?
  - What probability threshold was selected on validation, and how does it
    behave on the held-out 2022 test split?
  - How much full-grid area is predicted positive inside the retained
    plausible-kelp domain?
  - Does the model reduce assumed-background leakage relative to the ridge
    annual-max threshold diagnostics?
- Keep P1-19 separate: do not add probability calibration, isotonic/Platt
  scaling, or final production threshold selection unless the implementation
  plan explicitly narrows that work and the user approves it.

## Expected Reporting Semantics

Use these definitions unless the implementation plan revises them:

- `target_label = annual_max_ge_10pct`.
- `target_threshold_fraction = 0.10`.
- `target_threshold_area = 90.0`.
- `selection_split = validation`: probability operating threshold is selected
  on validation rows only.
- `selection_year = 2021`.
- `test_split = test`: 2022 rows are held-out audit/evaluation rows and are not
  used to choose model hyperparameters or probability thresholds.
- `mask_status = plausible_kelp_domain`: retained P1-12/P1-14 plausible-domain
  rows.
- `evaluation_scope = model_input_sample`: masked sampled rows used for
  model fitting/evaluation.
- `evaluation_scope = full_grid_masked`: complete retained plausible-kelp
  domain for configured full-grid report rows.
- `label_source = kelpwatch_station`: rows with Kelpwatch annual-max support.
- `label_source = assumed_background`: rows treated as zero canopy because they
  are not Kelpwatch-supported in the full-grid/background sample.

Avoid unqualified labels such as `present` or `positive` in persisted outputs.
Use labels that encode the annual-max rule and probability threshold context.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py tests/test_baselines.py
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv'; df=pd.read_csv(p); print(df.head()); print(df[['split','year','label_source','target_label','auroc','auprc','precision','recall','f1','probability_threshold']].to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "from pathlib import Path; p=Path('/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md'); text=p.read_text(); start=text.find('## Balanced Binary Presence Model'); print(start >= 0); print(text[start:text.find('##', start + 3)])"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: configured 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Binary target: `annual_max_ge_10pct`, defined by
  `kelp_fraction_y >= 0.10` / `kelp_max_y >= 90 m2`.
- Probability-threshold selection split/year: validation 2021.
- Held-out test split/year: test 2022, not used for training, hyperparameter
  selection, or operating-threshold selection.
- Label input: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Mask: current P1-12/P1-14 plausible-kelp domain mask.
- Model context: current masked-sample ridge/reference predictions remain the
  comparison baseline.

## Acceptance Criteria

- A package-backed CLI command trains the binary model from
  `configs/monterey_smoke.yaml`.
- The binary target is explicitly `annual_max_ge_10pct` and is derived only from
  the current annual-max label columns.
- The primary model uses an explicit imbalance strategy, such as
  class-weighted logistic regression.
- Model fitting, hyperparameter selection, and probability-threshold selection
  do not use the 2022 test split.
- Sample prediction and full-grid prediction artifacts include target metadata,
  probability outputs, selected-class outputs, split/year/label-source fields,
  and mask metadata when available.
- Metrics include AUROC, AUPRC, precision, recall, F1, selected probability
  threshold, predicted positive rate, and false-positive behavior.
- Full-grid summaries quantify predicted positive area and assumed-background
  leakage inside `full_grid_masked`.
- The Phase 1 report includes a package-generated "Balanced Binary Presence
  Model" section.
- The report compares validation/test binary metrics and full-grid area behavior
  against current ridge/reference diagnostics without claiming ecological truth.
- Tests cover target construction, class-weighted fitting or training data
  preparation, validation-only threshold selection, metric calculations,
  full-grid area summaries, report-section output, and no-positive edge cases.
- Validation commands pass.

## Known Constraints Or Non-Goals

- Do not change the annual max label input or add alternative temporal targets.
- Do not use annual mean, fall-only, winter-only, persistence, or seasonal
  labels as targets.
- Do not tune on the 2022 test split.
- Do not treat `annual_max_ge_10pct` as a final production threshold; it is the
  next candidate target for P1-18.
- Do not add probability calibration, isotonic regression, Platt scaling, or
  final threshold calibration; P1-19 handles that.
- Do not use bathymetry, DEM, depth, elevation, or mask reason as model
  predictors in this task.
- Do not change mask thresholds or resample the training table.
- Do not start full West Coast scale-up.
- Do not claim ecological truth. This model predicts a Kelpwatch-style
  annual-max binary target.
