# Task 27: Calibrate Binary Probabilities And Thresholds

## Goal

Calibrate the P1-18 balanced binary annual-max model so ranking skill,
classification thresholding, and full-grid area behavior are reported as
separate decisions. The calibration must be fit only on the 2021 validation
split and evaluated on the held-out 2022 test split.

The target remains the P1-17/P1-18 annual-max binary target:

```text
annual_max_ge_10pct = kelp_fraction_y >= 0.10
annual_max_ge_10pct = kelp_max_y >= 90 m2
```

This task should not treat calibrated probabilities as independent ecological
presence truth. They are calibrated probabilities for reproducing
Kelpwatch-style annual-max weak labels within the current Monterey Phase 1
domain.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Binary model sample predictions:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet`.
- Binary model full-grid predictions:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`.
- Binary model metrics and threshold-selection outputs:
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_thresholded_model_comparison.csv`
- Current Phase 1 model-analysis report outputs under `reports.outputs`.
- Current P1-12/P1-14 plausible-kelp reporting domain mask metadata already
  carried on binary prediction rows when available.

Current P1-18 anchors to preserve unless upstream artifacts are intentionally
refreshed:

- Model: `logistic_annual_max_ge_10pct`.
- Validation year: 2021.
- Test year: 2022.
- P1-18 validation-selected raw probability threshold: `0.91`.
- Primary full-grid reporting scope: retained plausible-kelp domain,
  `full_grid_masked`.
- Known map QA note: the P1-18 2022 binary map has a visible false-positive
  cluster in the nearshore area near or in the river mouth. Calibration should
  report how thresholds affect that leakage, but should not hide the spatial
  QA issue or solve it with an undocumented mask change.

## Outputs

- Package-backed calibration command, for example:
  `kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml`.
- New config paths under `models.binary_presence` or a nested
  `models.binary_presence.calibration` block.
- Serialized calibration model or compact metadata, for example:
  `/Volumes/x10pro/kelp_aef/models/binary_presence/logistic_annual_max_ge_10pct_calibration.joblib`.
- Calibrated sample predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_calibrated_sample_predictions.parquet`.
- Calibrated full-grid summary table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.csv`.
- Calibration metric table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv`.
- Calibration threshold-selection table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.csv`.
- Calibration diagnostic figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_calibration_curve.png`.
- Optional threshold comparison figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_calibrated_thresholds.png`.
- Calibration manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/binary_presence_calibration_manifest.json`.
- Updated Phase 1 Markdown, HTML, and PDF model-analysis report with a short
  calibrated binary presence section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Input sample predictions:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet`.
- Input full-grid predictions:
  `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`.
- Calibration split:
  `validation`.
- Calibration year:
  `2021`.
- Evaluation split:
  `test`.
- Evaluation year:
  `2022`.
- Candidate calibration methods:
  start with `platt` or `isotonic`, but keep the first implementation narrow.
- Candidate threshold policies:
  - validation max-F1 threshold on calibrated probabilities;
  - validation precision-constrained threshold, if the minimum precision target
    is explicitly configured;
  - validation area/prevalence-matching threshold, if the target prevalence is
    explicitly derived from validation rows.

## Plan/Spec Requirement

This is a model-calibration stage with new artifacts and report semantics.
Before implementation, write a brief implementation plan that confirms:

- Calibration method and why it is appropriate for a single validation year.
- Whether calibration is fit on all validation sample rows or only
  Kelpwatch-station rows.
- Whether assumed-background validation rows are treated as negatives for
  calibration. If they are included, the report must label this clearly.
- Which threshold policies are produced and which one is marked as the
  recommended diagnostic policy.
- Which outputs are row-level and which are compact aggregate summaries.
- How the full-grid calibrated area behavior is computed without using 2022
  test labels for threshold selection.
- How the river-mouth false-positive cluster is documented in report text or
  QA notes without changing the domain mask in this task.

## Implementation Plan

- Add a package-backed CLI command, for example:
  `kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml`.
- Load P1-18 sample predictions and validate that required columns exist:
  `split`, `year`, `label_source`, `binary_observed_y`,
  `pred_binary_probability`, `probability_threshold`, `longitude`, and
  `latitude`.
- Fit the chosen calibration model using 2021 validation rows only.
- Apply the fitted calibrator to sample predictions for train, validation, and
  test rows. Persist both raw and calibrated probabilities.
- Select candidate calibrated thresholds using 2021 validation rows only.
- Evaluate raw and calibrated probabilities on 2022 test rows, separating:
  ranking metrics such as AUROC/AUPRC, calibration metrics such as Brier score
  and expected calibration error, and threshold metrics such as precision,
  recall, F1, predicted-positive rate, and assumed-background false-positive
  rate.
- Apply the calibrator and selected calibrated thresholds to the existing
  binary full-grid prediction dataset without retraining the classifier.
- Write compact full-grid summaries by split, year, label source, mask status,
  evaluation scope, calibration method, and threshold policy. Avoid writing a
  second full row-level full-grid dataset unless the implementation plan
  justifies it.
- Write a calibration curve or reliability table/figure for validation and
  2022 test rows.
- Update the model-analysis report so it answers:
  - Did calibration improve probability reliability without changing ranking?
  - Which calibrated threshold is selected from validation rows?
  - How does calibrated thresholding change 2022 sample precision, recall, F1,
    and assumed-background false positives?
  - How does calibrated thresholding change 2022 masked full-grid predicted
    positive area?
  - Does the river-mouth false-positive cluster remain a QA concern?

## Expected Reporting Semantics

- `model_name = logistic_annual_max_ge_10pct`.
- `target_label = annual_max_ge_10pct`.
- `target_threshold_fraction = 0.10`.
- `target_threshold_area = 90.0`.
- `calibration_split = validation`.
- `calibration_year = 2021`.
- `test_split = test`.
- `test_year = 2022`.
- `probability_source = raw_logistic` or the calibrated method name.
- `threshold_policy` must identify how a threshold was selected.
- `mask_status = plausible_kelp_domain` for retained-domain full-grid rows.
- `evaluation_scope = model_input_sample` for sample rows.
- `evaluation_scope = full_grid_masked` for compact retained-domain full-grid
  summaries.

Keep ranking, calibration, threshold, and full-grid area metrics in separate
columns or tables. Do not imply that a better calibrated probability is a better
ranker unless AUROC/AUPRC also improve.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_binary_presence.py tests/test_model_analysis.py tests/test_package.py
uv run kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Training years inherited from P1-18: 2018-2020.
- Calibration year: 2021 validation split.
- Held-out audit year: 2022 test split.
- Label: Kelpwatch-style annual max canopy thresholded at `>=10%`.
- Feature model: existing P1-18 logistic probabilities, not a retrained
  classifier.

## Acceptance Criteria

- Calibration fitting uses only 2021 validation rows.
- 2022 test rows are never used to fit the calibrator or choose a threshold.
- Output tables include both raw-logistic and calibrated probability metrics,
  or otherwise preserve raw P1-18 metrics for comparison.
- The report separates:
  ranking skill, probability calibration, selected threshold behavior, and
  full-grid area behavior.
- The calibrated full-grid summary is compact and does not duplicate the full
  row-level prediction dataset unless explicitly justified.
- The river-mouth false-positive cluster is noted as a spatial QA concern in
  the report or manifest.
- `make check` passes after implementation.

## Known Constraints And Non-Goals

- Do not change the annual-max label target.
- Do not change the plausible-kelp domain mask thresholds.
- Do not retrain the binary classifier unless a later task explicitly requests
  model retraining.
- Do not tune calibration or threshold policies on 2022 test rows.
- Do not start a second region or West Coast scale-up.
- Do not treat Kelpwatch-derived calibrated probabilities as field-verified
  ecological presence probabilities.
