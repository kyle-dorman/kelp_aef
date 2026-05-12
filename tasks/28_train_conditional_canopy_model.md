# Task 28: Train Conditional Canopy Model

## Goal

Train the first Phase 1 conditional annual-max canopy model for cells that are
known positive or likely positive. This task should address the current ridge
failure mode where high-canopy Kelpwatch-style rows are shrunk toward the
background-heavy mean, while keeping presence detection and full-grid hurdle
composition separate.

The target remains the Monterey Phase 1 annual-max label input:

```text
kelp_fraction_y = kelp_max_y / 900 m2
kelp_max_y      = Kelpwatch-style annual max canopy area in one 30 m cell
```

This task should report whether a conditional canopy-amount model improves
positive-cell residuals against the ridge baseline. It should not claim
independent ecological biomass truth, and it should not become the final
full-grid prediction policy. P1-21 composes presence and conditional amount
into a hurdle prediction.

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
- Current ridge/reference outputs for comparison:
  - `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_observed_bin.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
- Current P1-18/P1-19 binary presence outputs:
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_calibrated_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.csv`
- Current Phase 1 model-analysis report outputs under `reports.outputs`.

Current anchors to preserve unless upstream artifacts are intentionally
refreshed:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Primary label input: Kelpwatch-style annual max, `kelp_fraction_y` /
  `kelp_max_y`.
- Primary binary target used for presence support:
  `annual_max_ge_10pct = kelp_fraction_y >= 0.10`.
- P1-19 recommended calibrated diagnostic threshold:
  `validation_max_f1_calibrated`, currently `0.40`.
- Primary full-grid reporting scope: retained plausible-kelp domain,
  `full_grid_masked`.

## Outputs

- Package-backed conditional canopy command, for example:
  `kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml`.
- New config paths under `models.conditional_canopy`.
- Serialized conditional canopy model, for example:
  `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_annual_max.joblib`.
- Row-level conditional sample predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/conditional_canopy_sample_predictions.parquet`.
- Optional compact full-grid likely-positive summary, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_full_grid_likely_positive_summary.csv`.
  Do not write a full row-level hurdle prediction dataset in this task.
- Positive or likely-positive residual table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv`.
- Conditional model metric table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv`.
- Conditional model comparison table against ridge on the same positive-cell
  evaluation rows, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_model_comparison.csv`.
- Optional residual diagnostic figure, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/conditional_canopy_positive_residuals.png`.
- Conditional model manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/conditional_canopy_manifest.json`.
- Updated Phase 1 Markdown, HTML, and PDF model-analysis report with a short
  conditional canopy section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Add a `models.conditional_canopy` block rather than overloading
  `models.baselines` or `models.binary_presence`.
- Use the masked model-input sample as the training/evaluation input:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Use calibrated binary sample predictions only to define likely-positive
  support or diagnostics, not to retrain the binary classifier.
- Set:
  - `target: kelp_fraction_y`
  - `target_area_column: kelp_max_y`
  - `positive_target_label: annual_max_ge_10pct`
  - `positive_target_threshold_fraction: 0.10`
  - `positive_target_threshold_area: 90.0`
  - `positive_support_policy: observed_positive_train_validation_test`
  - optional diagnostic policy:
    `likely_positive_policy: calibrated_probability_ge_validation_max_f1`
- Keep `features: A00-A63`.
- Start with a narrow continuous model such as ridge regression on
  positive-support rows. A second model class, such as random forest, should be
  out of scope unless the implementation plan keeps model selection clearly
  validation-only and report-visible.
- Add output paths for the model, sample predictions, metrics, positive-cell
  residuals, comparison table, manifest, and optional figure.

## Plan/Spec Requirement

This is a new model stage with new artifacts and report semantics. Before
implementation, write a brief implementation plan that confirms:

- Which rows are used to fit the conditional model.
  Default: training rows with observed `kelp_fraction_y >= 0.10`; do not train
  on assumed-background negatives.
- Whether validation/test evaluation is restricted to observed positive rows or
  also includes likely-positive rows from calibrated binary probabilities.
  Default: primary evaluation on observed positive Kelpwatch-supported rows,
  with likely-positive rows as diagnostics only.
- Whether the target is `kelp_fraction_y`, `kelp_max_y`, or a transformed
  positive-only target. Default: start with `kelp_fraction_y` and clip
  predictions to `[0, 1]`.
- Which model family is the first implementation.
  Default: ridge regression using AEF bands, with optional validation-only
  alpha selection.
- How ridge baseline rows are selected for apples-to-apples positive-cell
  comparison.
- How high-canopy residual bins are defined and reported.
- Which outputs are row-level sample predictions versus compact aggregate
  summaries.
- How the task avoids composing a full hurdle output or tuning anything on 2022
  test rows.

## Implementation Plan

- Add a package-backed CLI command, for example:
  `kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml`.
- Load the masked model-input sample and split manifest, preserving the
  existing train/validation/test split assignments.
- Derive the positive-support flag:
  `observed_annual_max_ge_10pct = kelp_fraction_y >= 0.10`.
- Build the primary training frame from train rows only where
  `observed_annual_max_ge_10pct` is true.
- Fit a positive-only continuous model using AEF bands `A00-A63`.
  Start with ridge regression and validation-only alpha selection if an alpha
  grid is configured.
- Apply the fitted conditional model to:
  - observed-positive train/validation/test sample rows for primary evaluation;
  - likely-positive rows identified by calibrated binary probabilities for
    diagnostics, if configured.
- Persist row-level conditional sample predictions with at least:
  `model_name`, `target`, `target_area_column`, `positive_support_policy`,
  `likely_positive_policy`, `split`, `year`, `label_source`, mask metadata,
  observed `kelp_fraction_y` / `kelp_max_y`, predicted conditional fraction,
  predicted conditional area, residual area, and support flags.
- Compare conditional predictions against current ridge predictions on exactly
  the same observed-positive evaluation rows.
- Write positive-cell residual rows grouped by `split`, `year`, `label_source`,
  `mask_status`, `evaluation_scope`, `observed_bin`, and support policy.
- Include high-canopy bins at minimum:
  - `annual_max_ge_10pct`
  - `annual_max_ge_50pct`
  - `annual_max_ge_90pct`
  - saturated or near-saturated rows where `kelp_max_y >= 810 m2`
- Write compact likely-positive full-grid summaries only if they help P1-21
  prepare hurdle composition. These summaries may count likely-positive rows
  and summarize predicted conditional canopy among those rows, but must not be
  presented as final full-grid canopy area.
- Update the Phase 1 model-analysis report so it answers:
  - Does the conditional model reduce high-canopy underprediction relative to
    ridge on held-out 2022 observed-positive rows?
  - Does improvement hold in validation and test, or only on training rows?
  - How do residuals differ across `annual_max_ge_10pct`,
    `annual_max_ge_50pct`, and saturated/near-saturated bins?
  - How many full-grid retained-domain rows would receive a conditional amount
    under the current calibrated presence threshold, without composing final
    hurdle predictions?

## Expected Reporting Semantics

- `model_name = ridge_positive_annual_max` or another explicit conditional
  model name.
- `model_family = conditional_canopy`.
- `target = kelp_fraction_y`.
- `target_area_column = kelp_max_y`.
- `positive_target_label = annual_max_ge_10pct`.
- `positive_target_threshold_fraction = 0.10`.
- `positive_target_threshold_area = 90.0`.
- `selection_split = validation`.
- `selection_year = 2021`.
- `test_split = test`.
- `test_year = 2022`.
- `positive_support_policy` identifies the row gate used for training or
  evaluation.
- `likely_positive_policy` identifies any calibrated-probability diagnostic
  gate.
- `mask_status = plausible_kelp_domain` for retained-domain rows.
- `evaluation_scope = conditional_observed_positive_sample` for primary
  sample metrics.
- `evaluation_scope = conditional_likely_positive_diagnostic` for optional
  likely-positive diagnostics.

Keep positive-cell canopy skill separate from binary presence skill and from
full-grid hurdle area behavior. Do not imply that a better conditional model is
a better full-grid model until P1-21 composes and evaluates the hurdle output.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_binary_presence.py tests/test_model_analysis.py tests/test_package.py
uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_model_comparison.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Training years: 2018-2020.
- Validation year: 2021.
- Held-out audit year: 2022.
- Label: Kelpwatch-style annual max canopy amount.
- Positive-support target: `annual_max_ge_10pct`.
- Features: current AEF annual 64-band embeddings.
- Domain: retained P1-12/P1-14 plausible-kelp domain.

## Acceptance Criteria

- Conditional model fitting uses only 2018-2020 training rows.
- Hyperparameter or model selection uses only 2021 validation rows.
- 2022 test rows are never used to fit the model, select hyperparameters, or
  choose likely-positive policies.
- Primary conditional training rows are observed-positive rows, not
  assumed-background negatives.
- Output tables include ridge comparison rows on the same positive-cell support
  used for conditional evaluation.
- The report separates:
  binary presence behavior, conditional positive-cell canopy amount, and future
  hurdle full-grid composition.
- Positive-cell residual tables include high-canopy and near-saturated bins.
- Any likely-positive full-grid output is compact and clearly labeled as a
  diagnostic, not a final full-grid annual-max prediction.
- `make check` passes after implementation.

## Known Constraints And Non-Goals

- Do not change the annual-max label input.
- Do not change the `annual_max_ge_10pct` binary target definition.
- Do not change the plausible-kelp domain mask thresholds.
- Do not retrain the binary presence model unless a later task explicitly asks
  for model retraining.
- Do not compose final hurdle predictions in this task.
- Do not write a full row-level full-grid conditional or hurdle prediction
  dataset unless a brief implementation plan justifies the cost and scope.
- Do not tune any model, gate, threshold, or report policy on 2022 test rows.
- Do not start a second region or West Coast scale-up.
- Do not treat Kelpwatch-derived conditional canopy predictions as
  field-verified ecological biomass.
