# Task 29: Compose First Hurdle Model

## Goal

Compose the first Phase 1 full-grid annual-max hurdle prediction by combining
the calibrated binary annual-max presence probability with the conditional
positive-cell canopy amount model.

The target remains the Monterey Phase 1 annual-max weak label:

```text
kelp_fraction_y = kelp_max_y / 900 m2
kelp_max_y      = Kelpwatch-style annual max canopy area in one 30 m cell
```

This task should answer whether the binary-plus-conditional composition reduces
retained-domain full-grid leakage and area bias relative to ridge and reference
baselines. It should not claim independent ecological biomass truth. It also
should not be expected to fix the P1-20 conditional high-canopy
underprediction: P1-20 showed that even with observed-positive support, simple
AEF ridge features still predict Kelpwatch-style canopy amount poorly.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current calibrated binary presence artifacts:
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/logistic_annual_max_ge_10pct_calibration.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.csv`
- Current conditional canopy artifacts:
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_annual_max.joblib`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_full_grid_likely_positive_summary.csv`
- Current ridge/reference outputs for comparison:
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
- Current Phase 1 report outputs under `reports.outputs`.

Current anchors to preserve unless upstream artifacts are intentionally
refreshed:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Primary label input: Kelpwatch-style annual max, `kelp_fraction_y` /
  `kelp_max_y`.
- Presence target: `annual_max_ge_10pct = kelp_fraction_y >= 0.10`.
- Calibrated probability policy: `validation_max_f1_calibrated`, currently
  threshold `0.40`.
- Primary full-grid reporting scope: retained plausible-kelp domain,
  `full_grid_masked`.
- Conditional amount model: `ridge_positive_annual_max`, selected only with
  2021 validation positives.

## Outputs

- Package-backed hurdle composition command, for example:
  `kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml`.
- New config paths under `models.hurdle`.
- Row-level retained-domain hurdle full-grid predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`.
- Hurdle prediction manifest, for example:
  `/Volumes/x10pro/kelp_aef/interim/hurdle_prediction_manifest.json`.
- Hurdle metrics table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.csv`.
- Hurdle full-grid area calibration table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`.
- Hurdle-vs-reference comparison table, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv`.
- Hurdle residual or leakage diagnostics, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_residual_by_observed_bin.csv`
  and
  `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`.
- Hurdle map figure for the 2022 test year, for example:
  `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png`.
- Updated Phase 1 Markdown, HTML, and PDF model-analysis report with a short
  first-hurdle section.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Add a `models.hurdle` block rather than overloading `models.baselines`,
  `models.binary_presence`, or `models.conditional_canopy`.
- Use the retained plausible-kelp domain as the primary composition scope.
- Point to the existing binary and conditional model payloads rather than
  retraining either model.
- Include:
  - `model_name: calibrated_probability_x_conditional_canopy`
  - `presence_probability_source: platt_calibrated`
  - `presence_threshold_policy: validation_max_f1_calibrated`
  - `presence_threshold: 0.40`
  - `conditional_model_name: ridge_positive_annual_max`
  - `target: kelp_fraction_y`
  - `target_area_column: kelp_max_y`
  - `cell_area_m2: 900.0`
  - `composition_policies: [expected_value, hard_gate]`
- The primary prediction should be the expected-value hurdle:

```text
pred_hurdle_fraction = calibrated_presence_probability
                       * pred_conditional_fraction_clipped
pred_hurdle_area_m2  = pred_hurdle_fraction * 900
```

- The hard-gated threshold output should be diagnostic:

```text
pred_hard_gate_fraction = pred_conditional_fraction_clipped
                          if calibrated_presence_probability >= 0.40
                          else 0
```

## Plan/Spec Requirement

This is a new full-grid prediction policy with new model semantics. Before
implementation, write a brief implementation plan that confirms:

- Which row-level binary probability source is used for full-grid rows.
  Default: use `binary_presence_full_grid_predictions.parquet`, apply the
  saved Platt calibration model, and select the `validation_max_f1_calibrated`
  threshold from the P1-19 table. Do not refit calibration.
- Which conditional amount model is used.
  Default: load `ridge_positive_annual_max.joblib` and predict conditional
  amount from AEF bands for retained-domain full-grid rows.
- Which composition policy is primary.
  Default: primary `expected_value = calibrated_probability * conditional_amount`;
  retain hard-gated rows as diagnostics only.
- Whether predictions are written for all configured years or only the 2022
  report year.
  Default: write all configured years inside the retained plausible-kelp domain
  if runtime is reasonable; at minimum support a fast/report-year mode.
- How ridge/reference baselines are selected for apples-to-apples comparison.
  Default: compare against existing retained-domain reference area calibration
  rows and ridge full-grid metrics.
- How maps and area-bias rows are labeled so expected-value and hard-gated
  hurdle outputs are not confused.
- How the report carries forward the P1-20 outcome that conditional amount
  remains weak on high-canopy rows.
- How the task avoids retraining, recalibrating, or selecting thresholds on the
  2022 test split.

## Implementation Plan

- Add a package-backed CLI command, for example:
  `kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml`.
- Load the configured binary full-grid predictions for retained-domain rows.
  If raw probabilities are stored without calibrated probabilities, load the
  existing Platt calibration payload and apply it row-wise.
- Load the configured conditional canopy model payload and stream AEF features
  from the full-grid inference table inside the retained domain.
- Join or align binary probabilities and conditional amount predictions by
  stable grid-year keys, preferably `year, aef_grid_cell_id`.
- For each retained-domain row, write at least:
  `model_name`, `model_family`, `composition_policy`, `split`, `year`,
  `label_source`, mask metadata, observed `kelp_fraction_y` / `kelp_max_y`,
  calibrated presence probability, probability threshold, predicted presence
  class, conditional predicted fraction/area, composed hurdle fraction/area,
  and residual fraction/area.
- Write both:
  - primary expected-value predictions; and
  - diagnostic hard-gated predictions using the validation-selected calibrated
    threshold.
- Build full-grid area calibration rows by split, year, label source,
  `mask_status`, `evaluation_scope`, and `composition_policy`.
- Compare the hurdle outputs against:
  - ridge regression;
  - previous-year annual max;
  - grid-cell climatology;
  - geographic ridge;
  - no-skill train mean, where present in the existing comparison table.
- Write a 2022 map figure for observed, predicted, and residual area. If both
  expected-value and hard-gated maps are too much for one task, make
  expected-value the map and keep hard-gated behavior tabular.
- Update the Phase 1 model-analysis report so it answers:
  - Does the expected-value hurdle reduce full-grid area overprediction or
    assumed-background leakage relative to ridge?
  - How much positive-cell canopy underprediction remains after composition?
  - Does the hard-gated diagnostic reduce leakage at the cost of false
    negatives or area underprediction?
  - Does the hurdle beat reference baselines on the 2022 retained-domain
    full-grid scope?

## Expected Reporting Semantics

- `model_family = hurdle`
- Primary `model_name = calibrated_probability_x_conditional_canopy`
- Diagnostic hard-gate `model_name = calibrated_hard_gate_conditional_canopy`
- `presence_model_name = logistic_annual_max_ge_10pct`
- `presence_probability_source = platt_calibrated`
- `presence_threshold_policy = validation_max_f1_calibrated`
- `presence_target_label = annual_max_ge_10pct`
- `presence_target_threshold_fraction = 0.10`
- `conditional_model_name = ridge_positive_annual_max`
- `conditional_target = kelp_fraction_y`
- `composition_policy = expected_value` for the primary hurdle rows.
- `composition_policy = hard_gate` for thresholded diagnostic rows.
- `selection_split = validation`
- `selection_year = 2021`
- `test_split = test`
- `test_year = 2022`
- `mask_status = plausible_kelp_domain`
- `evaluation_scope = full_grid_masked`

Keep presence skill, conditional amount skill, and final full-grid hurdle area
behavior separate in the report. If the hurdle improves full-grid leakage but
still misses dense canopy magnitude, say that directly.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_binary_presence.py tests/test_conditional_canopy.py tests/test_model_analysis.py tests/test_package.py
uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual output inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Training years for upstream fitted models: 2018-2020.
- Validation year for upstream calibration and thresholding: 2021.
- Held-out audit year: 2022.
- Label: Kelpwatch-style annual max canopy amount.
- Presence target: `annual_max_ge_10pct`.
- Features: current AEF annual 64-band embeddings.
- Domain: retained P1-12/P1-14 plausible-kelp domain.

## Acceptance Criteria

- The task does not refit the binary presence model, probability calibrator, or
  conditional canopy model.
- The task does not tune probability thresholds, composition policies, or
  report choices on 2022 test rows.
- Row-level hurdle predictions are restricted to the retained plausible-kelp
  domain unless an explicit off-domain audit table is added separately.
- The primary expected-value hurdle output and hard-gated diagnostic output are
  clearly labeled and not mixed in the same metric row.
- Area calibration compares the hurdle output against ridge and current
  reference baselines in the same retained-domain full-grid scope.
- The report explicitly separates:
  binary presence behavior, conditional positive-cell amount behavior, and
  final full-grid hurdle area behavior.
- The report carries forward the P1-20 outcome that conditional high-canopy
  underprediction remains a limitation even if the hurdle reduces leakage.
- `make check` passes after implementation.

## Known Constraints And Non-Goals

- Do not change the annual-max label input.
- Do not change the `annual_max_ge_10pct` presence target definition.
- Do not change the plausible-kelp domain mask thresholds.
- Do not retrain the binary presence model.
- Do not refit Platt calibration.
- Do not retrain the conditional canopy model.
- Do not tune anything on the 2022 test split.
- Do not start a second region or West Coast scale-up.
- Do not treat Kelpwatch-derived hurdle predictions as field-verified
  ecological biomass.
- Do not introduce new model families here; if the first hurdle fails, use the
  report to decide whether P1-22 capped-weight or stratified-background
  continuous modeling is still worth testing.
