# Monterey Phase 1 Model Analysis

## Executive Summary

This is the final closeout decision report for the Monterey annual-max Phase 1 workflow. It evaluates Kelpwatch-style weak labels, not independent field truth.

Current default policy: CRM-stratified, mask-first retained-domain sampling with `mask_status = plausible_kelp_domain` and `evaluation_scope = full_grid_masked`.

Ridge is no longer the best candidate. It still leaks too much retained full-grid background area: the primary full-grid area bias is `102.1%` and predicted area is `8.42 M m2`.

The expected-value hurdle is the selected AEF full-grid policy: primary RMSE is `0.0322`, R2 is `0.790`, and area bias is `-16.0%`.

Previous-year persistence remains a strong benchmark, with primary RMSE `0.0341` and area bias `-15.9%`.

Calibrated binary presence is strong for the Kelpwatch-style `annual_max_ge_10pct` target: primary test AUPRC is `0.943` and F1 is `0.853`.

The main weakness is still high-canopy amount prediction. In positive `annual_max_ge_50pct` rows, the conditional amount model mean residual is `185.4 m2`; positive residual means observed canopy exceeds predicted canopy.

Station-row ridge context remains useful but secondary: Kelpwatch-station test R2 is `0.664` and station-area bias is `-18.9%`.

## Current Default Policy And Data Scope

- Config: `configs/monterey_smoke.yaml`
- Label input: Kelpwatch annual max from `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Model input sample: `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
- Default sampling policy: `crm_stratified_mask_first_sample` (CRM-stratified mask-first default).
- Sampling-policy decision note: `docs/phase1_crm_stratified_sampling_policy_decision.md`
- Primary full-grid reporting scope: `full_grid_masked` inside `plausible_kelp_domain`.
- Primary split/year: `test` `2022`.
- Continuous baseline predictions: `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- Model comparison table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`

## 2022 Retained-Domain Model Scoreboard

Primary rows use `split = test`, `year = 2022`, `evaluation_scope = full_grid_masked`, and `label_source = all`.

| Model | Role | Rows | RMSE | R2 | F1 >=10% | Predicted area (M m2) | Area bias |
|---|---|---:|---:|---:|---:|---:|---:|
| Previous-year annual max | Reference persistence | 630151 | 0.0341 | 0.765 | 0.797 | 3.50 | -15.9% |
| Grid-cell climatology | Reference site memory | 630151 | 0.0429 | 0.627 | 0.708 | 3.69 | -11.5% |
| AEF ridge regression | AEF continuous baseline | 630151 | 0.0452 | 0.587 | 0.476 | 8.42 | 102.1% |
| Expected-value hurdle | AEF hurdle candidate | 630151 | 0.0322 | 0.790 | 0.812 | 3.50 | -16.0% |
| Hard-gated hurdle | AEF hurdle diagnostic | 630151 | 0.0336 | 0.771 | 0.825 | 3.49 | -16.1% |

![Pixel skill and area calibration](../figures/model_analysis_pixel_skill_area_calibration.png)

## What Improved Since Ridge

The clearest improvement is full-grid area behavior. AEF ridge predicts `8.42 M m2` inside the retained domain, while the expected-value hurdle predicts `3.50 M m2` by multiplying calibrated presence probability by conditional canopy amount.

Relative to ridge area bias `102.1%`, the expected-value hurdle area bias is `-16.0%`. Its RMSE `0.0322` is also lower than ridge `0.0452` in the same retained-domain scope.

The hard-gated hurdle is kept as a diagnostic because it has strong thresholded support (F1 `0.825`), but the expected-value row is the main full-grid canopy candidate because it preserves a continuous expected area estimate.

Persistence remains hard to beat: previous-year annual max has RMSE `0.0341` and area bias `-15.9%`. That benchmark sets the Phase 1 benchmark for AEF policy interpretation.

### Hurdle Observed, Predicted, And Error Map

![First hurdle 2022 map](../figures/hurdle_2022_observed_predicted_residual.png)

### Binary-Presence Diagnostic Map

Calibrated binary presence is the strongest classification component for the Kelpwatch-style `annual_max_ge_10pct` target. The primary test row has AUPRC `0.943`, precision `0.908`, recall `0.804`, and F1 `0.853` at the validation-selected threshold `0.36`. The map below is diagnostic support for where the binary gate still admits false positives or misses observed canopy.

![Binary presence 2022 map](../figures/binary_presence_2022_map.png)

## Remaining Failure Modes

Observed saturated count: `551`. Predicted p99: `204.4`. Predicted max: `900.0`. Mean prediction on observed `900 m2` rows: `668.7`.

Residual bins are sorted by numeric observed canopy area. Positive residuals mean `observed - predicted`, so positive values are underprediction. The pattern is shrinkage: zero rows have mean predicted area `8.8 m2`, while the highest observed bin `(810, 900]` has mean observed area `869.2 m2`, mean predicted area `566.4 m2`, and mean residual `302.8 m2`.

Primary mask-aware diagnostics use `mask_status = plausible_kelp_domain` and `evaluation_scope = full_grid_masked`. Rows are restricted to retained plausible-kelp cells; off-domain leakage stays in the separate audit table.
The largest retained group by row count is `retained_depth_0_60m` (`469521` rows). The retained mask reason with the largest absolute mean residual is `retained_ambiguous_coast` (`-16.2 m2`).
The largest underprediction concentration is `0_40m` / `subtidal_ocean` in `high_canopy_underprediction` rows, with `2270` rows and `2164` high-error rows.
Observed-zero false positives inside retained habitat account for `141242` rows and `5454792.5 m2` of predicted canopy area; the largest label-source group is `assumed_background`.
The retained-domain residuals point first to `false positives` inside the plausible habitat rather than off-domain leakage. Concentration in `retained_ambiguous_coast` and `0_40m` is a Phase 1 failure-mode finding, not a reason to tune mask thresholds in the closeout.

| Mask reason | Rows | Mean residual | Area bias | High-error rows |
|---|---:|---:|---:|---:|
| retained_depth_0_60m | 469521 | -3.5 | 1644382.6 | 15815 |
| retained_ambiguous_coast | 160630 | -16.2 | 2607924.0 | 15693 |

High-canopy amount remains the model bottleneck. On held-out observed-positive `annual_max_ge_50pct` rows, the conditional amount model mean residual is `185.4 m2`; on `near_saturated_ge_810m2` rows it is `208.8 m2`.

These failures define the Phase 1 closeout boundary: keep the annual-max label input, retained-domain mask, and validation-selected binary threshold fixed, and treat high-canopy amount underprediction as unresolved.

### Ridge Observed, Predicted, And Error Map

The three-panel model review map uses the latest static map for the primary `test` `2022` split. The rows are filtered to `plausible_kelp_domain` before plotting. The observed and predicted panels use the same canopy-area scale, and the error panel uses `observed - predicted`, so positive residuals are underprediction. The linked interactive map is `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.masked.html`.

![Observed, predicted, and residual map](../figures/ridge_2022_observed_predicted_residual.masked.png)

![Residual by observed bin](../figures/model_analysis_residual_by_observed_bin.png)

![Residual by persistence](../figures/model_analysis_residual_by_persistence.png)

![Residual by retained domain context](../figures/model_analysis_residual_by_domain_context.png)

## Phase 1 Closeout Decision

Selected Phase 1 AEF policy: keep `crm_stratified_mask_first_sample` as the default retained-domain sample policy and treat `calibrated_probability_x_conditional_canopy` as the best current AEF annual-max full-grid policy. This is a Monterey Phase 1 Kelpwatch-style weak-label policy, not independent biomass validation.

Decision evidence uses the primary retained-domain row (`split = test`, `year = 2022`, `evaluation_scope = full_grid_masked`, `label_source = all`). The selected expected-value hurdle has RMSE `0.0322`, R2 `0.790`, F1 at 10% annual max `0.812`, predicted canopy area `3.50 M m2`, and area bias `-16.0%`. It is stronger than AEF ridge on RMSE `0.0452`, F1 `0.476`, and area bias `102.1%`.

The previous-year persistence reference remains the meaningful non-AEF benchmark: its primary RMSE is `0.0341`, F1 is `0.797`, and area bias is `-15.9%`. The expected-value hurdle edges that benchmark on retained-domain RMSE and threshold skill, while landing at a similar total-area underprediction.

The hard-gated hurdle remains diagnostic, not the selected canopy-area policy. It has F1 `0.825` and area bias `-16.1%`, but the hard gate is a thresholded support decision rather than a continuous expected-area estimate. The P1-22 direct-continuous capped-weight and stratified-background experiments remain negative historical records and are not active maintained model policies.

Phase 1 is closed with unresolved limitations: the selected policy still underpredicts high-canopy Kelpwatch-style annual-max rows, and all results are limited to the Monterey 2018-2022 annual-max weak-label setting.

## Appendix

### Data Health And Scope Checks

| Check | Split | Year | Label source | Rows | Rate |
|---|---|---:|---|---:|---:|
| annual_label_rows | all | 2022 | all | 30232 | 1.000 |
| model_input_rows | all | 2022 | assumed_background | 32141 | 0.516 |
| model_input_rows | all | 2022 | kelpwatch_station | 30154 | 0.484 |
| split_manifest_rows | test | 2022 | all | 62295 | 1.000 |
| retained_model_rows | test | 2022 | all | 62295 | 1.000 |
| dropped_model_rows | test | 2022 | all | 0 | 0.000 |
| missing_feature_drop_rate | test | 2022 | all | 0 | 0.000 |
| prediction_rows | test | 2022 | assumed_background | 599997 | 0.952 |
| prediction_rows | test | 2022 | kelpwatch_station | 30154 | 0.048 |
| primary_report_prediction_rows | test | 2022 | all | 630151 | 1.000 |

| Stage | Split | Rows | Zero | Positive | 900 m2 |
|---|---:|---:|---:|---:|---:|
| annual_labels | all | 30232 | 17273 | 12959 | 552 |
| model_input_sample | all | 62295 | 49398 | 12897 | 551 |
| retained_model_rows | test | 62295 | 49398 | 12897 | 551 |

These diagnostics quantify Kelpwatch-style annual-max imbalance before any binary, balanced, hurdle, or conditional model is introduced. They do not choose a production threshold and do not tune on the primary test split.
In the current model-input sample, zero rows are `81.2%` and assumed-background rows are `51.6%` of rows. In the primary `full_grid_masked` report scope, zero rows are `98.0%` and assumed-background rows are `95.2%`.
Primary full-grid positives are `2.05%` of rows; high canopy rows at `>=450 m2` are `0.65%`, and saturated or near-saturated rows are `0.09%`. The mask removes off-domain background from the report scope, but the retained plausible domain still has strong within-domain target imbalance.
Later threshold, binary, balanced, hurdle, and conditional models should be evaluated against these rates so improvements are measured against the actual annual-max class mix, not introduced blindly.

| Scope | Split | Year | Rows | Positive | High canopy | Saturated | Assumed background |
|---|---|---:|---:|---:|---:|---:|---:|
| model_input_sample | test | 2022 | 62295 | 20.70% | 6.59% | 0.88% | 51.6% |
| split_manifest_retained | test | 2022 | 62295 | 20.70% | 6.59% | 0.88% | 51.6% |
| sample_predictions | test | 2022 | 62295 | 20.70% | 6.59% | 0.88% | 51.6% |
| full_grid_masked | test | 2022 | 630151 | 2.05% | 0.65% | 0.09% | 95.2% |

![Label distribution](../figures/model_analysis_label_distribution_by_stage.png)

![Annual-max class balance](../figures/model_analysis_class_balance.png)

### Threshold, Calibration, And Amount Diagnostics

Threshold selection uses only `validation` rows from `2021` in the sample-prediction table; `test` rows remain locked audit/context rows and are not used to choose the candidate. All thresholds are derived from the existing Kelpwatch annual-max fraction, so this does not change the label input or claim ecological truth.
The recommended next P1-18 candidate is `annual_max_ge_10pct` (10% annual max). Selection status is `selected_from_validation_support_floor` under policy `validation_only_prefer_highest_threshold_with_positive_support_then_f1`.

| Threshold | Positive | Predicted positive | Precision | Recall | F1 | Assumed-background FP area |
|---|---:|---:|---:|---:|---:|---:|
| annual_max_gt0 | 16.37% | 77.97% | 0.207 | 0.984 | 0.342 | 1037868.5 |
| annual_max_ge_1pct | 16.31% | 70.74% | 0.226 | 0.981 | 0.368 | 1023416.3 |
| annual_max_ge_5pct | 14.36% | 41.91% | 0.329 | 0.961 | 0.490 | 784149.2 |
| annual_max_ge_10pct | 12.26% | 23.94% | 0.474 | 0.925 | 0.626 | 502284.0 |

Detailed prevalence, comparison, and recommendation rows are written to `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_prevalence.csv`, `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_comparison.csv`, and `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_recommendation.csv`.

The balanced binary model predicts the Kelpwatch-style annual-max target `annual_max_ge_10pct`, defined as `kelp_fraction_y >= 0.10` or `kelp_max_y >= 90 m2`. The current implementation is a class-weighted logistic regression with `class_weight = balanced`; it is not calibrated and does not claim independent ecological presence truth.
The diagnostic operating threshold is `0.920` selected on validation rows only with status `selected_from_validation_max_f1`. The held-out 2022 test rows are used only for audit metrics.

| Split | Year | Rows | AUPRC | AUROC | Precision | Recall | F1 | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 2021 | 62295 | 0.939 | 0.988 | 0.839 | 0.872 | 0.855 | 12.75% |
| test | 2022 | 62295 | 0.943 | 0.988 | 0.902 | 0.814 | 0.856 | 12.28% |

Thresholded baseline comparison at `kelp_fraction_y >= 0.10` for the primary `test` `2022` rows:

| Model | Family | AUPRC | AUROC | Precision | Recall | F1 | Predicted positive | Assumed-bg FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_annual_max_ge_10pct | balanced_binary | 0.943 | 0.988 | 0.902 | 0.814 | 0.856 | 12.28% | 0.24% |
| geographic_ridge_lon_lat_year | thresholded_continuous_baseline | 0.320 | 0.747 | 0.000 | 0.000 | nan | 0.15% | 0.29% |
| grid_cell_climatology | thresholded_continuous_baseline | 0.731 | 0.918 | 0.649 | 0.778 | 0.708 | 16.30% | 0.00% |
| no_skill_train_mean | thresholded_continuous_baseline | 0.136 | 0.500 | nan | 0.000 | nan | 0.00% | 0.00% |
| previous_year_annual_max | thresholded_continuous_baseline | 0.850 | 0.913 | 0.841 | 0.758 | 0.797 | 25.33% | nan |
| ridge_regression | thresholded_continuous_baseline | 0.847 | 0.953 | 0.634 | 0.849 | 0.726 | 18.24% | 3.97% |

In the primary `full_grid_masked` 2022 report scope, the binary model predicts `1.31%` of retained rows as annual-max `>=10%`, or `7415100.0 m2` of 30 m cell area. Assumed-background predicted positives are `0.11%` of assumed-background rows.
The current ridge/reference comparison row for the same scope has `f1_ge_10pct = 0.476` and `predicted_canopy_area = 8415320.6 m2`. This compares binary ranking and operating-threshold behavior against the ridge leakage diagnostics without treating either output as final production calibration.

The calibrated binary stage fits Platt scaling on all `validation` rows from `2021`, including assumed-background rows as negatives. It calibrates the existing class-weighted logistic probabilities for the Kelpwatch-style `annual_max_ge_10pct` weak label; it does not create independent ecological presence truth.
The recommended diagnostic calibrated threshold is `0.360` from `validation_max_f1_calibrated`. Test-year rows are used only for evaluation, not for calibration fitting or threshold selection.

| Split | Source | Threshold policy | AUPRC | AUROC | Brier | ECE | Precision | Recall | F1 | Predicted positive |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | raw_logistic | p1_18_validation_raw_threshold | 0.939 | 0.988 | 0.0642 | 0.1130 | 0.839 | 0.872 | 0.855 | 12.75% |
| validation | platt_calibrated | validation_max_f1_calibrated | 0.939 | 0.988 | 0.0254 | 0.0069 | 0.848 | 0.863 | 0.856 | 12.48% |
| test | raw_logistic | p1_18_validation_raw_threshold | 0.943 | 0.988 | 0.0458 | 0.0719 | 0.902 | 0.814 | 0.856 | 12.28% |
| test | platt_calibrated | validation_max_f1_calibrated | 0.943 | 0.988 | 0.0299 | 0.0208 | 0.908 | 0.804 | 0.853 | 12.05% |

In the primary `full_grid_masked` 2022 scope, raw P1-18 thresholding predicts `1.31%` of retained rows positive. The recommended calibrated threshold predicts `1.28%` positive, or `7248600.0 m2` of 30 m cell area.
The known river-mouth false-positive cluster remains a spatial QA concern. This calibration reports threshold effects on leakage but does not change the P1-12/P1-14 plausible-kelp domain mask.

The conditional canopy model fits a positive-only ridge regression on training rows where the Kelpwatch-style annual max is `>=10%` of a 30 m cell. It predicts `kelp_fraction_y`, clips conditional amounts to `[0, 1]`, and leaves binary presence and full-grid hurdle composition to later tasks.
Model selection uses validation rows only. The 2022 test split is included here only as a held-out audit of positive-cell canopy amount, not for threshold or hyperparameter selection.

| Split | Model | Rows | MAE area | RMSE area | Mean residual area | Predicted area | Area bias |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | ridge_positive_annual_max | 7637 | 103.6 | 130.8 | 8.3 | 3319408.5 | -63220.5 |
| validation | ridge_regression | 7637 | 159.7 | 204.6 | 135.6 | 2347166.1 | -1035462.9 |
| test | ridge_positive_annual_max | 8478 | 149.1 | 191.8 | 79.1 | 3341480.0 | -670693.0 |
| test | ridge_regression | 8478 | 207.1 | 260.0 | 172.5 | 2549495.8 | -1462677.2 |

On held-out `test` observed-positive rows, the conditional model RMSE area is `191.8 m2` versus `260.0 m2` for ridge on the same rows. Positive residual means observed canopy exceeds predicted canopy, so large positive values still indicate high-canopy underprediction.
For held-out rows with annual max `>=50%`, mean residual area is `185.4 m2`; for near-saturated rows (`>=810 m2`), it is `208.8 m2`.
Under the current calibrated binary gate, `8054` retained-domain rows would receive a conditional amount in a future hurdle composition (`1.28%` of the scope). This is a count-only diagnostic, not a final full-grid canopy estimate.
These conditional amount diagnostics remain Kelpwatch-style weak-label results. They do not prove field biomass and should not be interpreted as final full-grid canopy area until the hurdle composition task combines them with calibrated presence probabilities.

The first hurdle model composes the validation-selected calibrated binary presence probability with the saved positive-only conditional canopy model. The primary prediction is the expected value `calibrated_probability * clipped_conditional_canopy`; the hard-gated threshold output is retained as a diagnostic.
This stage loads existing model and calibration payloads only. It does not retrain the binary model, refit Platt scaling, retrain the conditional model, or tune thresholds on 2022 test rows.

| Model | Rows | Observed area | Predicted area | Area bias | F1 >=10pct | Background predicted area |
|---|---:|---:|---:|---:|---:|---:|
| calibrated_probability_x_conditional_canopy | 630151 | 4163014.0 | 3495891.1 | -16.0% | 0.812 | 460496.5 |
| calibrated_hard_gate_conditional_canopy | 630151 | 4163014.0 | 3494314.0 | -16.1% | 0.825 | 285144.5 |

For the same `full_grid_masked` 2022 scope, ridge predicted `8415320.6 m2` against `4163014.0 m2` observed canopy area. The expected-value hurdle row above is the primary apples-to-apples full-grid comparison.
Positive-cell amount remains the limiting piece: the hurdle can reduce full-grid leakage by multiplying conditional canopy by calibrated presence probability, but it still inherits the P1-20 high-canopy underprediction. Mean residual area is `200.4 m2` for the `(450, 810]` observed-area bin and `218.4 m2` for the `(810, 900]` bin.

### Artifact Index

- Stage distribution table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_label_distribution_by_stage.csv`
- Prediction distribution table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_prediction_distribution.csv`
- Residual by observed bin table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_observed_bin.csv`
- Residual by persistence table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_persistence.csv`
- Residual by domain context table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv`
- Residual by mask reason table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_mask_reason.csv`
- Residual by depth/elevation bin table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_depth_bin.csv`
- Top residual domain-context table: `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.domain_context.csv`
- Class balance by split table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv`
- Target balance by label source table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_balance_by_label_source.csv`
- Background rate summary table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_background_rate_summary.csv`
- Binary threshold prevalence table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_prevalence.csv`
- Binary threshold comparison table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_comparison.csv`
- Binary threshold recommendation table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_binary_threshold_recommendation.csv`
- Binary presence metrics table: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv`
- Binary presence threshold-selection table: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv`
- Binary presence full-grid area summary: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv`
- Binary presence thresholded model comparison: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_thresholded_model_comparison.csv`
- Binary presence map figure: `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_2022_map.png`
- Binary presence calibration metrics table: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv`
- Binary presence calibrated threshold-selection table: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_threshold_selection.csv`
- Binary presence calibrated full-grid area summary: `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibrated_full_grid_area_summary.csv`
- Binary presence calibration manifest: `/Volumes/x10pro/kelp_aef/interim/binary_presence_calibration_manifest.json`
- Binary presence calibration curve figure: `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_calibration_curve.png`
- Binary presence calibrated threshold figure: `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_calibrated_thresholds.png`
- Conditional canopy metrics table: `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv`
- Conditional canopy positive residuals table: `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv`
- Conditional canopy model comparison table: `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_model_comparison.csv`
- Conditional canopy full-grid likely-positive summary: `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_full_grid_likely_positive_summary.csv`
- Conditional canopy residual figure: `/Volumes/x10pro/kelp_aef/reports/figures/conditional_canopy_positive_residuals.png`
- Conditional canopy manifest: `/Volumes/x10pro/kelp_aef/interim/conditional_canopy_manifest.json`
- Hurdle metrics table: `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.csv`
- Hurdle area-calibration table: `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`
- Hurdle-vs-reference comparison table: `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv`
- Hurdle residual by observed bin table: `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_residual_by_observed_bin.csv`
- Hurdle assumed-background leakage table: `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`
- Hurdle map figure: `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png`
- Hurdle manifest: `/Volumes/x10pro/kelp_aef/interim/hurdle_prediction_manifest.json`
- Phase 1 model comparison table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
- Sampling-policy audit table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`
- Phase 1 data-health table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_data_health.csv`
- Quarter mapping table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_quarter_mapping.csv`
- Reference fallback summary table: `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_fallback_summary.csv`
- Reference area calibration table: `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv`
- Phase 1 PDF report: `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`

Validation command:

```bash
make check
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```
