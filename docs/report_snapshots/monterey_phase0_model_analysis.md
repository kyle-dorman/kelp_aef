# Monterey Phase 0 Model Analysis

Snapshot copied into the repo after Phase 0 closeout. The active generated
report is now the Phase 1 model-analysis report under
`/Volumes/x10pro/kelp_aef/reports/model_analysis/`.

## Executive Summary

The Monterey smoke-test pipeline now runs from Kelpwatch labels and AEF features through alignment, ridge prediction, residual maps, and this report. Results should be interpreted as learning Kelpwatch-style labels, not field-verified kelp truth.

The strongest current finding is that the corrected background-inclusive setup exposes a failed ridge objective, not a successful baseline. On Kelpwatch-station rows in the configured test 2022 split, ridge RMSE is `0.2279` and station-area bias is `-36.7%`. On the background-inclusive sample, area bias is `16.6%`, which is tracked separately from full-grid map calibration. Zero-canopy rows receive mean predicted area `9.4 m2`, while the highest observed canopy bin has mean residual `601.4 m2`.

Recommended Phase 1 branch: **Sampling/objective calibration Phase 1**.

## Smoke-Test Scope And Artifacts

- Config: `configs/monterey_smoke.yaml`
- Annual labels: `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`
- Model input sample: `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`
- Predictions: `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- Metrics: `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`

## Pipeline Accomplishments

- Downloaded and inspected Kelpwatch source data.
- Queried and downloaded the Monterey AEF annual tiles.
- Built annual Kelpwatch max labels for 2018-2022.
- Corrected the initial station-center-only alignment by writing a full AEF-aligned 30 m grid artifact and a documented background-inclusive training sample.
- Trained no-skill and ridge baselines on the background-inclusive sample with a year holdout.
- Applied the trained ridge model back to the full-grid table with streamed inference.
- Wrote residual maps, area-bias tables, and this model-analysis report from the corrected artifacts.

## Full-Grid Background Correction

The first Phase 0 alignment only sampled AEF values at Kelpwatch station centers. Task 11 corrects that data contract: non-station cells are retained as weak-label `assumed_background` rows, and station-supported rows are flagged as `kelpwatch_station`.

| Artifact | Loaded rows | Kelpwatch station rows | Assumed background rows |
|---|---:|---:|---:|
| Model input sample | 1400809 | 150770 | 1250039 |
| Prediction rows in this report | 7458361 | 30154 | 7428207 |

Overall full-grid metrics are useful for calibration and map QA, but quarterly target-framing and persistence diagnostics are computed only on Kelpwatch-station rows because background cells do not have quarterly Kelpwatch support.

## Data And Label Distribution Findings

| Stage | Split | Rows | Zero | Positive | 900 m2 |
|---|---:|---:|---:|---:|---:|
| annual_labels | all | 30232 | 17273 | 12959 | 552 |
| model_input_sample | all | 280178 | 267281 | 12897 | 551 |
| retained_model_rows | test | 268541 | 255644 | 12897 | 551 |

![Label distribution](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_label_distribution_by_stage.png)

## Quarter And Season Grounding

Kelpwatch source quarter metadata is written to the quarter mapping table. In the current source, quarter 1 is Jan-Mar, quarter 2 is Apr-Jun, quarter 3 is Jul-Sep, and quarter 4 is Oct-Dec. This report treats configured winter as quarter 1 and configured fall as quarter 4.

- Quarter mapping table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_quarter_mapping.csv`

## Ridge Baseline Performance Recap

For the primary `test` split, the Kelpwatch-station ridge row has RMSE `0.2279`, R2 `0.3558`, and area percent bias `-36.70%`. The background-inclusive sample area percent bias is `16.58%`.

| Model | MAE | RMSE | R2 | Spearman | Area pct bias | F1 at 10% |
|---|---:|---:|---:|---:|---:|---:|
| no_skill_train_mean | 0.1561 | 0.3158 | -0.237 | nan | -90.06% | nan |
| ridge_regression | 0.1342 | 0.2279 | 0.356 | 0.770 | -36.70% | 0.770 |

On Kelpwatch-station rows, ridge only modestly improves over the train-mean baseline (RMSE reduction `27.8%`) and still misses most canopy area: station area bias is `-36.70%`. The background-inclusive sample shows the calibration tradeoff: no-skill area bias is `-11.46%` and ridge area bias is `16.58%`. This is still a sampling/objective calibration problem to revisit, but it is a more useful smoke baseline than the population-expanded weighted ridge.

![Observed vs predicted distribution](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_observed_vs_predicted_distribution.png)

## Observed, Predicted, And Error Map

The three-panel smoke-test map reuses the Task 09 static map for the primary `test` `2022` split. The observed and predicted panels use the same canopy-area scale, and the error panel uses `observed - predicted`, so positive residuals are underprediction. The linked interactive map is `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.html`.

![Observed, predicted, and residual map](/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png)

## Residual And Saturation Findings

Observed saturated count: `551`. Predicted p99: `86.1`. Predicted max: `406.0`. Mean prediction on observed `900 m2` rows: `300.5`.

Residual bins are sorted by numeric observed canopy area. Positive residuals mean `observed - predicted`, so positive values are underprediction. The pattern is shrinkage: zero rows have mean predicted area `9.4 m2`, while the highest observed bin `(810, 900]` has mean observed area `869.2 m2`, mean predicted area `267.8 m2`, and mean residual `601.4 m2`.

![Observed 900 predictions](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_observed_900_predictions.png)

![Residual by observed bin](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_observed_bin.png)

![Residual by persistence](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_persistence.png)

## Binary Threshold Sensitivity

Among the configured diagnostic thresholds, `10%` has the highest F1 (`0.211`) on the primary split. The highest threshold, `90%`, has recall `0.000`, which confirms poor high-canopy recall.

| Threshold | Observed positive | Predicted positive | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.2% | 31.7% | 0.005 | 0.996 | 0.011 |
| 1% | 0.2% | 24.6% | 0.007 | 0.995 | 0.013 |
| 5% | 0.1% | 6.6% | 0.020 | 0.979 | 0.038 |
| 10% | 0.1% | 0.8% | 0.120 | 0.877 | 0.211 |
| 50% | 0.1% | 0.0% | nan | 0.000 | nan |
| 90% | 0.0% | 0.0% | nan | 0.000 | nan |

Detailed threshold sensitivity is written to `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_threshold_sensitivity.csv`. These rows are diagnostics only; this task does not choose a production binary threshold.

## Alternative Target-Framing Findings

| Target | Kind | Spearman | Mean | Positive fraction |
|---|---|---:|---:|---:|
| annual_mean_area | continuous_area | 0.773 | 54.854 | 0.428 |
| annual_max_area | continuous_area | 0.770 | 138.058 | 0.428 |
| mean_presence_fraction | binary_or_fraction | 0.766 | 0.258 | 0.428 |
| any_presence | binary_or_fraction | 0.715 | 0.428 | 0.428 |
| presence_ge_gt0 | binary_or_fraction | 0.715 | 0.428 | 0.428 |
| presence_ge_01pct | binary_or_fraction | 0.712 | 0.406 | 0.406 |

The Spearman target-framing plot is a rank-agreement diagnostic for the existing annual-max ridge predictions. The model was not retrained for each target, so this does not prove that any alternative target is better. Current predictions rank `annual_mean_area` (`0.773`), `annual_max_area` (`0.770`), and `mean_presence_fraction` (`0.766`) similarly, suggesting the ridge output behaves like a general annual kelp-intensity or seasonal-regularity score rather than a clean saturated-peak predictor.

![Alternative target framings](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_alternative_target_framings.png)

## Feature Separability Findings

The largest PCA group-center distance from the zero group is `6.96`. This is a diagnostic for separability only; it is not model skill.

![Feature projection](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_feature_projection.png)

## Spatial Split And Scale-Up Readiness

`5` latitude bands meet the current minimum counts for a crude within-Monterey spatial holdout. This is useful for smoke-test diagnostics, but it is not enough to claim robust spatial generalization. Phase 1 should either add a second smoke region or broaden the California slice before making larger spatial claims.

![Spatial holdout readiness](/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_spatial_holdout_readiness.png)

## Baseline Completeness

Implemented baselines: train-mean no-skill and ridge regression. Missing reference baselines from the research plan: previous-year kelp, per-station climatology, lat/lon/year-only geographic baseline, and any non-AEF spectral-product baseline.

## Interpretation

Phase 0 now demonstrates that the end-to-end artifact path works, but the corrected background-inclusive ridge result should be treated as a failed first baseline. The Kelpwatch-station test R2 is `0.356` and the model still severely underpredicts canopy-support rows while leaking small positive predictions over a large assumed-background population. The next work should therefore focus on the learning objective, sampling/weighting policy, calibration, and reference baselines before interpreting stronger nonlinear models or 10 m spatial models as the main answer.

## Phase 1 Decision Matrix

| Branch | Evidence | Proposed next tasks |
|---|---|---|
| Sampling/objective calibration Phase 1 | strong: Kelpwatch-station ridge area bias is -36.7%, while background-inclusive sample area bias is 16.6%; the same model can underpredict observed canopy support while leaking positives across background. | Separate training objective weights from population-calibration metrics; test capped background weights, stratified losses, and post-fit calibration. |
| Derived-label Phase 1 | strong: Observed-900 rows have mean ridge prediction 300.5; annual max, annual mean, persistence, and threshold diagnostics need retrained target variants before changing models. | Implement evaluated label variants: mean canopy, persistent presence, fall/winter labels, and threshold diagnostics. |
| Baseline-hardening Phase 1 | strong: Current implemented references are train-mean no-skill and ridge; previous-year, climatology, and geographic baselines remain missing. | Add previous-year, station climatology, and lat/lon/year-only baselines; compare pixel skill and area calibration. |
| Stronger-tabular-model Phase 1 | moderate: Feature projection maximum distance from zero group is 6.96; use this only as separability evidence, not model skill. | Train a tree-based or histogram-gradient boosting baseline after target, baseline, and calibration checks. |
| Spatial-evaluation Phase 1 | moderate: 5 latitude bands meet minimum holdout-readiness counts in Monterey. | Design latitude/site holdout manifests and decide whether to add a second smoke region. |
| Ingestion/alignment-hardening Phase 1 | moderate: Missing-feature drops are 4.153% of split-manifest rows; label row counts should remain monitored. | Harden manifests, missing-feature diagnostics, and alignment comparison checks before scale-up if drop rates grow. |
| Scale-up Phase 1 | moderate: The smoke pipeline is runnable end to end, but target framing, baseline, and calibration gaps should be resolved before full West Coast processing. | Expand to a second region or broader California slice after label, baseline, and calibration choices are settled. |

## Appendix

- Stage distribution table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_label_distribution_by_stage.csv`
- Target framing table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_framing_summary.csv`
- Prediction distribution table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_prediction_distribution.csv`
- Residual by observed bin table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_observed_bin.csv`
- Residual by persistence table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_persistence.csv`
- Spatial readiness table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_spatial_holdout_readiness.csv`
- Feature separability table: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_feature_separability.csv`
- Phase 1 decision matrix: `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_decision_matrix.csv`

Validation command:

```bash
make check
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```
