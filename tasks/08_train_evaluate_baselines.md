# Task 08: Train And Evaluate First Simple Baselines

## Goal

Train and evaluate the first tabular baseline models for the Monterey smoke
test using the aligned AEF/Kelpwatch table. The first model should answer a
narrow question: do annual AEF embeddings contain enough signal to reproduce the
continuous Kelpwatch annual-max canopy fraction better than a no-skill baseline?

## First Model Choice

Use a `StandardScaler` + `Ridge` regression model as the first embedding model.

Fit target:

- `kelp_fraction_y`, the Kelpwatch annual max canopy area normalized by the
  900 m2 Landsat pixel area.

Feature columns:

- `A00` through `A63` only.

Primary split:

- Train: 2018-2020.
- Validation: 2021.
- Test: 2022.

Tune only the ridge `alpha` on the validation year from a small fixed grid, then
report final test metrics on 2022. For the first smoke result, do not refit on
the validation year before testing; keep test evaluation easy to reason about.

## Why This Model First

- The current target is continuous annual max canopy, not a settled binary
  presence label. Ridge lets us avoid choosing a binary threshold too early.
- The feature set is only 64 AEF embedding bands, so a regularized linear model
  is fast, stable, and enough to catch alignment or label issues.
- Ridge handles correlated embedding bands better than unregularized linear
  regression.
- Coefficients and residuals are easier to inspect than a high-capacity model.
- This gives a defensible baseline before deciding whether nonlinear models add
  useful skill.

## Why Not The Other Model Types Yet

- Logistic regression: defer because it requires choosing a binary Kelpwatch
  threshold. We have not yet decided whether `>0`, 1%, 5%, 10%, or another
  threshold is the right target.
- Random forest: defer until after ridge because it is higher capacity, slower,
  less transparent, and easier to overfit to station/year quirks in a small
  smoke test.
- Gradient boosting, XGBoost, LightGBM, or neural nets: defer because they add
  model complexity and possibly dependencies before we know the linear baseline
  is sane.
- CNNs or other spatial models: defer because the current artifact is a flat
  parquet table. Spatial models should use a chip/tensor product that preserves
  10 m neighborhoods.
- Lasso or elastic net: defer because the first goal is not feature selection;
  AEF bands are correlated and lasso-style sparsity can be unstable in that
  setting.
- Geographic or station-history models: useful later, but they answer a
  different question about persistence or location priors rather than AEF
  embedding signal.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Aligned table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`.
- Alignment manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table_manifest.json`.
- Split years from `configs/monterey_smoke.yaml`.

## Outputs

- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Trained ridge model:
  `/Volumes/x10pro/kelp_aef/models/baselines/ridge_kelp_fraction.joblib`.
- Baseline predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_predictions.parquet`.
- Model metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- Training/evaluation manifest:
  `/Volumes/x10pro/kelp_aef/interim/baseline_eval_manifest.json`.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions should cover:

- Input aligned table path.
- Split manifest path.
- Model output directory.
- Prediction output path.
- Metrics output path.
- Target column, initially `kelp_fraction_y`.
- Feature columns or band range, initially `A00-A63`.
- Ridge alpha grid.
- Whether to drop rows with missing features.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should state the
files to change, model pipeline, split handling, output schema, metrics, and how
missing AEF feature rows are handled.

## Proposed CLI

```bash
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
```

## Implementation Notes

- Add scikit-learn as the modeling dependency when implementing this task.
- Use a package-backed CLI command, not a notebook-only workflow.
- Build split assignments from the configured train, validation, and test years.
- Drop rows with missing AEF feature values for the first run and report the
  dropped-row count by split.
- Fit a no-skill train-mean baseline for context.
- Fit ridge candidates on train years only.
- Select ridge `alpha` using validation RMSE.
- Evaluate both no-skill and selected ridge on train, validation, and test
  splits.
- Clip predictions to `[0, 1]` only for reporting derived canopy area and binary
  threshold diagnostics; keep raw model predictions in the prediction table.
- Convert predicted fractions to area with `pred_kelp_max_y = pred_fraction *
  900`.
- Use structured logging for paths, split sizes, selected alpha, metrics, and
  dropped rows.

## Prediction Schema

Minimum columns:

- `year`
- `split`
- `kelpwatch_station_id`
- `longitude`
- `latitude`
- `kelp_fraction_y`
- `kelp_max_y`
- `model_name`
- `pred_kelp_fraction_y`
- `pred_kelp_fraction_y_clipped`
- `pred_kelp_max_y`
- `residual_kelp_fraction_y`
- `residual_kelp_max_y`

## Metrics

Regression metrics:

- MAE on `kelp_fraction_y`.
- RMSE on `kelp_fraction_y`.
- R2 on `kelp_fraction_y`.
- Pearson correlation when defined.
- Spearman correlation when defined.
- Mean observed canopy fraction.
- Mean predicted canopy fraction.
- Aggregate observed canopy area.
- Aggregate predicted canopy area.
- Aggregate area bias and percent bias.

Threshold diagnostics:

- Use observed and predicted clipped fractions at 1%, 5%, and 10% thresholds.
- Report precision, recall, F1, and positive rate by split as diagnostics only.
- Do not use these diagnostics to choose a production binary threshold in this
  task.

## Validation Command

```bash
make check
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
```

Add focused tests with tiny synthetic aligned tables to verify:

- Split assignment follows configured years.
- Missing feature rows are dropped and counted.
- The no-skill baseline predicts the train mean.
- Ridge training writes model, predictions, metrics, and manifest artifacts.
- Validation alpha selection is deterministic.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Train years: 2018-2020.
- Validation year: 2021.
- Test year: 2022.

## Acceptance Criteria

- The command writes the split manifest, model file, prediction parquet, metrics
  CSV, and evaluation manifest.
- Metrics include no-skill and ridge rows for train, validation, and test
  splits.
- The selected ridge alpha is recorded in the metrics and manifest.
- The prediction table includes both raw and clipped predictions.
- Dropped rows with missing AEF features are reported by split.
- `make check` passes.

## Known Constraints And Non-Goals

- Do not choose the final binary kelp threshold in this task.
- Do not train logistic regression or a classifier in this task.
- Do not train random forests, boosted trees, neural nets, CNNs, or spatial
  models in this task.
- Do not add latitude/longitude as model features in the first embedding model;
  keep this baseline focused on AEF embedding bands.
- Do not produce predicted maps or residual maps in this task. The next task
  should consume `baseline_predictions.parquet` for mapping and area-bias QA.
