# Task 50: Cross-Evaluate Monterey And Big Sur Training Regimes

## Goal

Evaluate every current Monterey/Big Sur training-regime and evaluation-region
combination for the annual-max model chain:

- Monterey-only model evaluated on Monterey.
- Monterey-only model evaluated on Big Sur.
- Big Sur-only model evaluated on Big Sur.
- Big Sur-only model evaluated on Monterey.
- Pooled Monterey+Big Sur model evaluated on Big Sur.
- Pooled Monterey+Big Sur model evaluated on Monterey.

This task should answer whether local, transfer, or pooled training gives the
best Kelpwatch-style annual maximum predictions in each region. It should also
test the new hypothesis from Task 49: target-local tuning is especially
important for predicting the amount of kelp, not just binary support.

## Inputs

- Configs:
  - `configs/monterey_smoke.yaml`
  - `configs/big_sur_smoke.yaml`
- Monterey model-input and inference artifacts:
  - `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`
- Big Sur model-input and inference artifacts:
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_split_manifest.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask_manifest.json`
- Existing same-region and transfer comparison anchors:
  - Monterey-only on Monterey: current Monterey model-analysis comparison table
    from `configs/monterey_smoke.yaml`.
  - Monterey-only on Big Sur:
    `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_monterey_transfer_primary_summary.csv`.
  - Big Sur-only on Big Sur:
    `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_only_primary_summary.csv`.

Current Big Sur primary anchors on 2022 retained-domain test rows:

- Monterey-only on Big Sur expected-value hurdle: F1 `0.849834`, area bias
  `-22.1124%`.
- Big Sur-only on Big Sur expected-value hurdle: F1 `0.859563`, area bias
  `-2.8414%`.
- Monterey-only on Big Sur hard-gated hurdle: F1 `0.849308`, area bias
  `-19.2801%`.
- Big Sur-only on Big Sur hard-gated hurdle: F1 `0.857286`, area bias
  `-0.4462%`.
- Monterey-only on Big Sur AEF ridge: F1 `0.771748`, area bias `+5.5720%`.
- Big Sur-only on Big Sur AEF ridge: F1 `0.649054`, area bias `+73.7367%`.

## Outputs

- Pooled model-input artifacts:
  - `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_split_manifest.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_training_manifest.json`
  - `/Volumes/x10pro/kelp_aef/reports/tables/pooled_monterey_big_sur_training_summary.csv`
- Big Sur-only-on-Monterey transfer artifacts:
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_transfer_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_transfer_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_transfer_eval_manifest.json`
- Pooled artifacts evaluated on Big Sur:
  - `/Volumes/x10pro/kelp_aef/models/baselines/pooled_monterey_big_sur_ridge_kelp_fraction.big_sur_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/pooled_monterey_big_sur_logistic_annual_max_ge_10pct.big_sur_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/pooled_monterey_big_sur_logistic_annual_max_ge_10pct_calibration.big_sur_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/pooled_monterey_big_sur_ridge_positive_annual_max.big_sur_target.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_pooled_monterey_big_sur_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_pooled_monterey_big_sur_eval_manifest.json`
- Pooled artifacts evaluated on Monterey:
  - `/Volumes/x10pro/kelp_aef/models/baselines/pooled_monterey_big_sur_ridge_kelp_fraction.monterey_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/pooled_monterey_big_sur_logistic_annual_max_ge_10pct.monterey_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/pooled_monterey_big_sur_logistic_annual_max_ge_10pct_calibration.monterey_target.joblib`
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/pooled_monterey_big_sur_ridge_positive_annual_max.monterey_target.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_pooled_monterey_big_sur_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_pooled_monterey_big_sur_eval_manifest.json`
- Combined cross-regime comparison artifacts for P2-09:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_training_regime_comparison_manifest.json`

The combined comparison must use canonical labels:

- `training_regime in {monterey_only, big_sur_only, pooled_monterey_big_sur}`
- `model_origin_region in {monterey, big_sur, monterey_big_sur}`
- `evaluation_region in {monterey, big_sur}`

Existing Task 48 rows may still carry `training_regime = monterey_transfer` in
their raw sidecar. The combined comparison should normalize those rows to
`training_regime = monterey_only`, `model_origin_region = monterey`, and
`evaluation_region = big_sur` without rewriting the Task 48 artifact.

## Config File

Use both `configs/monterey_smoke.yaml` and `configs/big_sur_smoke.yaml`.

Implementation may either:

- add path-distinct transfer/pooled sidecar blocks to the existing configs; or
- add thin derived configs such as
  `configs/monterey_pooled_monterey_big_sur_smoke.yaml` and
  `configs/big_sur_pooled_monterey_big_sur_smoke.yaml`.

Do not overwrite Monterey-only, Big Sur-only, or Monterey-transfer artifacts.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task or a
small decision note that confirms:

- The exact six train/evaluate cells being produced.
- How raw existing rows are normalized into canonical training-regime labels.
- How Monterey and Big Sur samples are combined for pooled fitting.
- Which validation rows tune each target-region policy.
- Whether pooled target-specific model payloads are separate files.
- The command sequence and output paths.

## Implementation Plan

1. Define the cross-regime matrix.

   | Training regime | Model origin | Evaluation region | Status |
   | --- | --- | --- | --- |
   | `monterey_only` | `monterey` | `monterey` | Reuse current Monterey same-region outputs. |
   | `monterey_only` | `monterey` | `big_sur` | Reuse Task 48 transfer outputs, normalized in the combined table. |
   | `big_sur_only` | `big_sur` | `big_sur` | Reuse Task 49 Big Sur-only outputs. |
   | `big_sur_only` | `big_sur` | `monterey` | New frozen Big Sur-on-Monterey transfer evaluation. |
   | `pooled_monterey_big_sur` | `monterey_big_sur` | `big_sur` | New pooled fit with Big Sur target-local validation. |
   | `pooled_monterey_big_sur` | `monterey_big_sur` | `monterey` | New pooled fit with Monterey target-local validation. |

2. Add or reuse package-backed support for target transfer in both directions.
   - `evaluate-transfer --config configs/big_sur_smoke.yaml --source-config configs/monterey_smoke.yaml`
     already covers Monterey-only on Big Sur.
   - Add the reciprocal output config needed for
     `evaluate-transfer --config configs/monterey_smoke.yaml --source-config configs/big_sur_smoke.yaml`
     without overwriting Monterey same-region artifacts.
3. Add or reuse a package-backed pooled-sample builder.
   - Combine Monterey and Big Sur masked sample tables.
   - Add `training_region` or `source_region` for auditing.
   - Keep region metadata out of model feature columns.
   - Preserve annual-max target columns, label-source columns, retained-domain
     mask metadata, and sample weights.
4. Train pooled target-specific model payloads.
   - Fit rows for both target policies: Monterey + Big Sur 2018-2020.
   - Big Sur target policy: use Big Sur 2021 for ridge alpha selection,
     binary calibration, threshold selection, and conditional amount alpha
     selection.
   - Monterey target policy: use Monterey 2021 for ridge alpha selection,
     binary calibration, threshold selection, and conditional amount alpha
     selection.
   - Keep the pooled fit/evaluation paths separate by target region because
     target-local validation may select different thresholds, calibrators, and
     conditional amount models.
5. Extend conditional canopy sidecar support if needed.
   - Baseline, binary, calibration, and hurdle sidecars already have most of
     the needed shape.
   - Conditional canopy sidecars currently support model reuse; this task needs
     refit support so the pooled conditional amount model can be selected from
     pooled train rows with target-local validation rows.
6. Compose pooled expected-value and hard-gated hurdle predictions for both
   target regions.
   - Big Sur pooled predictions use the Big Sur full-grid inference table and
     Big Sur retained-domain mask.
   - Monterey pooled predictions use the Monterey full-grid inference table and
     Monterey retained-domain mask.
7. Build a combined comparison table.
   - Primary filters for each region:
     - `split = test`
     - `year = 2022`
     - `evaluation_scope = full_grid_masked`
     - `label_source = all`
     - `mask_status = plausible_kelp_domain`
   - Include AEF ridge, previous-year persistence, expected-value hurdle, and
     hard-gated hurdle rows at minimum.
   - Include binary support summary rows or a linked binary metrics table if the
     schemas do not fit cleanly into the area-calibration comparison table.
8. Interpret the matrix with amount calibration as a first-class criterion.
   - Compare expected-value area bias, hard-gated area bias, RMSE, R2, and F1.
   - Explicitly call out whether pooled training improves or damages each
     region's canopy amount calibration.
   - Distinguish binary support transfer from conditional amount transfer.
9. Record the command sequence and primary outcomes in `docs/todo.md`.

## Validation Command

For docs-only task-plan edits:

```bash
git diff --check
```

For implementation:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run kelp-aef evaluate-transfer --config configs/monterey_smoke.yaml --source-config configs/big_sur_smoke.yaml
uv run kelp-aef build-pooled-region-sample --config configs/big_sur_smoke.yaml
uv run kelp-aef train-baselines --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
```

If implementation keeps pooled sidecars inside the existing configs instead of
derived configs, use the equivalent sidecar-producing command sequence and
document it in the task outcome.

## Smoke-Test Region And Years

- Training regions: Monterey, Big Sur, and pooled Monterey+Big Sur.
- Evaluation regions: Monterey and Big Sur.
- Label input: Kelpwatch-style annual max canopy.
- Features: AEF annual embedding bands `A00-A63`.
- Fit years: 2018-2020.
- Target-local validation year: 2021.
- Held-out test year: 2022.
- Primary evaluation scope: retained plausible-kelp-domain full grid for each
  target region.

## Acceptance Criteria

- The task emits a six-cell training-regime by evaluation-region comparison
  with canonical provenance labels.
- Big Sur-only models are evaluated on Monterey without using Monterey test
  rows for fitting, calibration, threshold selection, or amount-model
  selection.
- Pooled models are evaluated on both Monterey and Big Sur with target-local
  2021 validation for each target policy.
- Big Sur 2022 and Monterey 2022 rows are final held-out evaluation rows only.
- Region metadata is retained for auditing but is not used as a model
  predictor.
- Monterey-transfer and Big Sur-only sidecars from Tasks 48 and 49 are not
  overwritten.
- The outcome explicitly answers whether pooled training improves each region's
  expected-value hurdle area bias versus local-only and transfer alternatives.

## Known Constraints And Non-Goals

- Do not tune anything on 2022 held-out rows for either region.
- Do not change the Phase 2 target away from Kelpwatch annual max.
- Do not add bathymetry, DEM, coastline, source-region labels, or other region
  identifiers as predictors.
- Do not rewrite the integrated Phase 2 report here; P2-09 owns report
  synthesis.
- Do not start a broad model-family search here. Random forest or gradient
  boosting belongs in a later task if Phase 2 recommends it.

## Outcome

Completed on 2026-05-14.

Implemented package-backed pooled-region and reciprocal transfer support:

- Added `build-pooled-region-sample` to combine retained-domain Monterey and
  Big Sur masked sample tables with a `source_region` audit column and no
  region metadata in the model feature matrix.
- Added target-region filtering for validation/test evaluation so pooled
  model fitting uses both regions for 2018-2020 while ridge selection,
  binary calibration, threshold selection, conditional amount selection, and
  2022 final scoring use only the target region.
- Added reciprocal Big Sur-only-on-Monterey transfer sidecars without
  overwriting Monterey same-region outputs.
- Added target-specific pooled configs and model payload paths for Big Sur and
  Monterey.
- Added `compare-training-regimes` to normalize existing same-region,
  transfer, and pooled rows into canonical
  `training_regime` / `model_origin_region` / `evaluation_region` labels.

Primary artifacts:

- `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_training_table.masked.parquet`
- `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_split_manifest.parquet`
- `/Volumes/x10pro/kelp_aef/interim/pooled_monterey_big_sur_training_manifest.json`
- `/Volumes/x10pro/kelp_aef/reports/tables/pooled_monterey_big_sur_training_summary.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_transfer_model_comparison.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_pooled_monterey_big_sur_model_comparison.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_pooled_monterey_big_sur_model_comparison.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_model_comparison.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
- `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_training_regime_comparison_manifest.json`

Pooled sample summary:

- Combined rows: `459,840`
- Big Sur rows: `231,345`
- Monterey rows: `228,495`
- Fit years: `2018-2020`
- Target-local validation year: `2021`
- Final held-out test year: `2022`

Primary 2022 retained-domain comparison:

| Evaluation region | Training regime | Expected-value hurdle F1 | Expected-value area bias | Hard-gated hurdle F1 | Hard-gated area bias |
| --- | --- | ---: | ---: | ---: | ---: |
| Big Sur | Monterey-only transfer | `0.849834` | `-22.1124%` | `0.849308` | `-19.2801%` |
| Big Sur | Big Sur-only | `0.859563` | `-2.8414%` | `0.857286` | `-0.4462%` |
| Big Sur | Pooled Monterey+Big Sur | `0.850056` | `-20.1402%` | `0.854451` | `-16.8870%` |
| Monterey | Big Sur-only transfer | `0.779440` | `-13.6822%` | `0.777221` | `-12.2153%` |
| Monterey | Monterey-only | `0.840780` | `-27.5095%` | `0.853000` | `-26.3099%` |
| Monterey | Pooled Monterey+Big Sur | `0.830924` | `-22.1988%` | `0.830019` | `-21.5465%` |

Interpretation:

- For Big Sur, pooled training did not improve over Big Sur-only local tuning.
  It modestly improved expected-value area bias over Monterey-only transfer
  (`-20.1402%` versus `-22.1124%`) but remained far worse than Big Sur-only
  (`-2.8414%`).
- For Monterey, pooled training improved expected-value area bias relative to
  Monterey-only (`-22.1988%` versus `-27.5095%`) and improved RMSE/R2, but it
  reduced F1 relative to Monterey-only. Big Sur-only transfer had the smallest
  Monterey area bias (`-13.6822%`) but much worse F1 and RMSE than the
  Monterey-tuned and pooled models.
- The outcome supports the Task 49 hypothesis: target-local amount tuning is
  important. Pooled training alone did not replace target-local calibration,
  especially for Big Sur canopy amount.

Binary support follow-up:

The binary-only comparison was recomputed from the 2022 retained-domain
full-grid binary prediction tables and each policy's frozen Platt calibrator
and recommended threshold. This keeps transfer, local, and pooled rows on the
same `full_grid_masked`-style support-detection scope.

| Evaluation region | Training regime | Binary F1 | Precision | Recall | AUPRC | Predicted positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Big Sur | Monterey-only transfer | `0.846453` | `0.829083` | `0.864566` | `0.897458` | `4.6844%` |
| Big Sur | Big Sur-only | `0.857286` | `0.875459` | `0.839853` | `0.934899` | `4.3095%` |
| Big Sur | Pooled Monterey+Big Sur | `0.854451` | `0.887361` | `0.823895` | `0.927124` | `4.1709%` |
| Monterey | Big Sur-only transfer | `0.777221` | `0.866996` | `0.704293` | `0.824119` | `1.6142%` |
| Monterey | Monterey-only | `0.852555` | `0.880181` | `0.826610` | `0.930024` | `1.8662%` |
| Monterey | Pooled Monterey+Big Sur | `0.829966` | `0.909602` | `0.763152` | `0.920076` | `1.6672%` |

Binary interpretation:

- Binary support transfers much better than canopy amount. Monterey-only on
  Big Sur is close to Big Sur-only in F1 (`0.846453` versus `0.857286`) and
  has slightly higher recall, but lower precision and AUPRC.
- Big Sur-only on Monterey transfers less well: F1 drops to `0.777221`,
  mostly because recall falls to `0.704293`.
- Pooled binary support is not a clear overall win. It is the most conservative
  policy in both regions: highest precision, lowest predicted-positive rate,
  and lower recall than the local model.
- The main Phase 2 signal is therefore not binary separability. The larger
  region-specific failure is conditional canopy amount and hurdle area
  calibration.

Command sequence run:

```bash
uv run kelp-aef build-pooled-region-sample --config configs/big_sur_smoke.yaml
uv run kelp-aef evaluate-transfer --config configs/monterey_smoke.yaml --source-config configs/big_sur_smoke.yaml
uv run kelp-aef train-baselines --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_pooled_monterey_big_sur_smoke.yaml
uv run kelp-aef compare-training-regimes --config configs/big_sur_smoke.yaml
```
