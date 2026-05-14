# Task 49: Train And Evaluate Big Sur-Only Models

## Goal

Train the current annual-max model chain on Big Sur rows only, then evaluate on
held-out Big Sur test rows. This task should answer whether local Big Sur
training improves Big Sur performance relative to the Monterey-transfer
baseline from Task 48.

Use the same target and policy family as the closed Monterey Phase 1 workflow:
Kelpwatch-style annual maximum canopy on the retained plausible-kelp domain,
with AEF annual embeddings as predictors.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Big Sur model-input artifacts from P2-03c / Task 45:
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_split_manifest.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask_manifest.json`
- Monterey-transfer comparison artifacts from Task 48:
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_monterey_transfer_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_monterey_transfer_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_monterey_transfer_eval_manifest.json`
- Existing Big Sur model path blocks in `configs/big_sur_smoke.yaml`:
  `models.baselines`, `models.binary_presence`,
  `models.binary_presence.calibration`, `models.conditional_canopy`, and
  `models.hurdle`.

Current comparison anchor from Task 48 primary Big Sur 2022 retained-domain
test rows:

- Monterey-transfer AEF ridge: F1 `0.771748`, area bias `+5.5720%`.
- Monterey-transfer expected-value hurdle: F1 `0.849834`, area bias
  `-22.1124%`.
- Monterey-transfer hard-gated hurdle: F1 `0.849308`, area bias `-19.2801%`.
- Frozen Monterey binary support: AUROC `0.992484`, AUPRC `0.897458`,
  precision `0.829083`, recall `0.864566`, F1 `0.846453`.

## Outputs

- Big Sur-only baseline model artifacts:
  - `/Volumes/x10pro/kelp_aef/models/baselines/big_sur_ridge_kelp_fraction.joblib`
  - `/Volumes/x10pro/kelp_aef/models/baselines/big_sur_geographic_ridge_lon_lat_year.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_baseline_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_baseline_metrics.csv`
- Big Sur-only binary-presence artifacts:
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/big_sur_logistic_annual_max_ge_10pct.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_binary_presence_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_binary_presence_threshold_selection.csv`
- Big Sur-only binary calibration artifacts:
  - `/Volumes/x10pro/kelp_aef/models/binary_presence/big_sur_logistic_annual_max_ge_10pct_calibration.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_calibrated_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_binary_presence_calibration_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_binary_presence_calibrated_threshold_selection.csv`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_binary_presence_calibration_manifest.json`
- Big Sur-only conditional canopy artifacts:
  - `/Volumes/x10pro/kelp_aef/models/conditional_canopy/big_sur_ridge_positive_annual_max.joblib`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_conditional_canopy_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_conditional_canopy_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_conditional_canopy_model_comparison.csv`
- Big Sur-only hurdle artifacts:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_hurdle_prediction_manifest.json`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_hurdle_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_hurdle_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_hurdle_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/figures/big_sur_hurdle_2022_observed_predicted_residual.png`
- A compact Big Sur-only comparison or primary-summary artifact if the existing
  tables are not already sufficient for P2-09, labeled with:
  - `training_regime = big_sur_only`
  - `model_origin_region = big_sur`
  - `evaluation_region = big_sur`
- Big Sur-only training-regime comparison sidecars for P2-09:
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_only_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_only_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_only_eval_manifest.json`

## Config File

Use `configs/big_sur_smoke.yaml`.

The existing Big Sur model blocks should be treated as the canonical
Big Sur-only output paths. Do not overwrite any
`big_sur_monterey_transfer_*` artifacts from Task 48. If new summary paths are
needed for apples-to-apples comparison with transfer, add a clearly named
Big Sur-only summary or manifest path rather than changing Monterey-transfer
paths.

## Plan / Spec Requirement

Before implementation or artifact reruns, write a brief implementation plan
that confirms:

- Whether existing commands can run the Big Sur-only chain without code changes.
- The exact command sequence and which outputs each command refreshes.
- The split policy: 2018-2020 train, 2021 validation, 2022 held-out test.
- The binary target:
  `annual_max_ge_10pct = kelp_fraction_y >= 0.10`, equivalent to
  `kelp_max_y >= 90 m2`.
- The calibration split and threshold-selection policy. Calibration and
  threshold selection must use validation rows only.
- How Big Sur-only rows will be labeled for P2-09 training-regime comparison.
- How the Task 48 Monterey-transfer rows will be used as comparison context
  without reusing Monterey models or labels for Big Sur-only training.

## Implementation Plan

Expected command sequence if the existing package-backed stages work unchanged:

```bash
uv run kelp-aef train-baselines --config configs/big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_smoke.yaml
```

During implementation:

1. Verify the Big Sur masked sample and full-grid inference table exist and
   carry the expected AEF feature columns, split labels, label-source fields,
   sample weights, and plausible-domain mask metadata.
2. Train Big Sur-only ridge and geographic baselines on Big Sur training rows
   only. Select ridge alpha on the Big Sur validation split only.
3. Predict baseline outputs over the Big Sur retained-domain full-grid scope.
4. Train the Big Sur-only class-weighted logistic binary-presence model on
   Big Sur training rows only.
5. Select any raw binary threshold diagnostics on Big Sur validation rows only.
6. Fit the Big Sur Platt calibrator on Big Sur validation rows only, and select
   calibrated thresholds from Big Sur validation rows only.
7. Train the Big Sur-only conditional positive-canopy model using the existing
   positive-support policy and validation-only model selection.
8. Compose Big Sur-only expected-value and hard-gated hurdle predictions on the
   Big Sur retained-domain full grid.
9. Emit or verify report-visible rows with the same primary filters as Task 48:
   `split = test`, `year = 2022`, `evaluation_scope = full_grid_masked`,
   `label_source = all`, and `mask_status = plausible_kelp_domain`.
10. Record the completed outcome and primary Big Sur-only metrics in
    `docs/todo.md`.

## Outcome

Completed on 2026-05-14 with the configured Big Sur-only chain:

```bash
uv run kelp-aef train-baselines --config configs/big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_smoke.yaml
```

The chain used Big Sur training rows from 2018-2020, Big Sur validation rows
from 2021, and held-out Big Sur test rows from 2022. The Platt calibrator was
fit on 46,269 validation rows, and validation-only calibrated threshold
selection chose `0.41` for `validation_max_f1_calibrated`.

Primary held-out Big Sur 2022 `full_grid_masked` / `all` outcomes:

- Big Sur-only AEF ridge: F1 `0.649054`, RMSE `0.068505`, R2 `0.746215`,
  area bias `+73.7367%`.
- Big Sur-only expected-value hurdle:
  F1 `0.859563`, RMSE `0.044095`, R2 `0.894855`, area bias `-2.8414%`.
- Big Sur-only hard-gated hurdle:
  F1 `0.857286`, RMSE `0.048916`, R2 `0.870606`, area bias `-0.4462%`.
- Big Sur-only calibrated binary support on held-out test rows:
  AUROC `0.980687`, AUPRC `0.955301`, precision `0.909196`,
  recall `0.839853`, F1 `0.873150`.

Compared with the Task 48 Monterey-transfer primary rows, Big Sur-only training
improved the expected-value hurdle from F1 `0.849834` to `0.859563` and reduced
area bias from `-22.1124%` to `-2.8414%`. It also improved the hard-gated
hurdle from F1 `0.849308` to `0.857286` and reduced area bias from
`-19.2801%` to `-0.4462%`. The Big Sur-only AEF ridge underperformed the
Monterey-transfer ridge on F1 and area calibration, so the local-training gain
is concentrated in the calibrated binary + conditional hurdle chain rather than
the ridge-only baseline.

## Validation Command

For docs-only task-plan edits:

```bash
git diff --check
```

For implementation or artifact refreshes, run the relevant code checks plus the
Big Sur-only model chain:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run kelp-aef train-baselines --config configs/big_sur_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/big_sur_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/big_sur_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/big_sur_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/big_sur_smoke.yaml
```

If code changes are made, prefer `make check` after focused validation unless
unrelated formatting drift is explicitly recorded.

## Smoke-Test Region And Years

- Region: Big Sur.
- Training regime: Big Sur-only.
- Label input: Kelpwatch-style annual max canopy.
- Features: AEF annual embedding bands `A00-A63`.
- Split: train 2018-2020, validation 2021, test 2022.
- Primary evaluation scope: 2022 retained plausible-kelp-domain full grid.

## Acceptance Criteria

- Big Sur-only baseline, binary, calibration, conditional, and hurdle artifacts
  are generated from `configs/big_sur_smoke.yaml`.
- No Monterey rows or Monterey model payloads are used for Big Sur-only model
  fitting, calibration fitting, or threshold selection.
- Ridge alpha, binary thresholds, binary calibration, and conditional-model
  selection use Big Sur validation rows only.
- Held-out Big Sur 2022 test rows remain final evaluation rows and are not used
  for training, calibration, threshold selection, or model-policy selection.
- Big Sur-only metrics are clearly labeled with `training_regime =
  big_sur_only` or equivalent provenance in any new comparison table or
  manifest needed for P2-09.
- Primary Big Sur-only rows are directly comparable to Task 48
  Monterey-transfer rows on the same split, year, mask, evaluation scope, and
  label-source filters.
- `docs/todo.md` records the completed command sequence and primary outcome
  metrics after artifacts are regenerated.

## Known Constraints And Non-Goals

- Do not tune anything on the 2022 Big Sur test split.
- Do not change the Phase 2 target away from Kelpwatch annual max.
- Do not add bathymetry, DEM, coastline, or region identifiers as predictors in
  this task.
- Do not train pooled Monterey + Big Sur models here; that belongs in P2-08.
- Do not rewrite the narrative Phase 2 report here beyond concise task outcome
  notes; P2-09 owns the integrated report.
- Do not overwrite Monterey-transfer sidecars or reinterpret Task 48 results.
