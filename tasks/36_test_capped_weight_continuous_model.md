# Task 36: Test Capped-Weight Continuous Model

## Goal

Test one direct continuous AlphaEarth model whose training objective uses capped
retained-background expansion weights.

The target remains the Monterey Phase 1 annual-max weak label:

```text
kelp_fraction_y = kelp_max_y / 900 m2
kelp_max_y      = Kelpwatch-style annual max canopy area in one 30 m cell
```

This task should answer whether a simpler one-stage continuous objective can
compete with the current expected-value hurdle model without collapsing toward
near-zero predictions or leaking small positives across retained background
cells.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current default masked model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Current masked sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current retained-domain comparison anchors:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`

Current anchors to preserve:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label input: Kelpwatch-style annual max, `kelp_fraction_y` / `kelp_max_y`.
- Features: `A00-A63`.
- Primary reporting scope: `full_grid_masked` inside `plausible_kelp_domain`.
- Default sample policy: `crm_stratified_mask_first_sample`.

## Outputs

- Package-backed command or sidecar mode, for example:
  `kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment capped-weight`.
- Config block for the capped-weight continuous experiment.
- Serialized model, for example:
  `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_capped_weight.joblib`.
- Row-level retained-domain full-grid predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/continuous_objective_capped_weight_full_grid_predictions.parquet`.
- Sample prediction and metric artifacts, for example:
  - `/Volumes/x10pro/kelp_aef/processed/continuous_objective_capped_weight_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/interim/continuous_objective_capped_weight_manifest.json`
- Updated Phase 1 report and model-comparison rows that compare the capped-weight
  continuous model against ridge, expected-value hurdle, hard-gated hurdle, and
  reference baselines.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Add an explicit experiment block rather than silently changing
  `models.baselines`.
- Keep the active default sample path:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Use a training-only capped weight column or policy derived from `sample_weight`.
- Keep evaluation metrics unweighted unless a table explicitly labels a weighted
  diagnostic row. Full-grid area calibration should be computed from row-level
  full-grid predictions, not from weighted sample metrics.
- Start with one conservative cap selected from validation-only reasoning, for
  example `fit_weight_cap: 5.0` or another pre-declared value. If multiple caps
  are compared, select by 2021 validation metrics only and report the chosen cap.
- Preserve `features: A00-A63` and `target: kelp_fraction_y`.

## Plan/Spec Requirement

This is a new model objective and artifact family. Before implementation, write a
brief implementation plan that confirms:

- Whether the implementation extends baseline sidecars or creates a dedicated
  continuous-objective command.
- The exact capped fit-weight formula, for example:

```text
fit_weight = min(sample_weight, fit_weight_cap)
```

- Whether Kelpwatch-supported rows keep weight `1.0` and only
  assumed-background expansion weights are capped.
- Which cap value or validation-only cap grid is used.
- How fit weights are kept separate from unweighted station metrics and
  full-grid area calibration.
- How predictions are clipped to `[0, 1]` before canopy-area summaries.
- Which comparison table becomes authoritative for P1-22a.
- How the task avoids selecting caps, thresholds, or report choices on 2022 test
  rows.

## Implementation Plan

- Load the current masked sample, split assignments, and configured AEF feature
  columns.
- Derive a capped fit-weight column from `sample_weight`, keeping positive and
  Kelpwatch-supported rows from being downweighted below `1.0`.
- Fit a ridge regression on train rows with the capped fit weights.
- Select any hyperparameters using validation rows only.
- Predict on sample rows and retained-domain full-grid rows for all configured
  years.
- Write row-level predictions with enough metadata to distinguish:
  `model_family`, `model_name`, `objective_policy`, `fit_weight_policy`,
  `fit_weight_cap`, `sample_policy`, `split`, `year`, `label_source`,
  `mask_status`, and `evaluation_scope`.
- Report station skill separately from background-inclusive sample diagnostics.
- Report assumed-background leakage for 2022 retained-domain rows.
- Report full-grid area calibration for the same scope as the current
  scoreboard.
- Update the Phase 1 report so P1-22a answers:
  - Does capped weighting reduce ridge background leakage?
  - Does it preserve Kelpwatch-station skill?
  - Does it approach or beat the expected-value hurdle on RMSE, F1, and area
    calibration?

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment capped-weight
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

If the implementation reuses `train-baselines` instead of a new command, update
the command name in this task when the plan is accepted.

Full validation:

```bash
make check
```

Manual inspection:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_area_calibration.csv'; print(pd.read_csv(p).to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_assumed_background_leakage.csv'; print(pd.read_csv(p).to_string(index=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings, `A00-A63`.
- Reporting scope: retained plausible-kelp domain with
  `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- The capped-weight model is a direct continuous model, not a binary or hurdle
  model.
- The active annual-max target, mask, split, feature set, and sample policy are
  unchanged.
- Fit weights are documented in config and the output manifest.
- Fit weights do not silently replace unweighted evaluation metrics.
- The report includes a capped-weight comparison row for 2022
  `full_grid_masked`.
- The report includes Kelpwatch-station skill, assumed-background leakage, and
  full-grid area calibration.
- The comparison clearly states whether the capped-weight model beats ridge,
  competes with the expected-value hurdle, or fails.
- No cap, threshold, or policy is tuned on 2022 test rows.
- `make check` passes.

## Completion Notes

Completed on 2026-05-13.

Implemented as a dedicated `train-continuous-objective` command with an explicit
`models.continuous_objective.experiments.capped-weight` config block. The fitted
model is a direct ridge regression on `A00-A63` and `kelp_fraction_y`, using:

```text
fit_weight = 1.0 for Kelpwatch-supported or positive rows
fit_weight = min(max(sample_weight, 1.0), 5.0) for assumed-background rows
```

The cap is `5.0`; ridge alpha was selected on the 2021 validation split only.
The selected alpha was `0.01`. No 2022 test rows were used for cap, threshold,
or model-selection decisions.

Generated artifacts:

- `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_capped_weight.joblib`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_capped_weight_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_capped_weight_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_metrics.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_area_calibration.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_capped_weight_assumed_background_leakage.csv`
- `/Volumes/x10pro/kelp_aef/interim/continuous_objective_capped_weight_manifest.json`

Primary 2022 retained-domain result:

| Model | RMSE | R2 | F1 >=10% | Predicted area | Area bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| AEF ridge regression | 0.0452 | 0.587 | 0.476 | 8.42 M m2 | +102.1% |
| Capped-weight ridge | 0.0493 | 0.507 | 0.590 | 8.64 M m2 | +107.5% |
| Expected-value hurdle | 0.0322 | 0.790 | 0.812 | 3.50 M m2 | -16.0% |

Station-skill context: capped-weight ridge Kelpwatch-station test RMSE was
`0.1969`, worse than ridge `0.1647`. Assumed-background leakage was
`5.69 M m2` over `599,997` retained assumed-background rows, with predicted
positive rate `0.87%`.

Conclusion: capped weighting failed to beat ridge on the combined RMSE and
area-bias check and does not compete with the expected-value hurdle. Continue to
P1-22b only as a separate stratified-background direct-continuous experiment.

### Cap Sweep Follow-up

Follow-up run on 2026-05-13 after checking whether the cap itself explained the
poor cap-5 result. The sweep varied only `fit_weight_cap`; it kept the same
CRM-stratified mask-first sample, train/validation/test years, annual-max
target, `A00-A63` feature set, ridge alpha grid, and retained-domain full-grid
inference table fixed.

Additional experiments were added under
`models.continuous_objective.experiments`: `cap-1`, `cap-2`, `cap-10`,
`cap-20`, and `cap-100`. The existing `capped-weight` experiment remains the
cap-5 run. Cap 100 is effectively uncapped for the current sample because the
maximum raw assumed-background `sample_weight` is just under 100.

| Cap | Alpha | Val sample RMSE | Test station RMSE | Test full-grid RMSE | Area bias | Pred area (M m2) | Background leak (M m2) | Background positive rate | Train background weight share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.01 | 0.1016 | 0.1623 | 0.0452 | +102.1% | 8.42 | 5.04 | 1.95% | 51.6% |
| 2 | 0.01 | 0.1072 | 0.1720 | 0.0459 | +107.8% | 8.65 | 5.33 | 1.59% | 68.1% |
| 5 | 0.01 | 0.1226 | 0.1969 | 0.0493 | +107.5% | 8.64 | 5.69 | 0.87% | 84.2% |
| 10 | 0.01 | 0.1359 | 0.2180 | 0.0520 | +84.4% | 7.67 | 5.17 | 0.39% | 90.6% |
| 20 | 0.01 | 0.1388 | 0.2228 | 0.0523 | +62.8% | 6.78 | 4.47 | 0.37% | 93.3% |
| 100 | 0.01 | 0.1419 | 0.2288 | 0.0529 | +43.0% | 5.95 | 3.83 | 0.32% | 95.2% |

Sweep interpretation: cap 5 is not failing because it forgot to cap weights.
The cap is active, and changing it moves the expected tradeoff. Lower caps
protect station skill and full-grid RMSE but still leak enough small
background predictions to overpredict area by about 100%. Higher caps suppress
some full-grid area bias and lower the background positive rate, but the
background rows dominate the fit and station skill degrades sharply. The direct
continuous objective remains the issue; no cap in this sweep produces a
competitive retained-domain all-label result.

Additional generated cap-sweep artifacts follow the pattern:

- `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_capped_weight_cap_{1,2,10,20,100}.joblib`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_cap_{1,2,10,20,100}_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_cap_{1,2,10,20,100}_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_cap_{1,2,10,20,100}_metrics.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_cap_{1,2,10,20,100}_area_calibration.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_cap_{1,2,10,20,100}_assumed_background_leakage.csv`
- `/Volumes/x10pro/kelp_aef/interim/continuous_objective_cap_{1,2,10,20,100}_manifest.json`

Validation passed:

```bash
make check
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment cap-1
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment cap-2
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment cap-10
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment cap-20
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment cap-100
uv run pytest tests/test_continuous_objective.py tests/test_model_analysis.py
uv run mypy src/kelp_aef/evaluation/continuous_objective.py src/kelp_aef/cli.py src/kelp_aef/evaluation/model_analysis.py
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment capped-weight
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Known Constraints Or Non-Goals

- Do not use CRM depth, elevation, depth bin, or mask reason as predictors.
- Do not change the plausible-kelp mask.
- Do not change the annual-max label input.
- Do not retrain binary presence, probability calibration, conditional canopy, or
  hurdle artifacts unless a later task explicitly asks for that.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
- Do not start full West Coast scale-up.
