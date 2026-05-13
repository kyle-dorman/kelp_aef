# Task 37: Test Stratified-Background Continuous Model

## Goal

Test one direct continuous AlphaEarth model whose training objective balances
assumed-background contribution by retained-domain strata.

The target remains the Monterey Phase 1 annual-max weak label:

```text
kelp_fraction_y = kelp_max_y / 900 m2
kelp_max_y      = Kelpwatch-style annual max canopy area in one 30 m cell
```

This task should answer whether a single continuous model can reduce retained
background leakage and preserve station skill by controlling the loss across
background strata, without needing a two-stage hurdle composition.

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
- CRM depth, depth bins, and mask reasons are sampling/objective context only,
  not model predictors.

## Outputs

- Package-backed command or sidecar mode, for example:
  `kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment stratified-background`.
- Config block for the stratified-background continuous experiment.
- Serialized model, for example:
  `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_stratified_background.joblib`.
- Row-level retained-domain full-grid predictions, for example:
  `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_full_grid_predictions.parquet`.
- Sample prediction and metric artifacts, for example:
  - `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_sample_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/interim/continuous_objective_stratified_background_manifest.json`
- Updated Phase 1 report and model-comparison rows that compare the
  stratified-background continuous model against ridge, capped-weight continuous
  if available, expected-value hurdle, hard-gated hurdle, and reference
  baselines.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Add an explicit experiment block rather than silently changing
  `models.baselines`.
- Keep the active default sample path:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Define the training strata from retained-domain context, for example:
  `label_source`, `domain_mask_reason`, `depth_bin`, and `year`.
- Use stratum context only to compute fit weights or losses. Do not append CRM
  depth, depth bin, elevation, or mask reason to the model feature matrix.
- Keep evaluation metrics unweighted unless a table explicitly labels a weighted
  diagnostic row. Full-grid area calibration should be computed from row-level
  full-grid predictions.
- Preserve `features: A00-A63` and `target: kelp_fraction_y`.

## Plan/Spec Requirement

This is a new model objective and artifact family. Before implementation, write a
brief implementation plan that confirms:

- Whether the implementation extends baseline sidecars or creates a dedicated
  continuous-objective command.
- The exact stratum definition used for fit-weight construction.
- Whether Kelpwatch-supported rows are one stratum or are always kept at
  baseline weight `1.0`.
- How assumed-background strata are balanced, for example:

```text
each retained background stratum contributes the same total training weight per year
```

- Whether the objective combines stratum balancing with a global background cap.
  If yes, keep that cap fixed from the plan or select it on validation only.
- How stratum weights are kept separate from unweighted station metrics and
  full-grid area calibration.
- How predictions are clipped to `[0, 1]` before canopy-area summaries.
- Which comparison table becomes authoritative for P1-22b.
- How the task avoids selecting strata, weights, thresholds, or report choices on
  2022 test rows.

## Implementation Plan

- Load the current masked sample, split assignments, and configured AEF feature
  columns.
- Build training strata from retained-domain metadata already present in the
  sample.
- Derive a fit-weight column that balances assumed-background contribution across
  the configured strata while retaining Kelpwatch-supported rows.
- Fit a ridge regression on train rows with the stratified fit weights.
- Select any hyperparameters using validation rows only.
- Predict on sample rows and retained-domain full-grid rows for all configured
  years.
- Write row-level predictions with enough metadata to distinguish:
  `model_family`, `model_name`, `objective_policy`, `fit_weight_policy`,
  `stratum_columns`, `sample_policy`, `split`, `year`, `label_source`,
  `mask_status`, and `evaluation_scope`.
- Report station skill separately from background-inclusive sample diagnostics.
- Report assumed-background leakage for 2022 retained-domain rows.
- Report full-grid area calibration for the same scope as the current
  scoreboard.
- Update the Phase 1 report so P1-22b answers:
  - Does stratified-background weighting reduce retained-domain false positives?
  - Does it preserve Kelpwatch-station skill and high-canopy magnitude better
    than ridge?
  - Does it approach or beat the expected-value hurdle on RMSE, F1, and area
    calibration?

## Resolved Implementation Plan

- Extend the dedicated `train-continuous-objective` command from Task 36 with a
  new `stratified-background` experiment rather than changing
  `models.baselines`.
- Use configured stratum columns:
  `year`, `label_source`, `domain_mask_reason`, and `depth_bin`.
- Keep Kelpwatch-supported rows and any positive annual-max rows at baseline
  fit weight `1.0`.
- Balance only assumed-background rows. Within each year, each retained
  background stratum contributes the same total training weight:

```text
raw_weight = max(sample_weight, 1.0)
target_stratum_total_year = sum(raw_weight for background rows in year) / number_of_background_strata_in_year
fit_weight = raw_weight * target_stratum_total_year / sum(raw_weight in row stratum)
```

- Do not combine stratum balancing with a global background cap in this task.
  The capped-weight objective remains the Task 36 comparison row.
- Keep fit weights out of unweighted station metrics and full-grid area
  calibration. Full-grid area calibration remains computed from row-level
  retained-domain predictions clipped to `[0, 1]`.
- Treat `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
  as the authoritative P1-22b comparison table.
- Use only the 2021 validation split for ridge-alpha selection. Do not choose
  stratum columns, fit weights, thresholds, or report choices from 2022 test
  rows.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment stratified-background
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
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_area_calibration.csv'; print(pd.read_csv(p).to_string(index=False))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_assumed_background_leakage.csv'; print(pd.read_csv(p).to_string(index=False))"
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

- The stratified-background model is a direct continuous model, not a binary or
  hurdle model.
- The active annual-max target, mask, split, feature set, and sample policy are
  unchanged.
- Stratum definitions and fit weights are documented in config and the output
  manifest.
- CRM-derived context affects training weights only; it is not used as a
  predictor.
- Fit weights do not silently replace unweighted evaluation metrics.
- The report includes a stratified-background comparison row for 2022
  `full_grid_masked`.
- The report includes Kelpwatch-station skill, assumed-background leakage, and
  full-grid area calibration.
- The comparison clearly states whether the stratified-background model beats
  ridge, competes with the capped-weight continuous model, competes with the
  expected-value hurdle, or fails.
- No stratum policy, weight, threshold, or report choice is tuned on 2022 test
  rows.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not use CRM depth, elevation, depth bin, or mask reason as predictors.
- Do not change the plausible-kelp mask.
- Do not change the annual-max label input.
- Do not retrain binary presence, probability calibration, conditional canopy, or
  hurdle artifacts unless a later task explicitly asks for that.
- Do not claim independent ecological truth; this remains Kelpwatch-style
  weak-label modeling.
- Do not start full West Coast scale-up.

## Completion Notes

Completed on 2026-05-13.

Implemented as the `stratified-background` experiment on the existing
`kelp-aef train-continuous-objective` command. The fitted model is a direct ridge
regression on `A00-A63` and `kelp_fraction_y`; CRM-derived `domain_mask_reason`
and `depth_bin` are used only to construct fit weights.

The final configured stratum columns are:

```text
year, label_source, domain_mask_reason, depth_bin
```

Kelpwatch-supported and positive annual-max rows keep `fit_weight = 1.0`.
Assumed-background rows are balanced so each retained background stratum
contributes the same total fit weight within each year. The resulting real
sample check showed each background stratum-year total at `199999.0`, with no
global background cap. Ridge alpha was selected on the 2021 validation split
only; the selected alpha was `0.01`.

Generated artifacts:

- `/Volumes/x10pro/kelp_aef/models/continuous_objective/ridge_stratified_background.joblib`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/continuous_objective_stratified_background_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_metrics.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_area_calibration.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/continuous_objective_stratified_background_assumed_background_leakage.csv`
- `/Volumes/x10pro/kelp_aef/interim/continuous_objective_stratified_background_manifest.json`

Primary 2022 retained-domain result:

| Model | RMSE | R2 | F1 >=10% | Predicted area | Area bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| AEF ridge regression | 0.0452 | 0.587 | 0.476 | 8.42 M m2 | +102.1% |
| Capped-weight ridge | 0.0493 | 0.507 | 0.590 | 8.64 M m2 | +107.5% |
| Stratified-background ridge | 0.0542 | 0.405 | 0.694 | 5.85 M m2 | +40.4% |
| Expected-value hurdle | 0.0322 | 0.790 | 0.812 | 3.50 M m2 | -16.0% |

Interpretation: stratified-background weighting reduces retained assumed
background leakage to `3.80 M m2` and improves the 10% annual-max F1 relative to
ridge and capped-weight ridge, but it worsens full-grid RMSE and station RMSE
(`0.2360`). It fails to beat ridge or capped-weight ridge on the combined RMSE
and area-bias check and does not compete with the expected-value hurdle.

Validation passed:

```bash
uv run pytest tests/test_continuous_objective.py tests/test_model_analysis.py
uv run kelp-aef train-continuous-objective --config configs/monterey_smoke.yaml --experiment stratified-background
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
make check
```

## Sweep Addendum

Completed on 2026-05-13 after the first stratified-background result showed a
large station-skill penalty.

Durable write-up:
`docs/phase1_stratified_background_sweep_results.md`.

Added sweep controls to the stratified-background objective:

- `stratum_balance_gamma`: shrinks weights partway from raw sample weights
  toward equal background-stratum totals.
- `background_weight_budget_multiplier`: optionally caps total background fit
  weight per year as a multiple of Kelpwatch-supported rows.

Ran these sidecar experiments:

- `stratified-gamma-025`
- `stratified-gamma-050`
- `stratified-gamma-075`
- `stratified-gamma-025-bg5`
- `stratified-gamma-050-bg5`
- `stratified-gamma-050-bg2`

All sweep variants selected `alpha=0.01` on 2021 validation rows.

Primary 2022 retained-domain result:

| Model | RMSE | R2 | F1 >=10% | Predicted area | Area bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| AEF ridge regression | 0.0452 | 0.587 | 0.476 | 8.42 M m2 | +102.1% |
| Stratified gamma 0.25 + bg5 | 0.0472 | 0.549 | 0.503 | 8.23 M m2 | +97.8% |
| Stratified gamma 0.50 + bg5 | 0.0472 | 0.548 | 0.511 | 8.20 M m2 | +96.9% |
| Stratified gamma 0.50 + bg2 | 0.0466 | 0.560 | 0.449 | 8.55 M m2 | +105.5% |
| Stratified gamma 0.25 | 0.0532 | 0.426 | 0.674 | 5.93 M m2 | +42.5% |
| Stratified gamma 0.50 | 0.0535 | 0.419 | 0.680 | 5.91 M m2 | +41.9% |
| Stratified gamma 0.75 | 0.0538 | 0.412 | 0.686 | 5.88 M m2 | +41.2% |
| Full stratified background | 0.0542 | 0.405 | 0.694 | 5.85 M m2 | +40.4% |
| Expected-value hurdle | 0.0322 | 0.790 | 0.812 | 3.50 M m2 | -16.0% |

Interpretation: the sweep did not find a direct continuous stratified-background
variant that beats ridge or competes with the expected-value hurdle. Gamma-only
variants preserve the leakage improvement but keep the station/RMSE penalty.
Budgeted variants recover station skill, especially `gamma=0.50, bg2` with test
station RMSE `0.1641`, but they also restore the full-grid overprediction
failure (`+105.5%` area bias). The tradeoff appears structural for this
one-stage direct continuous objective under the current features and label
framing.
