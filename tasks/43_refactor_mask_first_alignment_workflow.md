# Task 43: Refactor Mask-First Alignment Workflow

## P2 Mapping

P2-03a: Refactor the alignment workflow so the masked model-input path is the
native path, then verify the full Monterey pipeline end to end.

## Goal

Make the alignment scripts match the artifact dependency graph we actually use:
build a full-grid target artifact first, build or refresh the domain mask from
that full grid, then build the retained-domain CRM-stratified model-input
sample from the full grid plus mask.

Do not solve this by adding compatibility flags such as
`--skip-domain-sample` or `--domain-sample-only`. Backward compatibility with
old unmasked Monterey artifacts is not required; Monterey can be rebuilt.

## Inputs

- Active checklist: `docs/todo.md`.
- Monterey config: `configs/monterey_smoke.yaml`.
- Current full-grid alignment implementation:
  `src/kelp_aef/alignment/full_grid.py`.
- Current CRM/domain-mask implementations:
  `src/kelp_aef/domain/crm_alignment.py`.
  `src/kelp_aef/domain/domain_mask.py`.
- Current CLI wiring: `src/kelp_aef/cli.py`.
- Current tests:
  - `tests/test_full_grid_alignment.py`.
  - `tests/test_crm_alignment.py`.
  - `tests/test_domain_mask.py`.
  - `tests/test_baselines.py`.
  - `tests/test_binary_presence.py`.
  - `tests/test_conditional_canopy.py`.
  - `tests/test_hurdle.py`.
  - `tests/test_model_analysis.py`.
- Canonical artifact root: `/Volumes/x10pro/kelp_aef`.

## Outputs

Code/config outputs:

- Refactored package-backed CLI commands and helpers that make the native order:
  1. full-grid alignment,
  2. CRM alignment,
  3. domain mask,
  4. retained-domain model-input sample.
- Monterey config updated to use the refactored command/data contract.
- Tests updated to cover the new command boundaries and mask-first sample
  contract.

Refreshed Monterey artifact outputs:

- Full-grid alignment:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Full-grid manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json`.
- Aligned CRM:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- Plausible-kelp mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Mask-first CRM-stratified model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Refreshed Monterey model, prediction, residual, visualizer, and report
  artifacts produced by the downstream Phase 1 commands.

## Config File

Use:

```bash
configs/monterey_smoke.yaml
```

## Implementation Plan

1. Inspect current alignment command boundaries:
   - Confirm where `align-full-grid` writes full-grid rows, legacy background
     sample rows, and configured domain-mask sidecars.
   - Confirm downstream commands read the masked sample by default.
   - Identify any tests that encode the old coupled behavior.
2. Refactor command semantics:
   - Make full-grid alignment responsible for full-grid features, labels,
     identifiers, row counts, and full-grid manifest only.
   - Move retained-domain sample construction into a separate script/helper/CLI
     path with a clear name, such as `build-model-input-sample`, unless a
     better repo-local naming pattern is already present.
   - Have the sample builder consume the full-grid table and plausible-kelp mask
     table, then write the configured CRM-stratified mask-first sample and its
     manifest/summary.
   - Remove the old implicit sidecar behavior from `align-full-grid` rather
     than adding flags to avoid it.
3. Update the config contract:
   - Keep the masked sample as the default model input.
   - Move sample-builder settings out of any place that implies the sample is a
     side effect of full-grid alignment if that makes the contract clearer.
   - Preserve the artifact paths downstream commands already consume unless the
     refactor requires a deliberate rename.
4. Update tests:
   - Test that full-grid alignment does not require an existing mask.
   - Test that the model-input sample builder requires a mask and writes the
     CRM-stratified retained-domain sample.
   - Test dropped-positive reporting and sample-weight recomputation against
     the retained mask domain.
   - Update downstream fixtures only where they encode the old sidecar
     behavior.
5. Rerun the full Monterey workflow:
   - Rebuild or refresh all required inputs and downstream outputs from the
     refactored commands.
   - Rerun the full model/report path so any contract break is caught on real
     artifacts, not only fixtures.
6. Record the outcome in this task file:
   - Refactored command names and artifact order.
   - Monterey full-grid row counts by year and label source.
   - Monterey mask retained/dropped counts and dropped-positive counts.
   - Monterey masked sample counts by year, label source, mask reason, and
     split where available.
   - Downstream model/report commands rerun and their output paths.
   - Any report metric changes that result only from the refactor.
7. Update `docs/todo.md` when the real Monterey rerun passes.

## Suggested Command Order

Exact command names may change in this task. The intended order is:

```bash
uv run kelp-aef build-labels --config configs/monterey_smoke.yaml
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml
uv run kelp-aef build-model-input-sample --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Validation Command

Code validation:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Full Monterey artifact validation:

```bash
make check
```

Manual artifact inspection examples:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/aligned_full_grid_training_table_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.masked_summary.csv'; print(pd.read_csv(p).to_string())"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label target: Kelpwatch annual max canopy.
- Mask: configured Monterey `plausible_kelp_domain`.
- Model-input policy: `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- The native workflow no longer treats the retained-domain sample as a hidden
  side effect of full-grid alignment.
- `align-full-grid` can run from labels and AEF assets without requiring an
  existing domain mask.
- The retained-domain model-input sample is produced by an explicit
  package-backed script/helper/CLI path after the mask exists.
- Monterey artifacts are rebuilt through the full downstream path listed above.
- Row counts, retained/dropped mask counts, dropped-positive counts, sample
  weights, and model-input sample counts are reportable from generated
  artifacts.
- Tests cover the new command boundaries and pass.
- No Big Sur model training, model evaluation, or performance interpretation is
  included in this task.

## Known Constraints And Non-Goals

- Do not add staging flags as the primary solution.
- Do not preserve old unmasked sample sidecar behavior for compatibility.
- Do not change the annual-max label target.
- Do not add CRM, depth, mask reason, or other domain variables as model
  predictors.
- Do not start Big Sur artifact generation until this Monterey refactor task is
  complete.

## Outcome

Completed 2026-05-14.

The native Monterey workflow now runs as:

1. `align-full-grid`: writes only the full-grid alignment table, summary, and
   manifest.
2. `align-noaa-crm`: aligns CRM values to the full-grid target cells.
3. `build-domain-mask`: writes the plausible-kelp retained-domain mask.
4. `build-model-input-sample`: consumes the full-grid table plus the domain mask
   and writes the CRM-stratified mask-first model-input sample.

The retained-domain sample is no longer a hidden side effect of
`align-full-grid`. The sample settings moved to top-level `model_input_sample`
config in `configs/monterey_smoke.yaml` and `configs/big_sur_smoke.yaml`, and
`build-model-input-sample` is package-backed through the `kelp-aef` CLI.

Refreshed Monterey full-grid alignment counts:

| Year | Label source | Rows | Complete feature rows | Missing feature rows | Observed canopy area |
| --- | --- | ---: | ---: | ---: | ---: |
| 2018 | assumed_background | 7,428,207 | 7,082,546 | 345,661 | 0 |
| 2018 | kelpwatch_station | 30,154 | 30,154 | 0 | 5,671,776 |
| 2019 | assumed_background | 7,428,207 | 7,082,546 | 345,661 | 0 |
| 2019 | kelpwatch_station | 30,154 | 30,154 | 0 | 2,215,941 |
| 2020 | assumed_background | 7,428,207 | 7,082,546 | 345,661 | 0 |
| 2020 | kelpwatch_station | 30,154 | 30,154 | 0 | 3,168,796 |
| 2021 | assumed_background | 7,428,207 | 7,082,546 | 345,661 | 0 |
| 2021 | kelpwatch_station | 30,154 | 30,154 | 0 | 3,501,294 |
| 2022 | assumed_background | 7,428,207 | 7,082,546 | 345,661 | 0 |
| 2022 | kelpwatch_station | 30,154 | 30,154 | 0 | 4,163,014 |

Refreshed Monterey mask counts:

| Mask class | Cells | Fraction |
| --- | ---: | ---: |
| retained overall | 630,151 | 0.084489 |
| dropped overall | 6,828,210 | 0.915511 |
| retained ambiguous coast | 160,630 | 0.021537 |
| retained depth 0-60 m | 469,521 | 0.062952 |
| dropped definite land | 5,692,768 | 0.763273 |
| dropped deep water | 1,135,442 | 0.152237 |

The model-input sample manifest records `command = build-model-input-sample`,
`artifact = model_input_sample_domain_mask`,
`masked_sample_policy = crm_stratified_mask_first_sample`,
`mask_first = true`, `population_row_count = 3,150,755`,
`sample_row_count = 311,475`, `mask_dropped_row_count = 34,141,050`, and
`mask_dropped_positive_row_count = 0`.

The refreshed sample has 62,295 rows per year. Within each year:

| Label source | Mask reason | Depth bin | Sample rows | Population rows |
| --- | --- | --- | ---: | ---: |
| assumed_background | retained_ambiguous_coast | ambiguous_coast | 18,981 | 158,170 |
| assumed_background | retained_depth_0_60m | 0_40m | 11,655 | 291,362 |
| assumed_background | retained_depth_0_60m | 40_60m | 1,505 | 150,465 |
| kelpwatch_station | retained_ambiguous_coast | ambiguous_coast | 2,460 | 2,460 |
| kelpwatch_station | retained_depth_0_60m | 0_40m | 27,692 | 27,692 |
| kelpwatch_station | retained_depth_0_60m | 40_60m | 2 | 2 |

The sample artifact does not carry a `split` column; the configured split is
assigned downstream by year: 2018-2020 train, 2021 validation, and 2022 test.
Sample weights are recomputed in the retained domain: observed Kelpwatch rows
stay at `1.0`, background ambiguous-coast rows are weighted about `8.333`,
background 0-40 m rows about `24.999`, and background 40-60 m rows about
`99.977`.

The full Monterey workflow was rerun through:

- `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.
- `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`.

No model interpretation was changed as part of this refactor. The refreshed
2022 `full_grid_masked` report rows for `label_source = all` include:

| Model | Rows | RMSE | F1 ge 10pct | Predicted area | Area pct bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| ridge_regression | 630,151 | 0.045152 | 0.475989 | 8,415,321 | 1.021449 |
| previous_year_annual_max | 630,151 | 0.034070 | 0.797394 | 3,501,294 | -0.158952 |
| grid_cell_climatology | 630,151 | 0.042899 | 0.707638 | 3,685,504 | -0.114703 |
| calibrated_hard_gate_conditional_canopy | 630,151 | 0.033617 | 0.825024 | 3,494,314 | -0.160629 |
| calibrated_probability_x_conditional_canopy | 630,151 | 0.032162 | 0.812370 | 3,495,891 | -0.160250 |

Validation passed:

- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest` (`101 passed`)

`make check` still stops at `uv run ruff format --check .` because the
pre-existing, untouched file `src/kelp_aef/viz/source_coverage.py` needs Ruff
formatting. The touched files were format-checked separately after
`tests/test_full_grid_alignment.py` was reformatted.
