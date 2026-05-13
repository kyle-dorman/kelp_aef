# Task 38: Remove Failed P1-22 Direct-Continuous Code Paths

## Goal

Remove the two failed P1-22 direct-continuous model code paths from the active
pipeline:

- P1-22a capped-weight direct continuous ridge, including the cap sweep.
- P1-22b stratified-background direct continuous ridge, including the gamma and
  background-budget sweep.

Both paths were useful negative tests, but neither produced a competitive Phase
1 model. After the results are recorded in the task history and sweep note, the
active codebase should not keep these experimental training/reporting paths as
maintained pipeline stages.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- P1-22a record: `tasks/36_test_capped_weight_continuous_model.md`.
- P1-22b record: `tasks/37_test_stratified_background_continuous_model.md`.
- Sweep result note:
  `docs/phase1_stratified_background_sweep_results.md`.
- Active report artifacts, especially:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`

## Outputs

- Removed active CLI/code path for `kelp-aef train-continuous-objective`.
- Removed `models.continuous_objective` experiment config blocks from
  `configs/monterey_smoke.yaml`.
- Removed active model-analysis ingestion and report sections for
  `model_family = continuous_objective`.
- Removed or rewritten tests that only validate the failed continuous-objective
  implementation.
- Updated artifact documentation so P1-22 continuous-objective artifacts are
  historical result artifacts, not active maintained outputs.
- Refreshed Phase 1 report and comparison table without active capped-weight,
  cap-sweep, stratified-background, or stratified-sweep rows.

## Config File

Use `configs/monterey_smoke.yaml`.

Remove the active `models.continuous_objective` block and all P1-22 experiment
entries, including:

- `capped-weight`
- `cap-1`
- `cap-2`
- `cap-10`
- `cap-20`
- `cap-100`
- `stratified-background`
- `stratified-gamma-025`
- `stratified-gamma-050`
- `stratified-gamma-075`
- `stratified-gamma-025-bg5`
- `stratified-gamma-050-bg5`
- `stratified-gamma-050-bg2`

Do not change the active annual-max label, retained-domain mask, default sample
policy, binary model, conditional canopy model, hurdle model, ridge baseline, or
reference baselines.

## Plan/Spec Requirement

Before implementation, write a short removal plan that lists:

- Exact source files and config blocks to delete.
- Any shared helpers that must stay because other model/report paths use them.
- How the report should refer to P1-22 after removal, if at all.
- How generated `/Volumes/x10pro/kelp_aef` artifacts will be treated. Default:
  do not delete generated model artifacts unless explicitly requested.
- Validation commands and expected report/table checks.

## Implementation Plan

- Remove the `train-continuous-objective` CLI command wiring from
  `src/kelp_aef/cli.py` and command-list tests.
- Delete `src/kelp_aef/evaluation/continuous_objective.py` if no active code
  imports it after CLI removal.
- Remove continuous-objective artifact discovery, comparison-row creation,
  sampling rows, and report narrative from
  `src/kelp_aef/evaluation/model_analysis.py`.
- Remove `models.continuous_objective` from `configs/monterey_smoke.yaml`.
- Remove `tests/test_continuous_objective.py`.
- Update `tests/test_model_analysis.py` so it no longer expects
  continuous-objective rows or report sections.
- Update docs that currently describe P1-22 artifacts as active outputs. Keep
  the negative-result records in `tasks/36...`, `tasks/37...`, and
  `docs/phase1_stratified_background_sweep_results.md`.
- Regenerate the Phase 1 report and verify the retained-domain scoreboard still
  compares the active ridge/reference/binary/conditional/hurdle paths correctly.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_package.py tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual checks:

```bash
rg -n "train-continuous-objective|models.continuous_objective|continuous_objective|capped-weight|stratified-background" src tests configs/monterey_smoke.yaml
rg -n "capped-weight|stratified-background|continuous_objective" /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md /Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv
```

The first manual check should have no active source/config/test hits except any
deliberate historical references. The second check may include historical report
text only if the implementation intentionally keeps a brief negative-result
note; it should not include active scoreboard rows for the failed P1-22 models.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings, `A00-A63`.
- Reporting scope: retained plausible-kelp domain with
  `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- `kelp-aef train-continuous-objective` is no longer an available command.
- `configs/monterey_smoke.yaml` has no active `models.continuous_objective`
  block or P1-22 experiment entries.
- Active model-analysis code no longer reads continuous-objective artifacts or
  adds capped-weight, cap-sweep, stratified-background, or stratified-sweep rows
  to comparison tables.
- The refreshed Phase 1 report does not present the failed P1-22 direct
  continuous models as active maintained candidates.
- Historical records remain available in:
  - `tasks/36_test_capped_weight_continuous_model.md`
  - `tasks/37_test_stratified_background_continuous_model.md`
  - `docs/phase1_stratified_background_sweep_results.md`
- Generated `/Volumes/x10pro/kelp_aef` model and table artifacts are not deleted
  unless a separate explicit cleanup task requests artifact deletion.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not remove or change the standard AEF ridge baseline.
- Do not remove or change binary presence, probability calibration,
  conditional canopy, or hurdle model code.
- Do not change the plausible-kelp mask, annual-max target, train/test split,
  feature set, or default sample policy.
- Do not erase the P1-22 negative-result documentation.
- Do not start a new model search in this cleanup task.
- Do not delete generated artifacts under `/Volumes/x10pro/kelp_aef` without an
  explicit follow-up request.
