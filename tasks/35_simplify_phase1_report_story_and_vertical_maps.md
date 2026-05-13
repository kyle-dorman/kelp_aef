# Task 35: Simplify Phase 1 Report Story And Vertical Maps

## Goal

Refactor the active Monterey Phase 1 model-analysis report from a chronological
task log into a clear decision report, and make the report map figures easier to
read by stacking map panels vertically instead of horizontally.

The report has accumulated useful diagnostics while Phase 1 work progressed, but
the main story is now harder to read than the evidence warrants. The next pass
should preserve the generated artifacts and audit tables while making the report
answer:

- What is the current default data/model policy?
- Which model is strongest right now?
- What improved relative to ridge and reference baselines?
- Where is the current modeling effort still failing?
- What should the next modeling task focus on?

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Active report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`
- Current report generator:
  `src/kelp_aef/evaluation/model_analysis.py`.
- Current map generators used by the report:
  - `src/kelp_aef/viz/residual_maps.py`
  - `src/kelp_aef/evaluation/hurdle.py`
  - `src/kelp_aef/evaluation/binary_presence.py`
- Current report tests:
  - `tests/test_model_analysis.py`
  - `tests/test_residual_maps.py`
  - `tests/test_hurdle.py`
  - `tests/test_binary_presence.py`
- Core tables to preserve as evidence:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv`

## Outputs

- A shorter, clearer Phase 1 report with a top-level structure close to:
  1. Executive Summary
  2. Current Default Policy And Data Scope
  3. 2022 Retained-Domain Model Scoreboard
  4. What Improved Since Ridge
  5. Remaining Failure Modes
  6. Decision / Next Modeling Step
  7. Appendix
- A compact main scoreboard table for the primary `test` / `2022` /
  `full_grid_masked` retained-domain scope. Include at least:
  - previous-year annual max;
  - grid-cell climatology;
  - AEF ridge regression;
  - expected-value hurdle;
  - hard-gated hurdle.
- Main-report prose that states the current modeling interpretation:
  - CRM-stratified, mask-first retained-domain sampling is the default policy.
  - Ridge still leaks too much full-grid background area.
  - Calibrated binary presence is strong for the Kelpwatch-style
    `annual_max_ge_10pct` target.
  - The expected-value hurdle is the main AEF full-grid candidate right now.
  - High-canopy amount prediction remains the main weakness.
  - Previous-year persistence remains a strong benchmark.
- Regenerated map figures with vertical panel layout for report-embedded maps:
  - ridge observed / predicted / residual map;
  - hurdle observed / predicted / residual map;
  - binary-presence diagnostic map panels.
- Updated tests for the new report structure and map orientation.
- Regenerated report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`

## Config File

Use `configs/monterey_smoke.yaml`.

Do not change model inputs, sampling policy, thresholds, or fitted artifacts in
this task. This is a reporting and visualization clarity pass.

## Plan/Spec Requirement

This is a multi-file reporting change. Before editing, write a brief
implementation plan that confirms:

- Which current report sections will remain in the main body.
- Which sections will be shortened, merged, or moved to appendix.
- The exact scoreboard rows and metrics used for the primary 2022 retained
  scope.
- Which map figures will switch from horizontal to vertical panels.
- How image-orientation tests will verify the regenerated maps are taller and
  more readable instead of only changing report text.

## Suggested Report Cleanup

Keep the main report focused on decisions, not task chronology.

Recommended main-body changes:

- Replace the current executive summary with a concise current-state summary.
- Replace the separate `Model Comparison` and `Reference Baseline Ranking`
  narrative with one primary scoreboard plus a short reference-baseline note.
- Keep `Observed, Predicted, And Error Map`, but show a larger vertical map.
- Keep one compact residual/failure-mode section focused on background leakage
  and high-canopy underprediction.
- Keep binary and hurdle content only where it explains the current candidate
  model; move threshold-selection and calibration process detail to appendix.
- Move or shorten `Data Health And Label Distribution`,
  `Quarter And Season Grounding`, `Class And Target Balance`,
  `Annual-Max Binary Threshold Comparison`, `Balanced Binary Presence Model`,
  `Calibrated Binary Presence Probabilities`, and `Conditional Canopy Amount
  Model`.
- Remove stale wording in `Phase 1 Coverage Gaps` and `Interpretation` that
  still describes already-completed Phase 1 work as future work.

## Map Layout Requirement

The report-embedded maps should be legible in Markdown, HTML, and PDF without
requiring the reader to zoom into a wide strip of small panels.

For the report map figures:

- stack observed, predicted, and residual panels vertically;
- use a taller figure size with stable colorbar placement;
- preserve shared spatial extents and the existing observed/predicted/residual
  semantics;
- keep `observed - predicted` as the residual sign convention;
- avoid changing the underlying prediction values or map filtering;
- update tests to confirm regenerated report map PNG dimensions are vertical
  for the stacked-panel figures.

For the binary-presence map, use a similar vertical or two-column layout that
makes each panel materially larger in the report. If the four binary panels do
not read well as a single vertical stack, use a 2x2 layout only if the rendered
panel size is demonstrably larger than the current horizontal strip.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py tests/test_residual_maps.py tests/test_hurdle.py tests/test_binary_presence.py
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual report review:

```bash
rg -n "^## " /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
rg -n "2022 Retained-Domain Model Scoreboard|Remaining Failure Modes|Decision / Next Modeling Step" /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
```

Manual visual checks should open or inspect the regenerated PNG dimensions for:

```text
/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.masked.png
/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_2022_map.png
/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`.
- Reporting scope: retained plausible-kelp domain with the default
  `crm_stratified_mask_first_sample` policy.

## Acceptance Criteria

- The main report reads as a current-state decision report, not a sequential
  task diary.
- The main report has a single clear primary scoreboard for the 2022
  retained-domain model comparison.
- Stale “future work” wording is removed or rewritten to reflect implemented
  Phase 1 reality.
- Detailed threshold-selection, calibration, data-health, and artifact-index
  material remains available, but no longer dominates the main body.
- The report explicitly states what is going well and where the current model
  still struggles.
- Report-embedded map panels are materially larger and vertically stacked or
  otherwise no longer rendered as thin horizontal strips.
- Tests cover the report structure and map orientation.
- Regenerated Markdown, HTML, and PDF reports exist.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not retrain or retune model artifacts except where existing commands
  regenerate unchanged report figures.
- Do not change the current default sampling policy.
- Do not tune thresholds or quota values on the 2022 test split.
- Do not remove generated audit CSVs from `/Volumes/x10pro/kelp_aef`.
- Do not claim independent ecological truth; all conclusions remain
  Kelpwatch-style weak-label results.
- Do not start full West Coast scale-up.
