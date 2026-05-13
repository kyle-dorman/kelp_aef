# Task 39: Close Phase 1 Model Policy And Report

## Goal

Close Monterey Phase 1 as the final Phase 1 task.

Select the best current Phase 1 model policy, or explicitly document that no
model policy is good enough to promote beyond Monterey annual-max feasibility.
The closeout should be a durable repo-tracked record of what Phase 1 proved,
what failed, and what remains unresolved, without adding new post-Phase-1 next
steps to the report.

The decision must stay framed as learning Kelpwatch-style annual-max labels from
AlphaEarth embeddings, not proving independent kelp biomass.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current active Phase 1 report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`
- Current authoritative comparison tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_calibration_metrics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv`
- Current Phase 1 historical records:
  - `tasks/36_test_capped_weight_continuous_model.md`
  - `tasks/37_test_stratified_background_continuous_model.md`
  - `tasks/38_remove_failed_p1_22_continuous_paths.md`
  - `docs/phase1_stratified_background_sweep_results.md`
- Core docs that must be brought into a coherent closeout state:
  - `docs/product.md`
  - `docs/architecture.md`
  - `docs/data_artifacts.md`
  - `docs/research_plan.md`
  - `docs/todo.md`
  - `docs/backlog.md`
  - `docs/phase1_model_domain_hardening.md`

## Outputs

- A final Phase 1 closeout section in the generated model-analysis report.
- A tracked Markdown snapshot of the final report, for example:
  `docs/report_snapshots/monterey_phase1_closeout_model_analysis.md`.
- A tracked decision note, for example:
  `docs/phase1_closeout_model_policy_decision.md`.
- Updated docs directory state:
  - Phase 1 marked closed in `docs/product.md`, `docs/research_plan.md`, and
    `docs/todo.md`.
  - Active-vs-historical wording cleaned up in `docs/data_artifacts.md`,
    `docs/architecture.md`, `docs/backlog.md`, and
    `docs/phase1_model_domain_hardening.md`.
  - Stale or duplicate Phase 1 planning docs either updated, archived as
    historical context, or removed if their content is preserved elsewhere.
- Removed unused config keys and dead code paths that are not part of the
  selected closeout policy or plausible next-phase starting point.
- Updated tests for the final report/report-snapshot behavior.

## Config File

Use `configs/monterey_smoke.yaml`.

The task may remove config keys only when they are no longer consumed by active
commands or by the final closeout report. It must not change:

- Monterey region and 2018-2022 years.
- Annual-max label input.
- `A00-A63` feature set.
- Year holdout split.
- Plausible-kelp retained-domain mask semantics.
- Default `crm_stratified_mask_first_sample` sample policy.
- Active ridge, binary-presence, calibrated-binary, conditional-canopy, and
  hurdle artifact paths unless they are deliberately relabeled as historical in
  the closeout.

## Plan/Spec Requirement

Before implementation, write a short closeout plan that lists:

- Candidate policies being compared:
  - Reference previous-year annual max.
  - AEF ridge regression.
  - Calibrated binary presence.
  - Positive-only conditional canopy.
  - Expected-value hurdle.
  - Hard-gated hurdle diagnostic.
- The exact comparison rows and metrics that will determine whether any AEF
  policy is selected.
- Which report sections will remain, be rewritten, or be removed.
- Which docs will become historical snapshots versus active project guidance.
- Which config/code paths are considered unused for the closeout state.
- How the final tracked report snapshot will be produced.

## Implementation Plan

- Query the authoritative Phase 1 comparison tables before editing prose.
- Decide the final Phase 1 policy from explicit evidence:
  - If expected-value hurdle is selected, say why it is the best AEF Phase 1
    candidate and where it still fails against previous-year persistence or
    high-canopy amount.
  - If no AEF policy is selected, say that directly and use the report to
    explain the failure.
- Rewrite the generated Phase 1 report as a closeout report:
  - Keep the current scope, data contract, selected policy, evidence table,
    successes, failures, and limitations.
  - Remove "next modeling task", "next steps", and open-ended future-work
    language from the report body.
  - Keep failures explicit: ridge area overprediction, direct-continuous P1-22
    failures, high-canopy underprediction, and the meaning of Kelpwatch-style
    weak labels.
  - Keep successes explicit: end-to-end AlphaEarth/Kelpwatch pipeline, domain
    mask, CRM-stratified mask-first sample policy, binary calibration,
    conditional canopy amount model, hurdle composition, and report harness.
- Generate the final Markdown/HTML/PDF report with `analyze-model`.
- Copy the final Markdown report into a tracked closeout snapshot under
  `docs/report_snapshots/`.
- Write or update the Phase 1 closeout decision note.
- Clean docs:
  - Mark Phase 1 closed and P1-23 complete in `docs/todo.md`.
  - Update product/research/architecture wording so Phase 1 is no longer
    described as active planning.
  - Move durable future-phase ideas to `docs/backlog.md` only if already present
    or necessary for context; do not add a "next steps" section to the report.
  - Remove stale duplicated Phase 1 planning text when it conflicts with the
    closeout decision.
- Clean active config/code:
  - Search for unused or stale Phase 1 keys and code paths after the selected
    policy is known.
  - Remove only clearly dead paths. Preserve historical task files and tracked
    report snapshots.
- Update tests so they assert the final report is a closeout artifact and no
  longer advertises post-Phase-1 next steps.

## Validation Command

Focused validation:

```bash
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
uv run pytest tests/test_model_analysis.py tests/test_package.py
```

Full validation:

```bash
make check
```

Manual checks:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv'; df=pd.read_csv(p); print(df[df['split'].eq('test') & df['year'].astype(str).eq('2022')].to_string(index=False))"
rg -n "next step|next task|future work|post-Phase|scale-up" /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md docs/report_snapshots/monterey_phase1_closeout_model_analysis.md
rg -n "Phase 1 is active|active planning|P1-23" docs
```

The second manual check should not find open-ended next-step language in the
final report snapshot. The third check should only find intentional historical
or completed-task references.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings, `A00-A63`.
- Reporting scope: retained plausible-kelp domain with
  `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- P1-23 is the last Phase 1 task in `docs/todo.md`, and Phase 1 is marked
  closed when the task is complete.
- The final report states a clear model-policy decision:
  - select a policy, or
  - explicitly document that no Phase 1 policy is good enough to promote.
- The decision is backed by exact 2022 retained-domain comparison rows and does
  not rely on chart impressions alone.
- The report discusses Phase 1 successes and failures in the body of the report.
- The report does not include a "next steps", "next modeling task", or
  post-Phase-1 work queue.
- A tracked Markdown snapshot of the final report exists under
  `docs/report_snapshots/`.
- The decision note and report snapshot are consistent with the generated
  external report.
- Docs no longer describe Phase 1 as active planning after closeout.
- Unused config keys and dead code paths not needed for the selected policy or
  likely next-phase starting point are removed.
- Historical P1-22 failure records are preserved.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not start a new model search.
- Do not change the annual-max label input.
- Do not tune on the 2022 test split.
- Do not add alternative temporal labels, full West Coast scale-up, spatial deep
  models, or new source ingestion work.
- Do not delete generated artifacts under `/Volumes/x10pro/kelp_aef` unless a
  separate explicit cleanup request asks for it.
- Do not claim independent ecological truth or real biomass validation from
  Kelpwatch-style weak labels.
- Do not erase task history. Historical task files and report snapshots should
  remain available for audit.
