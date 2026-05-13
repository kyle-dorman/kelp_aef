# Task 34: Retire Sampling-Policy Sidecar Reporting

## Goal

Remove the temporary current-vs-CRM sampling-policy comparison section from the
active Phase 1 report after CRM-stratified, mask-first sampling becomes the
default policy.

Task 31 needed a side-by-side report section because two sample policies were
active at once. After Task 32 documents the decision and Task 33 promotes the
policy into the default path, the active report should focus on the selected
default model artifacts rather than continuing to show a sidecar experiment as
an ongoing comparison.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Task 32 decision note:
  `docs/phase1_crm_stratified_sampling_policy_decision.md`.
- Task 33 regenerated default artifacts and report outputs.
- Current report generator:
  `src/kelp_aef/evaluation/model_analysis.py`.
- Current report tests:
  `tests/test_model_analysis.py`.
- Temporary sidecar comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`.
- Active Phase 1 report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.

## Outputs

- Updated report-generation code that no longer renders a recurring
  current-vs-CRM sidecar section for the default Monterey Phase 1 report.
- Updated tests that remove sidecar-section expectations or convert them into
  narrow legacy helper tests if the code is retained for audit.
- Updated report language that identifies the active default sampling policy in
  normal model-comparison, data-health, or methods/context sections.
- Regenerated Phase 1 report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`
- Optional archival pointer from the report to the Task 32 decision note if a
  reader needs the retired side-by-side evidence.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config state:

- The promoted CRM-stratified policy is the default sample producer.
- Sidecar paths are not required for the main report.
- If sidecar path fields remain temporarily, they should be marked as archival
  or migration-only and should not drive headline report sections.

## Plan/Spec Requirement

This is a report cleanup task after a policy promotion. Before implementation,
write a brief plan that confirms:

- Task 32's decision note exists and contains the side-by-side evidence.
- Task 33 has regenerated default artifacts under the promoted policy.
- Which report helper functions and tests are sidecar-only and can be removed.
- Whether the CSV comparison table remains generated for audit or is retired.
- Where the active default sampling policy is described in the report after the
  temporary section is removed.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Full validation:

```bash
make check
```

Manual inspection:

```bash
rg -n "CRM-Stratified Sampling Policy Comparison|Current masked sample|CRM-stratified background" /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
```

The grep should not find the retired temporary section in the active report.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Binary target: `annual_max_ge_10pct`.
- Reporting scope: retained plausible-kelp domain with the promoted default
  sampling policy.

## Acceptance Criteria

- The active Phase 1 report no longer includes the temporary side-by-side
  CRM-stratified sampling-policy section.
- The default model-comparison sections still show the selected model behavior,
  pixel skill, area calibration, and background leakage.
- The report or a methods/context section states that default training uses the
  promoted CRM-stratified, mask-first retained-domain sample.
- Historical metric evidence remains available in the Task 32 decision note.
- Tests pass for the updated report-generation behavior.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not remove the design decision note or historical evidence.
- Do not delete generated sidecar artifacts from `/Volumes/x10pro/kelp_aef`
  unless a later generated-artifact cleanup task explicitly requests that.
- Do not change model training behavior in this cleanup task.
- Do not start full West Coast scale-up.
