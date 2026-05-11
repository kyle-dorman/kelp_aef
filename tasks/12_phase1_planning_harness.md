# Task 12: Phase 1 Planning And Report Harness

## Goal

Implement the Phase 1 planning-and-harness work from `docs/todo.md` P1-01
through P1-03 as one agent-sized task.

The goal is to make the existing `analyze-model` report a Phase 1-ready
comparison harness without changing model behavior. After this task, each later
baseline, domain-mask, or imbalance-aware model change should be visible in a
stable report structure that separates:

- Kelpwatch-station pixel skill.
- Background-inclusive sample behavior.
- Full-grid area calibration.
- Data health and row-count/drop-rate checks.
- Current and future mask status.

This task should update the report framing from Phase 0 branch selection to the
selected Phase 1 theme: Monterey annual-max model and domain hardening.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Phase 1 plan:
  `docs/phase1_model_domain_hardening.md`.
- Active Phase 1 checklist:
  `docs/todo.md`.
- Current model-analysis implementation:
  `src/kelp_aef/evaluation/model_analysis.py`.
- Current baseline metric implementation:
  `src/kelp_aef/evaluation/baselines.py`.
- Current model-analysis tests:
  `tests/test_model_analysis.py`.
- Annual labels:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Model input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`.
- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Baseline metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- Full-grid baseline predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`.
- Closed Phase 0 model-analysis report snapshot:
  `docs/report_snapshots/monterey_phase0_model_analysis.md`.

## Outputs

- Updated Phase 1 model-analysis report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.
- Updated standalone Phase 1 HTML report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html`.
- Updated standalone Phase 1 PDF report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf`.
- Updated model-analysis manifest:
  `/Volumes/x10pro/kelp_aef/interim/model_analysis_manifest.json`.
- New compact model-comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`.
- New data-health table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_data_health.csv`.
- Updated unit tests for the new report tables and report text.
- Updated `docs/todo.md` checkboxes for P1-01, P1-02, and P1-03 after the task
  is implemented and validated.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions under `reports.outputs`:

```yaml
model_analysis_report: /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
model_analysis_html_report: /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html
model_analysis_pdf_report: /Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf
model_analysis_phase1_model_comparison: /Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv
model_analysis_data_health: /Volumes/x10pro/kelp_aef/reports/tables/model_analysis_data_health.csv
```

The active report paths now use the Phase 1 filename. The closed Phase 0 report
is preserved in `docs/report_snapshots/monterey_phase0_model_analysis.md`.

## Plan/Spec Requirement

No separate decision note is required before implementation. This task file is
the implementation plan.

Before editing code, confirm the current `model_analysis.py` report flow and
test fixture still match this task contract. If the current implementation has
already added equivalent tables or fields, reuse the existing surfaces instead
of adding duplicates.

## Implementation Notes

Add a new Phase 1 harness section to the report without changing model training,
prediction generation, or the selected annual-max label target.

Minimum model-comparison CSV fields:

- `model_name`
- `split`
- `year`
- `mask_status`
- `evaluation_scope`
- `label_source`
- `row_count`
- `mae`
- `rmse`
- `r2`
- `spearman`
- `f1_ge_10pct`
- `observed_canopy_area`
- `predicted_canopy_area`
- `area_pct_bias`

Minimum data-health CSV fields:

- `check_name`
- `split`
- `year`
- `label_source`
- `row_count`
- `reference_row_count`
- `rate`
- `detail`

Initial `mask_status` should be `unmasked` for all rows. The field exists so
future bathymetry/DEM domain-filter tasks can add `masked` or mask-specific
rows without changing the table contract.

Initial `evaluation_scope` values should include:

- `kelpwatch_station_sample` for Kelpwatch-supported metric rows.
- `background_inclusive_sample` for overall sampled metric rows.
- `full_grid_prediction` for full-grid prediction calibration rows.

The data-health table should include, at minimum:

- Annual label row counts by year.
- Model-input sample row counts by year and label source.
- Split-manifest retained and dropped counts by split/year.
- Missing-feature drop rates from `used_for_training_eval`.
- Prediction row counts by split/year and label source.
- Primary report split/year row counts used by `analyze-model`.

Update report text so:

- The executive summary frames the report as the active Phase 1 hardening
  report, rather than recommending a branch.
- The active Phase 1 theme is annual-max model and domain hardening.
- Phase 0 branch-selection notes are removed from the active report.
- A new `Phase 1 Harness Status` section links the two new tables and explains
  what future tasks should add to them.

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

No `train-baselines`, `predict-full-grid`, or `map-residuals` rerun is required
unless the existing report inputs are missing or stale. This task should only
reshape analysis/report outputs from existing inputs.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split policy: train 2018-2020, validation 2021, test 2022.
- Primary report split/year: test 2022.
- Label target: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.

## Acceptance Criteria

- `analyze-model` writes the two new CSV tables.
- The compact model-comparison table includes current `no_skill_train_mean` and
  `ridge_regression` rows without changing existing metric values.
- The compact model-comparison table includes a full-grid calibration row for
  the current ridge predictions.
- The data-health table reports row counts and missing-feature drop rates from
  labels, model input, split manifest, and predictions.
- The report includes a `Phase 1 Harness Status` section.
- The report clearly states the selected Phase 1 theme and omits the old
  Phase 0 branch-selection section.
- Existing ridge/no-skill reference results are preserved.
- The HTML and PDF reports include the compact model-comparison content, not
  just the separate CSV table.
- `tests/test_model_analysis.py` covers the new outputs.
- Focused validation and `make check` pass.
- P1-01, P1-02, and P1-03 are marked complete in `docs/todo.md` only after
  implementation and validation succeed.

## Known Constraints Or Non-Goals

- Do not change model training behavior.
- Do not change full-grid prediction behavior.
- Do not add previous-year, climatology, or geographic baselines in this task.
- Do not add bathymetry/DEM inputs or masks in this task.
- Do not choose a binary annual-max threshold in this task.
- Do not evaluate alternative temporal label inputs beyond annual max.
- Keep the closed Phase 0 report as a repo snapshot instead of overwriting it
  with active Phase 1 report content.
- Do not tune on the 2022 test split.
