# Task 65: Add False-Negative Probability-Margin Diagnostics

## Goal

Explain whether false negatives are near the calibrated binary decision boundary
or are genuinely low-confidence misses.

New positives and one-quarter spikes may sit just below the selected probability
threshold. If so, the issue may be a calibration or threshold tradeoff. If they
are far below threshold, the issue is more likely feature ambiguity, label
ambiguity, or missing temporal/ecological signal. This task must quantify the
margin, not just count FNs.

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Existing calibrated binary prediction artifacts and threshold metadata.
- Existing temporal context fields, including persistence and previous-year
  annual-max class, from the Phase 2 diagnostic pipeline.
- Existing Phase 2 diagnostic frame cache if fresh:
  `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
training_regime = pooled_monterey_big_sur
evaluation_region in {big_sur, monterey}
binary_outcome = FN
```

## Outputs

Write report-visible sidecar tables under
`/Volumes/x10pro/kelp_aef/reports/tables/`, with paths recorded in the Phase 2
manifest. Preferred names:

- `monterey_big_sur_fn_probability_margin_by_persistence.csv`
- `monterey_big_sur_fn_probability_margin_by_previous_year.csv`
- `monterey_big_sur_fn_probability_margin_summary.csv`

Minimum columns:

- `evaluation_region`;
- `training_regime`;
- `context_type`, such as `persistence` or `previous_year_class`;
- `context_value`;
- `fn_count`;
- `observed_positive_count`;
- `fn_share_of_observed_positive`;
- `probability_threshold`;
- `mean_calibrated_probability`;
- `median_calibrated_probability`;
- `p10_calibrated_probability`;
- `p25_calibrated_probability`;
- `p75_calibrated_probability`;
- `p90_calibrated_probability`;
- `mean_probability_margin`;
- `median_probability_margin`;
- `near_threshold_fn_count`;
- `near_threshold_fn_share`;
- `far_below_threshold_fn_count`;
- `far_below_threshold_fn_share`;
- observed canopy bins or summary quantiles for the same rows.

`probability_margin` must be defined as:

```text
calibrated_presence_probability - probability_threshold
```

FN margins are expected to be negative. The task must bin FNs by distance below
threshold, with a documented near-threshold band such as `[-0.05, 0)` or another
config-backed value.

Regenerate the Phase 2 Markdown/HTML/PDF report so the report states whether FNs
in persistence and previous-year classes are mostly near-threshold or
low-confidence misses.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add output
paths and any near-threshold margin setting under the existing Phase 2 report or
diagnostics blocks.

## Plan / Spec Requirement

Before implementation, add a short implementation note to this task file or the
PR/commit message that confirms:

- exact probability column and threshold source;
- exact margin definition and near-threshold band;
- exact persistence classes;
- exact previous-year annual-max classes;
- how one-quarter spike and new-positive rows are represented;
- how the report will distinguish calibration/threshold tradeoff from
  low-confidence ambiguity without changing the threshold.

## Validation Command

Use cached diagnostics when valid:

```bash
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
```

If diagnostic frames are stale or missing:

```bash
uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
```

For code changes, also run:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

## Smoke-Test Region And Years

- Regions: Big Sur and Monterey Phase 2 retained-domain rows.
- Year: held-out `2022`.
- Primary context: pooled Monterey+Big Sur evaluated separately on Big Sur and
  Monterey.

## Acceptance Criteria

- FN probability margins are reported by persistence class and previous-year
  annual-max class.
- New-positive and one-quarter spike cases are explicitly visible in the output
  or called out as unavailable with a concrete reason.
- The report states whether FNs are mostly near-threshold or far below the
  calibrated threshold.
- The margin definition and threshold source are recorded in the manifest or
  report.
- No calibration, threshold, model, label, mask, or sample policy is changed.

## Known Constraints And Non-Goals

- Do not retune the binary threshold from held-out margin diagnostics.
- Do not reclassify FNs as acceptable because they are near threshold; this task
  only diagnoses the tradeoff.
- Do not collapse persistence and previous-year diagnostics into a single
  unlabeled temporal bucket.
- Do not use these diagnostics to choose a Phase 3 model policy without a
  separate decision task.
