# Task 63: Add Crossed Context Diagnostics For Pooled Failures

## Goal

Replace one-way context summaries with explicit crossed-bin diagnostics for the
failure modes the current report cannot answer.

The one-way pooled plots are useful, but the likely failures are conditional:
high annual max plus intermittent or persistent temporal context, high annual
max near mat edges, and observed canopy interacting with shallow CRM depth. This
task must cross the relevant bins instead of summarizing each context axis
separately.

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Required Crosses

Implement all of these as separate output tables or one table with an explicit
`cross_type` column:

1. Observed annual canopy bin x temporal persistence.
2. Observed annual canopy bin x edge class.
3. CRM depth bin x observed annual canopy bin.
4. Observed annual canopy bin x fine CRM depth bin.

The report should interpret the conditional patterns directly. It is not enough
to link the existing one-way pooled context figure.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Existing pooled context diagnostics:
  `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_binary_context_diagnostics.csv`
  and
  `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_amount_context_diagnostics.csv`.
- Existing component-failure/pooled diagnostic frame cache if fresh:
  `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`.
- Prediction, label, CRM/domain, persistence, and edge/context columns already
  available to the Phase 2 diagnostic pipeline.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
training_regime = pooled_monterey_big_sur
evaluation_region in {big_sur, monterey}
```

## Outputs

Write report-visible sidecar tables under
`/Volumes/x10pro/kelp_aef/reports/tables/`, with paths recorded in the Phase 2
manifest. Preferred names:

- `monterey_big_sur_cross_observed_canopy_persistence.csv`
- `monterey_big_sur_cross_observed_canopy_edge.csv`
- `monterey_big_sur_cross_depth_observed_canopy.csv`
- `monterey_big_sur_cross_observed_canopy_fine_depth.csv`

Minimum metrics in each crossed table:

- `evaluation_region`;
- `training_regime`;
- `cross_type`;
- the two crossed bin columns;
- `rows`;
- `observed_positive_count`;
- `predicted_positive_count`;
- `tp_count`;
- `fp_count`;
- `fn_count`;
- `precision`;
- `recall`;
- `f1`;
- `amount_under_count`;
- `amount_under_rate`;
- `composition_shrink_count`;
- `composition_shrink_rate`;
- `mean_hurdle_residual_m2`;
- `median_hurdle_residual_m2`;
- `hurdle_area_bias_pct`.

Regenerate the Phase 2 Markdown/HTML/PDF report with a concise crossed-context
section or appendix table that names the strongest conditional failure bins.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add output
paths under the existing Phase 2 report or diagnostics blocks. Keep bin
definitions config-backed or shared with the existing pooled-context
diagnostics.

## Plan / Spec Requirement

Before implementation, add a short implementation note to this task file or the
PR/commit message that confirms:

- exact observed-canopy bins and whether they match the current pooled-context
  labels;
- exact temporal persistence classes;
- exact edge classes;
- exact broad and fine CRM depth bins;
- denominator for `amount_under_rate` and `composition_shrink_rate`;
- how sparse crossed cells are represented without hiding them.

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

- The report includes crossed-bin evidence, not only one-way context plots.
- Observed canopy x persistence, observed canopy x edge class, CRM depth x
  observed canopy, and observed canopy x fine CRM depth are all present as
  generated tables.
- The report explicitly identifies whether high annual max plus
  intermittent/persistent context, edge status, or shallow depth concentrates
  binary misses or amount shrinkage.
- Sparse crossed cells remain auditable through row counts.
- No thresholds, labels, masks, sample quotas, or model predictions are changed.

## Known Constraints And Non-Goals

- Do not collapse these crosses into a single visual impression without tables.
- Do not tune model policy from held-out crossed-bin results.
- Do not download new data.
- Do not replace the existing one-way pooled context diagnostics; this task adds
  conditional context needed to interpret them.
