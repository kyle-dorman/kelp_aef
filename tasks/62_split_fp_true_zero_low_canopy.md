# Task 62: Split False Positives Into True-Zero And Low-Canopy Cases

## Goal

Make binary false positives interpretable instead of treating every FP as the
same failure.

The current `annual_max_ge_10pct` binary target marks cells positive at
`kelp_max_y >= 90 m2`. A predicted positive with observed canopy below `90 m2`
can therefore be either:

- a true-zero false positive: observed annual max is `0 m2`;
- a low-canopy-below-threshold false positive: observed annual max is greater
  than `0 m2` but below `90 m2`.

These are different model questions. True-zero FPs are support leakage. Low-but
real canopy FPs are threshold, target, calibration, or weak-label boundary
issues. The Phase 2 report must separate them explicitly.

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Existing pooled and local binary/hurdle prediction artifacts declared in the
  Phase 2 config.
- Existing component-failure and pooled-context diagnostic frame cache if fresh:
  `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`.
- Current Phase 2 report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

Write a report-visible table under `/Volumes/x10pro/kelp_aef/reports/tables/`,
with a manifest entry. Preferred name:

- `monterey_big_sur_fp_subtype_diagnostics.csv`

Minimum columns:

- `evaluation_region`;
- `training_regime`;
- `model_origin_region`;
- `binary_threshold_m2`;
- `fp_total`;
- `fp_true_zero_count`;
- `fp_low_canopy_below_threshold_count`;
- `fp_true_zero_share`;
- `fp_low_canopy_below_threshold_share`;
- observed-canopy subtype bins, including `0`, `(0, 90)`, and any finer bins
  used for review;
- optional context columns such as `depth_bin`, `crm_depth_m_bin`, `edge_class`,
  and `label_source` if they are already available.

Regenerate the Phase 2 Markdown/HTML/PDF report so the main text states whether
pooled Big Sur FPs are mostly true-zero leakage or low-canopy-below-threshold
cases.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add any new
output path under the existing Phase 2 report or diagnostics output blocks. Do
not hard-code `/Volumes/x10pro/kelp_aef` paths in implementation code.

## Plan / Spec Requirement

Before implementation, add a short implementation note to this task file or the
PR/commit message that confirms:

- which cached diagnostic frame or prediction files supply `observed_area_m2`
  and binary predictions;
- the exact FP subtype definitions;
- whether the table covers pooled-only rows or all local, transfer, and pooled
  contexts;
- how the report will distinguish threshold/target issues from true support
  leakage.

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
- Primary review context: pooled Monterey+Big Sur evaluated separately on Big
  Sur and Monterey.

## Acceptance Criteria

- The report no longer uses one undifferentiated FP count when discussing
  binary support errors.
- True-zero FPs and low-canopy-below-threshold FPs are counted separately for
  the pooled Big Sur and pooled Monterey evaluation contexts.
- The report explicitly says whether low-canopy-below-threshold FPs are a
  material share of FPs.
- The table preserves the `90 m2` binary threshold so readers can verify that
  "false positive" means "below threshold," not necessarily "zero canopy."
- No model threshold, target, sample quota, mask, or prediction is changed by
  this diagnostic task.

## Known Constraints And Non-Goals

- Do not retune the calibrated binary threshold from held-out rows.
- Do not redefine `annual_max_ge_10pct`.
- Do not treat low-canopy-below-threshold FPs as field truth errors; they are
  Kelpwatch-style label/target diagnostics.
- Do not merge this work into a broad report-polish task without the explicit
  FP subtype table.
