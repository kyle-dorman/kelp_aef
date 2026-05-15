# Task 64: Add Patch-Level Component Evaluation

## Goal

Determine whether Phase 2 errors miss whole kelp patches or mostly shift patch
boundaries.

Cell-level edge rates show many FNs are edge cells, but that does not answer the
patch question. A model can have many edge-cell FNs while still finding every
patch, or it can miss entire small/intermittent components. This task must
evaluate connected observed and predicted components directly.

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Existing Phase 2 diagnostic frame cache if fresh:
  `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`.
- Prediction, label, retained-domain mask, row/column, transform, and
  calibrated binary prediction columns already available to the Phase 2
  diagnostic pipeline.

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

- `monterey_big_sur_observed_patch_detection.csv`
- `monterey_big_sur_predicted_patch_overlap.csv`
- `monterey_big_sur_patch_component_summary.csv`

Observed-patch table minimum columns:

- `evaluation_region`;
- `training_regime`;
- `observed_component_id`;
- `observed_component_cells`;
- `observed_component_area_m2`;
- `observed_canopy_sum_m2`;
- `observed_canopy_max_m2`;
- `component_canopy_bin`;
- `component_edge_cell_share`;
- `detected_cell_count`;
- `detected_cell_share`;
- `missed_cell_count`;
- `missed_cell_share`;
- `overlapping_predicted_component_count`;
- `patch_detection_class`, with values such as `fully_missed`,
  `mostly_missed`, `boundary_shift`, `mostly_detected`, and `fully_detected`.

Predicted-patch table minimum columns:

- `evaluation_region`;
- `training_regime`;
- `predicted_component_id`;
- `predicted_component_cells`;
- `predicted_component_area_m2`;
- `overlapping_observed_component_count`;
- `overlap_observed_positive_cells`;
- `predicted_patch_class`, with values such as `true_zero_leakage_patch`,
  `low_canopy_threshold_patch`, `boundary_extension`, and `matched_patch`.

Summary table minimum metrics:

- observed patch counts by detection class;
- observed canopy area and cell share by detection class;
- predicted patch counts by predicted patch class;
- patch-level recall by component count and by observed canopy area;
- patch-level false-discovery summaries by predicted component count and area.

Regenerate the Phase 2 Markdown/HTML/PDF report so the report states whether
misses are mainly whole-patch misses or boundary shifts.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add output
paths under the existing Phase 2 report or diagnostics blocks. Do not hard-code
artifact paths in implementation code.

## Plan / Spec Requirement

Before implementation, add a short implementation note to this task file or the
PR/commit message that confirms:

- connected-component neighborhood choice, defaulting to 8-connected cells on
  the retained 30 m grid unless there is a strong reason otherwise;
- how row/column adjacency is restricted to each evaluation region and year;
- how observed positive components are defined from `annual_max_ge_10pct`;
- how predicted positive components are defined from the calibrated binary
  decision;
- detection-class thresholds, such as the cell-share cutoffs for `fully_missed`,
  `mostly_missed`, `boundary_shift`, and `mostly_detected`;
- how component outputs are kept diagnostic-only and out of model training.

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

- The report distinguishes whole observed-patch misses from boundary-shift
  misses.
- Observed component detection is summarized by component count and observed
  canopy area, not only by 30 m cell count.
- Predicted components are classified by whether they overlap true-zero,
  low-canopy-below-threshold, or observed-positive cells.
- Component IDs are stable within each generated artifact and auditable through
  row/column or cell identifiers.
- No threshold, mask, model, or sample policy is changed.

## Known Constraints And Non-Goals

- Do not infer ecological truth beyond Kelpwatch-style observed components.
- Do not use connected components as model features in this task.
- Do not replace cell-level metrics; patch-level evaluation complements them.
- Do not hide tiny patches. Summarize them separately if needed, but keep them
  auditable.
