# Task 59: Cache Phase 2 Diagnostics For Fast Report Iteration

## Goal

Separate expensive Phase 2 diagnostic derivation from report rendering so
`analyze-model` can iterate on report text, layout, and plots without rebuilding
full-grid component-failure and pooled-context diagnostics every run.

The current Big Sur Phase 2 report cycle is too slow because report generation
owns heavy diagnostic table construction. A measured run on
`configs/big_sur_smoke.yaml` showed:

```text
TOTAL analyze-model stage timing: 133.17s
build_analysis_tables: 128.79s
write_analysis_figures: 2.33s
write_report: 0.74s
load_analysis_data: 0.73s
write_analysis_tables: 0.51s
```

Inside `build_analysis_tables`, the slow builders were:

```text
build_component_failure_tables: 70.56s
build_pooled_context_tables: 50.68s
build_binary_presence_hex_map_tables: 0.40s
all other table builders: mostly <1s each
```

The goal is to preserve the current diagnostics and report-visible evidence
while reducing normal report iteration from roughly two minutes to a few
seconds when diagnostic inputs have not changed.

Frame all outputs as Kelpwatch-style annual maximum reproduction diagnostics,
not independent field-truth biomass validation.

## Inputs

- Config:
  - `configs/big_sur_smoke.yaml`
- Existing report-generation code:
  - `src/kelp_aef/evaluation/model_analysis.py`
- Existing Phase 2 diagnostic builders:
  - `src/kelp_aef/evaluation/component_failure.py`
  - `src/kelp_aef/evaluation/pooled_context.py`
  - `src/kelp_aef/evaluation/binary_presence_hex.py`
- Existing Phase 2 model/prediction artifacts declared in
  `configs/big_sur_smoke.yaml`, especially:
  - local, transfer, and pooled hurdle full-grid predictions;
  - local, transfer, and pooled binary full-grid predictions;
  - pooled ridge full-grid predictions;
  - Big Sur and Monterey annual label tables.
- Current generated diagnostic CSVs and manifests:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_by_label_context.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_by_domain_context.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_by_spatial_context.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_by_model_context.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_edge_effect_diagnostics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_temporal_label_context.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_component_failure_manifest.json`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_context_model_performance.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_binary_context_diagnostics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_amount_context_diagnostics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_prediction_distribution_by_context.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_pooled_context_diagnostics_manifest.json`

Primary report filters remain:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

Add explicit config-driven cached diagnostic artifacts under
`/Volumes/x10pro/kelp_aef/interim/`. Suggested outputs:

- `/Volumes/x10pro/kelp_aef/interim/phase2_component_failure_frames/`
  - Parquet dataset with one annotated row-level component-failure frame per
    configured context.
  - Must include enough columns to rebuild all component-failure summary,
    label, domain, spatial, model, edge, and temporal tables.
- `/Volumes/x10pro/kelp_aef/interim/phase2_pooled_context_frames/`
  - Parquet dataset with one annotated row-level pooled-context frame per
    pooled evaluation context.
  - Must include expected-value hurdle fields, ridge predictions, calibrated
    binary probabilities, binary outcomes, context-family columns, and model
    surface alignment fields needed to rebuild pooled context tables and plots.
- `/Volumes/x10pro/kelp_aef/interim/phase2_diagnostics_cache_manifest.json`
  - Includes input paths, input modification times and sizes or stable content
    hashes, primary filters, tolerance values, grid size, code/config version
    breadcrumbs, output paths, row counts, and whether cache reuse was allowed.
- Existing report table outputs should still be written exactly where the
  config declares them.
- Existing Phase 2 model-analysis report outputs should remain:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`

## Config File

Use `configs/big_sur_smoke.yaml`.

Add explicit output keys rather than hard-coding cache paths in source code.
Suggested config keys under `reports.outputs` or under the existing
`training_regime_comparison` diagnostic blocks:

```yaml
phase2_component_failure_frame_cache: /Volumes/x10pro/kelp_aef/interim/phase2_component_failure_frames
phase2_pooled_context_frame_cache: /Volumes/x10pro/kelp_aef/interim/phase2_pooled_context_frames
phase2_diagnostics_cache_manifest: /Volumes/x10pro/kelp_aef/interim/phase2_diagnostics_cache_manifest.json
```

Keep path naming consistent with the existing `monterey_big_sur_*` Phase 2
artifacts if that reads better during implementation.

## Plan / Spec Requirement

Before implementation, write a brief design note in this task file or the PR
description that confirms:

- the exact cache artifact schema and partitioning;
- whether cache freshness is checked by path + size + mtime, file hash, config
  hash, or a combination;
- which `component_failure.py` functions produce reusable row-level frames;
- how `pooled_context.py` reuses the pooled component-failure frames instead of
  recomputing them;
- how `analyze-model` behaves when cache artifacts are missing, stale, or
  explicitly refreshed;
- how report-only iteration is invoked;
- the validation timing target.

## Design Note

- Cache schema and partitioning: write one annotated Parquet frame per
  configured context under a component-failure cache directory and one
  annotated Parquet frame per pooled evaluation context under a pooled-context
  cache directory. File names are `<context_id>.parquet`. Component frames keep
  the existing row-level component-failure columns needed to rebuild summary,
  label, domain, spatial, model, edge, and temporal tables. Pooled frames start
  from the matching component frame and add pooled ridge predictions,
  calibrated binary probabilities, refreshed binary outcomes, failure classes,
  and pooled temporal context fields needed to rebuild pooled performance,
  binary, amount, and prediction-distribution tables.
- Freshness check: use a manifest freshness hash built from the config file
  hash, relevant source-code file hashes, primary filters, tolerance/bin/grid
  settings, configured cache/table output paths, and input path metadata
  (`exists`, file size, mtime; recursive file count/size/max-mtime for Parquet
  directories). Heavy Parquet inputs are not content-hashed.
- Component functions: expose component-frame build/write/read helpers and a
  `build_component_failure_tables_from_frames` path so table aggregation no
  longer requires raw prediction reads once frames are cached.
- Pooled functions: expose pooled-frame build/write/read helpers and allow
  `read_pooled_context_frame` to reuse the matching cached component frame
  before joining ridge and calibrated binary surfaces.
- `analyze-model` behavior: the default path remains the uncached rebuild path.
  `--reuse-phase2-diagnostics` loads cached CSV diagnostic tables only when the
  manifest proves freshness; stale or missing caches fail explicitly. The
  `--refresh-phase2-diagnostics` path rebuilds frames, tables, and the cache
  manifest before rendering.
- Report-only iteration: run
  `uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml`
  after model/input changes, then run
  `uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics`
  for text, layout, and plot iteration.
- Timing target: cached Big Sur report iteration should avoid component and
  pooled row annotation and target under 15 seconds for `analyze-model` when
  diagnostic inputs have not changed.

## Implementation Notes

The current profiling points to two concrete issues:

1. Component-failure derivation is row-level and repeated over six contexts.
   - Raw parquet reads were cheap: about `0.1s` per context.
   - `annotate_component_failure_frame` cost about `6-9s` per context.
   - `add_component_spatial_context` cost about `0.8-1.1s` per context.
   - Grouped aggregate table construction was only about `10s` total.

2. Pooled-context derivation duplicates work already done by component failure.
   - `build_pooled_context_tables` calls `read_component_failure_frame` again
     for the two pooled contexts.
   - Those repeated component-frame reads/annotations cost about `20s`.
   - `annotate_aligned_pooled_context` then adds another `7-11s` per pooled
     context.
   - Context-family table aggregation costs about `11s` total.

The clean target shape:

```text
build-phase2-diagnostics
  -> annotated component-failure frame cache
  -> annotated pooled-context frame cache
  -> component-failure CSVs/manifests
  -> pooled-context CSVs/manifests

analyze-model
  -> load normal model-analysis inputs
  -> load cached Phase 2 diagnostic CSVs or cached frames
  -> rebuild only cheap report tables/figures
  -> write Markdown/HTML/PDF/manifest
```

Prefer package-backed CLI commands. A reasonable implementation is:

```bash
uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
```

If adding a flag is too awkward, use a config setting such as:

```yaml
reports:
  model_analysis:
    reuse_phase2_diagnostics: true
```

Do not silently reuse stale diagnostics. Stale cache behavior should be
explicit:

- default: rebuild if missing or stale;
- report-iteration mode: reuse only if the manifest proves freshness;
- optional refresh flag: force rebuild even when cache is fresh.

## Suggested Code Changes

- Add a small cache/manifest helper in `src/kelp_aef/evaluation/` rather than
  burying freshness checks inside report prose functions.
- In `component_failure.py`:
  - expose a function that writes annotated component-failure frames;
  - expose a function that reads those frames and builds the current
    `ComponentFailureTables`;
  - keep the existing table fields stable.
- In `pooled_context.py`:
  - allow `read_pooled_context_frame` to start from a cached pooled
    component-failure frame when available;
  - write/read the fully aligned pooled-context frame after ridge and binary
    joins;
  - keep the existing pooled context CSV fields stable.
- In `model_analysis.py`:
  - let `build_analysis_tables` load cached Phase 2 diagnostic tables or
    cached frames when valid;
  - keep normal Phase 1/summary table builders unchanged;
  - include cache reuse status and cache artifact paths in the
    model-analysis manifest.
- In the CLI:
  - add `build-phase2-diagnostics`, or add narrowly scoped flags to
    `analyze-model` if that better fits current command wiring.
- In tests:
  - add synthetic fixtures proving cached and uncached paths produce identical
    diagnostic rows;
  - add a stale-cache test by modifying an input path/mtime/hash in the
    manifest and confirming rebuild or failure behavior;
  - assert report generation can reuse diagnostic outputs without invoking the
    expensive row-level builders.

## Validation Command

Focused validation:

```bash
uv run ruff check src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/evaluation/component_failure.py src/kelp_aef/evaluation/pooled_context.py tests/test_model_analysis.py
uv run mypy src
uv run pytest tests/test_model_analysis.py
uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
git diff --check
```

## Outcome

Completed. The implementation adds:

- `kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml`
  to build or validate Phase 2 component-failure and pooled-context caches.
- `kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics`
  for report-only iteration against a fresh manifest.
- `kelp-aef analyze-model --config configs/big_sur_smoke.yaml --refresh-phase2-diagnostics`
  to rebuild the cache during report generation.
- Config-declared cache paths:
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_component_failure_frames/`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_pooled_context_frames/`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`

The real Big Sur cache build wrote six component context frames and two pooled
context frames. The cache manifest row counts match the current diagnostic
tables: `6` component summary rows, `267` component label-context rows, `1,510`
component model-context rows, `216` pooled performance rows, `72` pooled binary
context rows, `144` pooled amount rows, and `144` pooled prediction-distribution
rows.

The cached report iteration path was validated with the current Big Sur config.
After the first post-build report run refreshed ordinary sidecars, a second
steady-state cached rerun completed from `12:53:06` to `12:53:16` and the
model-analysis manifest recorded `phase2.diagnostics_cache.status = "reused"`.

Validation passed:

```bash
uv run ruff check src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/evaluation/component_failure.py src/kelp_aef/evaluation/pooled_context.py src/kelp_aef/evaluation/phase2_diagnostics_cache.py src/kelp_aef/cli.py tests/test_model_analysis.py
uv run mypy src
uv run pytest tests/test_model_analysis.py -q
uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
git diff --check
```

If the final CLI shape differs, update the command block in this task and in
`docs/todo.md`.

Include a timing check in the outcome. Minimum useful timing evidence:

```text
uncached build-phase2-diagnostics runtime:
cached/reuse analyze-model runtime:
```

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Diagnostic contexts:
  - Big Sur local.
  - Monterey local.
  - Monterey-trained transfer on Big Sur.
  - Big-Sur-trained transfer on Monterey.
  - pooled Monterey+Big Sur evaluated on Big Sur.
  - pooled Monterey+Big Sur evaluated on Monterey.
- Primary year: 2022 held-out test rows.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: Kelpwatch-style annual max canopy and `annual_max_ge_10pct`.

## Acceptance Criteria

- `analyze-model` no longer recomputes expensive Phase 2 diagnostic frames
  during normal report-only iteration when a valid cache exists.
- Cached diagnostic reuse preserves the current component-failure CSV rows,
  pooled-context CSV rows, report figures, report prose, and manifests.
- Pooled-context derivation does not call the full component-failure row
  annotation path again for pooled contexts when cached pooled component frames
  are available.
- Cache freshness is explicit and conservative; stale or incompatible caches
  are rebuilt or rejected with a clear log message.
- The model-analysis manifest records whether Phase 2 diagnostics were rebuilt
  or reused and records cache artifact paths.
- The fast report path is substantially faster than the current baseline.
  Target: cached report iteration under `15s` on the current Big Sur config,
  excluding intentional full diagnostic refresh.
- Generated report artifacts remain:
  - Markdown, HTML, PDF;
  - current pooled context metric-breakdown figure;
  - current binary hex figure;
  - current component-failure and pooled-context tables.
- Tests cover both cache miss/rebuild and cache hit/reuse behavior.

## Known Constraints / Non-Goals

- Do not change model training, predictions, masks, thresholds, features,
  labels, sample quotas, or model policy.
- Do not reduce diagnostic content just to make the report fast.
- Do not skip stale-cache validation.
- Do not track generated cache artifacts in git.
- Do not make Phase 3 modeling decisions in this task.
- Do not optimize the large pooled context plot unless it becomes a measured
  bottleneck; current figure/report writing is not the slow part.
