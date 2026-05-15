# Task 66: Fix Phase 2 Diagnostics Cache Invalidation

## Goal

Make Phase 2 report-only iteration actually fast.

The current cache is too conservative for report work. It can mark the expensive
Phase 2 diagnostics cache stale after edits that only affect report prose,
layout, or rendering because the freshness payload hashes the whole
`model_analysis.py` file and the whole coordinating config. That defeats the
point of Task 59: report-level edits should not force full-grid component and
pooled-context diagnostic rebuilds.

Fix the cache boundary so expensive diagnostics are rebuilt only when diagnostic
inputs, diagnostic settings, diagnostic output schemas/paths, or diagnostic
builder code change. Report text, section ordering, appendix wording, and
non-diagnostic figure/report rendering changes must be able to reuse a fresh
diagnostic cache.

Frame all report outputs as Kelpwatch-style annual maximum reproduction
diagnostics, not independent field-truth biomass validation.

## Inputs

- Config: `configs/big_sur_smoke.yaml`.
- Current Phase 2 diagnostics cache implementation:
  - `src/kelp_aef/evaluation/phase2_diagnostics_cache.py`
  - `src/kelp_aef/evaluation/model_analysis.py`
  - `src/kelp_aef/evaluation/component_failure.py`
  - `src/kelp_aef/evaluation/pooled_context.py`
- Current cache artifacts:
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_diagnostics_cache_manifest.json`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_component_failure_frames/`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_phase2_pooled_context_frames/`
- Current Phase 2 diagnostic CSV outputs under
  `/Volumes/x10pro/kelp_aef/reports/tables/`.
- Current report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`

Primary filters remain:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

- Refactored cache freshness logic that separates:
  - diagnostic cache inputs and settings;
  - diagnostic frame/table builder code;
  - report renderer/prose/layout code.
- Updated Big Sur report iteration behavior so the normal report workflow can
  reuse fresh diagnostics for report-only changes. Prefer the existing config
  setting if appropriate:

```yaml
reports:
  model_analysis:
    reuse_phase2_diagnostics: true
```

- Updated tests proving cache reuse survives report-only changes but still
  rejects truly stale diagnostics.
- Regenerated Phase 2 report and manifest showing
  `phase2.diagnostics_cache.status = "reused"` after a report-only rerun.
- No generated cache frames, CSVs, reports, or PDFs tracked in git.

## Config File

Use `configs/big_sur_smoke.yaml`.

If the implementation sets `reports.model_analysis.reuse_phase2_diagnostics:
true`, keep the explicit CLI flags:

- `--reuse-phase2-diagnostics` should still force reuse/fail-if-stale behavior.
- `--refresh-phase2-diagnostics` should still force a rebuild.
- The two flags should remain mutually exclusive.

## Plan / Spec Requirement

Before implementation, add a short design note to this task file or the
PR/commit message that confirms:

- exactly which fields are included in the diagnostic freshness payload;
- exactly which fields are excluded because they are report-only;
- whether `model_analysis.py` cache helpers need to move into
  `phase2_diagnostics_cache.py` or another narrow module to avoid hashing the
  report renderer;
- which source files invalidate cached component-failure frames;
- which source files invalidate cached pooled-context frames;
- how config changes are classified as diagnostic-affecting versus report-only;
- how the command behaves when cache artifacts are missing, stale, or explicitly
  refreshed;
- the expected fast-path runtime target.

## Implementation Notes

Current behavior to fix:

- `analyze-model` only uses the cache when `--reuse-phase2-diagnostics` or the
  config setting requests reuse.
- The current freshness payload includes a content hash for the entire
  coordinating config.
- The current code-path payload includes `model_analysis.py`, so report prose or
  layout edits can invalidate the expensive diagnostic cache.
- The cache manifest can be fresh and reusable, but small report edits still
  make the developer choose between a stale-cache failure and a slow rebuild.

The desired boundary:

- Diagnostic cache freshness should include:
  - component-failure and pooled-context input paths with existence, size, and
    mtime metadata;
  - primary filters and diagnostic tolerance/bin/grid settings;
  - configured cache frame paths and diagnostic table output paths that must be
    loaded from cache;
  - source files or version tokens for code that actually builds cached
    diagnostic frames and cached diagnostic CSV tables.
- Diagnostic cache freshness should not include:
  - report prose functions;
  - report section ordering;
  - Markdown/HTML/PDF rendering helpers;
  - unrelated `reports.outputs` keys;
  - visualizer config blocks;
  - report-only figure paths that are not produced by the cached diagnostic
    builders.

If `model_analysis.py` still contains both report-rendering and cache-fingerprint
logic, split the fingerprint code into a smaller module before deciding which
code paths to hash. Do not solve the problem by disabling freshness validation.

## Validation Command

Run focused checks first:

```bash
uv run ruff check src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/evaluation/phase2_diagnostics_cache.py tests/test_model_analysis.py
uv run pytest tests/test_model_analysis.py
```

Then validate real report behavior:

```bash
uv run kelp-aef build-phase2-diagnostics --config configs/big_sur_smoke.yaml
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
```

For final code changes, also run:

```bash
uv run mypy src
uv run pytest
git diff --check
```

## Smoke-Test Region And Years

- Regions: Big Sur and Monterey Phase 2 retained-domain rows.
- Year: held-out `2022`.
- Primary context: pooled Monterey+Big Sur diagnostics plus the existing
  six-context component-failure cache.

## Acceptance Criteria

- A report-only edit to `model_analysis.py` no longer invalidates the Phase 2
  diagnostics cache.
- A report-only config change no longer invalidates the Phase 2 diagnostics
  cache.
- A change to diagnostic input paths, diagnostic path metadata, diagnostic
  settings, component-failure builder code, or pooled-context builder code still
  invalidates or refreshes the cache explicitly.
- `uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml` can run
  as the normal fast report iteration command when a fresh cache exists, or the
  task records why the explicit `--reuse-phase2-diagnostics` flag remains
  required.
- The real Big Sur report rerun records
  `phase2.diagnostics_cache.status = "reused"` in
  `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`.
- Cached report iteration remains under `15s` on the current Big Sur config when
  diagnostic inputs have not changed.
- Stale-cache validation remains covered by tests.

## Known Constraints And Non-Goals

- Do not silently reuse stale diagnostics.
- Do not remove cache validation to get speed.
- Do not rebuild model predictions, labels, masks, samples, or full-grid
  alignment artifacts.
- Do not change model policy, thresholds, labels, masks, or Phase 2 diagnostic
  interpretation.
- Do not track generated cache or report artifacts in git.
