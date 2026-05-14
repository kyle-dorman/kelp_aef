# Task 41: Add Big Sur Config And Source-Manifest Plan

## P2 Mapping

P2-01: Add Big Sur config and source-manifest plan.

## Goal

Introduce `big_sur` as the second small Phase 2 region without breaking the
closed Monterey Phase 1 config, outputs, or report snapshots.

This task should define the Big Sur config strategy, region footprint path,
source-manifest paths, and review gates needed before P2-02 verifies coverage
and runs early visual QA. It should not download new source data, build aligned
tables, train models, or interpret Big Sur performance.

## Inputs

- Phase 2 plan: `docs/phase2_big_sur_generalization.md`.
- Active checklist: `docs/todo.md`.
- Existing config contract: `configs/monterey_smoke.yaml`.
- Artifact contract: `docs/data_artifacts.md`.
- Architecture notes: `docs/architecture.md`.
- User-provided Big Sur AlphaEarth STAC feature:
  - STAC item id: `8957`.
  - CRS: `EPSG:32610`.
  - Bbox:
    `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`.
  - Example 2022 AEF asset:
    `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`.
- Canonical artifact root: `/Volumes/x10pro/kelp_aef`.

## Outputs

Expected planning outputs:

- A Big Sur config strategy:
  - preferred first implementation: `configs/big_sur_smoke.yaml`;
  - acceptable alternative only if justified: a region-aware config extension
    that keeps Monterey and Big Sur paths explicit.
- A planned Big Sur footprint path, for example:
  `/Volumes/x10pro/kelp_aef/geos/big_sur_aef_10n_0000_8192_footprint.geojson`.
- Region-scoped source-manifest path plan for:
  - AEF catalog query, query summary, and tile manifest;
  - Kelpwatch source manifest and annual-label outputs;
  - NOAA CRM query/source manifests;
  - NOAA CUDEM, NOAA CUSP, and USGS 3DEP query/source manifests where relevant.
- Region-scoped downstream path plan for the artifacts that P2-03 and later
  tasks will write:
  - labels;
  - full-grid alignment;
  - plausible-kelp domain mask;
  - CRM-stratified mask-first sample;
  - baseline, binary, conditional, and hurdle artifacts;
  - report outputs;
  - results visualizer outputs.
- A short source-review gate documenting that P2-02 must verify AEF,
  Kelpwatch, and domain-source coverage before training or metric
  interpretation.
- Updated `docs/todo.md` entry for P2-01 with this task plan path.

## Config File

Use `configs/big_sur_smoke.yaml` as the planned Big Sur config unless the
implementation finds a strong reason to use a region-aware config extension.

The first Big Sur config should be a small smoke/generalization config, not a
full West Coast config. It should copy the Monterey config shape only where the
same loader contract applies, and it must replace output paths with
region-scoped Big Sur names.

## Planned Path Policy

Use `big_sur` in every new Big Sur artifact path. Do not reuse Monterey output
paths.

Prefer one of these two patterns consistently:

- flat files with a `big_sur_` prefix inside the existing artifact directories;
- `big_sur/` subdirectories under existing artifact directories.

The first implementation should choose the lower-risk pattern after checking
current loaders. If a loader assumes parent directories already exist, the task
should either choose flat `big_sur_` filenames or add explicit directory
creation in the relevant follow-on task.

Monterey paths in `configs/monterey_smoke.yaml` should remain unchanged.

## Plan/Spec Requirement

This task is itself the plan/spec for P2-01. No separate decision note is
required unless the implementer decides not to use `configs/big_sur_smoke.yaml`
or proposes a larger multi-region config refactor.

If the implementation changes config architecture, update `docs/architecture.md`
and `docs/data_artifacts.md` in the same task.

## Implementation Plan

1. Inspect the current Monterey config for loader-consumed keys and output path
   conventions.
2. Choose the P2 config shape:
   - default: add `configs/big_sur_smoke.yaml`;
   - avoid a broad multi-region config refactor unless it is clearly simpler
     than a second small config.
3. Define the Big Sur region block using the user-provided STAC geometry,
   bbox, CRS, and 2022 source tile URI.
4. Define all Big Sur output paths with `big_sur` in the filename or directory
   so reruns cannot overwrite Monterey artifacts.
5. Add source-manifest paths for AEF, Kelpwatch, NOAA CRM, CUDEM, CUSP, and
   USGS 3DEP. These are reviewable manifest targets only; P2-01 should not
   perform downloads.
6. Reserve early visual-QA output paths for P2-02:
   - Big Sur Kelpwatch labels;
   - Big Sur AEF coverage or footprint;
   - Big Sur CRM/domain context.
7. Record which commands P2-02 should run first, likely query/inspect/visual-QA
   commands rather than model-training commands.
8. Update `docs/todo.md` so P2-01 points to this task plan.
9. Run docs-only validation and inspect the diff.

## Implementation Outcome

Chosen config strategy:

- Add `configs/big_sur_smoke.yaml`.
- Do not refactor the config loader or introduce a multi-region config schema
  in P2-01.
- Keep `configs/monterey_smoke.yaml` unchanged.

Chosen path policy:

- Use flat `big_sur_` filenames under the existing artifact roots.
- Keep shared raw source mirrors such as `/Volumes/x10pro/kelp_aef/raw/aef`,
  `/Volumes/x10pro/kelp_aef/raw/kelpwatch`, and
  `/Volumes/x10pro/kelp_aef/raw/domain/` available for source reuse, but write
  all Big Sur manifests, derived tables, model outputs, figures, report tables,
  reports, and visualizer outputs with `big_sur` in the artifact name.

Big Sur region metadata in `configs/big_sur_smoke.yaml`:

- Region shorthand: `big_sur`.
- Planned footprint:
  `/Volumes/x10pro/kelp_aef/geos/big_sur_aef_10n_0000_8192_footprint.geojson`.
- STAC item id: `8957`.
- STAC source CRS: `EPSG:32610`.
- WGS84 bbox:
  `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`.
- Example 2022 AEF asset:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`.

Primary source manifest paths:

- AEF catalog query:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_catalog_query.parquet`.
- AEF catalog query summary:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_catalog_query_summary.json`.
- AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_tile_manifest.json`.
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_kelpwatch_source_manifest.json`.
- NOAA CRM query/source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_crm_query_manifest.json` and
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_crm_source_manifest.json`.
- NOAA CUDEM query/source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cudem_tile_query_manifest.json` and
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cudem_tile_manifest.json`.
- NOAA CUSP query/source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cusp_query_manifest.json` and
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cusp_source_manifest.json`.
- USGS 3DEP query/source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_usgs_3dep_query_manifest.json` and
  `/Volumes/x10pro/kelp_aef/interim/big_sur_usgs_3dep_source_manifest.json`.

Primary downstream path families reserved for P2-03 and later:

- Labels: `/Volumes/x10pro/kelp_aef/interim/big_sur_labels_annual.parquet`.
- Full-grid alignment:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table.parquet`.
- Plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask.parquet`.
- CRM-stratified mask-first sample:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked.parquet`.
- Baseline, binary, conditional, and hurdle model families use `big_sur_`
  prefixes under `/Volumes/x10pro/kelp_aef/models/`,
  `/Volumes/x10pro/kelp_aef/processed/`, and
  `/Volumes/x10pro/kelp_aef/reports/tables/`.
- Model-analysis report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`.
- Results visualizer:
  `/Volumes/x10pro/kelp_aef/reports/interactive/big_sur_results_visualizer.html`.

Source-review gate for P2-02:

- Materialize or verify the planned Big Sur footprint before running source
  query commands.
- Run source query and dry-run inspection commands first, including AEF,
  Kelpwatch, CRM, CUDEM, CUSP, and 3DEP coverage where relevant.
- Visually QA Big Sur labels, AEF coverage, and CRM/domain context before any
  model training, metric interpretation, or threshold/quantity comparison.
- Do not train Big Sur models or interpret Big Sur performance until P2-02
  records source coverage and visual coherence outcomes.

## Validation Command

Docs-only validation:

```bash
git diff --check
rg -n "P2-01|big_sur|tasks/41_big_sur_config_source_manifest_plan.md" docs tasks
```

If the implementation also creates `configs/big_sur_smoke.yaml`, add a focused
config sanity check that does not download data. Prefer a dry-run or metadata
inspection command if one exists for the touched command path. Do not run real
downloads in P2-01.

## Smoke-Test Region And Years

- Region: Big Sur.
- Region shorthand: `big_sur`.
- Initial years: 2018-2022, if source verification in P2-02 confirms the same
  overlap window as Monterey.
- Primary planned report year: 2022, if verified.

## Acceptance Criteria

- `docs/todo.md` links P2-01 to this task plan.
- The plan identifies the intended Big Sur config strategy.
- The planned Big Sur artifact paths are region-scoped and cannot overwrite
  Monterey artifacts.
- The plan records the user-provided STAC item id, bbox, CRS, and example AEF
  asset.
- The plan covers AEF, Kelpwatch, and domain-source manifests.
- The plan reserves early visual-QA outputs for labels, AEF coverage, and
  domain context.
- P2-02 remains responsible for actually verifying source coverage and visual
  coherence.

## Known Constraints And Non-Goals

- Do not download AEF, Kelpwatch, CRM, CUDEM, CUSP, or 3DEP source data in this
  task.
- Do not train or evaluate models.
- Do not build Big Sur aligned tables, masks, samples, predictions, or reports.
- Do not change the Monterey smoke config output paths.
- Do not start full West Coast scale-up.
- Do not change the Phase 2 annual-max label target.
- Do not add bathymetry/DEM as model predictors.
