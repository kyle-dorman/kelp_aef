# Task 20: Build The First Plausible-Kelp Domain Mask

## Goal

Build the first static plausible-kelp domain mask for the Monterey 30 m target
grid using the aligned NOAA CRM support layer from P1-11.

This task should produce a conservative candidate-area filter for Phase 1
full-grid reporting and later masked training. It should remove definite land
and very deep offshore cells while retaining nearshore and ambiguous coastal
cells until QA proves they are safe to drop.

The first mask should be framed as a physically plausible Kelpwatch-style
mapping domain, not ecological truth and not a model predictor. The first-pass
depth cutoff should be permissive, approximately 100 m, so the mask can reduce
obvious background leakage without prematurely discarding Kelpwatch-supported
canopy.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Source decision note:
  `docs/phase1_bathymetry_dem_source_decision.md`.
- Aligned NOAA CRM table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- Aligned NOAA CRM manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm_manifest.json`.
- Aligned NOAA CRM QA summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_noaa_crm_summary.csv`.
- Aligned domain-source comparison QA:
  `/Volumes/x10pro/kelp_aef/reports/tables/aligned_domain_source_comparison.csv`.
- NOAA CUSP source manifest, for shoreline vector provenance:
  `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json`.
- Existing full-grid target table, for label-source and Kelpwatch-positive QA:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Existing full-grid target manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json`.

P1-11 result to preserve in this task: the aligned CRM output has 7,458,361
unique target-grid cells with valid CRM samples. CUDEM QA covers 3,231,173 cells
and should remain QA only. 3DEP is land-side context only.

## Outputs

- Package-backed mask module, for example:
  `src/kelp_aef/domain/domain_mask.py`.
- CLI command wired through `src/kelp_aef/cli.py`, for example:
  `kelp-aef build-domain-mask`.
- Static mask table under:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Mask manifest under:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json`.
- Mask coverage summary under:
  `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_summary.csv`.
- Kelpwatch-positive retention QA table under:
  `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_kelpwatch_retention.csv`.
- Depth/elevation-bin QA table under:
  `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_depth_bins.csv`.
- Optional fast-path outputs using `.fast` suffixes or explicit config fields.
- Unit tests for config loading, rule classification, reason-code precedence,
  Kelpwatch-positive retention summaries, manifest construction, and fast-path
  behavior.

## Config File

Use `configs/monterey_smoke.yaml`.

Add concrete mask paths and thresholds under a narrow config block such as
`domain.plausible_kelp_mask`.

Expected config fields:

- Input aligned CRM table path.
- Input aligned CRM manifest path.
- Input full-grid table path.
- Input full-grid manifest path.
- Optional CUSP source manifest path.
- Output mask table path.
- Output mask manifest path.
- Output coverage summary path.
- Output Kelpwatch-retention summary path.
- Output depth-bin summary path.
- Broad maximum depth threshold, initially `100.0` m.
- Definite-land elevation threshold, initially `5.0` m.
- Ambiguous-coast elevation band, initially `[-5.0, 5.0]` m.
- Nearshore shallow depth upper bound for QA bins, initially `40.0` m.
- Intermediate QA depth bound, initially `50.0` m.
- Fast-path row/column window inherited from `alignment.full_grid.fast`.

Do not add model, prediction, residual-map, or retraining paths in this task.
Those belong to P1-13 and P1-14.

## Plan/Spec Requirement

This task creates a new mask artifact contract. Before implementation, write a
brief implementation plan in the working notes or PR description that confirms:

- The exact output schema and reason-code vocabulary.
- The reason-code precedence order.
- Whether this pass uses only CRM elevation/depth or also implements a first
  CUSP shoreline-side classifier.
- How Kelpwatch-positive retention will be measured by year and depth bin.
- How fast-path outputs map to the configured full-grid fast window.
- Which downstream commands will remain unchanged until P1-13.

If the implementation attempts true shoreline-side classification with CUSP
LineStrings, add a short decision note first. CUSP side classification is more
algorithmically sensitive than depth filtering and should not be hidden inside a
simple threshold task.

## Implementation Plan

- Add a small package-backed domain-mask module under `src/kelp_aef/domain/`.
- Add one CLI command that reads the aligned CRM support table, applies the
  first mask rules, writes one static row per target-grid cell, and writes
  summary artifacts.
- Preserve target-grid identifiers from the aligned CRM table:
  `aef_grid_row`, `aef_grid_col`, `aef_grid_cell_id`, `longitude`, and
  `latitude`.
- Keep CRM columns needed for diagnostics:
  `crm_elevation_m`, `crm_depth_m`, `crm_source_product_id`,
  `crm_vertical_datum`, and `crm_value_status`.
- Write mask columns such as:
  `is_plausible_kelp_domain`, `domain_mask_reason`,
  `domain_mask_detail`, `domain_mask_version`,
  `domain_mask_depth_threshold_m`, and `domain_mask_rule_set`.
- Start with a conservative reason-code vocabulary:
  - `retained_depth_0_100m`: CRM-valid cell with depth between 0 and 100 m and
    not definite land.
  - `retained_ambiguous_coast`: CRM-valid cell within the ambiguous elevation
    band, retained for first-pass caution.
  - `dropped_deep_water`: CRM-valid cell with depth greater than 100 m.
  - `dropped_definite_land`: CRM-valid cell with elevation greater than 5 m.
  - `qa_missing_crm`: no valid CRM value. This should be rare or absent after
    P1-11, and should be retained or failed explicitly by policy rather than
    silently dropped.
- Use explicit rule precedence. A reasonable first precedence is:
  1. `qa_missing_crm`
  2. `retained_ambiguous_coast`
  3. `dropped_definite_land`
  4. `dropped_deep_water`
  5. `retained_depth_0_100m`
- Treat the first mask as CRM-depth/elevation based unless a separate shoreline
  side classifier is specified. Record CUSP manifest validation/provenance in
  the mask manifest, but do not infer landward/oceanward side from CUSP unless
  the implementation plan explicitly defines and tests that algorithm.
- Join only the minimal full-grid columns needed for Kelpwatch-positive QA:
  `year`, `aef_grid_cell_id`, `label_source`, `kelp_max_y`,
  `kelp_fraction_y`, and binary annual-max labels if available.
- Write a Kelpwatch-retention table by year and label source, including:
  total Kelpwatch-positive cells, retained positives, dropped positives,
  retained positive fraction, and dropped positives by mask reason.
- Write depth-bin QA summaries, including Kelpwatch-positive counts in:
  land-positive, ambiguous coast, 0-40 m, 40-50 m, 50-100 m, and greater than
  100 m.
- Include CUDEM and 3DEP status in QA summaries only as context. Do not use
  CUDEM or 3DEP to override the primary CRM mask in this first pass unless a
  later decision changes the source order.
- Write a manifest with input artifacts, thresholds, rule version, row counts,
  retained/dropped counts, Kelpwatch-positive retention, CUSP provenance, output
  schema, and QA table paths.
- Add a fast mode that filters the aligned CRM table by the configured
  `alignment.full_grid.fast` row/column window and writes fast output paths.
- Make reruns idempotent by replacing existing output artifacts and logging the
  overwrite policy.

## Expected Output Schema

The mask table should be static: one row per target-grid cell, not one row per
cell-year.

Minimum columns:

```text
aef_grid_row
aef_grid_col
aef_grid_cell_id
longitude
latitude
crm_elevation_m
crm_depth_m
crm_source_product_id
crm_vertical_datum
crm_value_status
is_plausible_kelp_domain
domain_mask_reason
domain_mask_detail
domain_mask_version
domain_mask_depth_threshold_m
domain_mask_rule_set
```

Optional QA columns may include:

```text
cudem_value_status
usgs_3dep_value_status
depth_bin
elevation_bin
```

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_domain_mask.py
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml --fast
```

Full validation if code/config changes are made:

```bash
make check
```

Real data command:

```bash
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml
```

After the real run, inspect row counts and Kelpwatch-positive retention:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet'; df=pd.read_parquet(p); print(len(df), df.aef_grid_cell_id.nunique()); print(df.domain_mask_reason.value_counts(dropna=False))"
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: mask is static, but Kelpwatch-positive retention QA should summarize
  the configured 2018-2022 annual-max full-grid rows by year.
- Fast path: use the existing `alignment.full_grid.fast` row/column window.

## Acceptance Criteria

- Adds one package-backed mask command with deterministic config-backed paths.
- Produces one static mask row per unique aligned CRM target-grid cell.
- Full mask row count matches the aligned CRM row count: 7,458,361 cells for
  the current Monterey artifact.
- Mask output includes stable target-grid identifiers, CRM elevation/depth,
  boolean retain/drop flag, reason codes, thresholds, and rule version.
- Reason-code precedence is explicit and covered by tests.
- Manifest records input artifacts, thresholds, rule version, row counts,
  retained/dropped counts, CUSP provenance, and output schema.
- Coverage summary shows retained and dropped cells by reason and CRM source
  product.
- Kelpwatch-retention QA shows retained/dropped positive cells by year and
  reason before any downstream masking is applied.
- Positives deeper than 40-50 m are reported for inspection and not silently
  used to tighten the first-pass threshold.
- Fast-path validation writes small mask, manifest, and summary artifacts.
- Re-running the command overwrites outputs only according to an explicit,
  logged policy.

## Known Constraints Or Non-Goals

- Do not apply the mask to predictions, maps, or reports in this task. That is
  P1-13.
- Do not retrain or resample model inputs in this task. That is P1-14.
- Do not use bathymetry, elevation, or mask status as model predictors in
  Phase 1.
- Do not treat CRM/CUDEM/3DEP as field truth.
- Do not use CUDEM gaps as off-domain evidence.
- Do not promote 3DEP or CUDEM to the primary mask source in this task.
- Do not implement a complex shoreline-side classifier without a short decision
  note and dedicated tests.
- Do not add GEBCO, Copernicus, or West Coast scale-up behavior.
