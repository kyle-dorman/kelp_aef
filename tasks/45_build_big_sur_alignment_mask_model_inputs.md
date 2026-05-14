# Task 45: Build Big Sur Alignment, Mask, And Model-Input Artifacts

## P2 Mapping

P2-03c: Apply the refactored mask-first and Kelpwatch-native target-grid
workflow to Big Sur.

## Goal

Use the refactored alignment workflow from Task 43 and the general
Kelpwatch-native UTM target-grid policy from Task 44 to build Big Sur annual
labels, full-grid alignment, CRM support, a plausible-kelp retained-domain
mask, a CRM-stratified mask-first model-input sample, and a split manifest.

Do not train or evaluate Big Sur models in this task.

## Inputs

- Workflow prerequisite: `tasks/43_refactor_mask_first_alignment_workflow.md`.
- Target-grid prerequisite: `tasks/44_use_kelpwatch_native_utm_target_grid.md`.
- Phase 2 plan: `docs/phase2_big_sur_generalization.md`.
- Active checklist: `docs/todo.md`.
- Big Sur config: `configs/big_sur_smoke.yaml`.
- Source verification task and outcome:
  `tasks/42_verify_big_sur_source_coverage_visual_qa.md`.
- Big Sur AEF tile manifest:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_tile_manifest.json`.
- Big Sur Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_kelpwatch_source_manifest.json`.
- Big Sur source coverage summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_source_coverage_summary.csv`.
- Big Sur domain-source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_crm_source_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cusp_source_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_usgs_3dep_source_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cudem_tile_manifest.json`.
- Canonical artifact root: `/Volumes/x10pro/kelp_aef`.

P2-02 source state to preserve:

- AEF selected and downloaded valid `10N` assets for 2018-2022.
- Kelpwatch QA found 32,927 valid Big Sur stations per year and 78,759
  nonzero annual-max station-years inside the footprint.
- CRM is the primary broad topo-bathy source.
- CUSP is shoreline vector provenance/QA only.
- USGS 3DEP has Big Sur land-side support.
- NOAA CUDEM selected zero tiles from the configured index; record this caveat
  but do not block while CRM support is valid.

## Outputs

Big Sur annual label artifacts:

- Annual labels:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_labels_annual.parquet`.
- Annual label manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_labels_annual_manifest.json`.

Big Sur full-grid artifacts:

- Full-grid aligned table:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table.parquet`.
- Full-grid aligned manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table_manifest.json`.
- Full-grid aligned summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_full_grid_training_table_summary.csv`.

Big Sur domain and mask artifacts:

- Aligned NOAA CRM support table:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_noaa_crm.parquet`.
- Aligned NOAA CRM manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_noaa_crm_manifest.json`.
- Aligned NOAA CRM QA summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_noaa_crm_summary.csv`.
- Domain-source comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_domain_source_comparison.csv`.
- Plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask.parquet`.
- Plausible-kelp domain mask manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask_manifest.json`.
- Mask coverage summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_plausible_kelp_domain_mask_summary.csv`.
- Kelpwatch-positive mask retention table:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_plausible_kelp_domain_mask_kelpwatch_retention.csv`.
- Mask depth-bin summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_plausible_kelp_domain_mask_depth_bins.csv`.
- Mask visual QA figure:
  `/Volumes/x10pro/kelp_aef/reports/figures/big_sur_plausible_kelp_domain_mask_qa.png`.

Big Sur model-input and split artifacts:

- CRM-stratified mask-first sample:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked.parquet`.
- Masked sample manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked_manifest.json`.
- Masked sample summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_background_sample_training_table.masked_summary.csv`.
- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_split_manifest.parquet`.

## Config File

Use:

```bash
configs/big_sur_smoke.yaml
```

Keep the Big Sur config region-scoped and explicit. Do not refactor Monterey
and Big Sur into a multi-region config in this task.

## Implementation Plan

1. Confirm Task 43 is complete:
   - Full-grid alignment no longer requires an existing mask.
   - The retained-domain model-input sample is built by the explicit refactored
     sample command/helper.
   - Monterey was rerun through the full downstream workflow.
2. Confirm Task 44 is complete:
   - The default full-grid target-grid policy is Kelpwatch-native UTM 30 m.
   - Monterey was rebuilt and reviewed under that policy.
   - Big Sur config selects the same general target-grid policy.
3. Build Big Sur annual labels:
   - Run `build-labels` against `configs/big_sur_smoke.yaml`.
   - Confirm annual labels cover 2018-2022 and preserve the Phase 1 annual-max
     target columns, especially `kelp_max_y` and `kelp_fraction_y`.
4. Build Big Sur full-grid alignment:
   - Run the general Kelpwatch-native target-grid alignment from Task 44 against
     `configs/big_sur_smoke.yaml`.
   - Confirm the full-grid manifest reports the target-grid policy, CRS,
     origin, spacing, snap residuals, AEF support diagnostics, and
     out-of-coverage counts.
   - Do not implement a Big Sur-only target-grid special case in this task.
5. Align CRM to the Big Sur full grid:
   - Use CRM as the primary topo-bathy source.
   - Keep CUDEM and 3DEP as QA/context where the existing command supports
     them; do not use them as predictors.
   - Keep CUSP as shoreline-vector provenance, not raster depth input.
6. Build the Big Sur plausible-kelp domain mask:
   - Preserve the configured Big Sur mask thresholds and reason-code
     vocabulary.
   - Report retained/dropped cell counts by mask reason and CRM source product.
   - Report Kelpwatch-positive retained and dropped counts by year and reason.
   - Stop and inspect before accepting if any annual-max positive row is
     dropped unexpectedly.
7. Build the Big Sur CRM-stratified mask-first sample:
   - Use the refactored model-input sample builder from Task 43.
   - Verify the sample includes all retained Kelpwatch-observed rows required
     by policy and a CRM-stratified background sample from retained mask cells.
   - Recompute sample weights/population counts using the retained mask domain,
     not the unmasked tile.
8. Write a split manifest without training:
   - Preserve split semantics: train 2018-2020, validation 2021, test 2022.
   - Include `has_complete_features`, `has_target`, `used_for_training_eval`,
     and `drop_reason` fields consistent with downstream model commands.
9. Record the outcome in this task file:
   - Annual label row counts.
   - Full-grid row counts by year and label source.
   - Target-grid manifest summary, including snap residuals and AEF support
     diagnostics produced by the general Task 44 policy.
   - CRM alignment coverage.
   - Mask retained/dropped counts by reason.
   - Kelpwatch-positive retained/dropped counts.
   - Masked sample counts by year, split, label source, and mask reason.
   - Split manifest counts and dropped-feature counts.
   - Any caveat that should carry into P2-04.
10. Update `docs/todo.md` only after the real Big Sur artifacts are written and
   reviewed.

## Suggested Command Order

Exact sample command depends on the Task 43 refactor. Intended order:

```bash
uv run kelp-aef build-labels --config configs/big_sur_smoke.yaml
uv run kelp-aef align-full-grid --config configs/big_sur_smoke.yaml
uv run kelp-aef align-noaa-crm --config configs/big_sur_smoke.yaml
uv run kelp-aef build-domain-mask --config configs/big_sur_smoke.yaml
uv run kelp-aef build-model-input-sample --config configs/big_sur_smoke.yaml
uv run kelp-aef write-split-manifest --config configs/big_sur_smoke.yaml
```

## Validation Command

Focused code validation if Big Sur-specific code changes are needed:

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/test_full_grid_alignment.py tests/test_crm_alignment.py tests/test_domain_mask.py
```

Manual artifact inspection examples:

```bash
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_full_grid_training_table_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/big_sur_plausible_kelp_domain_mask_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/big_sur_plausible_kelp_domain_mask_kelpwatch_retention.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/big_sur_aligned_background_sample_training_table.masked_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/interim/big_sur_split_manifest.parquet'; df=pd.read_parquet(p); print(len(df)); print(df[['split','year','label_source','used_for_training_eval']].value_counts().sort_index())"
```

## Smoke-Test Region And Years

- Region: Big Sur.
- Region shorthand: `big_sur`.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label target: Kelpwatch annual max canopy.
- Mask: configured Big Sur `plausible_kelp_domain`.
- Model-input policy: `crm_stratified_mask_first_sample` inside the retained
  plausible-kelp domain.

## Acceptance Criteria

- Big Sur annual labels exist for 2018-2022 and preserve the Phase 1 annual-max
  schema.
- Big Sur full-grid alignment uses the general Kelpwatch-native UTM target-grid
  policy from Task 44.
- Full-grid manifests include the target-grid and snap-diagnostic fields
  defined by Task 44.
- Big Sur aligned CRM support exists and has valid coverage for the target
  grid, with CUDEM zero-tile support recorded as a source QA caveat.
- Big Sur plausible-kelp mask exists with stable retain/drop flags, reason
  codes, thresholds, and manifest provenance.
- Mask summaries report retained and dropped cells by mask reason and CRM source
  product.
- Kelpwatch-positive retention is explicit by year; any dropped positive rows
  are counted, reviewed, and either accepted with a documented caveat or treated
  as a blocker before P2-04.
- Big Sur CRM-stratified mask-first sample exists at the configured `.masked`
  path and reports counts by year, label source, retained/dropped status, mask
  reason, and sample policy.
- Split manifest exists and uses the configured year holdout: 2018-2020 train,
  2021 validation, 2022 test.
- No model training, model evaluation, calibration, threshold tuning, or Big Sur
  performance interpretation happens in this task.

## Known Constraints And Non-Goals

- Do not start full West Coast scale-up.
- Do not implement a Big Sur-only target-grid policy; Task 44 owns the general
  Kelpwatch-native UTM target-grid implementation.
- Do not switch away from Kelpwatch annual max.
- Do not add annual mean, fall-only, winter-only, or persistence labels.
- Do not train baselines, binary models, conditional canopy models, or hurdle
  models on Big Sur.
- Do not tune thresholds, sample quotas, calibration, or model policy on Big Sur
  held-out rows.
- Do not use CRM, CUDEM, CUSP, USGS 3DEP, depth, elevation, or mask reason as
  model predictors.
- Do not overwrite Monterey artifacts as part of this Big Sur task.
- Do not treat Kelpwatch-style reproduction as independent field-truth biomass
  validation.
