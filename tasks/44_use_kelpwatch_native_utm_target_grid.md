# Task 44: Use Kelpwatch-Native UTM Target Grid

## P2 Mapping

P2-03b: Make the Kelpwatch-native UTM 30 m lattice the general target-grid
policy before Big Sur artifact generation.

## Goal

Refactor full-grid alignment so the canonical 30 m target grid is derived from
the Kelpwatch station lattice in the correct UTM zone, then map AEF, CRM, and
mask metadata onto that grid.

This is a general alignment-policy change, not a Big Sur-only special case.
Task 43 proved the workflow refactor is behavior-preserving. Use that stable
baseline to make this grid-policy change separately and measure whether it
changes Monterey outputs before applying the same policy to Big Sur.

## Inputs

- Completed workflow refactor:
  `tasks/43_refactor_mask_first_alignment_workflow.md`.
- Monterey config: `configs/monterey_smoke.yaml`.
- Big Sur config: `configs/big_sur_smoke.yaml`.
- Current full-grid alignment implementation:
  `src/kelp_aef/alignment/full_grid.py`.
- Current station alignment helpers:
  `src/kelp_aef/alignment/feature_label_table.py`.
- Current CRM/domain-mask implementations:
  - `src/kelp_aef/domain/crm_alignment.py`.
  - `src/kelp_aef/domain/domain_mask.py`.
- Current tests:
  - `tests/test_full_grid_alignment.py`.
  - `tests/test_crm_alignment.py`.
  - `tests/test_domain_mask.py`.
  - downstream model/report tests as needed.
- Canonical artifact root: `/Volumes/x10pro/kelp_aef`.

## Outputs

Code/config outputs:

- A package-backed target-grid mode that builds a region target grid from the
  projected Kelpwatch station lattice rather than from AEF 3 x 3 block origins.
- Config wiring that makes this the default target-grid policy for Monterey and
  Big Sur full-grid alignment.
- Manifest fields that report:
  - target grid policy,
  - target CRS,
  - grid origin and spacing,
  - Kelpwatch station snap residuals,
  - AEF-to-target-grid phase diagnostics,
  - any stations outside selected AEF coverage.
- Tests covering target-grid construction, label snapping, and AEF mapping onto
  a Kelpwatch-native grid.

Refreshed Monterey artifact outputs:

- Full-grid alignment:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Full-grid manifest:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json`.
- Aligned CRM:
  `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`.
- Plausible-kelp mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Mask-first CRM-stratified model-input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`.
- Refreshed Monterey model, prediction, residual, visualizer, and report
  artifacts if the target-grid change alters row ids, sample rows, or model
  inputs.

## Config File

Use Monterey first:

```bash
configs/monterey_smoke.yaml
```

Update Big Sur config only enough to select the same general target-grid policy;
do not generate Big Sur artifacts in this task.

## Implementation Plan

1. Write a short implementation note before editing:
   - Current AEF-derived target-grid behavior.
   - Proposed Kelpwatch-native UTM target-grid contract.
   - Files to change.
   - Expected artifacts and validation.
   - How to determine whether Monterey changed.
2. Add a reusable target-grid builder:
   - Project Kelpwatch station centers into the configured UTM CRS for the
     selected region.
   - Infer the 30 m lattice origin and row/column coordinates from the station
     centers.
   - Snap stations to integer lattice cells without interpolating Kelpwatch
     labels.
   - Record snap residuals and fail or warn if residuals exceed a small
     tolerance.
3. Map AEF features onto the Kelpwatch-native target grid:
   - Do not assume each Kelpwatch target cell is exactly a 3 x 3 block of AEF
     pixels.
   - Use maintained raster operations, such as Rasterio/GDAL resampling or
     area-weighted aggregation, to produce target-grid feature values.
   - Preserve `aef_expected_pixel_count`, `aef_valid_pixel_count`, and
     `aef_missing_pixel_count` diagnostics or replace them with equivalent
     target-cell support diagnostics.
4. Attach labels:
   - Keep one row per target-grid cell and year.
   - Assign `label_source = kelpwatch_station` only for snapped station cells.
   - Assign `label_source = assumed_background` for in-coverage target cells
     without Kelpwatch annual labels.
   - If multiple stations snap to one target cell, aggregate labels using the
     existing deterministic behavior unless the implementation note justifies a
     safer alternative.
5. Align CRM and mask to the same target grid:
   - Ensure CRM alignment reads the target grid from the refreshed full-grid
     artifact or its manifest.
   - Ensure the plausible-kelp mask row keys match the refreshed target-grid
     cell ids.
6. Rebuild Monterey through the full downstream path if model inputs change:
   - `build-labels`.
   - `align-full-grid`.
   - `align-noaa-crm`.
   - `build-domain-mask`.
   - `build-model-input-sample`.
   - Downstream model/report commands as needed to verify whether conclusions
     changed.
7. Record the outcome in this task file:
   - Kelpwatch target-grid CRS, origin, spacing, row/column range, and row
     count.
   - Snap residual summary by region/year.
   - AEF phase/support diagnostics.
   - Full-grid row counts by year and label source.
   - Mask retained/dropped counts and dropped positives.
   - Model-input sample counts and sample weights.
   - Whether the Monterey report changed relative to the Task 43 byte-identical
     closeout result.
8. Update `docs/todo.md` only after Monterey has been rebuilt and reviewed.

## Suggested Command Order

Exact commands may change with implementation, but the intended Monterey order
is:

```bash
uv run kelp-aef build-labels --config configs/monterey_smoke.yaml
uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml
uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml
uv run kelp-aef build-model-input-sample --config configs/monterey_smoke.yaml
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml
uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml
uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Validation Command

Code validation:

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/test_full_grid_alignment.py tests/test_crm_alignment.py tests/test_domain_mask.py
```

Full validation when Monterey artifacts are refreshed:

```bash
uv run pytest
```

Manual artifact checks:

```bash
uv run python -c "import json; p='/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json'; print(json.dumps(json.load(open(p)).get('target_grid', {}), indent=2))"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/aligned_full_grid_training_table_summary.csv'; print(pd.read_csv(p).to_string())"
uv run python -c "import pandas as pd; p='/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.masked_summary.csv'; print(pd.read_csv(p).to_string())"
```

## Smoke-Test Region And Years

- Primary region: Monterey Peninsula.
- Secondary config update only: Big Sur.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label target: Kelpwatch annual max canopy.
- Mask: configured `plausible_kelp_domain`.
- Model-input policy: `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- The default full-grid target-grid policy is Kelpwatch-native UTM 30 m, not
  AEF-derived 3 x 3 block centers.
- Kelpwatch labels are snapped to their native UTM lattice without label-value
  interpolation.
- AEF features are mapped to the Kelpwatch-native target grid with maintained
  geospatial/raster operations.
- Full-grid manifests report target-grid policy, CRS, origin, spacing, snap
  residuals, AEF phase diagnostics, and out-of-coverage counts.
- Monterey artifacts are rebuilt through the mask-first model-input sample, and
  downstream model/report artifacts are refreshed if model inputs changed.
- The task outcome explicitly states whether Monterey report metrics changed
  relative to the Task 43 refactor result.
- Big Sur artifact generation remains out of scope and is left for Task 45.

## Known Constraints And Non-Goals

- Do not start Big Sur artifact generation in this task.
- Do not start full West Coast scale-up.
- Do not change the annual-max label target.
- Do not add CRM, depth, mask reason, or other domain variables as model
  predictors.
- Do not tune sampling quotas, thresholds, calibration, or model policy.
- Do not treat Kelpwatch-style reproduction as independent field-truth biomass
  validation.
