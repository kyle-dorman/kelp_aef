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

## Implementation Note

Current behavior before this task: `align-full-grid` builds the full-grid target
from the AEF raster transform scaled by the label/feature resolution ratio. For
the current 10 m AEF and 30 m Kelpwatch pairing, that means target cells are
AEF 3 x 3 block origins, and Kelpwatch station labels are snapped into that
AEF-derived grid. CRM and the plausible-kelp mask then inherit those
`aef_grid_*` row keys from the full-grid artifact.

New contract for this task: `alignment.full_grid.target_grid_policy` defaults
to `kelpwatch_native_utm_30m` for Monterey and Big Sur. Full-grid alignment
projects the selected Kelpwatch station centers into the AEF/UTM CRS, infers
the 30 m station lattice from projected center coordinates, snaps station
labels to integer cells without interpolating label values, and uses Rasterio
average resampling to map AEF bands onto that Kelpwatch-native target grid.
The existing `aef_grid_row`, `aef_grid_col`, and `aef_grid_cell_id` columns stay
as downstream row keys, but they now identify cells in the Kelpwatch-native
target grid rather than AEF 3 x 3 blocks.

Files to change:

- `src/kelp_aef/alignment/feature_label_table.py` for a reusable
  Kelpwatch-native target-grid builder.
- `src/kelp_aef/alignment/full_grid.py` for full-grid policy loading, snapping,
  AEF support diagnostics, and manifest fields.
- `configs/monterey_smoke.yaml` and `configs/big_sur_smoke.yaml` for explicit
  target-grid policy wiring.
- `tests/test_full_grid_alignment.py` plus CRM/mask tests if inherited row-key
  behavior changes.

Expected validation:

- Code validation:
  `uv run ruff check .`,
  `uv run mypy src`, and
  `uv run pytest tests/test_full_grid_alignment.py tests/test_crm_alignment.py tests/test_domain_mask.py`.
- Monterey artifact validation after the code path is in place:
  rebuild labels, full-grid alignment, CRM, domain mask, model-input sample,
  and downstream model/report artifacts only if row ids or model inputs change.

How Monterey change will be determined: compare the refreshed full-grid
manifest/summary, mask manifest, masked sample summary, and final model
comparison table against the Task 43 outputs. If row ids or sample rows change,
rerun the downstream model/report path and record the metric movement here.

## Outcome

Completed on 2026-05-14.

Implementation:

- Added `alignment.full_grid.target_grid_policy`, with Monterey and Big Sur set
  to `kelpwatch_native_utm_30m`.
- Added a reusable Kelpwatch-native target-grid builder that projects station
  centers to the AEF/UTM CRS, infers the 30 m lattice from station centers, and
  snaps labels to integer cells without interpolating label values.
- Kept downstream row-key columns as `aef_grid_row`, `aef_grid_col`, and
  `aef_grid_cell_id`, but these now identify the Kelpwatch-native target grid
  when the policy is `kelpwatch_native_utm_30m`.
- Mapped AEF bands onto the Kelpwatch-native grid with Rasterio
  `WarpedVRT(..., resampling=Resampling.average)` and target-cell support
  diagnostics.
- Full-grid manifests now include target-grid policy, target CRS, origin,
  spacing, transform, station snap residuals, AEF phase diagnostics, and station
  out-of-coverage counts.
- Monterey hurdle composition required updating the locked validation-selected
  calibrated presence threshold from `0.36` to `0.37`, matching the refreshed
  validation threshold table. This used validation rows only and did not tune on
  2022 test rows.

Refreshed Monterey target-grid diagnostics:

- Target CRS: `EPSG:32610`.
- Target-grid policy: `kelpwatch_native_utm_30m`.
- Grid size: `718` columns x `2619` rows, `1,880,442` lattice cells.
- Center origin/range:
  - `x_min = 582089.9999999346`, `x_max = 603599.9999999346`.
  - `y_max = 4092569.999998226`, `y_min = 4014029.999998226`.
- Corner origin: left `582074.9999999346`, top `4092584.999998226`.
- Spacing: `30.0 m` x `30.0 m`.
- Snap residuals: max XY residual `8.78216077301111e-08 m`, p95
  `7.974449545145035e-08 m`.
- AEF phase diagnostic for 2022: target corner/center is offset about `+5 m`
  in x and `-5 m` in y relative to the 10 m AEF source grid.
- Stations outside selected AEF coverage: `78` per year.

Refreshed Monterey artifact counts:

- Full-grid rows per year:
  - `1,512,411` assumed-background rows.
  - `30,154` Kelpwatch-station rows.
  - `0` missing-feature rows.
- Plausible-kelp mask:
  - `1,542,565` cells.
  - `426,640` retained cells.
  - `1,115,925` dropped cells.
  - `58,497` Kelpwatch-positive cell-year rows retained.
  - `0` Kelpwatch-positive cell-year rows dropped.
- Mask-first model-input sample:
  - `2,133,200` retained-domain population rows.
  - `228,495` sampled rows.
  - `5,579,625` mask-dropped full-grid rows.
  - `0` dropped positives.
  - Per year: `30,154` Kelpwatch-station rows retained, and `15,545`
    assumed-background rows sampled from `396,486` retained-background rows.
  - Background sample weights by stratum: `5.028`, `24.998353`, and
    `99.972860`; Kelpwatch-station sample weight remains `1.0`.

Report comparison against Task 43:

- The Monterey report changed because the target-grid row ids, retained-domain
  population, model-input sample, validation-selected calibrated threshold, and
  full-grid prediction population changed.
- Primary 2022 `full_grid_masked` / `all` rows moved from `630,151` retained
  rows to `426,640` retained rows. Observed 2022 canopy area stayed
  `4,163,014.0 m2`.
- Selected expected-value hurdle:
  - F1 at 10% changed from `0.812370` to `0.840780`.
  - Predicted canopy area changed from `3,495,891 m2` to `3,017,791 m2`.
  - Area bias changed from `-16.0250%` to `-27.5095%`.
- Diagnostic hard-gated hurdle:
  - F1 at 10% changed from `0.825024` to `0.853000`.
  - Predicted canopy area changed from `3,494,314 m2` to `3,067,728 m2`.
  - Area bias changed from `-16.0629%` to `-26.3099%`.
- Ridge baseline:
  - F1 at 10% changed from `0.475989` to `0.602267`.
  - Predicted canopy area changed from `8,415,321 m2` to `7,753,785 m2`.
  - Area bias changed from `+102.1449%` to `+86.2541%`.

Commands run:

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

Validation passed:

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/test_full_grid_alignment.py tests/test_crm_alignment.py tests/test_domain_mask.py
uv run pytest
```

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
