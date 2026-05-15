# Task 57: Hex-Aggregate The Binary-Presence Diagnostic Map

## Goal

Replace the unreadable point-level binary-presence diagnostic map with a
readable 1 km hex summary map for pooled binary predictions.

The current 30 m point/scatter map is too dense to interpret in the report.
The map should instead summarize observed and predicted binary support over
larger spatial neighborhoods so reviewers can see where pooled support is
missing, leaking, or spatially shifted.

The main question is:

```text
Where does the pooled binary support model overpredict or underpredict
Kelpwatch-style annual-max presence at an interpretable spatial scale?
```

Frame results as Kelpwatch-style annual maximum reproduction, not independent
field-truth biomass validation.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- Pooled binary prediction artifacts:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
- Existing coordinate and grid columns in prediction/full-grid tables:
  - `longitude`;
  - `latitude`;
  - `aef_grid_row`;
  - `aef_grid_col`;
  - any projected target-grid x/y columns already present;
  - `kelp_fraction_y`;
  - `kelp_max_y`;
  - calibrated binary probability and selected binary class columns;
  - split, year, evaluation scope, label source, and mask-status columns.
- Existing binary map/report code:
  - `src/kelp_aef/evaluation/binary_presence.py`
  - `src/kelp_aef/evaluation/model_analysis.py`

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

Write figure, table, and manifest artifacts under `/Volumes/x10pro/kelp_aef`.
Expected outputs:

- `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_binary_presence_hex_1km.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_binary_presence_hex_1km.csv`
- `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_pooled_binary_presence_hex_manifest.json`

The exact names can vary if they fit the existing report-output naming better,
but they must be distinct from the old point-level map artifacts.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Add
path-explicit figure/table/manifest outputs under the existing report or binary
presence output block. Do not hard-code output paths in implementation code.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- where the hex aggregation lives, such as the binary-presence map helper
  versus the Phase 2 report helper;
- the projected coordinate system used for hex construction;
- the exact 1 km hex definition;
- the binary probability/class columns used;
- how Big Sur and Monterey panels are separated;
- whether the old point-level map is retained as a legacy artifact.

Implementation note:

- Hex aggregation lives in `src/kelp_aef/evaluation/binary_presence_hex.py`,
  with `src/kelp_aef/evaluation/model_analysis.py` only loading config,
  writing outputs during `analyze-model`, and embedding the figure in the
  Phase 2 report.
- Hex construction uses projected `EPSG:32610` coordinates from
  longitude/latitude via Rasterio; if input projected columns are added later,
  the current explicit transform can be replaced without changing the CSV
  contract.
- Each hex is a deterministic flat-top regular hexagon with `1,000 m`
  flat-to-flat diameter, side length `1,000 / sqrt(3) m`, global origin
  `(0, 0)`, and cube-rounded axial assignment so each retained 30 m cell maps
  to one hex.
- The observed target is `binary_observed_y`; the probability input is raw
  `pred_binary_probability`, converted through the configured binary
  calibration payload, and the selected class is
  `calibrated_probability >= validation_max_f1_calibrated threshold`.
- Big Sur and Monterey are separate configured pooled inputs and appear as
  separate figure rows plus `evaluation_region` CSV values.
- The old point-level binary map is retained as a legacy same-region appendix
  artifact, but the Phase 2 report body uses the 1 km pooled hex map.

## Required Analysis

Build 1 km hex bins in projected meters:

- Use the configured target-grid CRS, expected to be `EPSG:32610`, or transform
  from longitude/latitude to that CRS if projected x/y columns are not already
  present.
- Define each hexagon as 1,000 m across flats.
- Assign each retained 30 m cell to exactly one hex.
- Keep the hex geometry or center coordinate in the CSV for later mapping.

Aggregate these values for each hex:

- `evaluation_region`;
- `year`;
- `hex_id`;
- hex center longitude/latitude and projected x/y;
- `n_cells`;
- observed positive count and observed positive rate for
  `annual_max_ge_10pct`;
- mean calibrated binary probability;
- predicted positive count and predicted positive rate;
- true-positive, false-positive, false-negative, and true-negative counts;
- FP rate and FN rate where denominators are nonzero;
- predicted positive rate minus observed positive rate.

The main figure should be readable in the HTML report:

- show Big Sur and Monterey separately, either as stacked region rows or
  separate titled panels;
- include observed positive rate;
- include predicted positive rate or mean predicted probability;
- include predicted-minus-observed rate difference;
- use shared color scales across regions;
- keep point-level 30 m cells out of the main figure;
- avoid four tiny panels that require zooming to understand.

Default color policy:

- rate/probability panels use a shared sequential scale, capped for readability
  but with true values preserved in the CSV;
- difference panel uses a diverging scale centered at zero;
- all clipping/capping values are recorded in the manifest and figure caption.

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_binary_presence.py tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

If the hex helper is implemented outside existing tested modules, add focused
synthetic tests for:

- deterministic hex assignment;
- one-cell-to-one-hex membership;
- observed/predicted rate aggregation;
- empty or denominator-zero FP/FN handling.

For the generated PNG, perform a lightweight nonblank image check and visually
inspect the report figure.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Model context: pooled Monterey+Big Sur binary model evaluated separately on
  each region.
- Primary year: 2022 held-out test rows.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: `annual_max_ge_10pct`.
- Hex size: 1 km across flats.

## Acceptance Criteria

- The Phase 2 report uses the 1 km hex map instead of the unreadable 30 m
  point-level scatter map.
- The hex CSV contains enough fields to audit observed rate, predicted rate,
  probability, FP/FN counts, and residual direction.
- Big Sur and Monterey are clearly separated and labeled.
- The map is readable in the HTML report without zooming into individual
  pixels.
- The manifest records CRS, hex size, filters, input paths, output paths, and
  color-scale clipping/capping rules.
- Held-out 2022 rows are diagnostic only. No threshold, model, mask, or sample
  policy is changed from this map.

## Known Constraints / Non-Goals

- Do not change the binary model, calibration threshold, or selected model
  policy.
- Do not aggregate Monterey and Big Sur into one unlabeled map.
- Do not treat hex aggregation as a new evaluation target.
- Do not download new basemap or source data.
- Do not remove legacy point-level artifacts unless the existing report path
  requires replacement; demoting them to appendix/debug output is fine.

## Outcome

Completed with a Phase 2 `analyze-model` sidecar diagnostic. The regenerated
report embeds the 1 km pooled binary hex figure and no longer uses the old
Big Sur point-level binary map in the Phase 2 report body.

Generated artifacts:

- `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_binary_presence_hex_1km.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_binary_presence_hex_1km.csv`
- `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_pooled_binary_presence_hex_manifest.json`

Final artifact check: `1,058` hex rows, `698,670` retained 30 m cells, both
`big_sur` and `monterey` present, nonblank `1800 x 2700` PNG, and
`Pooled Binary Support Hex Map` present in
`/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`.
Follow-up visual polish set Monterey above Big Sur, changed the panel
background from white to light water-blue, removed the local NOAA CUSP
shoreline overlay after review, and widened panel padding so the tall/narrow
Monterey footprint does not look clipped.
