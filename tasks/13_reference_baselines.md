# Task 13: Reference Baselines

## Goal

Implement the Phase 1 reference-baseline work from `docs/todo.md` P1-04
through P1-08 as a sequence of small, validated changes.

The goal is to make the Phase 1 report answer this question directly:

```text
Do AlphaEarth embeddings add value beyond one-year persistence, site memory,
and spatial/temporal location for Monterey annual-max Kelpwatch-style labels?
```

This task should keep the label input fixed as Kelpwatch annual max,
`kelp_max_y` / `kelp_fraction_y`, and compare all baselines against the current
AEF ridge model using the existing year holdout:

- Train: 2018-2020.
- Validation: 2021.
- Test: 2022.

## Fast/Simple Implementation Direction

Do this with cached baseline state and small summary artifacts, not by writing a
giant full-grid prediction dataset for every reference model.

The fast shape is:

- Fit each baseline once and cache the model/output state needed for reuse.
- Use Polars lazy scans for large Parquet inputs whenever possible.
- Build station/sample predictions and metrics on the existing sampled model
  input.
- Build full-grid area calibration as aggregate summary rows, not row-level
  predictions for every reference baseline.
- Keep `predict-full-grid` as the map-oriented inference path for models that
  need row-level full-grid maps. Do not make it materialize no-skill,
  previous-year, climatology, and geographic prediction rows across the full
  grid just to compute area totals.
- Keep `analyze-model` as the analysis/report consumer. It should read cached
  model, prediction, fallback, and area-calibration artifacts; it should not
  retrain models or redo expensive full-grid joins.

Why this matters:

- The full-grid table is tens of millions of rows. Multiplying it by every
  reference model turns a simple baseline comparison into a huge write.
- Previous-year and climatology need cell-key lookups, but those lookups should
  be built once and cached, or computed once as a lazy aggregate. They should
  not be rebuilt or joined inside every streamed `predict-full-grid` batch.
- Full-grid calibration only needs sums, counts, and errors by model, split,
  year, and label source. It does not require a per-cell prediction artifact for
  every reference baseline.

Prefer a Polars implementation for the large-table pieces:

- Add `polars` as a dependency if it is not already present.
- Use `pl.scan_parquet(...)` for full-grid inputs.
- Use lazy joins/group-bys and streaming collection for aggregate outputs.
- Convert to pandas only at small boundaries where existing report code expects
  pandas DataFrames.
- Keep pandas acceptable for small sample artifacts and tests.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Phase 1 plan: `docs/phase1_model_domain_hardening.md`.
- Active Phase 1 checklist: `docs/todo.md`.
- Current baseline training and full-grid prediction implementation:
  `src/kelp_aef/evaluation/baselines.py`.
- Current model-analysis implementation:
  `src/kelp_aef/evaluation/model_analysis.py`.
- Current residual map and area-bias implementation:
  `src/kelp_aef/viz/residual_maps.py`.
- Current CLI command wiring: `src/kelp_aef/cli.py`.
- Current tests:
  - `tests/test_baselines.py`.
  - `tests/test_model_analysis.py`.
- Model input sample:
  `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet`.
- Full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current sample predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`.
- Current full-grid predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`.
- Current metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- Active Phase 1 report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`.

## Outputs

- Updated sample prediction table containing the current no-skill/ridge rows
  plus reference-baseline rows:
  `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`.
- Cached reference-baseline artifacts for fast reuse:
  - train-mean no-skill scalar in the ridge model payload;
  - sample prediction rows for no-skill, ridge, previous-year, climatology, and
    geography;
  - fallback summary rows for previous-year and climatology availability;
  - geographic lat/lon/year model payload;
  - compact full-grid area-calibration rows with a freshness check so
    `analyze-model` can reuse the CSV when model/config/inference inputs have
    not changed.
- Updated baseline metrics table with matching station/sample metrics for all
  models:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- New reference-baseline fallback summary table:
  `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_fallback_summary.csv`.
- New reference-baseline full-grid area-calibration summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.csv`.
  This should be a compact aggregate table, not a row-level prediction dataset.
- Existing ridge map/full-grid prediction artifacts remain map-oriented:
  `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`.
  Do not expand this dataset to contain every reference baseline unless a later
  task explicitly asks for full-grid maps for those baselines.
- Updated Phase 1 model-comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`.
- Updated Phase 1 model-analysis report in Markdown, HTML, and PDF.
- Updated manifests for baseline-state caching, reference area calibration, and
  model analysis.
- Unit tests covering previous-year alignment, climatology fallback behavior,
  geographic baseline training, full-grid area-calibration summaries, and
  report ranking.
- Updated `docs/todo.md` checkboxes for P1-04 through P1-08 only after the
  corresponding implementation and validation pass.

## Config File

Use `configs/monterey_smoke.yaml`.

Keep the existing `models.baselines` input/output paths unless implementation
shows a strong reason to split reference outputs into separate files. The first
implementation should prefer one shared prediction schema with a `model_name`
column, because the existing metrics, maps, and report code already group by
model name.

Expected config addition under `reports.outputs`:

```yaml
reference_baseline_fallback_summary: /Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_fallback_summary.csv
reference_baseline_area_calibration: /Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.csv
```

If a separate geographic model artifact is needed, add it under
`models.baselines` rather than creating a new top-level config section:

```yaml
geographic_model: /Volumes/x10pro/kelp_aef/models/baselines/geographic_ridge_lon_lat_year.joblib
```

Do not add a new CLI command unless the implementation becomes awkward without
one. The preferred shape is:

- `train-baselines`: fit/cache model state and sample predictions.
- `analyze-model`: consume cached model state, sample metrics, fallback
  summaries, and full-grid area-calibration summaries for reporting.
- `predict-full-grid`: continue to produce row-level map predictions for the
  configured map model, not every reference baseline.

## Plan/Spec Requirement

No separate decision note is required before implementation. This task file is
the implementation plan.

Before editing code, confirm that the current prediction schema, model-analysis
table contract, and residual map grouping still match this task. If the current
checkout has already added equivalent baseline surfaces, reuse them instead of
adding duplicates.

Also check whether a prior interrupted `predict-full-grid` run left a partial
`baseline_full_grid_predictions.parquet` dataset. Treat such output as invalid
until it is regenerated by the corrected map-oriented path.

Write a separate decision note only if implementation changes one of these
policy choices:

- Using annual max as the only label input.
- Using the 2018-2020 / 2021 / 2022 year split.
- Treating bathymetry/DEM only as future domain-filter inputs.
- Using the existing continuous regression metric/report loop for these
  reference baselines.

## Model Names

Use stable, explicit model names in prediction and metric artifacts:

- `no_skill_train_mean`: existing global train-mean no-skill row.
- `ridge_regression`: existing AEF ridge model.
- `previous_year_annual_max`: one-year persistence baseline.
- `grid_cell_climatology`: grid-cell/site-memory baseline fit from training
  years.
- `geographic_ridge_lon_lat_year`: lat/lon/year-only baseline using the same
  regularized regression family as the AEF ridge baseline.

If a model name changes during implementation, update the report text, tests,
and this task file in the same change.

## Implementation Plan

### P1-04: Previous-Year Station Baseline

Add a previous-year annual-max baseline on retained Kelpwatch-station rows.

Implementation notes:

- Add reference-baseline helpers in a small module such as
  `src/kelp_aef/evaluation/reference_baselines.py`, or split helpers out of
  `baselines.py` if that is cleaner.
- Prefer Polars lazy scans/joins for large lookup construction and aggregation.
  Pandas is fine for the sampled model frame if that keeps the existing metric
  code simple.
- Use a stable cell key for year-to-year joins. Prefer `aef_grid_cell_id`; fall
  back to `aef_grid_row` plus `aef_grid_col`; fail clearly if neither key is
  available.
- For each row in year `Y`, predict `kelp_fraction_y` from the same cell in
  year `Y - 1`.
- For validation, require predictions to use 2020 rows for 2021 labels.
- For test, require predictions to use 2021 rows for 2022 labels.
- For train metrics, emit rows only where a previous in-scope year exists, so
  2019 can use 2018 and 2020 can use 2019. Do not fabricate 2018 persistence
  predictions.
- Preserve the existing prediction schema:
  `model_name`, `split`, `year`, target columns, clipped prediction columns,
  residual columns, and provenance columns.
- Add a fallback/missing-history summary row for previous-year prediction
  counts, even if all validation/test rows are covered.
- Keep reusable outputs cached. Do not add a separate lookup-state directory
  unless the lazy full-grid calibration becomes a measured bottleneck.

Focused validation:

```bash
uv run pytest tests/test_baselines.py
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
```

Acceptance:

- `baseline_sample_predictions.parquet` includes
  `previous_year_annual_max` rows.
- Validation rows document 2020 -> 2021 persistence.
- Test rows document 2021 -> 2022 persistence.
- Metrics are grouped by split and label source where available.

### P1-05: Previous-Year Full-Grid Area Calibration

Extend the previous-year baseline to full-grid area calibration without writing
row-level full-grid predictions for the previous-year model.

Implementation notes:

- Do not extend `predict-full-grid` to write `previous_year_annual_max` rows
  across the full grid.
- Use Polars lazy scans over
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Compute previous-year full-grid area calibration with one lazy self-join, or
  an equivalent cached lookup, using the same stable cell key selected for
  P1-04.
- Aggregate directly to summary rows by model, split, year, and label source.
- Do not emit 2018 full-grid persistence rows unless a configured 2017 input is
  available, which is out of scope for Phase 1.
- Keep `label_source` and `is_kelpwatch_observed` provenance in aggregate
  grouping where available.
- Write previous-year rows to
  `reference_baseline_area_calibration.csv`.
- Keep the existing `baseline_full_grid_predictions.parquet` path reserved for
  row-level map predictions.

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Acceptance:

- The compact area-calibration table includes `previous_year_annual_max`.
- The test-year full-grid calibration uses 2021 -> 2022.
- The report or companion tables separate Kelpwatch-station rows from
  assumed-background rows for area calibration.
- `baseline_full_grid_predictions.parquet` is not multiplied by reference
  models.

### P1-06: Grid-Cell Climatology Baseline

Add a grid-cell/site-memory climatology baseline.

Implementation notes:

- Fit climatology from training years only, 2018-2020.
- For validation and test, predict each cell as the mean training-year
  `kelp_fraction_y` for the same cell.
- For train rows, use leave-one-year-out training climatology when possible, so
  a row's own label is not part of its prediction.
- Define and implement a deterministic fallback order for rows without enough
  cell history:
  1. cell training mean;
  2. label-source training mean when `label_source` is present;
  3. global training mean.
- Write fallback counts by split, year, label source, and fallback reason to
  `reference_baseline_fallback_summary.csv`.
- Use the existing prediction schema for station/sample predictions.
- For full-grid calibration, aggregate lazily to
  `reference_baseline_area_calibration.csv` instead of writing row-level
  climatology predictions.
- Cache the fallback summary and full-grid area-calibration output. Rebuild the
  climatology lookup lazily only when the compact calibration table is missing
  or stale.

Focused validation:

```bash
uv run pytest tests/test_baselines.py tests/test_model_analysis.py
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Acceptance:

- Sample predictions include `grid_cell_climatology`.
- Full-grid area calibration includes `grid_cell_climatology`.
- Fallback counts are written and linked from the report.
- Area calibration is reported for validation and test years.

### P1-07: Lat/Lon/Year Geographic Baseline

Add a location-only geographic baseline.

Implementation notes:

- Train a ridge pipeline on `longitude`, `latitude`, and `year`, using the same
  split policy and target as the AEF ridge model.
- Use the existing ridge alpha grid unless a separate
  `geographic_alpha_grid` is added to config.
- Select alpha on validation RMSE only. Do not tune on the 2022 test split.
- Save a compact model payload for reuse.
- Emit `geographic_ridge_lon_lat_year` rows into sample predictions.
- For full-grid calibration, either:
  - use Polars expressions from cached linear coefficients if straightforward;
  - or stream feature batches through the cached model and aggregate metrics
    without writing row-level predictions.
- Keep missing-feature and missing-coordinate diagnostics explicit in the
  baseline manifest.
- Only write row-level full-grid geographic predictions if a later task asks
  for geographic baseline maps.

Focused validation:

```bash
uv run pytest tests/test_baselines.py
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Acceptance:

- Sample predictions include `geographic_ridge_lon_lat_year`.
- Full-grid area calibration includes `geographic_ridge_lon_lat_year`.
- Validation alpha selection is recorded.
- Metrics use the same split, label-source grouping, and thresholds as ridge.

### P1-08: Report Ranking And Interpretation

Update the Phase 1 report so it ranks reference baselines against the AEF ridge
model.

Implementation notes:

- Update `model_analysis_phase1_model_comparison.csv` so it includes every
  model listed above for station/sample metrics and full-grid area calibration.
- Read full-grid calibration from
  `reference_baseline_area_calibration.csv`, not from a giant multi-model
  full-grid prediction dataset.
- Add ranking columns or a stable sort policy that makes the key comparison
  explicit:
  - primary station skill: validation/test MAE or RMSE on
    `kelpwatch_station` rows;
  - threshold skill: F1 at 10 percent canopy where available;
  - full-grid calibration: absolute area percent bias for validation/test.
- Add a short report section named `Reference Baseline Ranking`.
- Interpret the comparison as Kelpwatch-style label reproduction, not field
  truth.
- State plainly whether ridge beats previous-year persistence, climatology, and
  geography on station skill and whether it is better calibrated on the full
  grid.
- Keep 2022 as the held-out test split. Do not rewrite model choices using
  2022 results.

Focused validation:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Acceptance:

- The report includes `Reference Baseline Ranking`.
- The comparison table includes no-skill, ridge, previous-year, climatology,
  and geography rows.
- Pixel skill and full-grid area calibration are visible for every baseline
  where the metric is meaningful.
- The report separates Kelpwatch-station and assumed-background interpretation.
- `analyze-model` uses cached/small artifacts and does not rerun expensive
  full-grid model prediction.
- P1-04 through P1-08 can be marked complete in `docs/todo.md`.

## Validation Command

Use focused validation while implementing each slice. Before closing the whole
Reference Baselines task, run:

```bash
make check
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Run `predict-full-grid` and `map-residuals` only when validating that the
configured map-oriented ridge prediction artifact still works. They should not
be required just to add or compare reference baselines.

Inspect the updated Phase 1 Markdown report and the model-comparison CSV after
the final `analyze-model` run.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split policy: train 2018-2020, validation 2021, test 2022.
- Primary report split/year: test 2022.
- Label target: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Features for AEF ridge: AEF annual bands `A00-A63`.
- Features for geographic baseline: `longitude`, `latitude`, `year`.

## Acceptance Criteria

- Previous-year station predictions exist and use 2020 -> 2021 for validation
  and 2021 -> 2022 for test.
- Previous-year full-grid area calibration is reported by split/year and label
  source without materializing row-level previous-year full-grid predictions.
- Grid-cell climatology predictions exist for sample outputs, and full-grid
  climatology area calibration exists as compact summary rows.
- Climatology fallback counts are written and reported.
- The geographic baseline trains only on lat/lon/year and uses the same split
  policy as ridge.
- The geographic baseline has sample predictions and full-grid area
  calibration, without row-level full-grid geographic predictions unless maps
  are explicitly needed.
- The model-comparison table ranks no-skill, ridge, previous-year,
  climatology, and geography under the same metric contract.
- The Phase 1 report includes a short interpretation of whether AEF ridge adds
  value beyond persistence, site memory, and geography.
- Tests cover the new joins, fallback behavior, model rows, and report output.
- Full validation passes.
- `docs/todo.md` is updated to mark P1-04 through P1-08 complete only after the
  corresponding validated implementation has landed.

## Known Constraints Or Non-Goals

- Do not change the label input away from annual max.
- Do not add alternative seasonal targets in this task.
- Do not introduce bathymetry/DEM masking in this task.
- Do not add binary, hurdle, tree, MLP, or deep spatial models in this task.
- Do not expand `baseline_full_grid_predictions.parquet` into a multi-model
  reference-baseline dataset.
- Do not rebuild previous-year or climatology joins inside every streamed
  prediction batch.
- Do not start full West Coast scale-up.
- Do not bulk-download any new AlphaEarth collection.
- Do not tune model choices on the 2022 test split.
- Do not treat Kelpwatch-derived labels as independent field truth.
