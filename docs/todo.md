# Phase 1 TODO

Status: Phase 1 planning is active as of 2026-05-08.

Phase 1 theme: harden the Monterey annual-max pipeline before scale-up. The
goal is to make each model comparison answer whether AlphaEarth embeddings add
value beyond persistence, site memory, and geography inside a physically
plausible kelp domain.

Phase 1 plan:

```text
docs/phase1_model_domain_hardening.md
```

Closed Phase 0 report snapshot:

```text
docs/report_snapshots/monterey_phase0_model_analysis.md
```

Active Phase 1 report outputs:

```text
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf
```

Default validation loop for implemented Phase 1 tasks:

```bash
make check
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Use the relevant subset when a task touches only one stage, but every model or
masking change should end with an updated model-analysis report.

## Phase 1 Scope

- [x] Select Phase 1 direction.
  - Selected: model and domain hardening for the Monterey annual-max pipeline.
- [x] Keep the label input fixed.
  - Use Kelpwatch annual max canopy, `kelp_max_y` / `kelp_fraction_y`.
  - Alternative temporal labels such as annual mean, fall-only, winter-only, or
    multi-season persistence are out of scope for Phase 1.
  - Binary targets derived by thresholding annual max are in scope.
- [x] Keep the smoke region and years fixed unless a later task explicitly
  chooses a second small region.
  - Region: Monterey Peninsula.
  - Years: 2018-2022.
  - Split: train 2018-2020, validate 2021, test 2022.
- [x] Keep scale-up out of scope.
  - Do not start full West Coast processing in Phase 1.

## Planning And Harness

- [x] P1-01: Add a Phase 1 run/report harness.
  - Goal: Make each implemented change visible in the report without changing
    model behavior yet.
  - Output: Phase 1 report sections or tables that can show current model,
    reference baselines, domain mask status, pixel metrics, and area
    calibration.
  - Validation: rerun `analyze-model` and confirm the report still reproduces
    the current ridge reference result.
  - Plan: `tasks/12_phase1_planning_harness.md`.
  - Completed: added `Phase 1 Harness Status` to the Phase 1 model-analysis
    report.
- [x] P1-02: Add a compact model comparison table contract.
  - Goal: Standardize columns for model name, split, label source, mask status,
    pixel skill, threshold skill, and area calibration.
  - Output: report table contract and generated comparison table.
  - Validation: existing no-skill and ridge rows appear unchanged.
  - Plan: `tasks/12_phase1_planning_harness.md`.
  - Completed: wrote
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`.
- [x] P1-03: Add stage-to-stage row-count and drop-rate checks to the report.
  - Goal: Catch data plumbing changes before interpreting model changes.
  - Output: row-count, missing-feature, label-source, and mask-status summary.
  - Validation: report shows current row counts and missing-feature
    drop rates.
  - Plan: `tasks/12_phase1_planning_harness.md`.
  - Completed: wrote
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_data_health.csv`.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    and `make check`.

## Reference Baselines

- [x] P1-04: Add a previous-year annual-max baseline on Kelpwatch-station rows.
  - Goal: Test whether AEF beats one-year persistence.
  - Output: previous-year predictions and station-row metrics.
  - Validation: validation uses 2020 -> 2021 and test uses 2021 -> 2022.
  - Plan: `tasks/13_reference_baselines.md`.
- [x] P1-05: Extend previous-year baseline to full-grid area calibration.
  - Goal: Compare persistence against ridge on full-grid annual area totals.
  - Output: compact full-grid area-calibration rows, not row-level
    previous-year prediction rows.
  - Validation: report separates Kelpwatch-station and assumed-background rows.
  - Plan: `tasks/13_reference_baselines.md`.
- [x] P1-06: Add a station or grid-cell climatology baseline.
  - Goal: Test whether AEF beats site memory from training years.
  - Output: climatology predictions and fallback policy for cells without
    training history.
  - Validation: report documents fallback counts and area calibration.
  - Plan: `tasks/13_reference_baselines.md`.
- [x] P1-07: Add a lat/lon/year-only geographic baseline.
  - Goal: Test whether AEF beats spatial and temporal location alone.
  - Output: geographic baseline model, predictions, and metrics.
  - Validation: same splits and report metrics as ridge.
  - Plan: `tasks/13_reference_baselines.md`.
- [x] P1-08: Update the report to rank all reference baselines against ridge.
  - Goal: Make the "AEF adds value" question explicit.
  - Output: comparison table and short interpretation section.
  - Validation: report includes pixel skill and area calibration for every
    baseline.
  - Validation passed:
    `make check`,
    `uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml`,
    and `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`.
  - Plan: `tasks/13_reference_baselines.md`.

## Bathymetry And DEM Domain Filter

- [x] P1-09: Choose the bathymetry and DEM source inputs for Monterey.
  - Goal: Record allowed domain-filter inputs and thresholds before coding.
  - Plan: `tasks/14_bathymetry_dem_source_plan.md`.
  - Completed: `docs/phase1_bathymetry_dem_source_decision.md`.
  - Validation: docs-only diff inspection; no pipeline behavior changes yet.
- [ ] P1-10a: Add NOAA CUDEM / Coastal DEM query and download scripts.
  - Goal: Create package-backed query and downloader commands for the
    preferred Monterey topo-bathy source.
  - Plan: `tasks/15_download_noaa_cudem.md`.
  - Validation: `make check` and a dry-run command that writes a manifest to
    `/private/tmp`.
- [x] P1-10b: Add NOAA CUSP shoreline query and download scripts.
  - Goal: Create package-backed query and downloader commands for
    shoreline-side classification.
  - Plan: `tasks/16_download_noaa_cusp.md`.
  - Validation: `make check` and a dry-run command that writes a manifest to
    `/private/tmp`.
  - Validation passed:
    `uv run pytest tests/test_noaa_cusp.py`,
    `uv run kelp-aef query-noaa-cusp --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/noaa_cusp_query_manifest_dry_run.json`,
    `uv run kelp-aef download-noaa-cusp --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/noaa_cusp_query_manifest_dry_run.json --manifest-output /private/tmp/noaa_cusp_source_manifest_dry_run.json`,
    and `make check`.
- [x] P1-10c: Add USGS 3DEP land DEM fallback query and download scripts.
  - Goal: Create package-backed query and downloader commands for the U.S.
    land-side fallback.
  - Plan: `tasks/17_download_usgs_3dep.md`.
  - Validation: `make check` and a dry-run command that writes a manifest to
    `/private/tmp`.
  - Validation passed:
    `uv run pytest tests/test_usgs_3dep.py`,
    `uv run kelp-aef query-usgs-3dep --config configs/monterey_smoke.yaml --dry-run --manifest-output /private/tmp/usgs_3dep_query_manifest_dry_run.json`,
    `uv run kelp-aef download-usgs-3dep --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/usgs_3dep_query_manifest_dry_run.json --manifest-output /private/tmp/usgs_3dep_source_manifest_dry_run.json`,
    metadata-only query for inspection at `/private/tmp/usgs_3dep_query_manifest.json`,
    download dry-run for that query at `/private/tmp/usgs_3dep_source_manifest.json`,
    and `make check`.
- [ ] P1-11: Align bathymetry and DEM to the 30 m target grid.
  - Goal: Produce one depth/elevation row per existing full-grid cell.
  - Output: aligned bathymetry/DEM table or raster plus QA summary.
  - Validation: row counts match the full-grid target grid for configured
    years or the static grid.
- [ ] P1-12: Build the first plausible-kelp domain mask.
  - Goal: Exclude land, implausibly deep water, and other impossible cells.
  - Output: mask artifact with reason codes and coverage table.
  - Validation: report shows retained/dropped cells and retained Kelpwatch
    positives by year.
- [ ] P1-13: Apply the domain mask to full-grid inference/reporting first.
  - Goal: Measure how much area leakage is off-domain before retraining.
  - Output: masked full-grid area-bias rows and residual maps.
  - Validation: unmasked and masked area calibration appear side by side.
- [ ] P1-14: Apply the domain mask to training and sampling.
  - Goal: Train only on physically plausible cells unless explicitly comparing
    against the unmasked run.
  - Output: masked model-input sample and manifest.
  - Validation: rerun baselines and report the effect on station skill and
    full-grid calibration.
- [ ] P1-15: Add mask-aware residual diagnostics.
  - Goal: Explain false positives and underprediction by domain-mask reason,
    depth/elevation bin, label source, and observed canopy bin.
  - Output: residual taxonomy tables and report figures.
  - Validation: top residuals have mask/depth/elevation context.

## Imbalance-Robust Models

- [ ] P1-16: Add class and target-balance diagnostics for annual max.
  - Goal: Make imbalance visible before changing objectives.
  - Output: positive-rate, high-canopy-rate, and background-rate summaries by
    split, label source, and mask status.
  - Validation: report reproduces Phase 0 class imbalance.
- [ ] P1-17: Add annual-max binary threshold comparison on the validation year.
  - Goal: Choose candidate binary targets derived only from annual max.
  - Output: threshold table for 1%, 5%, 10%, and any retained diagnostic
    thresholds.
  - Validation: threshold choice uses validation data, not the 2022 test split.
- [ ] P1-18: Train a balanced binary presence model.
  - Goal: Reduce background leakage with an objective designed for imbalance.
  - Output: class-weighted or balanced-sampling classifier and probability
    predictions.
  - Validation: report includes AUROC, AUPRC, precision-recall, selected
    threshold, and full-grid positive-area behavior.
- [ ] P1-19: Calibrate binary probabilities and thresholds on validation.
  - Goal: Separate ranking skill from area calibration.
  - Output: calibrated probabilities or selected thresholds plus calibration
    tables.
  - Validation: calibration is fit on 2021 and evaluated on 2022.
- [ ] P1-20: Train a conditional canopy model for positive or likely-positive
  cells.
  - Goal: Address high-canopy shrinkage separately from presence detection.
  - Output: conditional continuous model and positive-cell residual table.
  - Validation: report shows high-canopy residual bins against the ridge
    baseline.
- [ ] P1-21: Compose a first hurdle model.
  - Goal: Combine presence probability and conditional canopy amount into a
    full-grid annual-max prediction.
  - Output: hurdle predictions, maps, metrics, and area-bias rows.
  - Validation: compare against previous-year, climatology, geographic, and
    ridge baselines.
- [ ] P1-22: Test one capped-weight or stratified-background continuous model.
  - Goal: Check whether a simpler continuous objective can compete with the
    hurdle model without collapsing or leaking positives.
  - Output: weighted or stratified model comparison row.
  - Validation: report shows station skill, background leakage, and full-grid
    area calibration.
- [ ] P1-23: Select the best Phase 1 model policy or document failure.
  - Goal: Close Phase 1 with a defensible model/mask/calibration decision.
  - Output: Phase 1 closeout section in the report and updated decision note.
  - Validation: the selected policy beats meaningful reference baselines or the
    report clearly explains why it does not.

## Explicit Non-Goals

- Do not evaluate alternative temporal label inputs beyond annual max.
- Do not use fall-only, winter-only, annual mean, or multi-season persistence
  labels as Phase 1 targets.
- Do not start full West Coast scale-up.
- Do not introduce deep spatial models before the tabular and calibration
  questions are resolved.
- Do not treat bathymetry/DEM as predictors in Phase 1 unless explicitly
  approved later; use them first for domain filtering and diagnostics.
- Do not tune on the 2022 test split.
