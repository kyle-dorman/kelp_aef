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
    higher-resolution topo-bathy QA source where CUDEM coverage exists.
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
- [x] P1-10d: Add NOAA CRM California mosaic query and download scripts.
  - Goal: Query the current target-grid footprint against NOAA CRM Southern
    California v2 and CRM Volume 7, then write a reviewable manifest before
    any real CRM download.
  - Plan: `tasks/18_query_download_noaa_crm.md`.
  - Output: CRM query manifest, selected product/subset plan, and later source
    manifest under `/Volumes/x10pro/kelp_aef`.
  - Validation: `make check` and a dry-run query command that writes a manifest
    to `/private/tmp`.
  - Constraint: run the query/manifest step first; do not download CRM source
    data until the manifest has been inspected.
  - Validation passed:
    `uv run pytest tests/test_noaa_crm.py tests/test_package.py`,
    `make check`,
    `uv run kelp-aef query-noaa-crm --config configs/monterey_smoke.yaml --skip-remote-checks --manifest-output /private/tmp/noaa_crm_query_manifest.json`,
    `uv run kelp-aef download-noaa-crm --config configs/monterey_smoke.yaml --dry-run --query-manifest /private/tmp/noaa_crm_query_manifest.json --manifest-output /private/tmp/noaa_crm_source_manifest_dry_run.json`,
    and metadata-only query with THREDDS checks at
    `/private/tmp/noaa_crm_query_manifest_remote.json`.
- [x] P1-11: Align NOAA CRM and validate domain sources to the 30 m target grid.
  - Goal: Produce one CRM-derived depth/elevation row per existing full-grid
    cell, using the broad California topo-bathy source selected in P1-10d, and
    validate against downloaded CUDEM, USGS 3DEP, and CUSP source coverage.
  - Plan: `tasks/19_align_noaa_crm_to_target_grid.md`.
  - Output: static aligned CRM table plus manifest, QA summary, and
    cross-source comparison table.
  - Validation: row counts match the full-grid target grid for configured
    years or the static grid; fast-path command succeeds.
  - Completed: added `kelp-aef align-noaa-crm`, wrote
    `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet`, and recorded
    CUSP as shoreline-vector validation rather than raster depth/elevation.
    The full output has 7,458,361 unique target-grid cells, all with valid CRM
    samples; CUDEM QA covers 3,231,173 cells.
  - Validation passed:
    `uv run pytest tests/test_crm_alignment.py tests/test_package.py`,
    `uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml --fast`,
    `uv run kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml`,
    full-output row-count inspection, and `make check`.
- [x] P1-12: Build the first plausible-kelp domain mask.
  - Goal: Exclude land, very deep water, and other impossible cells, starting
    with a permissive depth cutoff such as approximately 100 m.
  - Plan: `tasks/20_build_plausible_kelp_domain_mask.md`.
  - Output: mask artifact with reason codes and coverage table.
  - Validation: report shows retained/dropped cells and retained Kelpwatch
    positives by year.
  - Completed: added `kelp-aef build-domain-mask`, wrote
    `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`,
    and generated coverage, Kelpwatch-positive retention, depth-bin, manifest,
    and visual QA outputs.
  - Result: full mask has 7,458,361 unique target-grid cells, retains 999,519
    cells, drops 6,458,842 cells, and retains all 58,497 Kelpwatch-positive
    cell-year rows in 2018-2022. Five positive rows fall in the 40-50 m QA bin;
    none fall deeper than 50 m.
  - Validation passed:
    `uv run pytest tests/test_domain_mask.py`,
    `uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml --fast`,
    `uv run kelp-aef build-domain-mask --config configs/monterey_smoke.yaml`,
    and `make check`.
- [x] P1-13: Apply the domain mask to full-grid inference/reporting first.
  - Goal: Make the P1-12 plausible-kelp mask the default full-grid reporting
    domain before retraining.
  - Plan: `tasks/21_apply_domain_mask_to_reporting.md`.
  - Output: masked full-grid area-bias rows, masked residual maps, and updated
    report tables where the largest recurring area is the masked domain.
  - Validation: masked-domain area calibration is the primary report scope;
    any unmasked/off-domain numbers are isolated as migration/audit diagnostics
    instead of recurring `all` headline rows.
  - Completed: added reporting-only domain-mask config and shared mask joins for
    residual maps, compact reference area calibration, and model-analysis report
    rows. Training and sampling inputs remain unchanged for P1-14.
  - Result: masked 2022 residual-map/reporting scope retains 999,519 cells;
    masked full-grid report rows use `mask_status = plausible_kelp_domain` and
    `evaluation_scope = full_grid_masked`. Off-domain prediction leakage is
    isolated in
    `/Volumes/x10pro/kelp_aef/reports/tables/off_domain_prediction_leakage_audit.csv`.
  - Validation passed:
    `uv run pytest tests/test_residual_maps.py tests/test_model_analysis.py tests/test_baselines.py`,
    `uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    and `make check`.
- [x] P1-14: Apply the domain mask to training and sampling.
  - Goal: Train only on physically plausible cells unless explicitly comparing
    against the unmasked run.
  - Plan: `tasks/22_apply_domain_mask_to_training_sampling.md`.
  - Output: masked model-input sample and manifest.
  - Validation: rerun baselines and report the effect on station skill and
    masked-domain full-grid calibration. Here, full-grid calibration means all
    retained plausible-kelp mask cells, not the unmasked AEF tile.
  - Completed: added a mask-aware background-sample sidecar that joins the
    P1-12 mask by `aef_grid_cell_id`, carries mask metadata into model-input
    rows, recomputes retained-domain background sample weights, and points
    `train-baselines` at the masked sample by default.
  - Result: masked the existing sample from 1,400,809 rows to 313,954 retained
    plausible-domain rows, with 0 dropped Kelpwatch-observed rows and 0 dropped
    Kelpwatch-positive rows. Retrained baselines, refreshed full-grid
    predictions, residual maps, masked area-bias tables, reference area
    calibration, and the Phase 1 model-analysis report.
  - Validation passed:
    `uv run pytest tests/test_full_grid_alignment.py tests/test_baselines.py tests/test_model_analysis.py`,
    `uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml --fast`,
    masked-sample derivation from the existing full-grid/sample artifacts,
    `uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    and `make check`.
- [x] P1-15: Add mask-aware residual diagnostics.
  - Goal: Explain false positives and underprediction by domain-mask reason,
    depth/elevation bin, label source, and observed canopy bin.
  - Plan: `tasks/23_mask_aware_residual_diagnostics.md`.
  - Output: residual taxonomy tables and report figures.
  - Validation: top residuals have mask/depth/elevation context.
  - Completed: joined retained P1-12 mask context onto primary 2022 masked
    full-grid residual rows, added residual taxonomy summaries by domain
    context, mask reason, and CRM depth/elevation bin, extended top residuals
    with mask/depth/elevation context, and added a report section plus compact
    retained-domain residual figure.
  - Result: the refreshed report uses 999,519 retained plausible-domain
    prediction rows and writes 44 domain-context rows, 2 mask-reason rows, 4
    depth/elevation-bin rows, and 100 top residual context rows.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py tests/test_residual_maps.py`,
    `uv run ruff check .`, `uv run mypy src`,
    `uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    `make check`,
    and manual inspection of
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv`
    and
    `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.domain_context.csv`.
  - Note: this is the last task in the Bathymetry and DEM domain-filter block;
    after it, move to imbalance-aware model diagnostics and objectives.

## Imbalance-Robust Models

- [x] P1-16: Add class and target-balance diagnostics for annual max.
  - Goal: Make imbalance visible before changing objectives.
  - Plan: `tasks/24_class_target_balance_diagnostics.md`.
  - Output: positive-rate, high-canopy-rate, and background-rate summaries by
    split, label source, and mask status.
  - Validation: report reproduces Phase 0 class imbalance.
  - Completed: added package-generated class/target-balance tables and a
    compact class-balance figure to `analyze-model`, with rows for the masked
    model-input sample, retained split manifest, sample predictions, and
    masked full-grid report scope.
  - Result: the primary 2022 masked full-grid report scope has 999,519 rows,
    1.29% positive annual-max rows, 0.41% high-canopy rows, 0.06%
    saturated/near-saturated rows, 98.7% zero rows, and 97.0%
    assumed-background rows. The report now includes `Class And Target Balance`
    before threshold/model-objective changes.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py`,
    `uv run ruff check .`, `uv run mypy src`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    manual inspection of
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv`
    and the refreshed report section, and `make check`.
- [x] P1-17: Add annual-max binary threshold comparison on the validation year.
  - Goal: Choose candidate binary targets derived only from annual max.
  - Plan: `tasks/25_annual_max_binary_threshold_comparison.md`.
  - Output: threshold table for 1%, 5%, 10%, and any retained diagnostic
    thresholds.
  - Validation: threshold choice uses validation data, not the 2022 test split.
  - Completed: package-generated validation-first threshold comparison,
    prevalence, recommendation, figure, and report section. The current
    validation-only recommendation is `annual_max_ge_10pct` for P1-18 candidate
    modeling.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py`, `uv run ruff check .`,
    `uv run mypy src`, and
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`;
    full repo validation passed with `make check`.
- [x] P1-18: Train a balanced binary presence model.
  - Goal: Reduce background leakage with an objective designed for imbalance.
  - Plan: `tasks/26_train_balanced_binary_presence_model.md`.
  - Target: use P1-17's validation-backed `annual_max_ge_10pct`
    (`kelp_fraction_y >= 0.10`, `kelp_max_y >= 90 m2`) as the candidate binary
    target.
  - Output: class-weighted or balanced-sampling classifier and probability
    predictions.
  - Validation: report includes AUROC, AUPRC, precision-recall, selected
    threshold, and full-grid positive-area behavior.
  - Completed: added `kelp-aef train-binary-presence`, configured
    `models.binary_presence`, trained a class-weighted logistic regression for
    `annual_max_ge_10pct`, wrote sample and masked full-grid probability
    predictions, and added a balanced binary model section to the Phase 1
    report.
  - Follow-up: added a 2022 binary full-grid map and a thresholded model
    comparison table that evaluates the balanced binary classifier against the
    continuous reference baselines thresholded at the same `>=10%` annual-max
    target.
  - Result: validation selected probability threshold `0.91` by max F1.
    Held-out 2022 sample metrics are AUPRC `0.945`, AUROC `0.988`, precision
    `0.892`, recall `0.834`, and F1 `0.862`. In the 2022 masked full-grid
    scope, the model predicts 9,840 of 999,519 retained rows positive
    (`0.98%`), or 8,856,000 m2 in 30 m cell area, with assumed-background
    predicted-positive rate `0.21%`.
  - Validation passed:
    `uv run pytest tests/test_binary_presence.py tests/test_model_analysis.py tests/test_baselines.py`,
    `uv run mypy src`,
    `uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    manual inspection of
    `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_thresholded_model_comparison.csv`,
    `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_2022_map.png`,
    and `make check`.
- [x] P1-18a: Add CRM-stratified background sampling for binary model inputs.
  - Goal: Improve the masked binary model-input sample using CRM domain
    context before further binary calibration or hurdle iteration.
  - Plan: `tasks/30_crm_stratified_background_sampling.md`.
  - Output: sidecar CRM-stratified masked sample, manifest, summary table, and
    binary model comparison against the current P1-18 sample.
  - Validation: compare validation/test sample metrics and masked full-grid
    assumed-background false positives by `depth_bin` and `domain_mask_reason`.
  - Constraint: keep all Kelpwatch-observed rows, oversample hard coastal
    assumed-background strata such as `retained_ambiguous_coast`, downsample
    easy retained strata such as `50_100m`, and do not use CRM depth/elevation
    as model predictors.
  - Completed: the sidecar sample is
    `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet`
    with 314,280 rows, compared with 313,954 rows in the current masked sample.
    It keeps all retained Kelpwatch rows and samples assumed-background rows by
    `domain_mask_reason` and `depth_bin`.
  - Comparison:
    `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_crm_stratified_comparison.csv`.
    The sidecar reduces 2022 retained full-grid assumed-background predicted
    positives from 1,999 to 577, including ambiguous-coast assumed-background
    positives from 1,920 to 531. The tradeoff is lower Kelpwatch-station recall
    at the selected threshold: 2021 validation recall drops from 0.870 to 0.850,
    and 2022 test recall drops from 0.834 to 0.785.
  - Validation passed:
    `uv run pytest tests/test_full_grid_alignment.py tests/test_binary_presence.py tests/test_model_analysis.py`,
    `uv run ruff check src/kelp_aef/alignment/full_grid.py src/kelp_aef/evaluation/binary_presence.py tests/test_full_grid_alignment.py tests/test_binary_presence.py`,
    `uv run mypy src/kelp_aef/alignment/full_grid.py src/kelp_aef/evaluation/binary_presence.py`,
    `uv run kelp-aef align-full-grid --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    and `make check`.
- [x] P1-19: Calibrate binary probabilities and thresholds on validation.
  - Goal: Separate ranking skill from area calibration.
  - Plan: `tasks/27_calibrate_binary_probabilities_thresholds.md`.
  - Output: calibrated probabilities or selected thresholds plus calibration
    tables.
  - Validation: calibration is fit on 2021 and evaluated on 2022.
- [x] P1-20: Train a conditional canopy model for positive or likely-positive
  cells.
  - Goal: Address high-canopy shrinkage separately from presence detection.
  - Plan: `tasks/28_train_conditional_canopy_model.md`.
  - Output: conditional continuous model and positive-cell residual table.
  - Validation: report shows high-canopy residual bins against the ridge
    baseline.
  - Completed: added `kelp-aef train-conditional-canopy`, configured
    `models.conditional_canopy`, trained a positive-only ridge model on
    observed `annual_max_ge_10pct` training rows, wrote conditional sample
    predictions, residual bins, comparison rows, compact likely-positive
    diagnostics, and a Phase 1 report section.
  - Result: validation selected `alpha = 100.0`. On held-out 2022
    observed-positive rows, the conditional model reduced area RMSE from
    `248.2 m2` for ridge to `191.8 m2` on the same rows, while still
    underpredicting high-canopy and near-saturated rows.
  - Outcome: even with the strong prior that evaluation rows are already
    observed `annual_max_ge_10pct` positives, simple AEF ridge features still
    predict Kelpwatch-style canopy amount poorly. This improves the ridge
    baseline but does not solve high-canopy magnitude recovery; P1-21 may reduce
    full-grid leakage, but it should not be expected to fix conditional
    high-canopy underprediction by itself.
  - Validation passed:
    `uv run pytest tests/test_conditional_canopy.py tests/test_binary_presence.py tests/test_model_analysis.py tests/test_package.py`,
    `uv run mypy src/kelp_aef/evaluation/conditional_canopy.py src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/cli.py`,
    `uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml`,
    and
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`;
    full repo validation passed with `make check`.
- [x] P1-21: Compose a first hurdle model.
  - Goal: Combine presence probability and conditional canopy amount into a
    full-grid annual-max prediction.
  - Plan: `tasks/29_compose_first_hurdle_model.md`.
  - Output: hurdle predictions, maps, metrics, and area-bias rows.
  - Validation: compare against previous-year, climatology, geographic, and
    ridge baselines.
  - Completed: added `kelp-aef compose-hurdle-model`, configured
    `models.hurdle`, loaded the saved Platt-calibrated binary probability
    model and saved positive-only conditional ridge model without retraining,
    and wrote expected-value plus hard-gated retained-domain full-grid
    predictions.
  - Result: in the 2022 retained-domain complete-feature scope, the primary
    expected-value hurdle predicted `4,321,134 m2` against `4,163,014 m2`
    observed area (`3.8%` area bias, F1 `0.756`). The hard-gated diagnostic
    predicted `4,122,971 m2` (`-1.0%` area bias, F1 `0.774`). Both reduced
    ridge full-grid leakage sharply, but previous-year persistence still had
    better F1 and RMSE in this scope.
  - Output details: wrote
    `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`,
    `/Volumes/x10pro/kelp_aef/interim/hurdle_prediction_manifest.json`,
    `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_residual_by_observed_bin.csv`,
    `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv`,
    and
    `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png`.
  - Validation passed:
    `uv run pytest tests/test_hurdle.py tests/test_binary_presence.py tests/test_conditional_canopy.py tests/test_model_analysis.py tests/test_package.py`,
    `uv run mypy src/kelp_aef/evaluation/hurdle.py src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/cli.py`,
    `uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml`,
    and
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`;
    full repo validation passed with `make check`.
- [x] P1-21a: Complete CRM-stratified background sampling across all model
  families.
  - Goal: Make current versus CRM-stratified sample policy comparisons fair
    across continuous baselines, binary presence, calibration, conditional
    canopy, and hurdle outputs.
  - Plan: `tasks/31_crm_stratified_all_model_comparison.md`.
  - Output: sidecar model artifacts and a report-visible all-model comparison
    table keyed by `sample_policy`.
  - Completed: added current and CRM-stratified `sample_policy` sidecars for
    continuous baselines, binary calibration, conditional-canopy diagnostics,
    and hurdle composition. Current default paths remain intact; CRM sidecars
    are path-distinct.
  - Output details: wrote
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`
    with 528 rows covering `current_masked_sample` and
    `crm_stratified_background_sample` across continuous, binary, conditional,
    and hurdle model families. Conditional canopy reused
    `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_annual_max.joblib`;
    the reuse manifest confirms matching observed-positive support
    (`41,824` rows) at
    `/Volumes/x10pro/kelp_aef/interim/conditional_canopy_reuse_manifest.crm_stratified.json`.
  - Result: in the 2022 retained-domain test/all scope, the CRM-stratified
    ridge sidecar reduced continuous ridge predicted canopy area
    (`10.31M m2` versus `12.12M m2` current) and improved ridge RMSE
    (`0.0378` versus `0.0438`). The CRM calibrated binary sidecar selected
    the validation-max-F1 threshold `0.35` and predicted `7.31M m2` positive
    area versus `8.52M m2` current at the comparable policy. The CRM
    expected-value hurdle predicted `3.98M m2` against `4.16M m2` observed;
    the CRM hard-gated hurdle predicted `3.52M m2` with F1 `0.825`.
  - Validation: rerun the relevant model commands and the Phase 1 report, then
    compare pixel skill, assumed-background leakage, full-grid area behavior,
    and any Kelpwatch-station recall tradeoff under both sample policies.
  - Validation passed:
    focused `uv run ruff check` and `uv run mypy` for the touched evaluation
    modules and tests,
    `uv run pytest tests/test_baselines.py tests/test_binary_presence.py tests/test_conditional_canopy.py tests/test_hurdle.py tests/test_model_analysis.py -q`,
    `uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef calibrate-binary-presence --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    and `make check`.
  - Constraint: CRM depth/elevation context remains a sampling input only, not
    a model feature, and current default artifacts remain available until a
    later policy-selection decision.
- [x] P1-21b: Document the CRM-stratified sampling policy decision.
  - Goal: Promote the Task 31 CRM-stratified sample-policy result from sidecar
    experiment to an explicit Phase 1 design decision before changing default
    pipeline behavior.
  - Plan: `tasks/32_crm_sampling_policy_decision.md`.
  - Output: decision note documenting metric improvements, exact successful
    sidecar quota values, the move to mask-first sampling, the planned 60 m
    maximum-depth mask, and the replacement or reinterpretation of
    `background_rows_per_year`.
  - Validation: docs diff inspection plus metric spot checks against
    `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv`.
  - Completed: accepted
    `docs/phase1_crm_stratified_sampling_policy_decision.md` as the durable
    decision note. CRM-stratified, mask-first retained-domain sampling should
    become the default in P1-21c; executable config behavior was not changed in
    this docs-only task.
- [x] P1-21c: Promote CRM-stratified, mask-first sampling to the default
  masked model-input policy.
  - Goal: Replace the sidecar sampling path with a default retained-domain
    sampler that filters to the plausible-kelp mask before applying background
    quotas.
  - Plan: `tasks/33_promote_crm_stratified_sampling_default.md`.
  - Output: regenerated 60 m plausible-kelp mask, default masked sample,
    model artifacts, manifests, and report outputs using CRM-stratified
    retained-background quotas by default.
  - Validation: rerun the mask, alignment, baseline, binary, calibration,
    conditional, hurdle, and report commands, then run `make check`.
  - Constraint: remove the retained `50_100m` sampling stratum, replace the
    final retained depth bin with `40_60m`, and keep CRM depth/elevation out of
    the feature matrix.
  - Completed: promoted `crm_stratified_mask_first_sample` to the default
    masked sample path and regenerated the 60 m mask, alignment sample,
    baseline, binary, calibration, conditional, hurdle, and model-analysis
    artifacts. The default masked sample manifest records mask-first sampling,
    3,150,755 retained-domain population rows, 311,475 sampled rows, zero
    mask-dropped Kelpwatch-positive rows, and
    `background_rows_per_year_controls_default_masked_workflow: false`.
    Retained depth strata are now `0_40m` and `40_60m`; the historical
    CRM sidecar paths remain disabled for audit until P1-21d.
- [x] P1-21d: Retire sidecar sampling-policy reporting after promotion.
  - Goal: Remove the temporary current-vs-CRM report section once the design
    decision is documented and the CRM-stratified policy is the default.
  - Plan: `tasks/34_remove_sampling_policy_sidecar_reporting.md`.
  - Output: updated model-analysis report and tests that describe the active
    default sampling policy without recurring side-by-side sidecar tables.
  - Validation: `uv run pytest tests/test_model_analysis.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`, and
    `make check`.
  - Completed: removed the temporary report section from the active Phase 1
    report, added the default `crm_stratified_mask_first_sample` policy and
    decision-note pointer to the normal artifact context, and kept the
    all-model sampling-policy CSV as an audit table rather than a headline
    report section. Disabled P1-21a sidecar config paths are now marked as
    audit-only.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    manual grep confirmed the retired section text is absent from the refreshed
    Markdown report, and `make check`.
- [x] P1-21e: Simplify the Phase 1 report story and make maps larger.
  - Goal: Turn the accumulated Phase 1 report into a current-state decision
    report with a clear 2022 retained-domain scoreboard, concise interpretation,
    and readable report-embedded maps.
  - Plan: `tasks/35_simplify_phase1_report_story_and_vertical_maps.md`.
  - Output: refreshed Markdown, HTML, and PDF reports with a simplified main
    body and vertically stacked or otherwise larger map panels for ridge,
    binary presence, and hurdle diagnostics.
  - Validation: focused report/map tests, regenerate map and report artifacts
    with the Monterey smoke config, visually check map orientation, then run
    `make check`.
  - Completed: rewrote the active Phase 1 report as a decision report with a
    primary 2022 retained-domain scoreboard, moved detailed diagnostic
    chronology into the appendix, and regenerated ridge, binary-presence, and
    hurdle maps with larger vertical or two-column layouts.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py tests/test_residual_maps.py tests/test_hurdle.py tests/test_binary_presence.py`,
    `uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef train-binary-presence --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`,
    manual report-heading and PNG-dimension checks, and `make check`.
- [x] P1-22a: Test a capped-weight continuous model.
  - Plan: `tasks/36_test_capped_weight_continuous_model.md`.
  - Goal: Check whether capped retained-background fit weights let a simple
    continuous model compete with the hurdle model without collapsing or leaking
    positives.
  - Output: capped-weight continuous model comparison row.
  - Validation: report shows station skill, background leakage, and full-grid
    area calibration.
  - Completed: added `kelp-aef train-continuous-objective --experiment
    capped-weight`, wrote the capped-weight ridge model/prediction/metric
    artifacts, refreshed the Phase 1 report, and added the capped-weight row to
    the retained-domain scoreboard.
  - Result: capped-weight ridge failed to beat ridge or compete with the
    expected-value hurdle on the 2022 retained-domain all-label row. RMSE was
    `0.0493` and area bias was `+107.5%`, compared with ridge `0.0452` /
    `+102.1%` and expected-value hurdle `0.0322` / `-16.0%`.
  - Follow-up cap sweep: caps `1`, `2`, `5`, `10`, `20`, and `100` all selected
    `alpha=0.01`. Cap `1` had the best full-grid RMSE (`0.0452`) and station
    RMSE (`0.1623`) but still overpredicted area by `+102.1%`; cap `100`
    reduced area bias to `+43.0%` but worsened full-grid RMSE to `0.0529` and
    station RMSE to `0.2288`. No cap resolves the direct-continuous objective
    tradeoff.
  - Validation passed:
    `uv run pytest tests/test_continuous_objective.py tests/test_model_analysis.py`,
    focused `ruff`/`mypy` checks, `uv run kelp-aef
    train-continuous-objective --config configs/monterey_smoke.yaml
    --experiment capped-weight`, and `uv run kelp-aef analyze-model --config
    configs/monterey_smoke.yaml`; final full validation passed with
    `make check`.
- [x] P1-22b: Test a stratified-background continuous model.
  - Plan: `tasks/37_test_stratified_background_continuous_model.md`.
  - Goal: Check whether retained-domain stratum-balanced background weighting
    lets a simple continuous model compete with the hurdle model without
    collapsing or leaking positives.
  - Output: stratified-background continuous model comparison row.
  - Validation: report shows station skill, background leakage, and full-grid
    area calibration.
  - Completed: added the `stratified-background` experiment to
    `kelp-aef train-continuous-objective`, wrote the stratified ridge
    model/prediction/metric artifacts, refreshed the Phase 1 report, and added
    the stratified-background row to the retained-domain scoreboard.
  - Result: the stratified-background ridge reduced retained assumed-background
    leakage to `3.80 M m2` and reduced full-grid area bias to `+40.4%`, compared
    with ridge `+102.1%` and capped-weight ridge `+107.5%`. It did not beat the
    direct continuous baselines on RMSE: 2022 retained-domain RMSE was `0.0542`,
    versus ridge `0.0452`, capped-weight ridge `0.0493`, and expected-value
    hurdle `0.0322`. Station RMSE worsened to `0.2360`, so this does not
    compete with the expected-value hurdle for P1-23.
  - Validation passed:
    `uv run pytest tests/test_continuous_objective.py tests/test_model_analysis.py`,
    `uv run kelp-aef train-continuous-objective --config
    configs/monterey_smoke.yaml --experiment stratified-background`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`, and
    `make check`.
  - Sweep addendum: tested gamma-shrunk and background-budgeted
    stratified-background variants. No variant fixed the one-stage continuous
    tradeoff. Gamma-only variants reduced leakage but retained poor RMSE and
    station skill; budgeted variants recovered station skill but restored
    full-grid overprediction. The best budgeted test station row was
    `ridge_stratified_gamma_050_bg2` (`station RMSE 0.1641`), but its retained
    full-grid area bias was `+105.5%`. The expected-value hurdle remains the
    stronger Phase 1 AEF candidate.
  - Write-up: `docs/phase1_stratified_background_sweep_results.md`.
- [x] P1-22c: Remove failed direct-continuous experiment code paths.
  - Plan: `tasks/38_remove_failed_p1_22_continuous_paths.md`.
  - Goal: Back out the active code/config/report paths for the failed P1-22a
    capped-weight and P1-22b stratified-background direct-continuous
    experiments while preserving the recorded negative results.
  - Output: no active `train-continuous-objective` command, no active
    `models.continuous_objective` config block, no active continuous-objective
    report rows, and updated tests/docs that treat P1-22 as historical.
  - Validation: `uv run pytest tests/test_package.py tests/test_model_analysis.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`, and
    `make check`.
  - Completed: removed the active continuous-objective CLI command, config
    block, implementation module, model-analysis ingestion/report rows, and
    direct-continuous tests. Refreshed the Phase 1 report and comparison table;
    the active scoreboard now contains no capped-weight, stratified-background,
    or continuous-objective rows.
  - Validation passed:
    `uv run pytest tests/test_package.py tests/test_model_analysis.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`, and
    `make check`.
- [ ] P1-23: Select the best Phase 1 model policy or document failure.
  - Plan: `tasks/39_close_phase1_model_policy_and_report.md`.
  - Goal: Close Phase 1 with a defensible model/mask/calibration decision.
  - Output: final Phase 1 closeout report section, tracked report snapshot,
    updated decision note, docs cleanup, and removal of unused active
    config/code paths that are not part of the closeout state.
  - Validation: the selected policy beats meaningful reference baselines or the
    report clearly explains why it does not; the final report does not include
    post-Phase-1 next-step language.

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
