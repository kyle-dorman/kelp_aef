# Project TODO

## Active Phase 2 Plan

Status: Phase 2 selected on 2026-05-14. P2-01 has a task plan; later checklist
items are still planning stubs until their `tasks/` files are written.

Phase 2 theme: test whether the closed Monterey Phase 1 annual-max policy
generalizes to neighboring Big Sur before choosing a broader Phase 3 direction.

Phase 2 plan:

```text
docs/phase2_big_sur_generalization.md
```

Selected Big Sur AEF context:

- STAC item id: `8957`.
- CRS: `EPSG:32610`.
- Bbox:
  `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`.
- Example 2022 AEF asset:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`.

Phase 2 should:

- verify Big Sur AEF, Kelpwatch, and domain-source coverage before model
  interpretation;
- run an early visual QA pass on Big Sur labels, AEF coverage, and domain
  context before model training;
- start with the Phase 1 annual-max target, CRM-stratified mask-first sample
  policy, calibrated binary support model, conditional canopy model, and
  expected-value hurdle policy fixed unless source verification forces a
  change;
- compare Monterey-trained transfer, Big Sur-only training, and pooled
  Monterey+Big Sur training when evaluating on Big Sur;
- compare Big Sur against Monterey on AEF ridge, previous-year persistence,
  expected-value hurdle, and hard-gated diagnostic policy;
- generate a Big Sur results visualizer;
- reserve space for report and visualizer changes, including region/year
  selection if that becomes the cleanest review path;
- close with a Phase 3 recommendation: broader scale-up, simple non-linear
  tabular modeling, temporal-label exploration, ingestion/domain hardening, or
  evaluation-tooling work.

Phase 2 non-goals:

- Do not start full West Coast scale-up.
- Do not switch away from Kelpwatch annual max inside Phase 2.
- Do not choose final thresholds, sample quotas, or model policy by tuning on
  held-out Big Sur test rows. Validation-driven Big Sur-only and pooled
  training comparisons are in scope.
- Do not add bathymetry/DEM as predictors without a later decision.
- Do not write implementation code in task-plan-only passes.

## Phase 2 Checklist

- [x] P2-01: Add Big Sur config and source-manifest plan.
  - Goal: Introduce `big_sur` as a second small region without breaking the
    closed Monterey config.
  - Outputs: region-scoped config paths, Big Sur footprint path, and reviewable
    AEF/Kelpwatch/domain-source manifest expectations.
  - Acceptance: the planned artifact names cannot overwrite Monterey outputs.
  - Plan: `tasks/41_big_sur_config_source_manifest_plan.md`.
  - Completed: added `configs/big_sur_smoke.yaml` with flat `big_sur_`
    artifact names, reviewable source-manifest paths, the user-provided AEF
    STAC metadata, and a P2-02 source-review gate. No source data was
    downloaded.
- [x] P2-02: Verify Big Sur source coverage and early visual QA.
  - Goal: Confirm Big Sur AEF, Kelpwatch, CRM/domain, and shoreline support
    before interpreting any model results, and visually check labels, AEF, and
    domain context before training.
  - Outputs: Big Sur coverage manifests, a short coverage summary, and quick
    visual QA artifacts for labels, AEF coverage, and domain context.
  - Acceptance: source years, domain coverage, and Kelpwatch-positive retention
    are explicit, and obvious visual source/alignment problems have been ruled
    out or recorded.
  - Plan: `tasks/42_verify_big_sur_source_coverage_visual_qa.md`.
  - Completed: Big Sur AEF selected and downloaded valid 2018-2022 `10N`
    assets; Kelpwatch QA found 32,927 valid stations per year and 78,759
    nonzero annual-max station-years inside the footprint; CRM/CUSP/3DEP
    support was verified with local source manifests; CUDEM selected zero tiles
    from the configured index and is recorded as a caveat, not a blocker.
    Comparable Big Sur and Monterey source-coverage CSV/PNG/HTML QA artifacts
    were written.
- [x] P2-03a: Refactor mask-first alignment workflow and verify Monterey.
  - Goal: Make the retained-domain sample an explicit native step after
    full-grid alignment, CRM alignment, and domain-mask construction.
  - Outputs: Refactored alignment/sample commands, refreshed Monterey
    full-grid/mask/model-input artifacts, and rerun downstream Monterey model
    and report artifacts.
  - Acceptance: Monterey reruns through the full downstream workflow, and row
    counts, dropped positives, mask reasons, and sample weights are reportable
    from generated artifacts.
  - Plan: `tasks/43_refactor_mask_first_alignment_workflow.md`.
  - Completed: `align-full-grid` now writes only full-grid artifacts, and
    `build-model-input-sample` explicitly builds the retained-domain
    CRM-stratified mask-first sample from the full-grid table plus plausible
    kelp mask. Monterey was rerun through labels, full-grid alignment, CRM
    alignment, domain mask, model-input sample, baseline/binary/conditional/
    hurdle models, residual maps, results visualizer, and model analysis.
    The refreshed sample has 311,475 rows from 3,150,755 retained-domain
    population rows, with 34,141,050 mask-dropped rows and zero dropped
    positives recorded in the manifest.
- [x] P2-03b: Make Kelpwatch-native UTM 30 m the general target grid.
  - Goal: Refactor full-grid alignment so Monterey and Big Sur use the
    Kelpwatch-native UTM 30 m label lattice as the target grid, then map AEF,
    CRM, and mask metadata onto that grid.
  - Outputs: General target-grid policy/config wiring, manifest snap and phase
    diagnostics, refreshed Monterey alignment/mask/model-input artifacts, and a
    report comparison against the Task 43 byte-identical result.
  - Acceptance: Monterey reruns under the Kelpwatch-native target-grid policy,
    Kelpwatch labels are snapped without label interpolation, AEF support
    diagnostics are recorded, and any report metric movement is documented.
  - Plan: `tasks/44_use_kelpwatch_native_utm_target_grid.md`.
  - Completed: added `kelpwatch_native_utm_30m` as the configured full-grid
    policy for Monterey and Big Sur, refreshed Monterey through full-grid
    alignment, CRM alignment, plausible-kelp mask, mask-first model-input
    sample, baseline/binary/conditional/hurdle models, residual maps,
    visualizer, and model analysis. The refreshed Monterey target grid is
    `EPSG:32610`, `718 x 2619` cells, with max station snap residual
    `8.78e-08 m`, AEF phase offset about `5 m`, and `78` stations per year
    outside selected AEF coverage. The retained mask has `426,640` cells, the
    masked sample has `228,495` rows from `2,133,200` retained-domain
    population rows, and dropped positives remain `0`. The selected
    expected-value hurdle changed on the primary 2022 `full_grid_masked` all
    row from F1 `0.812370` / area bias `-16.0250%` to F1 `0.840780` / area
    bias `-27.5095%`.
- [x] P2-03c: Build Big Sur alignment, mask, and model-input artifacts.
  - Goal: Apply the completed mask-first workflow and general
    Kelpwatch-native target-grid policy to Big Sur.
  - Outputs: Big Sur labels, full-grid alignment, plausible-kelp mask,
    CRM-stratified mask-first sample, and split manifest.
  - Acceptance: grid snap diagnostics, row counts, dropped positives, and mask
    reasons are reportable.
  - Plan: `tasks/45_build_big_sur_alignment_mask_model_inputs.md`.
  - Completed: Big Sur labels, Kelpwatch-native full-grid alignment, CRM
    alignment, plausible-kelp mask, retained-domain CRM-stratified model-input
    sample, and split manifest were written. The aligned full-grid has
    `3,309,208` rows per year (`32,389` Kelpwatch-station rows and `3,276,819`
    assumed-background rows), max station snap residual `6.25e-08 m`, and
    `538` stations per year outside selected AEF coverage. CRM is valid for all
    target cells from `crm_socal_v2_1as`; CUDEM remains a zero-tile QA caveat.
    The mask retains `272,030` cells, drops `3,037,178`, and drops `0`
    Kelpwatch-positive rows. The masked sample has `231,345` rows from
    `1,360,150` retained-domain population rows; the split manifest has no
    dropped-feature rows. No Big Sur models or predictions were trained.
- [x] P2-04: Improve results visualizer review layers.
  - Goal: Make the Monterey visualizer focus on Kelpwatch-observed canopy
    points, binary FNs, and high hurdle/conditional values instead of letting
    deep-water TNs dominate the default binary outcome view.
  - Outputs: revised visualizer point-selection and binary-outcome layer
    behavior, refreshed Monterey visualizer artifacts, and manifest bucket
    counts.
  - Acceptance: the existing 50k cap is tested first, any cap increase is
    recorded, and Kelpwatch positives/FNs remain visible for manual QA.
  - Plan: `tasks/46_improve_results_visualizer_review_layers.md`.
  - Completed: prioritized binary TP/FP/FN review rows, including FNs,
    Kelpwatch-positive rows, non-TN Kelpwatch-observed rows, high expected-value
    hurdle predictions, high conditional-canopy predictions, and large hurdle
    residuals (`abs(residual) >= 90 m2`) without filling the cap with small
    true-negative support rows; exposed binary outcomes as one TP/FP/FN/TN
    layer with TN hidden by default through the class filter; deduplicated popup
    fields; refreshed the Monterey visualizer artifacts. The existing `50,000`
    cap was enough:
    `9,432/9,432` TP/FP/FN rows, `1,470/1,470` binary FNs,
    `9,019/9,019` Kelpwatch-positive rows, `9,047/9,047` non-TN
    Kelpwatch-observed rows, `6,541/6,541` large hurdle residuals,
    `7,287/7,287` high hurdle rows, and `2,498/2,498` high conditional rows
    were included. True negatives are omitted unless they meet the large
    residual rule: `35/417,208` included.
- [x] P2-05: Add results visualizer layer filters and legends.
  - Goal: Let reviewers hide noisy binary outcome classes and low-scoring
    continuous/probability values per data type or layer, while replacing
    text-only legend guidance with visual swatches and ramps.
  - Outputs: active-layer filter controls, per-data-type legends, filter defaults
    in config/manifest, and refreshed Monterey visualizer artifacts.
  - Acceptance: filters are optional, layer-aware, and do not change model
    artifacts or tune thresholds from held-out rows.
  - Plan: `tasks/47_add_visualizer_layer_filters_and_legends.md`.
  - Completed: added UI-only active-layer filters with config-backed defaults
    and layer overrides, numeric controls for prediction/residual/probability
    layers, binary class checkboxes, active visual ramps/swatches, and manifest
    filter/legend metadata, and a `Kelpwatch observed label` layer. Defaults
    are `>= 1 m2` for observed labels, `>= 90 m2` for hurdle predictions,
    `abs(residual) >= 90 m2`, `>= 450 m2` for conditional ridge, probability
    `>= 0.10`, and TP/FP/FN visible in the binary outcome review layer. The
    Monterey visualizer artifacts were refreshed and the Task 46 inspection
    export remains `9,467` rows.
- [x] P2-06: Evaluate Monterey-trained transfer on Big Sur.
  - Goal: Apply the closed Monterey Phase 1 policy to Big Sur as the transfer
    baseline.
  - Outputs: Big Sur metrics for AEF ridge, expected-value hurdle, hard-gated
    hurdle, calibrated binary support, and reference baselines.
  - Acceptance: Big Sur held-out performance is reported without Big Sur
    training-driven policy changes.
  - Plan: `tasks/48_evaluate_monterey_transfer_on_big_sur.md`
  - Completed: added `kelp-aef evaluate-transfer --config
    configs/big_sur_smoke.yaml --source-config configs/monterey_smoke.yaml`
    and wrote path-distinct `big_sur_monterey_transfer_*` sidecars. The primary
    Big Sur `test`/2022 `full_grid_masked`/`all` row has 272,030 retained
    cells and 6,504,325 m2 observed canopy. Monterey-transfer AEF ridge reached
    F1 `0.771748` with `+5.5720%` area bias; expected-value hurdle reached F1
    `0.849834` with `-22.1124%` area bias; hard-gated hurdle reached F1
    `0.849308` with `-19.2801%` area bias. Frozen Monterey calibrated binary
    support used threshold `0.37` and reached AUROC `0.992484`, AUPRC
    `0.897458`, precision `0.829083`, recall `0.864566`, and F1 `0.846453`.
    The transfer manifest records no Big Sur model, calibrator, threshold, or
    conditional-model refit.
- [x] P2-07: Train and evaluate Big Sur-only models.
  - Goal: Test whether training on Big Sur improves held-out Big Sur
    performance relative to Monterey transfer.
  - Outputs: Big Sur-only baseline, binary, calibration, conditional, and
    hurdle artifacts.
  - Acceptance: thresholds and calibration use validation rows, and held-out
    Big Sur test rows remain final evaluation.
  - Plan: `tasks/49_train_evaluate_big_sur_only_models.md`.
  - Completed: reran `train-baselines`, `predict-full-grid`,
    `train-binary-presence`, `calibrate-binary-presence`,
    `train-conditional-canopy`, and `compose-hurdle-model` against
    `configs/big_sur_smoke.yaml`. Big Sur-only fitting used 2018-2020 rows,
    Platt calibration and threshold selection used 46,269 validation rows from
    2021, and held-out 2022 rows remained evaluation-only. The calibrated
    threshold selected from validation was `0.41`. New comparison sidecars were
    written to `big_sur_only_model_comparison.csv`,
    `big_sur_only_primary_summary.csv`, and
    `big_sur_only_eval_manifest.json` with `training_regime = big_sur_only`,
    `model_origin_region = big_sur`, and `evaluation_region = big_sur`.
    Primary 2022 retained-domain Big Sur results: AEF ridge F1 `0.649054` with
    `+73.7367%` area bias; expected-value hurdle F1 `0.859563` with
    `-2.8414%` area bias; hard-gated hurdle F1 `0.857286` with `-0.4462%`
    area bias. Relative to Monterey transfer, Big Sur-only training improved
    the expected-value hurdle F1 from `0.849834` to `0.859563` and area bias
    from `-22.1124%` to `-2.8414%`; the ridge-only baseline got worse.
- [x] P2-08: Cross-evaluate Monterey and Big Sur training regimes.
  - Goal: Evaluate Monterey-only, Big Sur-only, and pooled Monterey+Big Sur
    models on both Monterey and Big Sur.
  - Outputs: reciprocal transfer artifacts, pooled training artifacts, and a
    six-cell comparison table keyed by training regime, model origin region,
    and evaluation region.
  - Acceptance: the report can compare all Monterey/Big Sur local, transfer,
    and pooled combinations on the same 2022 retained-domain evaluation scope
    for each target region.
  - Plan: `tasks/50_train_evaluate_pooled_monterey_big_sur_models.md`.
  - Completed: added pooled-region sample building, reciprocal Big Sur-only on
    Monterey transfer evaluation, target-specific pooled configs, and canonical
    training-regime comparison outputs. The pooled sample has `459,840` rows
    (`231,345` Big Sur, `228,495` Monterey) and keeps `source_region` for
    auditing only, not as a predictor.
  - Primary outcome: on Big Sur, Big Sur-only expected-value hurdle remained
    best for amount calibration with F1 `0.859563` and `-2.8414%` area bias;
    pooled Big Sur evaluation had F1 `0.850056` and `-20.1402%` area bias,
    only slightly better than Monterey transfer area bias. On Monterey, pooled
    expected-value hurdle improved area bias versus Monterey-only
    (`-22.1988%` vs `-27.5095%`) and improved RMSE/R2, but F1 stayed below
    Monterey-only. Big Sur-only transfer had the smallest Monterey area bias
    (`-13.6822%`) with much worse F1/RMSE, so P2-09 should separate amount
    calibration from binary/support quality in the report.
  - Binary support note: Monterey-only transfers fairly well to Big Sur for
    binary support (F1 `0.846453` versus Big Sur-only `0.857286`), while Big
    Sur-only transfers less well to Monterey (F1 `0.777221` versus
    Monterey-only `0.852555`). Pooled support is more conservative in both
    regions, with higher precision but lower recall than the local model.
    This supports treating binary support as relatively transferable and
    canopy amount calibration as the larger region-specific issue.
- [x] P2-09: Update the model-analysis report for region and training-regime
      comparison.
  - Goal: Make Phase 2 outcomes visible without turning the report into a long
    chronology.
  - Outputs: compact Monterey-vs-Big-Sur and training-regime comparison
    sections.
  - Acceptance: the report answers whether Big Sur-only or pooled training
    improves Big Sur performance.
  - Plan: `tasks/51_update_phase2_model_analysis_report.md`.
  - Completed: `analyze-model` now switches the Big Sur report into a compact
    Phase 2 synthesis when `training_regime_comparison` is configured. The
    regenerated report leads with the P2-09 answer: Big Sur-only training is
    best for Big Sur canopy amount calibration; pooled Monterey+Big Sur does
    not beat Big Sur-only for Big Sur amount calibration; binary support
    transfers better than canopy amount and pooled support is more
    conservative. The new binary-support table is
    `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`,
    and the manifest records the training-regime inputs, binary prediction
    inputs, calibration payloads, primary filters, and output report paths.
- [ ] P2-10: Generate or extend the results visualizer for Big Sur.
  - Goal: Make Big Sur labels, predictions, residuals, and binary outcomes
    inspectable.
  - Outputs: Big Sur visualizer or a multi-region visualizer with region/year
    selection.
  - Acceptance: Big Sur can be selected or opened without confusing it with
    Monterey rows. Use the P2-09 split explicitly: binary support is relatively
    transferable, while canopy amount calibration and residual structure need
    Big Sur visual QA.
- [ ] P2-11: Close Phase 2 and recommend Phase 3.
  - Goal: Decide whether Phase 3 should broaden geography, test simple
    non-linear tabular models such as random forest or gradient boosting,
    explore deferred temporal labels, harden ingestion/domain coverage, or
    improve evaluation tooling.
  - Outputs: Phase 2 closeout decision note and updated docs.
  - Acceptance: the decision is grounded in Big Sur source coverage, transfer
    performance, Big Sur-only performance, pooled performance, the P2-09 report
    synthesis, and visual QA.

## Closed Phase 1 TODO

Status: Phase 1 is closed as of 2026-05-13.

Phase 1 theme: harden the Monterey annual-max pipeline before scale-up. The
closeout selected the expected-value hurdle as the best current AEF model
policy inside the physically plausible kelp domain, while preserving
high-canopy underprediction as unresolved.

Phase 1 plan:

```text
docs/phase1_model_domain_hardening.md
```

Closed Phase 0 report snapshot:

```text
docs/report_snapshots/monterey_phase0_model_analysis.md
```

Final Phase 1 closeout report outputs:

```text
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html
/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf
docs/report_snapshots/monterey_phase1_closeout_model_analysis.md
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
- [x] P1-10a: Add NOAA CUDEM / Coastal DEM query and download scripts.
  - Goal: Create package-backed query and downloader commands for the
    higher-resolution topo-bathy QA source where CUDEM coverage exists.
  - Plan: `tasks/15_download_noaa_cudem.md`.
  - Validation: `make check` and a dry-run command that writes a manifest to
    `/private/tmp`.
  - Completed: added package-backed `query-noaa-cudem` and
    `download-noaa-cudem` commands, config paths, and tests. Downstream CRM
    alignment uses the CUDEM tile manifest as an optional QA source.
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
- [x] P1-23: Select the best Phase 1 model policy or document failure.
  - Plan: `tasks/39_close_phase1_model_policy_and_report.md`.
  - Goal: Close Phase 1 with a defensible model/mask/calibration decision.
  - Output: final Phase 1 closeout report section, tracked report snapshot,
    updated decision note, docs cleanup, and removal of unused active
    config/code paths that are not part of the closeout state.
  - Validation: the selected policy beats meaningful reference baselines or the
    report clearly explains why it does not; the final report does not include
    post-Phase-1 next-step language.
  - Completed: selected `calibrated_probability_x_conditional_canopy` as the
    best current AEF Phase 1 model policy, retained
    `calibrated_hard_gate_conditional_canopy` as a diagnostic support policy,
    wrote `docs/phase1_closeout_model_policy_decision.md`, refreshed the final
    generated report, copied the tracked closeout snapshot, and removed disabled
    historical sidecar blocks from the active Monterey smoke config.
  - Final evidence: on the `test` / `2022` / `full_grid_masked` /
    `label_source = all` rows, the expected-value hurdle has RMSE `0.0322`,
    R2 `0.790`, F1 at 10% annual max `0.812`, predicted area `3.50M m2`, and
    area bias `-16.0%`. It improves over AEF ridge and slightly edges
    previous-year annual max on retained-domain RMSE and F1, while still
    underpredicting high-canopy rows.
  - Validation passed:
    `uv run pytest tests/test_model_analysis.py tests/test_package.py`,
    `uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml`, and
    `make check`.
- [x] P1-24: Create an interactive results visualizer.
  - Plan: `tasks/40_interactive_results_visualizer.md`.
  - Goal: Build a local zoomable map for inspecting Monterey retained-domain
    labels, predictions, residuals, and binary outcomes against a background
    map layer.
  - Output: generated HTML viewer, layer assets, and manifest under
    `/Volumes/x10pro/kelp_aef`.
  - Validation: viewer opens locally, exposes one radio-selected data layer at
    a time, exposes cell coordinates and values for Planet or Kelpwatch
    comparison, and does not change model artifacts.
  - Completed: added `kelp-aef visualize-results`, configured the Monterey
    viewer, and wrote a Leaflet HTML artifact focused on expected-value hurdle
    prediction/residual, conditional ridge, calibrated binary probability, and
    binary TP/FP/FN/TN outcome layers.
  - Output details: wrote
    `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`,
    `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer/`,
    `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`, and
    `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`.
  - Result: the viewer loaded 630,151 retained-domain 2022 rows for each
    configured visualizer source layer and exported 50,000 bounded inspection
    points for coordinate/value lookup. Browser layers are coordinate-based
    point layers instead of image overlays, stale PNG/TIF assets are removed on
    rerun, data-layer selection is radio-button exclusive, and popups use
    compact labels without label-source or mask-reason rows.
  - Validation passed:
    `uv run pytest tests/test_results_visualizer.py -q`, focused
    `ruff`/`mypy` checks,
    `uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml`,
    generated asset/manifest inspection, and `make check`.

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
