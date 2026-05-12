# Backlog

This is the durable high-level project queue. Keep short-term execution details
in `docs/todo.md`; add P1/P2 subtasks here only when a phase is close enough to
start.

## Status

- [x] Phase 0: Monterey feasibility milestone.
  - Completed for now on 2026-05-08.
  - Outputs include Kelpwatch labels, AEF asset discovery/download, full-grid
    alignment, background-inclusive sample training, streamed full-grid
    prediction, maps, metrics, and a Phase 0 report.
  - Final report snapshot:
    `docs/report_snapshots/monterey_phase0_model_analysis.md`.
- [x] Phase 1 direction selected.
  - Selected theme: model and domain hardening for the Monterey annual-max
    pipeline.
  - Plan: `docs/phase1_model_domain_hardening.md`.
  - Active checklist: `docs/todo.md`.

## Active Phase 1 Work

- [ ] Harden reference baselines.
  - Previous-year annual-max baseline.
  - Station or grid-cell climatology baseline.
  - Lat/lon/year-only geographic baseline.
  - Compare pixel skill and full-grid area calibration against ridge.
- [ ] Harden domain filtering.
  - Choose and manifest Monterey bathymetry and DEM inputs.
  - Align bathymetry/DEM to the 30 m target grid.
  - Build a plausible-kelp domain mask and report retained/dropped cells.
  - Apply the domain mask to full-grid reporting before retraining, making the
    masked plausible-kelp domain the default largest reporting area.
  - Keep unmasked/off-domain prediction leakage as an audit diagnostic rather
    than a recurring headline scope.
- [ ] Harden imbalance-aware modeling.
  - Add class and target-balance diagnostics.
  - Evaluate binary annual-max thresholds using validation-year data.
  - Train a balanced binary presence model.
  - Train a conditional canopy model or first hurdle model.
  - Compare against ridge and all reference baselines.
- [ ] Harden evaluation and reporting.
  - Keep every task report-visible.
  - Separate Kelpwatch-station pixel skill, background leakage, and full-grid
    area calibration.
  - Add mask-aware residual taxonomy.
  - Preserve a rerunnable CLI path for each stage.

## Phase 1 Non-Goals

- Do not evaluate alternative temporal label inputs beyond annual max.
- Do not use fall-only, winter-only, annual mean, or multi-season persistence
  labels as Phase 1 targets.
- Do not start full West Coast scale-up.
- Do not introduce deep spatial models before the tabular and calibration
  questions are resolved.
- Do not use bathymetry/DEM as predictors in Phase 1 unless explicitly approved
  later; use them first for domain filtering and diagnostics.
- Do not tune on the 2022 test split.

## Candidate Future Pipeline Milestones

- [ ] Harden ingestion and manifests beyond the Monterey smoke run.
  - Kelpwatch reader/downloader is implemented for the smoke path, but needs
    robustness before larger geography.
  - AEF STAC catalog query and download work for the smoke tile, but selection,
    retry, and manifest QA need hardening before scale-up.
  - Available-years/regions manifests remain candidate work.
- [ ] Explore derived labels after Phase 1.
  - Alternative temporal labels are intentionally out of scope while Phase 1
    hardens annual max.
  - Later candidates include annual mean, fall-only, winter-only,
    multi-season persistence, and seasonal-coverage diagnostics.
  - Binary thresholds derived from annual max are not deferred; they are part of
    Phase 1 imbalance-aware modeling.
- [ ] Harden feature/label alignment beyond the current Monterey grid.
  - Phase 0 has both station-centered and full-grid 30 m alignment artifacts.
  - Phase 1 should add domain-mask QA for Monterey, but larger-geography
    alignment QA remains future work.
  - Add stronger QA for target-grid identity, duplicate cells, missing features,
    and station-to-grid assignment before larger runs.
  - Treat parquet as the first end-to-end artifact, not the final data shape for
    every model family.
  - Later chip, tensor, or other spatial artifact for CNNs and neighborhood-aware
    models that need to preserve the 10 m spatial context.
- [ ] Expand split manifests after Phase 1 hardening.
  - Year holdout remains the Phase 1 default.
  - Latitude, site, state, or regional holdouts remain future candidates.
  - Random/block split should stay a sanity check only.
- [ ] Scale beyond the smoke test.
  - Full U.S. West Coast config.
  - Broader temporal/spatial holdouts.
  - Larger artifact-management and runtime strategy.
  - Do not start this until Phase 1 decides a defensible mask/model/calibration
    policy.

## Research Questions

- [ ] Resolve sampling/objective calibration for background-inclusive training.
  - Moved into Phase 1.
  - Phase 0 showed that population-expanded background weights collapse the
    continuous ridge model toward near-zero predictions.
  - The final Phase 0 baseline therefore uses an unweighted
    background-inclusive sample.
  - Full-grid calibration is still poor because small positive predictions over
    millions of assumed-background cells accumulate into large area
    overprediction.
  - Phase 1 should compare binary, hurdle, capped-weight, and
    stratified-background policies against reference baselines.
- [ ] Evaluate alternative Kelpwatch temporal label aggregations after Phase 1.
  - Deferred explicitly.
  - Keep annual max as the Phase 1 label input because the current signal is
    present but the data domain and model objective are not hardened yet.
- [ ] Evaluate binary Kelpwatch annual-max thresholds.
  - Moved into Phase 1.
  - Compare `kelp_max_y > 0` with fractional canopy thresholds such as 1%, 5%,
    and 10% of a 30 m Landsat pixel.
  - Use validation-year behavior to choose candidate thresholds before
    evaluating the 2022 test split.
- [ ] Evaluate higher-resolution label and prediction strategies after the
  Kelpwatch-native 30 m baseline.
  - Keep the first alignment path as AEF 10 m embeddings aggregated to
    Kelpwatch 30 m labels.
  - Later compare 10 m prediction workflows that either replicate parent
    Kelpwatch fractional cover to child cells or resample the 30 m fractional
    cover surface to 10 m with cubic interpolation.
  - Do not force CNN or other spatial model experiments through the flat parquet
    table; design a chip or tensor product when those models become a near-term
    task.
  - For any 10 m label experiment, validate predictions by aggregating them back
    to 30 m and comparing against the original Kelpwatch area labels.
  - Treat 10 m labels as weak/interpolated targets, not independent 10 m truth.

## Engineering

- [ ] Write a short `README.md` entrypoint for humans and agents.
- [ ] Add CI once the dependency and test surface stabilizes.
- [ ] Keep geospatial/data dependencies tied to concrete tasks and avoid
  hand-rolled geometry, raster, projection, or tabular logic when maintained
  libraries are available.
- [ ] Decide whether to track any tiny QA fixtures after real outputs exist.
