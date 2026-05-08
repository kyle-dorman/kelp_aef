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
  - Final report:
    `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase0_model_analysis.md`.

## Active Planning

- [ ] Decide the next phase explicitly.
  - Do not assume the next phase is scale-up, target engineering, calibration,
    or a stronger model until the Phase 0 report is reviewed.
  - When selected, move the relevant backlog item into `docs/todo.md` with an
    agent-sized task contract.

## Candidate Pipeline Milestones

- [ ] Harden ingestion and manifests beyond the Monterey smoke run.
  - Kelpwatch reader/downloader is implemented for the smoke path, but needs
    robustness before larger geography.
  - AEF STAC catalog query and download work for the smoke tile, but selection,
    retry, and manifest QA need hardening before scale-up.
  - Available-years/regions manifests remain candidate work.
- [ ] Explore derived labels.
  - Seasonal-to-annual Kelpwatch targets.
  - Continuous and binary label variants.
  - Missing-data and seasonal-coverage diagnostics.
  - Label QA maps for known kelp regions.
- [ ] Harden feature/label alignment.
  - Phase 0 has both station-centered and full-grid 30 m alignment artifacts.
  - Add stronger QA for target-grid identity, duplicate cells, missing features,
    and station-to-grid assignment before larger runs.
  - Treat parquet as the first end-to-end artifact, not the final data shape for
    every model family.
  - Later chip, tensor, or other spatial artifact for CNNs and neighborhood-aware
    models that need to preserve the 10 m spatial context.
- [ ] Expand split manifests.
  - Year holdout.
  - Latitude or spatial holdout.
  - State or regional holdout.
  - Random/block split only as a sanity check.
- [ ] Improve baselines and first embedding models.
  - No-skill and geographic baselines.
  - Previous-year and station-climatology baselines.
  - Ridge/logistic variants.
  - Random forest or similar tree model.
  - Defer deep spatial models until tabular baselines are informative.
- [ ] Improve evaluation and reporting.
  - Classification and regression metrics.
  - Area-bias and region/year summaries.
  - Predicted and residual maps.
  - Results summary for the feasibility spike.
- [ ] Scale beyond the smoke test.
  - Full U.S. West Coast config.
  - Broader temporal/spatial holdouts.
  - Larger artifact-management and runtime strategy.
  - Do not start this until the next phase decision explicitly chooses scale-up.

## Research Questions

- [ ] Resolve sampling/objective calibration for background-inclusive training.
  - Phase 0 showed that population-expanded background weights collapse the
    continuous ridge model toward near-zero predictions.
  - The final Phase 0 baseline therefore uses an unweighted
    background-inclusive sample.
  - Full-grid calibration is still poor because small positive predictions over
    millions of assumed-background cells accumulate into large area
    overprediction.
  - Candidate work: compare capped background weights, stratified losses,
    station-supported metrics, full-grid calibration metrics, and post-fit
    calibration.
- [ ] Evaluate alternative Kelpwatch temporal label aggregations after the first
  annual-max baseline.
  - Compare annual max canopy with fall-only, winter-only, and fall-to-winter
    difference or loss labels.
  - Test whether annual max labels count canopy that was later lost before
    fall or winter, and whether annual AEF embeddings learn that loss signal or
    mainly reproduce peak-season canopy.
  - Keep this as backlog until the initial annual-max label derivation,
    alignment table, and baseline evaluation exist.
- [ ] Evaluate binary Kelpwatch label thresholds after the continuous
  annual-max baseline.
  - Use Bell et al. 2019/2020 and Kelpwatch method notes as motivation for
    caution around low canopy area values and detection limits.
  - Compare `kelp_max_y > 0` with fractional canopy thresholds such as 1%, 5%,
    and 10% of a 30 m Landsat pixel.
  - Report threshold sensitivity in area totals, class balance, maps, and model
    metrics before choosing a production binary target.
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
