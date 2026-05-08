# Backlog

This is the durable high-level project queue. Keep short-term execution details
in `docs/todo.md`; add P1/P2 subtasks here only when a phase is close enough to
start.

## Active Planning

- [ ] Phase 0: lock the first feasibility milestone.
  - Confirm the smoke-test region, years, label target, feature source,
    alignment choice, split policy, expected outputs, and task contracts.

## Pipeline Milestones

- [ ] Phase 1: implement data ingestion.
  - Kelpwatch reader/downloader.
  - AlphaEarth sample or chip fetcher.
  - Source metadata inspection.
  - Available-years/regions manifest.
  - Tiny visual QA sample output.
- [ ] Phase 2: implement derived labels.
  - Seasonal-to-annual Kelpwatch targets.
  - Continuous and binary label variants.
  - Missing-data and seasonal-coverage diagnostics.
  - Label QA maps for known kelp regions.
- [ ] Phase 3: implement feature/label alignment.
  - AlphaEarth aggregation to the Kelpwatch 30 m grid for the first pass.
  - Grid-cell-by-year aligned parquet table for tabular baselines.
  - Treat parquet as the first end-to-end artifact, not the final data shape for
    every model family.
  - Later chip, tensor, or other spatial artifact for CNNs and neighborhood-aware
    models that need to preserve the 10 m spatial context.
- [ ] Phase 4: implement split manifests.
  - Year holdout.
  - Latitude or spatial holdout.
  - State or regional holdout.
  - Random/block split only as a sanity check.
- [ ] Phase 5: implement baseline and first embedding models.
  - No-skill and geographic baselines.
  - Logistic/ridge model.
  - Random forest or similar tree model.
  - Defer deep spatial models until tabular baselines are informative.
- [ ] Phase 6: implement evaluation and reporting.
  - Classification and regression metrics.
  - Area-bias and region/year summaries.
  - Predicted and residual maps.
  - Results summary for the feasibility spike.
- [ ] Phase 7: scale beyond the smoke test.
  - Full U.S. West Coast config.
  - Broader temporal/spatial holdouts.
  - Larger artifact-management and runtime strategy.

## Research Questions

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
- [ ] Add geospatial/data dependencies only when the first concrete data task
  needs them.
- [ ] Decide whether to track any tiny QA fixtures after real outputs exist.
