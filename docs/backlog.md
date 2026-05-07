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
  - Grid-cell-by-year aligned table.
  - Later chip product for spatial models.
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

## Engineering

- [ ] Write a short `README.md` entrypoint for humans and agents.
- [ ] Add CI once the dependency and test surface stabilizes.
- [ ] Add geospatial/data dependencies only when the first concrete data task
  needs them.
- [ ] Decide whether to track any tiny QA fixtures after real outputs exist.

