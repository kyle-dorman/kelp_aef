# First Feasibility Milestone TODO

This is the active planning checklist for Phase 0. The basic agentic setup work
is complete; use `docs/backlog.md` for the durable high-level queue.

Do not start data-heavy implementation until this plan is reviewed and the
smoke-test assumptions are updated.

## Scope Decisions

- [x] Choose the first smoke-test region.
  - Selected: Monterey Peninsula.
- [x] Define the region geometry.
  - Use a small GeoJSON footprint extracted from one AlphaEarth tile.
  - Config path:
    `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`.
  - Do not keep provisional bbox bounds as the active smoke geometry.
- [x] Choose the first year window.
  - Use the currently identified AlphaEarth/Kelpwatch overlap window:
    2018-2022.
- [x] Choose the initial label target.
  - Use `kelp_max_y`, Kelpwatch annual max canopy.
  - Defer binary `kelp_present_y` and thresholds until after source metadata and
    value ranges are inspected.
- [x] Choose the initial feature product and access path.
  - Use Source Cooperative AlphaEarth/AEF v1 annual GeoTIFFs for one `10N` grid
    tile.
  - Query the Source Cooperative AEF STAC GeoParquet catalog to identify the
    specific assets to download:
    `https://data.source.coop/tge-labs/aef/v1/annual/aef_index_stac_geoparquet.parquet`.
  - Mirror the S3 key layout under
    `/Volumes/x10pro/kelp_aef/raw/aef/v1/annual/{year}/10N/`.
  - Known source examples:
    `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/xdz8z3a9znk5b1j75-0000008192-0000008192.tiff`
    and
    `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000008192-0000008192.tiff`.
- [x] Confirm the first alignment policy.
  - Aggregate AlphaEarth 10 m embeddings to the Kelpwatch 30 m grid.
- [x] Choose the first split policy.
  - Year holdout: train 2018-2020, validate 2021, test 2022.

## Config And Artifacts

- [x] Revise `configs/monterey_smoke.yaml` after the scope decisions above.
  - Config now references the footprint GeoJSON and S3-mirrored AEF raw layout.
- [x] Confirm the expected output artifact paths.
  - AEF tile footprint GeoJSON.
  - AEF catalog query result.
  - AEF tile manifest.
  - Kelpwatch source manifest.
  - Metadata summary.
  - Annual label table.
  - AlphaEarth feature/sample table.
  - Aligned training table.
  - Split manifest.
  - Predicted map.
  - Residual map.
  - Area-bias summary.
- [x] Confirm whether any tiny QA samples should be tracked.
  - Current policy: none are tracked by default.

## Task Contracts

- [x] Write the Phase 0 implementation spec or decision note.
  - Include goal, scope, non-goals, selected config values, artifact paths,
    validation plan, and open questions.
- [x] Create one task file per agent-sized unit of work.
  - Suggested location: `tasks/`.
- [x] For each task file, include:
  - Goal.
  - Inputs.
  - Outputs.
  - Config file.
  - Plan/spec requirement.
  - Validation command.
  - Smoke-test region and years.
  - Acceptance criteria.
  - Known constraints or non-goals.

## Initial Task Sequence

- [x] Extract AlphaEarth tile footprint GeoJSON.
- [x] Query the AEF STAC GeoParquet catalog for Monterey smoke assets.
- [x] Download the selected AEF tile assets from the catalog query.
- [x] Inspect Kelpwatch source format and write downloader/reader.
- [x] Visualize downloaded Kelpwatch source data for Monterey QA.
- [x] Build initial annual label derivation.
  - Plan: `tasks/06_build_annual_labels.md`.
- [x] Align features and labels into the first parquet table.
  - Plan: `tasks/07_align_features_labels.md`.
- [ ] Train and evaluate first simple baselines.
  - Plan: `tasks/08_train_evaluate_baselines.md`.
- [ ] Make first predicted/residual maps and area-bias table.
