# First Feasibility Milestone TODO

This is the active planning checklist for Phase 0. The basic agentic setup work
is complete; use `docs/backlog.md` for the durable high-level queue.

Do not start data-heavy implementation until this plan is reviewed and the
smoke-test assumptions are updated.

## Scope Decisions

- [ ] Choose the first smoke-test region.
  - Current placeholder: Monterey Peninsula.
  - Alternatives to consider: Santa Barbara or another region with easier data
    access/QA.
- [ ] Define the region geometry.
  - Decide whether bounds live directly in config, in a small GeoJSON, or in a
    named region registry.
  - Review the provisional bounds in `configs/monterey_smoke.yaml`.
- [ ] Choose the first year window.
  - Confirm actual overlap between Kelpwatch and AlphaEarth availability.
  - Decide whether the first smoke test should be 2-3 years or the broader
    2018-2024 milestone window.
- [ ] Choose the initial label target.
  - Candidate from the research plan: `kelp_max_y`.
  - Decide whether binary `kelp_present_y` is needed for the first model.
  - Define any thresholds before implementation.
- [ ] Choose the initial feature product and access path.
  - Confirm whether AlphaEarth data will come through Earth Engine,
    Geo-Embeddings/Zarr tooling, exported chips, or another route.
  - Keep the first pull coastal and small.
- [ ] Confirm the first alignment policy.
  - Current research-plan default: aggregate AlphaEarth 10 m embeddings to the
    Kelpwatch 30 m grid.
  - Document any alternative before implementation.
- [ ] Choose the first split policy.
  - Current placeholder: year holdout.
  - Decide exact train/validation/test years only after confirming available
    data.

## Config And Artifacts

- [ ] Revise `configs/monterey_smoke.yaml` after the scope decisions above.
  - Treat the current file as a provisional scaffold, not final science.
- [ ] Confirm the expected output artifact paths.
  - Metadata summary.
  - Annual label table.
  - AlphaEarth feature/sample table.
  - Aligned training table.
  - Split manifest.
  - Predicted map.
  - Residual map.
  - Area-bias summary.
- [ ] Confirm whether any tiny QA samples should be tracked.
  - Current policy: none are tracked by default.

## Task Contracts

- [ ] Write the Phase 0 implementation spec or decision note.
  - Include goal, scope, non-goals, selected config values, artifact paths,
    validation plan, and open questions.
- [ ] Create one task file per agent-sized unit of work.
  - Suggested location: `tasks/`.
- [ ] For each task file, include:
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

- [ ] Inspect Kelpwatch source format and write reader.
- [ ] Inspect AlphaEarth access path and write sample/chip fetcher.
- [ ] Build initial annual label derivation.
- [ ] Align features and labels into the first table.
- [ ] Train and evaluate first simple baselines.
- [ ] Make first predicted/residual maps and area-bias table.

