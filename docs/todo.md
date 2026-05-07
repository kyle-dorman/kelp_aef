# Agentic Workflow Setup TODO

Work through these in order. The goal is to make the repo easy for agents to
understand, modify, validate, and hand off without rediscovering project intent
each time.

## 1. Project Docs

- [x] Convert `docs/research_plan.md` into a concise `docs/product.md`.
  - Include project goal, non-goals, first feasibility milestone, and key caveat
    that Kelpwatch is a weak label rather than independent field truth.
- [x] Create `docs/architecture.md`.
  - Define package/module boundaries, canonical data flow, expected artifacts,
    and the smoke-test workflow.
- [x] Keep `docs/research_plan.md` as the longer strategy/reference document.

## 2. Agent Instructions

- [x] Expand `AGENTS.md`.
  - Add project goal and current milestone.
  - Document `uv sync` setup and validation commands.
  - Tell agents not to start with the full West Coast.
  - Require narrow task contracts: inputs, outputs, validation command, smoke
    region, and acceptance criteria.
- [x] Fix the unclosed code fence in `AGENTS.md`.

## 3. Verification Workflow

- [x] Split mutating and non-mutating checks in `Makefile`.
  - `make check` should run only non-mutating validation.
  - `make fix` should run formatting and autofixes.
- [x] Make `make check` pass on a fresh clone.
- [x] Add a tiny test so `pytest` does not fail because zero tests are collected.
- [x] Adjust the `mypy` command so it does not fail on an empty `tests/`
  directory.

## 4. Toolchain Cleanup

- [x] Align Python versions across `pyproject.toml`.
  - Project requires Python 3.12 and remaining tool config now matches it.
- [x] Simplify formatter/linter configuration.
  - Prefer Ruff as the formatter/linter path.
  - Removed stale Black/isort/autopep8/flake8 config.
- [x] Trim VS Code recommendations to tools that matter for this project.

## 5. Project Skeleton

- [x] Create the package structure from the research plan:
  - `src/kelp_aef/io/`
  - `src/kelp_aef/labels/`
  - `src/kelp_aef/alignment/`
  - `src/kelp_aef/features/`
  - `src/kelp_aef/models/`
  - `src/kelp_aef/evaluation/`
  - `src/kelp_aef/viz/`
- [x] Add `scripts/` entry scripts or decide that all commands live under the
  package CLI.
- [x] Add `configs/monterey_smoke.yaml` before writing data-heavy code.
- [x] Add artifact directories or document creation commands for the external
  data root:
  - `/Volumes/x10pro/kelp_aef/raw/`
  - `/Volumes/x10pro/kelp_aef/interim/`
  - `/Volumes/x10pro/kelp_aef/processed/`
  - `/Volumes/x10pro/kelp_aef/models/`
  - `/Volumes/x10pro/kelp_aef/reports/figures/`
  - `/Volumes/x10pro/kelp_aef/reports/tables/`

## 6. CLI

- [ ] Replace the starter `kelp-aef` CLI with a real entrypoint.
- [ ] Add initial subcommands for the first milestone, likely:
  - `smoke`
  - `inspect-kelpwatch`
  - `fetch-aef-chip`
  - `build-labels`
  - `align`
- [ ] Make each subcommand accept a config path.
- [ ] Add tests for CLI import and `--help`.

## 7. Data And Artifact Contracts

- [ ] Define canonical paths and git-ignore policy for data artifacts.
  - Canonical data root: `/Volumes/x10pro/kelp_aef`.
- [ ] Update `.gitignore` for large local outputs:
  - raw data
  - downloaded raster/Zarr artifacts
  - temporary Earth Engine exports
  - model outputs
- [ ] Decide which tiny sample outputs are intentionally tracked for visual QA.
- [ ] Document expected outputs for the Monterey feasibility spike:
  - metadata summary
  - aligned training table
  - predicted map
  - residual map
  - area-bias summary

## 8. First Feasibility Milestone

- [ ] Lock the first smoke-test scope.
  - Region: Monterey Peninsula unless changed.
  - Years: small AlphaEarth/Kelpwatch overlap window.
  - Label: Kelpwatch annual max.
  - Features: AlphaEarth embeddings aggregated to 30 m.
  - Models: logistic regression plus random forest.
  - Split: year holdout.
- [ ] Create one task file per narrow agent-sized unit of work.
- [ ] For each task, include input files, output files, validation command, and
  acceptance criteria.
