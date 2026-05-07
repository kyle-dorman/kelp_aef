# AGENTS.md

## Project

This is a Python project managed with `uv`.

The project builds a reproducible pipeline that uses AlphaEarth/Satellite
Embedding features to learn Kelpwatch-style kelp canopy labels. Kelpwatch is a
Landsat-derived weak-label product, not independent field truth, so frame early
results as learning or reproducing Kelpwatch-style labels rather than proving
true kelp biomass.

Read these files before making project-shaping changes:

- `docs/product.md`
- `docs/architecture.md`
- `docs/research_plan.md`
- `docs/todo.md`

## Current Milestone

Start with the Monterey Peninsula feasibility spike:

```text
Kelpwatch label tile + AlphaEarth embedding tile -> aligned training table -> simple model -> map
```

Default scope unless a task says otherwise:

- Region: Monterey Peninsula.
- Years: small AlphaEarth/Kelpwatch overlap window.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings aggregated to the Kelpwatch
  30 m grid.
- Models: simple tabular baselines first, such as logistic or ridge regression
  and random forest.
- Split: year holdout.

Do not start with the full West Coast. Do not bulk-download the full AlphaEarth
collection.

## Data And Artifacts

Keep the code repo focused on source, configs, tests, and docs. Store raw data
and generated artifacts under:

```text
/Volumes/x10pro/kelp_aef
```

Expected artifact roots:

```text
/Volumes/x10pro/kelp_aef/raw/
/Volumes/x10pro/kelp_aef/interim/
/Volumes/x10pro/kelp_aef/processed/
/Volumes/x10pro/kelp_aef/models/
/Volumes/x10pro/kelp_aef/reports/figures/
/Volumes/x10pro/kelp_aef/reports/tables/
```

Only track tiny sample artifacts if the repo explicitly marks them as visual QA
fixtures.

## Setup

Run:

```bash
uv sync
```

## Validation

For docs-only changes, inspect the rendered Markdown or diff.

For code changes, run the relevant subset of:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

`make check` is intended to become the standard non-mutating validation command,
but the Makefile cleanup is tracked separately in `docs/todo.md`.

## Implementation Rules

- Prefer small, narrow modules under `src/kelp_aef/`.
- Prefer package-backed CLI commands over notebook-only workflows.
- Each command should accept a config path, starting with
  `configs/monterey_smoke.yaml`.
- Keep each step artifact-producing and restartable.
- Do not rely on hidden notebook state for pipeline behavior.
- Keep heavy geospatial downloads, model outputs, and generated reports out of
  git unless explicitly requested.

## Agent Task Contract

For every agent-sized task, define:

- Goal.
- Inputs.
- Outputs.
- Config file.
- Plan/spec requirement.
- Validation command.
- Smoke-test region and years.
- Acceptance criteria.
- Known constraints or non-goals.

Plan/spec requirements should scale with task size:

- Narrow mechanical tasks: no separate plan required.
- Multi-file or ambiguous tasks: write a brief implementation plan before
  editing, including files to change, expected artifacts, validation, and
  assumptions.
- New pipeline stages, data contracts, model/evaluation choices, or
  architecture-shaping work: write a spec or decision note before implementation
  unless the user explicitly asks to skip it.

Example:

```text
Goal: Inspect Kelpwatch metadata for the Monterey smoke config.
Input: Kelpwatch sample path declared in configs/monterey_smoke.yaml.
Output: /Volumes/x10pro/kelp_aef/interim/metadata_summary.json.
Plan/spec requirement: Brief implementation plan before editing.
Validation: uv run ruff check . and kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml.
Acceptance: Summary includes CRS, bounds, variables, seasons, years, units, and missing-data notes.
```
