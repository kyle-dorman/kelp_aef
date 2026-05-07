# Architecture

## Operating Principle

Keep each step narrow, artifact-producing, and runnable from the command line.
Agents should be able to pick up one task, read an explicit config, write known
outputs, run a validation command, and hand off without relying on hidden
notebook state.

`docs/research_plan.md` is the long strategy document. `docs/product.md` is the
short product contract. This document defines the intended technical shape of
the repo.

## Planned Package Layout

```text
src/kelp_aef/
  io/
  labels/
  alignment/
  features/
  models/
  evaluation/
  viz/
```

Responsibilities:

- `io/`: source-specific readers, download helpers, metadata inspection, and
  small sample loading.
- `labels/`: Kelpwatch seasonal-to-annual label derivation, binary targets,
  missing-data diagnostics, and label summaries.
- `features/`: AlphaEarth embedding selection, chip fetching, aggregation, and
  feature-table assembly.
- `alignment/`: CRS/grid checks, resampling or aggregation decisions, spatial
  joins, and aligned grid-cell-by-year outputs.
- `models/`: baselines and simple embedding models. Start with tabular models
  before spatial CNN/UNet work.
- `evaluation/`: split manifests, metrics, area-bias summaries, and comparison
  tables.
- `viz/`: sample QA plots, predicted maps, residual maps, and uncertainty maps.

## Planned Command Layout

Prefer package-backed CLI commands so code remains testable. Thin scripts are
acceptable when they only delegate into package functions.

Decision: user-facing workflow commands should live behind the `kelp-aef`
package CLI. Do not add standalone `scripts/` entrypoints unless they are thin
wrappers around package functions or there is a specific operational reason.

Mapped initial commands:

```text
kelp-aef smoke --config configs/monterey_smoke.yaml
kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml
kelp-aef fetch-aef-chip --config configs/monterey_smoke.yaml
kelp-aef build-labels --config configs/monterey_smoke.yaml
kelp-aef align --config configs/monterey_smoke.yaml
```

Each command should accept a config path and write deterministic artifact paths
declared in that config.

## Configs

Use YAML configs for repeatable runs.

Expected config files:

```text
configs/monterey_smoke.yaml
configs/west_coast_full.yaml
```

The smoke config should define:

- Data root, currently `/Volumes/x10pro/kelp_aef`.
- Region name and bounds.
- Years.
- Label target.
- Feature aggregation choice.
- Input paths.
- Output paths.
- Split policy.
- Runtime limits for small agent-safe runs.

## Data Flow

```text
raw source data
  -> source metadata summaries
  -> derived annual Kelpwatch labels
  -> AlphaEarth chips or samples
  -> aligned grid-cell-by-year training table
  -> split manifests
  -> trained baseline models
  -> metrics, maps, and tables
```

The first aligned table should use AlphaEarth embeddings aggregated to the
Kelpwatch 30 m grid. This matches the label resolution and avoids pretending
upsampled 30 m labels contain 10 m truth.

## Artifact Conventions

See `docs/data_artifacts.md` for the detailed data/artifact contract and git
tracking policy.

The canonical data and generated artifact root is outside the code repo:

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

Create the external artifact directories with:

```bash
mkdir -p /Volumes/x10pro/kelp_aef/raw/kelpwatch \
  /Volumes/x10pro/kelp_aef/raw/aef_samples \
  /Volumes/x10pro/kelp_aef/interim \
  /Volumes/x10pro/kelp_aef/processed \
  /Volumes/x10pro/kelp_aef/models \
  /Volumes/x10pro/kelp_aef/reports/figures \
  /Volumes/x10pro/kelp_aef/reports/tables
```

Early smoke-test artifacts:

- `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aef_samples.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`
- `/Volumes/x10pro/kelp_aef/reports/figures/sample_kelpwatch_vs_aef.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/predicted_map.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/residual_map.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_summary.csv`

Large raw data, downloaded raster/Zarr artifacts, model outputs, and temporary
exports should stay out of git unless the repo explicitly marks a tiny sample as
tracked visual QA.

## Validation

Verification should be non-mutating by default.

Expected commands after setup:

```bash
uv sync
make check
```

`make check` should eventually run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Use a separate mutating command such as `make fix` for formatting and autofixes.

## Agent Task Contract

Every agent-sized task should specify:

- Goal.
- Inputs.
- Outputs.
- Config file.
- Validation command.
- Smoke-test region and years.
- Acceptance criteria.
- Known constraints or non-goals.

Good task shape:

```text
Implement Kelpwatch metadata inspection for the Monterey smoke config.
Input: local Kelpwatch sample path from configs/monterey_smoke.yaml.
Output: /Volumes/x10pro/kelp_aef/interim/metadata_summary.json.
Validation: make check and kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml.
Acceptance: summary includes CRS, bounds, variables, seasons, years, units, and missing-data notes.
```

Avoid broad tasks such as "build the whole pipeline" until the smoke-test
contracts are stable.
