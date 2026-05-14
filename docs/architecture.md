# Architecture

## Operating Principle

Keep each step narrow, artifact-producing, and runnable from the command line.
Agents should be able to pick up one task, read an explicit config, write known
outputs, run a validation command, and hand off without relying on hidden
notebook state.

`docs/research_plan.md` is the long strategy document. `docs/product.md` is the
short product contract. This document defines the intended technical shape of
the repo.

## Package Layout

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

Phase 0 implemented the first practical slices of this layout under `labels/`,
`features/`, `alignment/`, `evaluation/`, and `viz/`. Some planned folders may
remain thin until the next selected phase gives them a concrete purpose.

## Command Layout

Prefer package-backed CLI commands so code remains testable. Thin scripts are
acceptable when they only delegate into package functions.

Decision: user-facing workflow commands should live behind the `kelp-aef`
package CLI. Do not add standalone `scripts/` entrypoints unless they are thin
wrappers around package functions or there is a specific operational reason.

Current CLI commands:

```text
kelp-aef smoke --config configs/monterey_smoke.yaml
kelp-aef query-aef-catalog --config configs/monterey_smoke.yaml
kelp-aef download-aef --config configs/monterey_smoke.yaml
kelp-aef inspect-kelpwatch --config configs/monterey_smoke.yaml
kelp-aef visualize-kelpwatch --config configs/monterey_smoke.yaml
kelp-aef fetch-aef-chip --config configs/monterey_smoke.yaml
kelp-aef build-labels --config configs/monterey_smoke.yaml
kelp-aef align --config configs/monterey_smoke.yaml
kelp-aef align-full-grid --config configs/monterey_smoke.yaml
kelp-aef build-model-input-sample --config configs/monterey_smoke.yaml
kelp-aef train-baselines --config configs/monterey_smoke.yaml
kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
kelp-aef train-conditional-canopy --config configs/monterey_smoke.yaml
kelp-aef compose-hurdle-model --config configs/monterey_smoke.yaml
kelp-aef map-residuals --config configs/monterey_smoke.yaml
kelp-aef visualize-results --config configs/monterey_smoke.yaml
kelp-aef analyze-model --config configs/monterey_smoke.yaml
kelp-aef query-noaa-cudem --config configs/monterey_smoke.yaml
kelp-aef download-noaa-cudem --config configs/monterey_smoke.yaml
kelp-aef query-noaa-cusp --config configs/monterey_smoke.yaml
kelp-aef download-noaa-cusp --config configs/monterey_smoke.yaml
kelp-aef query-noaa-crm --config configs/monterey_smoke.yaml
kelp-aef download-noaa-crm --config configs/monterey_smoke.yaml
kelp-aef align-noaa-crm --config configs/monterey_smoke.yaml
kelp-aef build-domain-mask --config configs/monterey_smoke.yaml
kelp-aef query-usgs-3dep --config configs/monterey_smoke.yaml
kelp-aef download-usgs-3dep --config configs/monterey_smoke.yaml
```

Each command should accept a config path and write deterministic artifact paths
declared in that config.

## Configs

Use YAML configs for repeatable runs.

Current config file:

```text
configs/monterey_smoke.yaml
```

Future configs should be added only when the next phase needs them. Do not add a
full West Coast config just because it was part of the early sketch.

The smoke config defines:

- Data root, currently `/Volumes/x10pro/kelp_aef`.
- Region name and a footprint GeoJSON path.
- Years.
- Label target.
- Feature aggregation choice.
- Input paths.
- Output paths.
- Split policy.
- Runtime limits for small agent-safe runs.
- A full-grid alignment product and an explicit retained-domain model-input
  sample product.
- The closed Phase 1 default mask/sample/model policy: retained-domain
  CRM-stratified mask-first sampling, AEF ridge/reference baselines, calibrated
  binary presence, positive-only conditional canopy, and the selected
  expected-value hurdle.

Phase 2 is the first point where a second region config or region-aware config
extension is expected. Keep Monterey reproducible while adding Big Sur. Prefer
explicit region-scoped artifact paths over reusing Monterey output names.

## Data Flow

```text
raw source data
  -> source metadata summaries
  -> derived annual Kelpwatch labels
  -> AlphaEarth chips or samples
  -> aligned full-grid cell-by-year target table
  -> aligned CRM support table
  -> plausible-kelp domain mask
  -> retained-domain model-input sample
  -> split manifests
  -> trained baseline models
  -> metrics, maps, and tables
```

The first aligned table should use AlphaEarth embeddings aggregated to the
Kelpwatch 30 m grid. This matches the label resolution and avoids pretending
upsampled 30 m labels contain 10 m truth.

Phase 0 now has three alignment-related artifacts with different meanings:

- `aligned_training_table.parquet`: station-centered alignment artifact retained
  mostly as a QA/reference product.
- `aligned_full_grid_training_table.parquet`: full 30 m target-grid artifact
  with both Kelpwatch-supported rows and assumed-background rows.
- `aligned_background_sample_training_table.parquet`: sampled training input
  used by the Phase 0 ridge baseline.

The current native alignment path builds `aligned_full_grid_training_table.parquet`
first, then builds CRM/domain-mask support, then writes the default
`aligned_background_sample_training_table.masked.parquet` through
`build-model-input-sample`. Phase 1 preserved the same artifact-producing shape
while adding the retained-domain mask, mask-first CRM-stratified model sample,
calibrated binary presence, positive-only conditional canopy, and hurdle
composition. The selected Phase 1 AEF policy is the expected-value hurdle
recorded in
`docs/phase1_closeout_model_policy_decision.md`.

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
/Volumes/x10pro/kelp_aef/geos/
/Volumes/x10pro/kelp_aef/interim/
/Volumes/x10pro/kelp_aef/processed/
/Volumes/x10pro/kelp_aef/models/
/Volumes/x10pro/kelp_aef/reports/figures/
/Volumes/x10pro/kelp_aef/reports/tables/
```

Create the external artifact directories with:

```bash
mkdir -p /Volumes/x10pro/kelp_aef/raw/kelpwatch \
  /Volumes/x10pro/kelp_aef/raw/aef \
  /Volumes/x10pro/kelp_aef/raw/aef_samples \
  /Volumes/x10pro/kelp_aef/geos \
  /Volumes/x10pro/kelp_aef/interim \
  /Volumes/x10pro/kelp_aef/processed \
  /Volumes/x10pro/kelp_aef/models \
  /Volumes/x10pro/kelp_aef/reports/figures \
  /Volumes/x10pro/kelp_aef/reports/tables
```

Early smoke-test artifacts:

- `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json`
- `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json`
- `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aef_samples.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet`
- `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`
- `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/figures/sample_kelpwatch_vs_aef.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`
- `docs/report_snapshots/monterey_phase0_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md`

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

Phase 0 closeout validation passed with:

```bash
make check
```

The full check covered formatting, linting, mypy, and the test suite.

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

## Phase 1 Boundary

Phase 1 model and domain hardening for the Monterey annual-max pipeline closed
on 2026-05-13. The planning note and closeout decision are:

```text
docs/phase1_model_domain_hardening.md
docs/phase1_closeout_model_policy_decision.md
```

The closed Phase 1 architecture:

- Added reference baselines under the existing model/evaluation path.
- Added bathymetry/DEM domain-filter artifacts and manifests under the external
  artifact root.
- Keeps bathymetry/DEM as filtering and diagnostic inputs unless explicitly
  promoted to model predictors later.
- Added imbalance-aware tabular models before any deep spatial model work.
- Keeps the same Monterey annual-max label input; alternative temporal labels
  are out of scope for Phase 1.
- Preserves a rerunnable model-analysis closeout report.

Do not assume this phase includes full West Coast scale-up.

## Phase 2 Boundary

Phase 2 is the Big Sur generalization gate for the closed Monterey Phase 1
policy. The planning note is:

```text
docs/phase2_big_sur_generalization.md
```

The Phase 2 architecture should:

- keep the Monterey config and artifacts reproducible;
- add Big Sur through a small config or region-aware config extension, not a
  full West Coast config;
- use the same package-backed CLI command pattern as Monterey;
- keep AEF, Kelpwatch, domain-source, model, report, and visualizer outputs
  region-scoped so Big Sur never overwrites Monterey;
- verify Big Sur domain-source coverage before model interpretation;
- run an early visual QA pass for Big Sur labels, AEF coverage, and domain
  context before model training;
- treat the selected Phase 1 expected-value hurdle as the Monterey-trained
  transfer baseline;
- support Big Sur-only and pooled Monterey+Big Sur training-regime artifacts
  without mixing them with transfer outputs;
- allow the report and visualizer to grow region and year selection if separate
  single-region artifacts become awkward to review.

Phase 2 should not introduce deep spatial models, alternative temporal labels,
or bathymetry/DEM predictors. Those are Phase 3 candidates depending on the Big
Sur result.
