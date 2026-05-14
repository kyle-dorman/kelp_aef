# Task 52: Generate Big Sur And Pooled Results Visualizer

## Goal

Generate or extend the interactive results visualizer so Phase 2 can inspect the
current local and pooled training-regime outputs without mixing Monterey and
Big Sur rows.

The primary review contexts are:

- Big Sur dataset with the Big Sur-only model.
- Monterey dataset with the Monterey-only model.
- Pooled Monterey+Big Sur model output, with Big Sur and Monterey evaluation
  rows still separated or explicitly selectable.

Use the P2-09 report split directly: binary support appears relatively
transferable, while canopy amount calibration and residual structure need Big
Sur visual QA. The visualizer should therefore make binary TP/FP/FN/TN support
reviewable, but it should emphasize expected-value hurdle amount predictions,
residuals, high-canopy underprediction, and spatial residual patterns.

This is a visual QA task. Do not tune thresholds, sample quotas, or model policy
from held-out 2022 visual inspection.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- Existing visualizer implementation:
  `src/kelp_aef/viz/results_visualizer.py`.
- Existing visualizer task context:
  - `tasks/40_interactive_results_visualizer.md`
  - `tasks/46_improve_results_visualizer_review_layers.md`
  - `tasks/47_add_visualizer_layer_filters_and_legends.md`
- Phase 2 report context and metrics:
  - `tasks/51_update_phase2_model_analysis_report.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`
- Big Sur local model predictions:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_full_grid_predictions.parquet`
- Monterey local model predictions:
  - `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
- Pooled model predictions:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
- Existing optional transfer diagnostics if needed for binary-support review:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_binary_presence_full_grid_predictions.parquet`
- Region masks, split manifests, and full-grid inference tables declared in the
  Monterey and Big Sur configs.

## Outputs

One of these output shapes is acceptable:

- A single multi-context visualizer with explicit region/training-regime
  selection.
- Separate HTML visualizers for the three main review contexts, plus a manifest
  that records the shared schema and context labels.

The outputs must include or refresh:

- Big Sur local visualizer:
  `/Volumes/x10pro/kelp_aef/reports/interactive/big_sur_results_visualizer.html`
- Monterey local visualizer:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
- Pooled Monterey+Big Sur visualizer or multi-context entry point, declared in
  config, for example:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_big_sur_pooled_results_visualizer.html`
- Region/training-regime-aware asset directories under:
  `/Volumes/x10pro/kelp_aef/reports/interactive/`
- Region/training-regime-aware manifests under:
  `/Volumes/x10pro/kelp_aef/interim/`
- Region/training-regime-aware inspection CSVs under:
  `/Volumes/x10pro/kelp_aef/reports/tables/`
- Tests covering config parsing, context selection, manifest labeling, and the
  generated HTML controls.

## Config Files

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config and preserve
`configs/monterey_smoke.yaml` as the Monterey-local config.

Expected config work:

- Keep the existing single-region `reports.results_visualizer` behavior working
  for Monterey.
- Extend the Big Sur config or add a small region-aware block that can declare
  multiple visualizer contexts.
- Each context must declare:
  - `context_id`
  - `display_name`
  - `evaluation_region`
  - `training_regime`
  - `model_origin_region`
  - `split`
  - `year`
  - full-grid prediction path
  - binary prediction or probability source path
  - output HTML, asset directory, manifest, and inspection CSV paths
- The pooled training regime must not collapse Monterey and Big Sur evaluation
  rows into one unlabeled point cloud. If one pooled visualizer is produced, it
  must expose a region selector/filter and record per-region row counts in the
  manifest.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should confirm:

- whether the implementation extends the existing single-config
  `visualize-results` command or adds a multi-context mode under the same
  command;
- whether outputs are one multi-context HTML or separate context-specific HTML
  files;
- the exact config schema for visualizer contexts;
- how Monterey and Big Sur grid rows are kept distinct in the browser, manifest,
  and inspection CSV;
- which model path is used for each context;
- whether transfer-model layers are included, and if so how they are labeled as
  optional diagnostics rather than the default local/pooled review layer;
- how the existing radio-style layer control, binary TP/FP/FN/TN layer,
  layer-aware filters, legends, click popups, and coordinate copy behavior are
  preserved;
- how row caps are allocated so Big Sur high-residual and binary FN rows are not
  crowded out by Monterey rows or true negatives;
- how the task records visual QA observations without changing model-selection
  conclusions.

## Suggested Implementation Plan

1. Inspect the current Big Sur visualizer config and generated artifacts.
2. Decide whether the least risky implementation is context-specific HTML files
   or one multi-context HTML.
3. Add a context loader that preserves the current single-region config path and
   accepts an explicit multi-context Phase 2 block.
4. Build the three required context groups:
   - `big_sur_local`: Big Sur rows with Big Sur-only predictions.
   - `monterey_local`: Monterey rows with Monterey-only predictions.
   - `pooled_monterey_big_sur`: pooled-model predictions, with Big Sur and
     Monterey evaluation rows separated by region.
5. Carry `evaluation_region`, `training_regime`, `model_origin_region`,
   `context_id`, and `display_name` into every manifest and inspection row.
6. Reuse the existing visualizer layer types for observed labels, expected-value
   hurdle prediction, residual, conditional ridge, binary probability, and
   binary outcome.
7. Preserve the existing default filters:
   - observed labels `>= 1 m2`;
   - expected-value hurdle predictions `>= 90 m2`;
   - residuals `abs(residual) >= 90 m2`;
   - conditional ridge predictions `>= 450 m2`;
   - binary probability `>= 0.10`;
   - binary outcome classes TP/FP/FN visible by default.
8. Regenerate the selected HTML, assets, inspection CSVs, and manifests.
9. Record in the task outcome whether the visual QA reveals spatial patterns
   that support P2-09's split between transferable binary support and
   region-specific canopy amount calibration.

## Outcome

Completed on 2026-05-14 with a multi-context extension to the existing
`visualize-results` command. The legacy single-config path remains intact, and
`configs/big_sur_smoke.yaml` now coordinates four context-specific HTML
visualizers plus an index page:

- Big Sur local:
  `/Volumes/x10pro/kelp_aef/reports/interactive/big_sur_results_visualizer.html`
- Monterey local:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
- Pooled Monterey+Big Sur evaluated on Big Sur:
  `/Volumes/x10pro/kelp_aef/reports/interactive/big_sur_pooled_monterey_big_sur_results_visualizer.html`
- Pooled Monterey+Big Sur evaluated on Monterey:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_pooled_monterey_big_sur_results_visualizer.html`
- Multi-context entry point:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_big_sur_pooled_results_visualizer.html`

The aggregate manifest is:
`/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_results_visualizer_manifest.json`.

Each context manifest and inspection CSV carries `context_id`,
`evaluation_region`, `training_regime`, and `model_origin_region`. The pooled
Big Sur and pooled Monterey outputs are separate context files, so the browser,
manifest, and inspection CSV never collapse the two evaluation regions into one
unlabeled point cloud.

Generated context row counts:

| Context | Evaluation region | Training regime | Layer rows | Inspection rows |
| --- | --- | --- | ---: | ---: |
| `big_sur_local` | `big_sur` | `big_sur_only` | `272,030` | `13,951` |
| `monterey_local` | `monterey` | `monterey_only` | `426,640` | `9,467` |
| `pooled_monterey_big_sur_on_big_sur` | `big_sur` | `pooled_monterey_big_sur` | `272,030` | `13,554` |
| `pooled_monterey_big_sur_on_monterey` | `monterey` | `pooled_monterey_big_sur` | `426,640` | `9,208` |

Inspection bucket checks show binary TP/FP/FN rows remain explicit while true
negatives stay optional or low-priority. Big Sur local exported `1,957` binary
FNs and `7,849` large-residual rows; pooled-on-Big-Sur exported `2,152` binary
FNs and `9,173` large-residual rows. This reinforces the P2-09 review framing:
binary support is directly inspectable, while canopy amount residual structure
still needs region-specific visual QA. No model artifacts, thresholds, masks,
labels, or training samples were changed.

Validation run:

```bash
uv run pytest tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/big_sur_smoke.yaml
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
rg -n "Big Sur|Monterey|pooled|Pooled|TP|FP|FN|TN|Observed|Residual" \
  /Volumes/x10pro/kelp_aef/reports/interactive/*.html
uv run ruff check .
uv run mypy src
uv run pytest
```

`make check` was also attempted after formatting the touched visualizer file,
but it is currently blocked by pre-existing Ruff format drift in unrelated
files: `src/kelp_aef/alignment/full_grid.py`,
`src/kelp_aef/evaluation/pooled_regions.py`, and
`src/kelp_aef/viz/source_coverage.py`.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/big_sur_smoke.yaml
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
```

If the implementation touches CLI dispatch or shared config helpers, also run:

```bash
uv run pytest tests/test_package.py
```

For final project validation, run:

```bash
make check
```

Manual visual checks:

```bash
rg -n "Big Sur|Monterey|pooled|Pooled|TP|FP|FN|TN|Observed|Residual" \
  /Volumes/x10pro/kelp_aef/reports/interactive/*.html
```

Open the generated HTML artifacts and confirm the first-screen context labels,
layer controls, legends, filters, basemap, popups, and coordinate copy behavior.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Training regimes:
  - Monterey-only.
  - Big Sur-only.
  - pooled Monterey+Big Sur.
- Primary review split/year: `test` / `2022`.
- Years available: 2018-2022.
- Label input: Kelpwatch-style annual max canopy,
  `kelp_fraction_y` / `kelp_max_y`.
- Primary scope: retained plausible-kelp domain,
  `evaluation_scope = full_grid_masked`.

## Acceptance Criteria

- Big Sur local review can be opened or selected with Big Sur rows and the
  Big Sur-only expected-value hurdle model as the default amount layer.
- Monterey local review can be opened or selected with Monterey rows and the
  Monterey-only expected-value hurdle model as the default amount layer.
- Pooled Monterey+Big Sur review can be opened or selected without collapsing
  Monterey and Big Sur evaluation rows into an unlabeled combined map.
- The viewer labels every active context with `evaluation_region`,
  `training_regime`, and `model_origin_region`.
- The manifest reports input paths, output paths, row counts, selected
  split/year, context labels, layer names, filter defaults, and selection-bucket
  counts for each context.
- The inspection CSV includes context columns so exported rows can be traced
  back to the correct region and training regime.
- The default review layers include observed annual-max labels,
  expected-value hurdle predictions, residuals, conditional ridge predictions,
  calibrated binary probability, and binary TP/FP/FN/TN outcomes.
- Binary outcome review keeps TP/FP/FN visible by default and keeps TN optional
  or filtered so true negatives do not dominate the map.
- The task outcome states whether visual patterns reinforce, complicate, or
  contradict the P2-09 split: transferable binary support versus
  region-specific canopy amount calibration.
- No model artifacts, thresholds, masks, annual-max labels, or training samples
  are changed by this task.

## Known Constraints And Non-Goals

- Do not retrain Monterey, Big Sur, or pooled models.
- Do not tune thresholds or model policy from held-out 2022 visual inspection.
- Do not change the annual-max label target.
- Do not change retained-domain mask thresholds.
- Do not start full West Coast visualizer work.
- Do not make the primary artifact a notebook-only viewer.
- Do not let transfer diagnostics become the default layer set unless the task
  explicitly records why they are needed for binary-support visual QA.
- Keep generated visualizer artifacts under `/Volumes/x10pro/kelp_aef`; do not
  track them in git.
