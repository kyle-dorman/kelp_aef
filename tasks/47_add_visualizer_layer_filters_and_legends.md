# Task 47: Add Visualizer Layer Filters And Legends

## Goal

Add optional, per-layer filtering and clearer legends to the results visualizer so
reviewers can hide visual noise without losing access to diagnostic rows.

The current point layers can become hard to interpret when low-scoring
continuous values, low-probability binary points, or noisy binary outcome
classes dominate the map. This task should add explicit controls that let the
reviewer filter those points differently for each data type or dataset while
keeping the underlying inspection export reproducible.

Improve legends at the same time. Text-only guidance is not enough; each active
data type should have a compact visual legend with swatches, color ramps, class
labels, thresholds, and units where relevant.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Existing visualizer implementation:
  `src/kelp_aef/viz/results_visualizer.py`.
- Follow-on review-layer task:
  `tasks/46_improve_results_visualizer_review_layers.md`.
- Current Monterey visualizer artifacts:
  - `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
  - `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`

## Outputs

- Updated `visualize-results` HTML with filter controls scoped to the active data
  layer.
- Configurable filter defaults for supported data types, such as:
  - continuous canopy area thresholds;
  - residual absolute-value thresholds;
  - binary probability thresholds;
  - binary outcome class toggles, including optional hiding of noisy classes;
  - dataset- or layer-specific thresholds where one global cutoff would be
    misleading.
- Improved per-data-type legends:
  - continuous prediction color ramp with units;
  - residual diverging color ramp with sign meaning;
  - probability ramp with probability ticks;
  - binary outcome swatches for TP, FP, FN, and TN;
  - visible threshold/filter state for the active layer.
- Updated manifest entries recording default filter settings and legend metadata.
- Tests covering generated controls, defaults, and legend payloads.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config shape can live under `reports.results_visualizer`, for example:

```yaml
filter_defaults:
  continuous_min_area_m2: 1.0
  residual_min_abs_area_m2: 1.0
  probability_min: 0.01
  binary_outcomes_visible: [TP, FP, FN]
layer_filter_defaults:
  conditional_ridge:
    continuous_min_area_m2: 90.0
```

The exact schema can differ if a cleaner implementation emerges, but it should
support per-data-type defaults and layer-specific overrides.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should specify:

- which filters are UI-only versus applied during inspection-point export;
- how layer-specific filter overrides are represented in config and manifest;
- the default binary outcome visibility, especially whether TN is hidden by
  default and whether FN can be toggled off when it clouds other information;
- the default thresholds for continuous, residual, and probability layers;
- how legends update when the active radio data layer changes;
- how color ramps and swatches are generated without making the HTML brittle.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
```

For final project validation, run:

```bash
make check
```

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Primary review split/year: `test` / `2022`.
- Label input: Kelpwatch-style annual max canopy,
  `kelp_fraction_y` / `kelp_max_y`.
- Primary scope: retained plausible-kelp domain,
  `evaluation_scope = full_grid_masked`.

## Acceptance Criteria

- The viewer has visible, layer-aware filter controls for the active data layer.
- Reviewers can hide low-scoring continuous values without changing the source
  prediction artifacts.
- Reviewers can toggle binary outcome classes, including noisy classes that cloud
  the map, while preserving access to TP, FP, FN, and TN when needed.
- Filters can differ by data type and, where needed, by specific layer/dataset.
- The active legend uses visual swatches or color ramps instead of text-only
  descriptions.
- The manifest records default filter settings and legend metadata.
- Existing click popups, coordinate copy, radio-style data-layer selection, and
  runtime basemap support still work.

## Known Constraints And Non-Goals

- Do not tune model thresholds or Phase 2 conclusions from visual filtering.
- Do not remove diagnostic rows from source prediction artifacts.
- Prefer UI filtering over export-time dropping unless the point cap forces a
  documented export-selection rule.
- Do not make Big Sur model-evaluation changes in this task, but keep the
  controls reusable for a Big Sur visualizer.
