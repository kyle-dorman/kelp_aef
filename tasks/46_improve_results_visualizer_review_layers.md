# Task 46: Improve Results Visualizer Review Layers

## Goal

Improve the results visualizer so manual QA focuses on biologically and
model-diagnostically useful cells instead of being dominated by deep-water true
negative points selected only because a support layer has nonzero values.

The next visualizer pass should prioritize:

- all Kelpwatch-observed canopy-area points for the configured split/year;
- binary false negatives where Kelpwatch indicates presence but the calibrated
  binary model predicts absence;
- high hurdle or conditional-canopy values that need review as possible offshore
  or deep-water artifacts;
- a clearer default layer state that does not draw every true negative in the
  binary outcome layer.

Start with the existing `50,000` point cap. Expand the cap only if the revised
selection cannot show the Kelpwatch area points and diagnostic FN/high-value
points needed for review, and record any cap increase in the manifest.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Existing visualizer implementation:
  `src/kelp_aef/viz/results_visualizer.py`.
- Existing visualizer task context:
  `tasks/40_interactive_results_visualizer.md`.
- Current Monterey retained-domain predictions:
  `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`.
- Current Monterey retained-domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current Monterey visualizer outputs:
  - `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
  - `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`

## Outputs

- Updated package-backed visualizer behavior through:
  `kelp-aef visualize-results --config configs/monterey_smoke.yaml`.
- Revised inspection-point selection that preserves the rows needed for
  Kelpwatch-positive review, binary FN review, and high-value hurdle/conditional
  review.
- Revised binary outcome display semantics, for example by hiding TNs by default
  or splitting TNs into a separate optional layer.
- Refreshed Monterey visualizer artifacts:
  - `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
  - `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer/`
  - `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`
- Tests covering the revised point-selection and binary-outcome layer behavior.

## Config File

Use `configs/monterey_smoke.yaml`.

Keep `reports.results_visualizer.max_inspection_points: 50000` as the initial
cap. If the implementation needs a larger value, update the config deliberately
and make the reason visible in the manifest or task outcome.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should specify:

- how Kelpwatch-observed canopy-area points are defined;
- whether "Kelpwatch area points" means all Kelpwatch-observed rows, all
  nonzero observed rows, or both as separate layers;
- the exact binary FN condition, including threshold/source columns;
- the high-value hurdle or conditional-canopy selection rule;
- how priority buckets are combined under the point cap without crowding out
  Kelpwatch positives or FNs;
- how TNs are represented so the deep-water gray mass is not the default visual
  impression;
- what manifest row counts will prove each diagnostic bucket was included.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
```

If shared report semantics or model-analysis text is touched, also run:

```bash
uv run pytest tests/test_model_analysis.py
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

- The default visualizer no longer shows a large deep-water TN mass as the
  dominant binary outcome layer.
- Kelpwatch-observed positive canopy cells for `test` / `2022` are represented
  in the inspection export unless the count exceeds the configured cap, in which
  case the manifest reports how many were included and omitted.
- Binary FNs are explicitly selectable or visible in a review layer.
- High hurdle or conditional-canopy values in deeper water remain reviewable
  without being confused with final selected hurdle predictions.
- The manifest reports row counts for the selection buckets used to build the
  inspection table.
- The refreshed viewer still has radio-style mutually exclusive data layers,
  click popups, coordinate copy, and runtime basemap support.
- The task outcome states whether the existing `50,000` point cap was enough or
  whether it was expanded.

## Known Constraints And Non-Goals

- This is a visualization and QA task, not a model-selection or threshold-tuning
  task.
- Do not tune on held-out 2022 test rows based on visual inspection.
- Do not change the annual-max label target.
- Do not change the retained-domain mask thresholds.
- Do not make Big Sur model or report changes in this task, but keep the
  visualizer behavior general enough to reuse for Big Sur later in Phase 2.
