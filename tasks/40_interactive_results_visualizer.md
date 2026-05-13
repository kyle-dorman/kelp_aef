# Task 40: Create Interactive Results Visualizer

## Goal

Create a local interactive map viewer for qualitative review of Monterey Phase 1
model results against Kelpwatch-style labels.

The viewer should make it easy to pan and zoom into retained-domain prediction
patterns, compare observed annual-max labels against one or more model
prediction layers, inspect residuals at specific cells, and use the displayed
coordinates as a bridge to external tools such as the Planet UI or Kelpwatch.

This is a visual QA and interpretation tool. It should not replace the
validation tables or tune model thresholds from the 2022 test split.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Current full-grid inference table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet`.
- Current plausible-kelp domain mask:
  `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet`.
- Current split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Current retained-domain prediction artifacts, as available:
  - `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
- Current Phase 1 model comparison table:
  `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_model_comparison.csv`.
- Existing report map outputs for visual cross-checking:
  - `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.masked.png`
  - `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png`
  - `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_2022_map.png`

Current anchors to preserve:

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Default review split/year: test 2022.
- Label input: Kelpwatch-style annual max, `kelp_fraction_y` / `kelp_max_y`.
- Primary review scope: retained plausible-kelp domain,
  `evaluation_scope = full_grid_masked`.
- Default sample policy: `crm_stratified_mask_first_sample`.

## Outputs

- Package-backed command, for example:
  `kelp-aef visualize-results --config configs/monterey_smoke.yaml`.
- Config block for the viewer under `reports.results_visualizer`.
- Local HTML entry point:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`.
- Web-friendly layer assets, for example:
  `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer/`.
- Viewer manifest:
  `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`.
- Optional cell-inspection extract or top-residual point layer if needed for
  browser performance, for example:
  `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions:

- Review split and year, defaulting to `test` and `2022`.
- Layer list with model names and source prediction paths.
- Output HTML path, asset directory, and manifest path.
- Background map provider URL/template, attribution, and an enable/disable flag.
- Layer opacity defaults and robust color-scale policy.
- Maximum rows or tile/rendering limits for any point-based inspection layer.

The background map layer may use external web tiles at viewer runtime, but the
command should not download or cache basemap tiles. The viewer should still open
with model/label layers if the basemap provider is unavailable.

## Planned Libraries And Format Choices

Keep the first implementation boring, local, and artifact-oriented.

Use Python only for deterministic data preparation:

- `pandas` / `pyarrow` for reading Parquet predictions, selecting the configured
  split/year, joining label, prediction, mask, and split metadata, and writing
  compact manifest or inspection-table outputs.
- `GeoPandas` / `Shapely` for footprint handling, optional point or polygon
  export, and any GeoJSON/GeoParquet-friendly inspection layers.
- Do not use the historical direct-continuous P1-22 artifacts as active viewer
  defaults. They remain audit artifacts only after the Phase 1 closeout cleanup.

Use direct Leaflet HTML/JavaScript for the browser viewer:

- `L.TileLayer` for the configurable background map.
- `L.ImageOverlay`, local generated tiles, or a bounded canvas/point layer for
  observed, predicted, and residual layers.
- `L.control.layers` for toggling basemaps and overlays.
- Leaflet click/hover events for cell inspection, coordinates, and copyable
  values for manual Planet or Kelpwatch lookup.

Avoid notebook-first or opaque generated-viewer paths for the MVP:

- Do not make the primary artifact an `ipyleaflet` notebook widget.
- Do not use `folium` unless it is only a quick prototype and the final output
  still has explicit, testable generated assets and HTML controls.
- Do not start with MapLibre, vector tiles, or PMTiles unless Leaflet image or
  local tile overlays are too slow or too blurry at useful zoom levels.

External basemap tiles are allowed only as a runtime viewer dependency. The
model/label/residual layers must be local generated artifacts, and the manifest
must record the basemap URL template, attribution, and no-basemap fallback.

## Plan/Spec Requirement

This is a new visualization tool and artifact family. Before implementation,
write a brief implementation plan that confirms:

- Whether the implementation extends the existing residual-map HTML helper or
  creates a dedicated `src/kelp_aef/viz/results_visualizer.py` module.
- The selected browser mapping library and why it is acceptable for a local
  generated artifact.
- Whether layers are emitted as raster overlays, XYZ tiles, PMTiles, vector
  tiles, or a bounded point layer.
- The first supported layers for the MVP.
- The exact join keys used to combine labels, predictions, mask context, and
  split metadata.
- How the tool avoids putting the full retained-domain table into one huge
  embedded JSON payload.
- How color scales remain comparable to the static report maps.
- Which external background map URL is used and how attribution is displayed.
- How manual visual observations will be kept separate from validation metric
  conclusions.

## Implementation Plan

- Implement the first pass as a dedicated
  `src/kelp_aef/viz/results_visualizer.py` module and a `visualize-results`
  CLI command.
- Use direct Leaflet HTML rather than notebook widgets or Folium so the output
  is a deterministic local artifact with explicit assets and testable controls.
- Emit local PNG image overlays for observed canopy, model predictions, and
  residuals. Use `aef_grid_row` / `aef_grid_col` to place values on a compact
  array, write transparent no-data pixels, and use Leaflet `ImageOverlay` bounds
  from the selected row longitude/latitude extent.
- Emit a bounded GeoJSON inspection layer for top residual and nonzero cells
  instead of embedding the full retained-domain table in the HTML. Keep the
  default point cap configurable.
- Start with the current primary hurdle expected-value model when available and
  the ridge baseline as the comparison model. Include binary-presence
  probability as an optional overlay when the artifact is available.
- Use `aef_grid_cell_id` as the primary join key for retained-domain mask
  context. Use `aef_grid_row` and `aef_grid_col` as the display grid keys.
- Keep color-scale logic aligned with the report maps: observed and predicted
  area share a robust canopy scale, and residuals use a diverging scale centered
  on zero.
- Use a configurable OpenStreetMap tile template as the default background
  layer, recorded with attribution in the manifest, and include a blank
  no-basemap option.
- Treat manual Planet or Kelpwatch comparisons as qualitative visual QA notes,
  not validation metrics or threshold-selection input.

- Add a package-backed `visualize-results` CLI command.
- Load the configured split/year and retained plausible-kelp mask.
- Read only the columns needed for visual review and cell inspection.
- Join retained-domain labels, prediction layers, split metadata, and optional
  domain context by stable grid-cell keys such as `aef_grid_cell_id`,
  `aef_grid_row`, and `aef_grid_col`.
- Build browser-friendly layers for:
  - observed Kelpwatch annual-max canopy area;
  - current primary model prediction;
  - ridge prediction as a baseline comparison;
  - residuals using `observed - predicted`;
  - binary presence probability or selected class if the artifact is present;
  - optional top overprediction and underprediction points.
- Include a configurable basemap layer such as OpenStreetMap or Esri imagery
  with attribution, plus a no-basemap fallback.
- Provide layer controls for model/label/residual overlays, opacity, and
  basemap toggling.
- Provide click or hover inspection with at least:
  `longitude`, `latitude`, `aef_grid_cell_id`, `year`, `split`,
  `label_source`, observed canopy area, predicted canopy area, residual,
  model name, mask status, and domain-mask reason when available.
- Add a coordinate/cell-id copy affordance so the same location can be checked
  in Planet or Kelpwatch manually.
- Write a manifest with input paths, row counts, layer names, color scales,
  basemap provider, split/year, and output paths.
- Add tests for config loading, layer-selection behavior, manifest contents, and
  the generated HTML containing the expected basemap/layer controls.

## Expected Review Semantics

- The viewer is for qualitative spatial QA: finding suspicious false positives,
  missed canopy patches, boundary artifacts, and places worth checking in
  external imagery.
- The authoritative model comparison still comes from the Phase 1 report tables,
  especially the retained-domain `test` / `2022` rows.
- Manual review notes should be recorded as follow-up observations or task
  outcomes. Do not use ad hoc Planet/Kelpwatch visual inspection to tune the
  2022 test metrics.
- The viewer should label outputs as Kelpwatch-style weak-label results, not
  independent ecological truth.

## Validation Command

Focused validation:

```bash
uv run pytest tests/test_results_visualizer.py tests/test_residual_maps.py
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
```

If the implementation shares helpers with `analyze-model`, include:

```bash
uv run pytest tests/test_model_analysis.py
```

Full validation:

```bash
make check
```

Manual visual review:

```bash
rg -n "OpenStreetMap|Esri|Layer|Observed|Predicted|Residual" /Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html
```

Then open the generated HTML locally and confirm:

- the basemap appears when network access is available;
- the model and label overlays can be toggled;
- zooming into Monterey kelp patches keeps the overlays spatially aligned;
- clicking or hovering returns usable coordinates and model/label values;
- the displayed coordinate can be checked manually in Planet or Kelpwatch.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Default viewer year: 2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Label: Kelpwatch annual max canopy.
- Primary review scope: retained plausible-kelp domain with
  `crm_stratified_mask_first_sample`.

## Acceptance Criteria

- `kelp-aef visualize-results --config configs/monterey_smoke.yaml` writes a
  local HTML viewer, layer assets, and manifest under `/Volumes/x10pro/kelp_aef`.
- The viewer supports panning and zooming over a background map layer.
- The viewer can toggle observed labels, model predictions, and residuals.
- At least the current primary model and ridge baseline are available as
  prediction layers when their artifacts exist.
- The viewer has a no-basemap fallback and does not download basemap tiles during
  generation.
- Cell inspection exposes coordinates, model values, observed labels, residuals,
  and mask context needed for manual comparison in Planet or Kelpwatch.
- The generated browser payload is bounded or tiled; it does not embed the full
  retained-domain table as one large JSON blob.
- Color scales are documented in the manifest and remain consistent enough to
  compare against the static report maps.
- Tests cover config parsing, layer selection, manifest output, and generated
  HTML controls.
- `make check` passes.

## Known Constraints Or Non-Goals

- Do not ingest, scrape, or automate the Planet UI or Kelpwatch UI in this task.
- Do not add Planet imagery or other proprietary basemap data to the repo or
  artifact root.
- Do not treat external visual comparison as independent validation of true kelp
  biomass.
- Do not change model training, sampling, masks, thresholds, or fitted artifacts.
- Do not tune or select models based on 2022 visual inspection.
- Do not start full West Coast scale-up.

## Completion Notes

Completed on 2026-05-13.

Implemented a dedicated `src/kelp_aef/viz/results_visualizer.py` module and
package-backed CLI command:

```bash
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
```

The implementation uses:

- `pandas` / `pyarrow.dataset` for bounded Parquet reads by split, year, and
  model name;
- the existing retained-domain mask helper for baseline rows that do not already
  carry mask metadata;
- `GeoPandas` / `Shapely` for the inspection-point GeoJSON;
- `rasterio` for georeferenced GeoTIFF sidecars;
- direct Leaflet HTML/JavaScript for the local browser viewer.

Generated artifacts:

- `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer.html`
- `/Volumes/x10pro/kelp_aef/reports/interactive/monterey_results_visualizer/`
- `/Volumes/x10pro/kelp_aef/interim/results_visualizer_manifest.json`
- `/Volumes/x10pro/kelp_aef/reports/tables/results_visualizer_inspection_points.csv`

The final simplified Monterey viewer loads `630,151` retained-domain `test` /
`2022` rows for the configured expected-value hurdle layer and emits only:

- expected-value hurdle prediction;
- expected-value hurdle residual;
- conditional positive-row ridge prediction;
- calibrated binary presence probability;
- binary TP/FP/FN/TN outcome;
- bounded inspection points.

The final browser view uses coordinate-based point layers for prediction,
residual, conditional ridge, binary probability, and binary TP/FP/FN/TN outcome
review instead of PNG image overlays. Data-layer selection is radio-button
exclusive so only one model/result layer is visible at a time. This avoids
stretching the source grid over a single WGS84 rectangle and keeps every visible
mark tied to its stored longitude/latitude. The conditional ridge, binary
probability, and binary outcome layers are read from the same hurdle prediction
table columns that feed the expected-value composition:
`pred_conditional_area_m2`, `calibrated_presence_probability`, and
`pred_presence_class`. It uses an OpenStreetMap tile template only at browser
runtime and includes a no-basemap option. The inspection layer is bounded to
`50,000` high-residual or nonzero-support cells with coordinates and model/label
values for manual Planet or Kelpwatch lookup. Area point layers only draw values
with at least `1 m2` absolute magnitude, and the binary probability layer draws
probabilities of at least `0.01`, so tiny nonzero predictions do not appear as a
broad green mask. The click popup omits label-source and mask-reason rows and
uses compact labels such as `Hurdle pred m2`, `Cond ridge m2`, `Binary prob`,
and `Binary outcome`.

The first implementation briefly included observed-label, ridge, capped-weight,
binary-probability overlays, raster image overlays, and per-layer opacity
sliders. Those were removed after visual review because they made the viewer too
cluttered, the slider behavior was unclear, and the image overlay was spatially
misleading against the basemap.

Validation passed:

```bash
uv run pytest tests/test_results_visualizer.py tests/test_residual_maps.py tests/test_package.py -q
uv run mypy src/kelp_aef/viz/results_visualizer.py src/kelp_aef/cli.py tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/monterey_smoke.yaml
make check
```

Generated asset inspection confirmed that the final asset directory contains
only expected-value hurdle prediction/residual overlays plus inspection-point
assets, and that the prediction/residual rasters are nonblank.
