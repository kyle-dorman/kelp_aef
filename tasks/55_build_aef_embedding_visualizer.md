# Task 55: Build AEF Embedding Visualizer

## Goal

Build a visual QA tool for AlphaEarth Feature (AEF) embeddings so Phase 2 can
inspect feature-space structure directly against labels, model residuals,
binary outcomes, region, year, bathymetry/domain context, and computed
edge/interior context.

The main question is:

```text
Do FP/FN edge pixels, high-canopy underpredictions, and region-specific failure
clusters occupy distinct AEF embedding neighborhoods, or do they look
feature-similar to nearby observed kelp and background pixels?
```

This task should help decide whether the next phase needs better evaluation
tooling, temporal-label work, simple non-linear tabular models, or broader
regional sampling. It should not create a new model predictor or tune model
policy from held-out rows.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- AEF feature-bearing tables for Monterey and Big Sur:
  - retained full-grid inference tables;
  - retained model-input samples;
  - AEF annual embedding bands `A00-A63`.
- Label and context columns:
  - `kelp_fraction_y`;
  - `kelp_max_y`;
  - `label_source`;
  - `is_kelpwatch_observed`;
  - `year`;
  - `split`;
  - `longitude`;
  - `latitude`;
  - `aef_grid_row`;
  - `aef_grid_col`;
  - `crm_depth_m`;
  - `crm_elevation_m`;
  - `depth_bin`;
  - `domain_mask_reason`;
  - `mask_status`;
  - `evaluation_scope`.
- Model outputs for overlays:
  - local and pooled hurdle predictions;
  - local and pooled binary probabilities/outcomes;
  - residuals and component failure classes from Task 53 if available.
- Existing results visualizer code:
  - `src/kelp_aef/viz/results_visualizer.py`.
- Existing source-coverage and residual visualization patterns under
  `src/kelp_aef/viz/` and `src/kelp_aef/evaluation/`.

## Outputs

Write path-explicit artifacts under `/Volumes/x10pro/kelp_aef/`.
Exact names can vary, but expected outputs are:

- AEF embedding diagnostic tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/aef_embedding_projection_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/aef_embedding_projection_sample.csv`
  - `/Volumes/x10pro/kelp_aef/interim/aef_embedding_visualizer_manifest.json`
- Feature-derived RGB outputs for Monterey and Big Sur:
  - PCA RGB table or raster/PNG for 2022 retained-domain cells;
  - optional t-SNE or UMAP RGB sample table for review points;
  - per-year RGB outputs if temporal feature drift is easy to compare.
- Interactive HTML visualizer or result-viewer extension:
  - `/Volumes/x10pro/kelp_aef/reports/interactive/aef_embedding_visualizer.html`
  - or separate region-specific HTML files with an index page.
- Overlay layers or linked context fields for:
  - observed annual max;
  - binary TP/FP/FN/TN;
  - expected-value hurdle residual;
  - high-canopy underprediction;
  - domain/depth context;
  - edge/interior classes from Task 53 when available.

## Config File

Use `configs/big_sur_smoke.yaml` as the coordinating config and
`configs/monterey_smoke.yaml` for Monterey-local paths.

Add a path-explicit visualizer block such as `reports.aef_embedding_visualizer`
or a new diagnostics block. Do not hard-code artifact paths in implementation
code.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- whether the tool is a new command or an extension of `visualize-results`;
- which dimensionality reductions are used for full-grid RGB and which are used
  only for sampled points;
- how RGB values are normalized consistently across Monterey and Big Sur;
- whether labels are used only as overlays or also for a supervised diagnostic
  projection;
- sample caps for t-SNE/UMAP or other expensive reductions;
- how stochastic projections are seeded and recorded;
- how the visualizer avoids treating projected coordinates as model predictors
  or tuned policy variables.

## Required Analysis

Start with deterministic feature-only projections:

- Fit PCA on a documented training sample from AEF bands `A00-A63`.
- Use the first three PCA components as RGB after robust percentile scaling.
- Apply the same PCA/RGB mapping to Monterey and Big Sur so color differences
  are comparable across regions.
- Record explained variance and feature scaling metadata.
- Produce 2022 retained-domain RGB maps for both regions.

Add a non-linear sampled projection if feasible:

- Use t-SNE or UMAP on a capped, documented sample of retained-domain cells.
- Include labels, residuals, binary outcomes, region, year, depth/domain, and
  edge/interior classes as point metadata.
- Keep the non-linear projection as a sample diagnostic. Do not imply it is a
  full-grid transform unless using a method with a valid transform path.
- Seed stochastic reducers and record parameters in the manifest.

Use labels and residuals as overlays, not as the primary RGB fit:

- observed annual max bins;
- `annual_max_ge_10pct`;
- high-canopy and near-saturated status;
- binary TP/FP/FN/TN;
- expected-value hurdle residual;
- conditional amount residual;
- component failure class from Task 53;
- depth and domain-mask context.

Analyze these questions explicitly:

- Do observed kelp mats appear as coherent AEF color regions?
- Do FP cells adjacent to kelp mats have AEF colors closer to observed
  positives than to far-background zeros?
- Do FN cells on mat edges look feature-similar to interior positives or to
  surrounding background?
- Do high-canopy underpredictions occupy the same AEF neighborhoods as moderate
  positives, suggesting target-scale/model-capacity limits?
- Does pooled training fail in embedding regions that are mostly Big Sur,
  mostly Monterey, or shared between regions?
- Are there AEF feature clusters that correspond to bathymetry/domain context
  rather than kelp labels?
- Are there year-specific AEF color shifts that could explain annual-max target
  mismatch?
- Do southern/central/northern ecological hypotheses require more regions, or
  can Monterey and Big Sur already show feature-space separation?

The visualizer should support practical QA actions:

- switch region and year;
- switch projection mode such as PCA RGB versus sampled t-SNE/UMAP;
- overlay observed labels, binary outcomes, residuals, depth/domain context,
  and edge/interior classes;
- click a point or cell to inspect AEF projection values, label values, model
  predictions, residuals, depth, domain context, and failure class;
- copy coordinates for follow-up map review.

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_results_visualizer.py
uv run kelp-aef visualize-results --config configs/big_sur_smoke.yaml
git diff --check
```

If a new command is added, include focused tests for config parsing, projection
metadata, RGB scaling, sample caps, and generated HTML controls. Run the new
command against `configs/big_sur_smoke.yaml`.

If raster or PNG outputs are produced, inspect dimensions and verify nonblank
pixel values with a lightweight image check.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Years: start with 2022 held-out test rows; optionally include 2018-2022 if
  temporal feature drift is cheap to render.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Features: AEF annual embedding bands `A00-A63`.
- Labels: Kelpwatch-style annual max canopy.

## Acceptance Criteria

- Monterey and Big Sur AEF embeddings can be inspected with a shared projection
  or explicitly documented region-specific projection.
- The RGB mapping and projection parameters are recorded in a manifest.
- Reviewers can compare AEF feature colors/projection neighborhoods against
  observed labels, binary outcomes, residuals, depth/domain context, and
  edge/interior classes.
- The output explicitly supports the FP/FN edge hypothesis and high-canopy
  underprediction hypothesis.
- The task outcome states whether AEF feature-space structure argues for
  better evaluation tooling, temporal-label experiments, simple non-linear
  tabular models, or cautious regional expansion.

## Known Constraints / Non-Goals

- Do not download new AEF assets.
- Do not train or select a predictive model from PCA, t-SNE, UMAP, or RGB
  projection coordinates.
- Do not use labels to fit the primary feature-only RGB projection.
- Do not tune thresholds, masks, sample quotas, or model policy from held-out
  visual inspection.
- Do not make t-SNE the only visualization path; it is stochastic and does not
  provide a simple full-grid transform.
- Do not collapse Monterey and Big Sur rows without explicit region labels.
