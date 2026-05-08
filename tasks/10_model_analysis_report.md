# Task 10: Model Analysis And Phase 0 Report

## Goal

Analyze the Monterey smoke-test model behavior deeply enough to decide what
Phase 1 should do next, then write a report-style Phase 0 closeout document.
The report should summarize what the smoke test accomplished, what the first
ridge baseline learned, where it failed, and whether the main failure mode looks
like label distribution, label aggregation, temporal mismatch, alignment drift,
model capacity, incomplete baselines, or split/scope limitations.

This task keeps Phase 0 open until the report is written and reviewed.

## Inputs

- Config: `configs/monterey_smoke.yaml`.
- Kelpwatch annual labels:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet`.
- Kelpwatch annual label manifest:
  `/Volumes/x10pro/kelp_aef/interim/labels_annual_manifest.json`.
- Aligned feature/label table:
  `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`.
- Split manifest:
  `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet`.
- Baseline predictions:
  `/Volumes/x10pro/kelp_aef/processed/baseline_predictions.parquet`.
- Baseline metrics:
  `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv`.
- Residual QA artifacts from Task 09:
  - `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/ridge_observed_vs_predicted.png`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.csv`.
- Optional source-level Kelpwatch NetCDF from the source manifest, if needed to
  compare annual labels against the original quarter-level source values.

## Outputs

- Final Phase 0 model-analysis report:
  `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase0_model_analysis.md`.
- Report manifest:
  `/Volumes/x10pro/kelp_aef/interim/model_analysis_manifest.json`.
- Label and target distribution summary tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_label_distribution_by_stage.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_framing_summary.csv`.
- Error and prediction diagnostic tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_observed_bin.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_persistence.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_prediction_distribution.csv`.
- Phase 1 decision and readiness tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_threshold_sensitivity.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_spatial_holdout_readiness.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_feature_separability.csv`.
  - `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_phase1_decision_matrix.csv`.
- Inline report figures:
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_label_distribution_by_stage.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_observed_vs_predicted_distribution.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_observed_bin.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_observed_900_predictions.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_persistence.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_alternative_target_framings.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_feature_projection.png`.
  - `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_spatial_holdout_readiness.png`.

## Config File

Use `configs/monterey_smoke.yaml`.

Expected config additions should cover:

- Report output directory, initially
  `/Volumes/x10pro/kelp_aef/reports/model_analysis`.
- Report markdown path.
- Report figure/table output paths or prefixes.
- Model name to analyze, initially `ridge_regression`.
- Primary test split/year, initially test year 2022.
- Observed-area bins for residual diagnostics.
- Binary threshold values for threshold-sensitivity diagnostics.
- Alternative target-framing settings, including which quarters should be
  treated as fall or winter after confirming the Kelpwatch quarter convention.

## Plan/Spec Requirement

Write a brief implementation plan before editing code. The plan should state:

- Files to change.
- Exact report sections.
- Plot and table list.
- How stage comparisons will be computed.
- How alternative target framings will be derived.
- How source-level Kelpwatch quarter data will be used, if used.
- How feature separability will be summarized without training a new model.
- How the report will translate findings into Phase 1 decision branches.
- Validation and artifact-inspection steps.

## Proposed CLI

```bash
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

## Main Questions

The report should answer:

- Did the end-to-end smoke-test pipeline work?
- Did alignment preserve the Kelpwatch label distribution, or did it alter the
  distribution through footprint filtering or missing-feature drops?
- Are zero-area labels still present in the aligned and modeled data?
- How common are saturated `900 m2` Kelpwatch labels by year and split?
- What is the observed label distribution for annual max canopy?
- What is the prediction distribution, and what is the maximum ridge prediction?
- For observed `900 m2` pixels, what does ridge predict?
- Is underprediction concentrated in saturated/high-canopy labels, specific
  years, specific coast segments, or persistent kelp locations?
- Does the ridge prediction align better with alternative target framings such
  as mean canopy, persistent presence, winter presence, fall presence, minimum
  nonzero canopy, or number of nonzero quarters?
- Does the evidence suggest annual max is a poor temporal target for annual AEF
  embeddings, or does it mainly suggest model-capacity limitations?
- Are current no-skill and ridge results enough, or does Phase 1 need stronger
  reference baselines such as previous-year, climatology, and geographic-only
  models before interpreting embedding skill?
- Is Monterey alone adequate for spatial holdout diagnostics, or should Phase 1
  expand to another region or a broader California/West Coast slice?
- Do the AEF features separate zero, positive, persistent, and saturated kelp
  labels visibly enough to justify stronger tabular models next?
- Which concrete Phase 1 branch is best supported by the evidence?

## Required Analyses

### Stage Distribution Comparison

Compare label distributions across the pipeline stages:

- Annual Kelpwatch label table.
- Aligned feature/label table.
- Split manifest retained rows.
- Ridge prediction table rows.

For each stage, year, and split when available, report:

- Row count.
- Station count.
- Missing-label count.
- Zero count and fraction.
- Positive count and fraction.
- `900 m2` count and fraction.
- Median, p90, p95, p99, max.
- Aggregate canopy area.

If source-level Kelpwatch quarter data is straightforward to load, also compare:

- Quarter-level source values within the configured footprint.
- Annual max labels derived from those quarter-level values.

### Kelpwatch Quarter And Season Grounding

Before interpreting fall or winter labels, confirm the Kelpwatch time coordinate
and quarter convention. The report should include a small table that maps:

- Source time coordinate values.
- Calendar dates or date ranges, if present.
- Derived year.
- Derived quarter.
- Any inferred season labels used by this task.

Do not interpret `area_q1` through `area_q4` as fall or winter until this
mapping is explicit in the report.

### Label Distribution Figures

Create histograms or ECDF-style figures showing:

- Annual `kelp_max_y` distribution by year.
- Annual `kelp_max_y` distribution by stage.
- A focused high-canopy view, including the count and fraction at `900 m2`.

The report should explicitly confirm whether saturated `900 m2` labels are rare
or common enough to be a first-order modeling issue.

### Binary Threshold Sensitivity

Summarize class balance, area totals, and threshold sensitivity for binary label
framings motivated by the project backlog and Kelpwatch detection-limit
questions. At minimum include:

- `kelp_max_y > 0`.
- `kelp_fraction_y >= 0.01`.
- `kelp_fraction_y >= 0.05`.
- `kelp_fraction_y >= 0.10`.
- High-canopy thresholds such as `kelp_fraction_y >= 0.50` and
  `kelp_fraction_y >= 0.90`.

For each threshold, report:

- Positive count and fraction by year and split.
- Aggregate canopy area represented by positive rows.
- Ridge prediction distribution for positives and negatives.
- Simple precision/recall-style diagnostics after converting clipped ridge
  predictions to the same threshold.

Do not choose a production binary threshold in this task; use this section to
inform whether threshold selection should be a Phase 1 priority.

### Prediction Distribution And Saturation

Compare observed and predicted canopy distributions for ridge:

- Overall and by split/year.
- Test-year observed vs predicted histograms.
- Predicted max, p95, p99, and p99.9.
- Distribution of ridge predictions conditioned on `kelp_max_y == 900`.
- Distribution of ridge predictions conditioned on high-canopy bins such as
  `kelp_max_y >= 450` and `kelp_max_y >= 810`.

This section should test the observed behavior that ridge does not want to
predict values near `900`.

### Baseline Completeness

Assess whether the current baseline evidence is strong enough to claim the
embedding model beats meaningful simple alternatives. The report should list
which baselines have been implemented and which are missing from the research
plan:

- Train-mean no-skill baseline.
- Previous-year kelp baseline.
- Per-station or per-location climatology baseline.
- Latitude/longitude/year-only geographic baseline.
- Any available non-AEF spectral-product baseline.

Do not implement missing baselines in this task unless the user explicitly
expands scope. Instead, state whether missing baselines should be part of Phase
1 before or alongside stronger embedding models.

### Residual Diagnostics

Summarize residuals by:

- Observed canopy area bins.
- Split/year.
- Latitude band or spatial segment, reusing Task 09 outputs where useful.
- Zero vs positive labels.
- High-canopy and saturated-label subsets.

Use residual sign consistently:

```text
residual = observed - predicted
```

Positive residuals mean underprediction. Negative residuals mean
overprediction.

### Alternative Target Framing Diagnostics

Use the annual label table's quarterly columns where possible:

- `area_q1`
- `area_q2`
- `area_q3`
- `area_q4`
- `valid_quarter_count`
- `nonzero_quarter_count`

Derive and summarize target framings such as:

- Annual max area: existing `kelp_max_y`.
- Annual mean area across valid quarters.
- Annual minimum area across valid quarters.
- Mean presence fraction:
  `nonzero_quarter_count / valid_quarter_count`.
- Persistent presence:
  kelp is present in every valid quarter.
- Any presence:
  kelp is present in at least one valid quarter.
- High-canopy presence under one or more explicit thresholds.
- Fall area or fall presence, after confirming the quarter mapping.
- Winter area or winter presence, after confirming the quarter mapping.
- Fall-minus-winter area, after confirming the quarter mapping.

For each framing, ask whether ridge predictions are more consistent with that
framing than with annual max. This can include:

- Correlation or rank correlation between ridge predictions and each framing.
- Residual summaries grouped by persistence/presence bins.
- Area-bias-like summaries for continuous target framings.
- Simple binary threshold diagnostics for presence-style framings.

Do not train new models for this task unless the user explicitly expands scope.
This is an analysis of the existing smoke-test ridge model and labels.

### Feature Separability Diagnostics

Summarize whether the AEF feature space visibly separates key target states
without fitting a new predictive model. Use lightweight diagnostic summaries
such as:

- PCA or another deterministic 2D projection of AEF features colored by zero,
  positive, persistent, high-canopy, and saturated labels.
- Per-label-bin summaries of the first few projection coordinates.
- Optional per-band or projection-coordinate standardized differences between
  zero, positive, persistent, and saturated groups.

This section should help distinguish:

- The ridge model is too limited for a separable feature signal.
- The annual-max or saturated labels are not separable in the annual AEF
  features.
- The feature signal is dominated by geography or other confounders.

Do not use this projection as evidence of field truth; it is only diagnostic
support for Phase 1 planning.

### Spatial Split And Scale-Up Readiness

Summarize whether the current Monterey smoke region can support useful spatial
holdout analysis and what scale-up should prioritize. Include:

- Row, station, positive-label, and saturated-label counts by latitude band.
- Area totals by latitude band.
- Whether each band has enough zero, positive, high-canopy, and saturated rows
  for a meaningful holdout.
- Whether top residuals are concentrated in one or more spatial segments.
- Whether the next data expansion should prioritize a second smoke region,
  broader California, the full West Coast, or more years if available.

This section should connect directly to the backlog split families: year
holdout, latitude/spatial holdout, state holdout, and region/site holdout.

### Phase 1 Decision Matrix

End the analysis with an explicit decision matrix that maps findings to Phase 1
branches. Include at least these branches:

- Derived-label Phase 1:
  choose this if annual max or saturated transient labels explain most error.
- Baseline-hardening Phase 1:
  choose this if current no-skill/geographic/climatology baselines are
  insufficient to interpret ridge skill.
- Stronger-tabular-model Phase 1:
  choose this if feature separability is visible but ridge underfits high
  canopy or nonlinear thresholds.
- Spatial-evaluation Phase 1:
  choose this if residuals are spatially clustered or Monterey alone cannot
  support meaningful generalization tests.
- Ingestion/alignment-hardening Phase 1:
  choose this if label distribution changes, missing-feature drops, or
  source-alignment issues materially affect conclusions.
- Scale-up Phase 1:
  choose this only if the smoke-test artifacts are stable and the key failure
  mode needs more geography or years to resolve.

For each branch, state:

- Triggering evidence from this report.
- Proposed next task or tasks.
- Expected new artifacts.
- What decision the branch would unlock.

## Report Structure

The final Markdown report should include:

- Executive summary.
- Smoke-test scope and artifacts.
- Pipeline accomplishments.
- Data and label distribution findings.
- Alignment and retained-row findings.
- Ridge baseline performance recap.
- Residual and saturation findings.
- Binary threshold-sensitivity findings.
- Alternative target-framing findings.
- Feature separability findings.
- Spatial split and scale-up readiness.
- Baseline completeness and missing reference baselines.
- Interpretation: most likely explanations for the observed failure mode.
- Phase 1 decision matrix and recommended next branch.
- Appendix with artifact paths and validation commands.
- Appendix with implemented vs missing backlog/research-plan items.

Use inline Markdown images for key figures so the report can be read as a
single document. Keep detailed tables as linked CSV artifacts and summarize the
important rows in prose.

## Validation Command

```bash
make check
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Add focused tests with small synthetic inputs to verify:

- Stage distribution tables preserve zero and saturated labels.
- Alignment-stage comparison does not silently drop zero labels.
- Missing-feature drops are counted separately from label filtering.
- Alternative target framings handle zero, missing, and all-quarter-present
  cases correctly.
- Residual-by-bin tables use `observed - predicted`.
- Threshold-sensitivity tables use configured thresholds and preserve class
  counts by split/year.
- Feature projection diagnostics are deterministic for a tiny synthetic input.
- Phase 1 decision matrix rows are written and included in the report.
- The Markdown report includes expected sections and inline figure links.
- Manifest, figures, tables, and report are written.

## Smoke-Test Region And Years

- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validation 2021, test 2022.
- Primary model: `ridge_regression`.

## Acceptance Criteria

- The command writes all configured model-analysis tables, figures, report, and
  manifest.
- The report includes inline plots for the core label, prediction, residual,
  and alternative-target diagnostics.
- The report explicitly states whether zero labels were retained through
  alignment and modeling.
- The report explicitly states how many `900 m2` labels exist by year and split.
- The report identifies whether the ridge underprediction is mainly associated
  with saturated annual-max labels, nonpersistent kelp, spatial clusters, or a
  broader model-capacity issue.
- The report states which baselines are implemented and which reference
  baselines remain missing.
- The report states whether Monterey is sufficient for spatial holdout analysis
  or whether Phase 1 should expand geography.
- The report includes a Phase 1 decision matrix with evidence-linked branches.
- The report gives a concrete recommendation for the next Phase 1 branch, not
  only the next model.
- `make check` passes.

## Known Constraints And Non-Goals

- Do not close Phase 0 before this report is reviewed.
- Do not train or tune new models in this task.
- Do not replace the annual label derivation in this task.
- Do not bulk-download any new external data.
- Do not frame results as field-verified kelp truth; this remains
  Kelpwatch-style label learning.
- Do not expand to the full West Coast.
