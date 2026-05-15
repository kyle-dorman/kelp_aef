# Task 58: Simplify Phase 2 Report For Pooled Diagnostics

## Goal

Turn the Phase 2 model-analysis report into a readable decision artifact that
keeps one compact six-context comparison at the top, then focuses the main body
on pooled Monterey+Big Sur diagnostics.

The current report still spends too much space comparing "trained on X and
applied to X/Y" contexts after that question is mostly understood for these two
adjacent grids. The report should instead help answer what fails under the
pooled workflow and what that implies for the next model/data task.

The main question is:

```text
For pooled Monterey+Big Sur evaluation, are remaining failures mostly binary
support, amount shrinkage, hurdle composition, edge/boundary mismatch,
depth/elevation context, temporal-label context, or model-capacity limits?
```

Frame results as Kelpwatch-style annual maximum reproduction, not independent
field-truth biomass validation.

## Inputs

- Config:
  - `configs/big_sur_smoke.yaml`
- Existing report outputs:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- Existing six-context comparison tables:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`
- Existing component-failure outputs from Task 53.
- New pooled context diagnostics from Task 56.
- New 1 km binary hex map from Task 57.
- Report-generation code in `src/kelp_aef/evaluation/model_analysis.py`.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

Regenerate the Phase 2 model-analysis report and manifest:

- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- existing Phase 2 report manifest path from `configs/big_sur_smoke.yaml`

If needed, write a compact top-summary table such as:

- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_phase2_top_model_summary.csv`

## Config File

Use `configs/big_sur_smoke.yaml`. Keep report paths path-explicit in config.
Do not hard-code artifact paths in report rendering code.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- the final report section order;
- which six-context table remains in the main body;
- which current tables move to appendix or artifact index;
- the exact definitions and denominators for the component-failure columns;
- how isolated, edge, interior, adjacent, and near categories are separated;
- how the report links from the main table area to appendix column
  definitions.

Implementation note:

- Final main-body section order: Executive Summary; Phase 2 Training-Regime
  Gate; Pooled Diagnostic Scope; Pooled Binary Support Failures; Pooled Amount
  And Hurdle Failures; Pooled Context Diagnostics; Remaining Failure Modes;
  Appendix.
- The only six-context table left in the main body is a compact gate table that
  joins the expected-value hurdle row from
  `monterey_big_sur_training_regime_primary_summary.csv` with the calibrated
  support row from `monterey_big_sur_binary_support_primary_summary.csv`.
- Full amount/model comparison rows, binary transfer rows, Big Sur same-region
  context, data-health tables, threshold/calibration/amount diagnostics,
  artifact paths, and full component-failure tables move to the appendix.
- `amount_under_rate` and `composition_shrink_rate` share the denominator of
  observed-positive rows where calibrated binary support is detected. This
  makes composition shrink dependent on the selected binary threshold.
- `positive_near_correct` uses all observed-positive rows as its denominator.
  Near/adjacent spatial classes use the retained 30 m grid: adjacent is the
  3x3 radius-1 neighborhood, and near is the 5x5 radius-2 ring outside the
  adjacent ring.
- FN isolated/edge/interior and FP predicted-isolated/predicted-edge/
  predicted-interior are presented as separate exhaustive topology
  denominators. FP adjacent/near-observed remains a separate observed-proximity
  denominator and is not added to predicted topology percentages.
- Main pooled binary and amount tables link to the appendix column definitions
  section.

## Required Report Structure

Use this main-body structure unless implementation reveals a clearer equivalent:

1. Executive Summary.
2. Phase 2 Training-Regime Gate.
   - One compact table comparing the six local/transfer/pooled contexts.
   - Short text explaining that Big Sur-only still calibrates Big Sur amount
     best, pooled support is conservative, and the rest of the report now uses
     pooled diagnostics as the forward-looking evaluation surface.
3. Pooled Diagnostic Scope.
   - State filters, target, pooled contexts, and model surfaces.
4. Pooled Binary Support Failures.
   - Use Task 56 binary context diagnostics.
   - Use Task 57 hex map.
   - Include FP and FN rates for isolated cells as well as edge/interior cells.
5. Pooled Amount And Hurdle Failures.
   - Split amount-underprediction from composition shrinkage.
   - Use the same denominator for amount-underprediction and composition
     shrinkage: observed-positive rows where binary support was detected.
   - Make clear that this denominator makes composition shrink depend on the
     binary threshold.
6. Pooled Context Diagnostics.
   - Observed canopy bins.
   - Quarterly persistence bins.
   - Fine CRM depth bins and elevation bins.
   - Prediction-distribution summaries for binary, ridge, and hurdle.
7. Remaining Failure Modes.
   - Use the refined context diagnostics instead of broad
     `retained_depth_0_60m` language.
8. Appendix.
   - Column definitions.
   - Artifact index.
   - Full tables and demoted context comparisons.

## Table And Column Definition Requirements

Move table and artifact definitions out of the main narrative and into the
appendix. Add a link near the main component-failure table that jumps to the
appendix definitions.

Define these columns explicitly:

- `amount_under_rate`: share of observed-positive, support-detected rows whose
  expected-value hurdle prediction is below the chosen tolerance.
- `composition_shrink_rate`: share of the same observed-positive,
  support-detected rows whose expected-value hurdle is materially shrunk by
  probability composition relative to conditional canopy.
- `positive_near_correct`: define the denominator explicitly. Default
  denominator should be all observed-positive rows unless the implementation
  keeps a stricter detected-positive denominator; the report must say which.
- `near`: define as the configured grid-neighborhood distance to an observed
  positive cell. Record the exact cell radius and meter distance used by the
  code.
- `fn_isolated`, `fn_edge`, `fn_interior`: exclusive categories that should add
  to 100% of FNs, excluding missing or unclassified rows if any.
- `fp_predicted_isolated`, `fp_predicted_edge`, `fp_predicted_interior`:
  exclusive predicted-topology categories that should add to 100% of FPs,
  excluding missing or unclassified rows if any.
- `fp_adjacent_observed`, `fp_near_observed`, `fp_far_from_observed`: separate
  observed-proximity categories. Do not present these as if they add to the
  predicted-topology categories.

This should fix the confusing current interpretation where FP isolated and
adjacent/near FP do not add to 100%. They are different breakdowns and should
be either separated or made explicitly exhaustive within their own denominator.

## Content To Remove Or Demote From Main Body

- Demote the oversized `Canopy Amount And Hurdle Calibration` table from the
  main body. Keep a compact summary or artifact link only.
- Demote train-mean and geographic-ridge same-region context from the main
  body. They can remain in appendix if still useful as historical reference.
- Do not keep long local/transfer/pooled comparison rows throughout every
  section. After the top gate, main analysis should focus on pooled-on-Big-Sur
  and pooled-on-Monterey.
- Do not use broad `retained_depth_0_60m` as the main failure-mode explanation
  when finer `crm_depth_m_bin` and `elevation_bin` are available.

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

Inspect the generated Markdown and HTML manually after regeneration.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Main diagnostic context: pooled Monterey+Big Sur evaluated separately on
  each region.
- Top gate context: six local/transfer/pooled evaluation rows.
- Primary year: 2022 held-out test rows.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: Kelpwatch-style annual max canopy and `annual_max_ge_10pct`.

## Acceptance Criteria

- The report starts with one compact six-context comparison and does not keep
  replaying local/transfer/pooled rows throughout the main body.
- The main body focuses on pooled binary, ridge, and hurdle diagnostics for Big
  Sur and Monterey.
- The binary diagnostic map is the 1 km hex summary from Task 57.
- Deep component-failure analysis is split into binary support versus
  amount/hurdle composition sections.
- FP and FN isolated/edge/interior percentages are either exhaustive within
  their denominator or clearly marked as separate proximity/topology
  diagnostics.
- Column definitions and artifact paths live in the appendix with a visible
  link near the relevant table.
- The report can be read as a Phase 2 decision artifact rather than a
  chronological artifact log.

## Outcome

Completed. The Phase 2 report renderer now uses this main-body order:
Executive Summary, Phase 2 Training-Regime Gate, Pooled Diagnostic Scope,
Pooled Binary Support Failures, Pooled Amount And Hurdle Failures, Pooled
Context Diagnostics, Remaining Failure Modes, and Appendix.

The main report keeps one six-row training-regime gate table. Full amount and
binary transfer comparisons, Big Sur same-region context, data-health checks,
threshold/calibration details, artifact links, and full component-failure tables
are in the appendix. The main pooled binary section includes the Task 57 1 km
hex map and separates FN topology, FP predicted topology, and FP observed
proximity. The main amount section uses the shared observed-positive,
support-detected denominator for amount underprediction and composition shrink.
The pooled context diagnostics section now uses a full metric-breakdown plot
instead of another wide table or a single strongest-row summary. It expands each
context family into bar charts by region and model surface: binary panels use
F1, while ridge and expected-value hurdle panels use RMSE.

Regenerated artifacts:

- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`
- `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_metric_breakdown.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_prediction_distribution.png`

Validation:

```bash
uv run ruff check src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/evaluation/component_failure.py tests/test_model_analysis.py
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

## Known Constraints / Non-Goals

- Do not retrain models.
- Do not change model thresholds, masks, sample quotas, labels, features, or
  model policy.
- Do not remove the six-context comparison entirely; keep it as the top gate.
- Do not turn appendix artifact links into the main narrative.
- Do not choose Phase 3 in this task; preserve the evidence for P2-14 closeout.
