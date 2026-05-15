# Task 53: Deep Model-Component Failure Analysis

## Goal

Before closing Phase 2, break down where the current model chain is failing and
which component is responsible.

This task should push beyond one aggregate local-vs-pooled metric table. It
should analyze binary support, calibrated probability, conditional canopy
amount, expected-value hurdle composition, and hard-gated hurdle output against
every secondary pixel context we already have or can compute from existing
artifacts.

The main question is:

```text
Are Phase 2 failures mostly binary support misses, binary support leakage,
conditional amount shrinkage, hurdle composition effects, kelp-mat edge
effects, bathymetry/domain effects, temporal-label effects, or training-regime
effects?
```

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- Current report and comparison artifacts:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`
- Local model predictions:
  - `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_full_grid_predictions.parquet`
- Pooled model predictions:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
- Existing full-grid tables, masks, CRM/domain columns, split manifests, and
  Kelpwatch label artifacts declared in the Monterey and Big Sur configs.
- Existing quarterly or annual label QA artifacts if they are already available
  locally. Do not download new label products in this task.
- Existing visualizer inspection CSVs:
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_pooled_monterey_big_sur_results_visualizer_inspection_points.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_pooled_monterey_big_sur_results_visualizer_inspection_points.csv`

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

Write report-visible sidecar tables under `/Volumes/x10pro/kelp_aef/reports/tables/`.
Exact names can vary, but the expected outputs are:

- `monterey_big_sur_component_failure_summary.csv`
- `monterey_big_sur_component_failure_by_label_context.csv`
- `monterey_big_sur_component_failure_by_domain_context.csv`
- `monterey_big_sur_component_failure_by_spatial_context.csv`
- `monterey_big_sur_component_failure_by_model_context.csv`
- `monterey_big_sur_edge_effect_diagnostics.csv`
- `monterey_big_sur_temporal_label_context.csv`, if quarterly/seasonal label
  context can be derived from existing artifacts without new downloads.
- Updated Phase 2 report Markdown/HTML/PDF from `analyze-model`, or a compact
  decision-note section if a separate report helper is cleaner.
- Manifest recording input prediction paths, context definitions, thresholds,
  primary filters, and output paths.

Optional figures if they make the report clearer:

- stacked bar chart of failure classes by evaluation region and training
  regime;
- component residual heatmap by observed canopy bin and edge/interior class;
- FP/FN edge-distance histograms;
- depth/domain-context failure chart.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. Preserve
`configs/monterey_smoke.yaml` as the Monterey-local source of paths.

If new paths are needed, add them under a report or diagnostics block in the
Big Sur config. Do not hard-code output paths in implementation code.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- which command owns the diagnostic output, such as extending `analyze-model`
  versus adding a separate `analyze-component-failures` command;
- the exact prediction contexts included;
- how binary, conditional, expected-value hurdle, and hard-gated hurdle
  failures are defined;
- which secondary pixel contexts are computed from existing data;
- how the edge/interior metrics are computed;
- how held-out visual QA and test rows are handled without tuning thresholds or
  selecting a new policy.

## Implementation Note

- Command ownership: extend `kelp-aef analyze-model --config
  configs/big_sur_smoke.yaml` rather than adding a separate command. Phase 2
  report synthesis, training-regime comparison, and binary-support comparison
  already live there, so component-failure diagnostics should be regenerated
  with the same report pass.
- Prediction contexts: include the four required contexts (`big_sur_only` on
  Big Sur, `monterey_only` on Monterey, pooled on Big Sur, pooled on Monterey)
  and the two reciprocal transfer contexts as supporting rows. Every output row
  must carry `evaluation_region`, `training_regime`, and `model_origin_region`.
- Failure definitions: use the validation-selected calibrated binary threshold
  stored in each hurdle artifact (`pred_presence_class` and
  `presence_probability_threshold`) for binary TP/FP/FN/TN, use
  `pred_conditional_area_m2` for conditional amount, use
  `pred_expected_value_area_m2` for expected-value hurdle residuals, and use
  `pred_hard_gate_area_m2` for hard-gated residuals. Use a fixed `90 m2`
  one-cell canopy tolerance for near-correct and large residual bins; do not
  retune thresholds from held-out rows.
- Secondary contexts: compute label bins, `annual_max_ge_10pct`,
  `annual_max_ge_50pct`, near-saturated status, label source, previous-year
  annual max class from existing prediction artifacts, retained domain reason,
  detail, CRM depth/elevation bins, and quarterly persistence context from the
  already-built annual label parquet where station-level quarterly fields are
  available.
- Edge/interior metrics: compute 3x3 and 5x5 retained-grid neighborhood
  positive counts/fractions, nearest observed/predicted positive distance in
  30 m grid cells/meters, observed connected-component id/area for
  `annual_max_ge_10pct` cells, and positive-edge/exterior-ring classes from
  existing `aef_grid_row`/`aef_grid_col` coordinates.
- Held-out discipline: primary diagnostics are restricted to
  `split = test`, `year = 2022`, `evaluation_scope = full_grid_masked`,
  `label_source = all`, and `mask_status = plausible_kelp_domain`. These rows
  are diagnostic only and cannot change thresholds, sample quotas, masks,
  labels, features, or the selected model policy.

## Required Analysis

Analyze all of these model contexts at minimum:

| Context | Evaluation region | Training regime | Required |
| --- | --- | --- | --- |
| Monterey local | `monterey` | `monterey_only` | yes |
| Big Sur local | `big_sur` | `big_sur_only` | yes |
| Pooled on Monterey | `monterey` | `pooled_monterey_big_sur` | yes |
| Pooled on Big Sur | `big_sur` | `pooled_monterey_big_sur` | yes |
| Monterey transfer on Big Sur | `big_sur` | `monterey_only` | optional but useful |
| Big Sur transfer on Monterey | `monterey` | `big_sur_only` | optional but useful |

For each required context, classify every retained-domain 2022 row into model
component failure fields:

- `binary_outcome`: TP, FP, FN, TN for `annual_max_ge_10pct`.
- `binary_probability_bin`: calibrated probability bins such as
  `[0, .1)`, `[.1, .25)`, `[.25, .5)`, `[.5, .75)`, `[.75, .9)`, `[.9, 1]`.
- `binary_threshold_margin`: calibrated probability minus the selected
  threshold, binned around the decision boundary.
- `conditional_prediction_bin`: predicted conditional canopy amount bins.
- `expected_value_residual_bin`: observed minus expected-value prediction.
- `hard_gate_residual_bin`: observed minus hard-gated prediction.
- `component_failure_class`, with at least:
  - `support_miss_positive`: observed positive but binary class is negative.
  - `support_leakage_zero`: observed zero but binary class is positive.
  - `amount_underprediction_detected_positive`: observed positive, binary
    support is positive, and expected-value hurdle amount is too low.
  - `amount_overprediction_low_or_zero`: observed low or zero and conditional
    or hurdle amount is too high.
  - `high_confidence_wrong`: probability near 0 or 1 but observed class
    disagrees.
  - `composition_shrinkage`: observed positive and binary support is positive,
    but expected-value composition materially shrinks a substantial conditional
    canopy prediction.
  - `near_correct`: residual is inside a declared tolerance such as one
    Kelpwatch canopy bin or 90 m2.

Break those failure classes down by label-derived context:

- observed annual max bins: zero, `(1, 90]`, `(90, 225]`, `(225, 450]`,
  `(450, 810]`, `(810, 900]`;
- binary target status for `annual_max_ge_10pct`;
- high-canopy status for `annual_max_ge_50pct`;
- near-saturated status for `kelp_max_y >= 810 m2`;
- `label_source`, especially `kelpwatch_station` versus `assumed_background`;
- previous-year annual max or persistence class if already available in the
  prediction/report inputs;
- annual label persistence or seasonality if quarterly labels are already
  available locally:
  - number of positive quarters;
  - annual mean canopy area;
  - annual max divided by annual mean;
  - one-quarter spike versus persistent canopy;
  - first/last positive quarter if easy to derive.

Break failures down by bathymetry and domain context:

- `domain_mask_reason`;
- `domain_mask_detail`;
- `depth_bin`;
- continuous `crm_depth_m` bins, including shallow, mid-depth, and 40-60 m
  bins;
- `crm_elevation_m` and `elevation_bin`;
- ambiguous coast versus subtidal ocean;
- distance to shoreline if existing CUSP/shoreline artifacts make it easy and
  deterministic without new downloads;
- mask-retained context only. Off-domain leakage should stay a separate audit.

Compute spatial context from grid coordinates:

- 3x3 and 5x5 local observed-positive count and fraction;
- 3x3 and 5x5 local predicted-positive count and fraction;
- distance in grid cells and meters to nearest observed positive cell;
- distance in grid cells and meters to nearest predicted positive cell;
- observed connected-component id and component area for positive kelp mats;
- distance-to-observed-component-edge for positive cells;
- exterior ring class for observed-zero cells adjacent to observed positives;
- edge class:
  - `positive_interior`;
  - `positive_edge`;
  - `isolated_positive`;
  - `zero_adjacent_to_positive`;
  - `near_positive_exterior`;
  - `far_zero_exterior`.

Use the spatial context to test the AEF blur or boundary-mismatch hypothesis:

- Are FPs mostly observed-zero cells adjacent to Kelpwatch positives?
- Are FNs mostly positive edge cells rather than positive interiors?
- Are high-confidence FPs outside mat edges paired with high-confidence FNs
  inside the same mat edge?
- Do edge/interior patterns differ between Big Sur local and pooled models?
- Do edge FPs/FNs cluster in specific depth or domain-mask contexts?
- Does the expected-value hurdle shrink edge positives more than interior
  positives?

Compare model components directly:

- binary support versus conditional amount: among observed positives, separate
  rows missed by support from rows detected by support but underpredicted by
  amount;
- calibrated probability versus hard gate: compare expected-value residuals to
  hard-gated residuals by probability margin;
- local versus pooled: determine whether pooled changes support behavior,
  amount behavior, or both;
- Monterey versus Big Sur: determine whether failures are shared across regions
  or concentrated in one region's bathymetry/domain/edge structure.

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

If a new command is added, include focused tests for the command and run it
against `configs/big_sur_smoke.yaml`.

If shared spatial-neighborhood helpers are added, include small synthetic tests
for edge/interior classes and nearest-positive distances.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Primary year: 2022 held-out test rows.
- Secondary context years: 2018-2022 only if needed for persistence or
  quarterly/seasonality context from existing local label artifacts.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: Kelpwatch-style annual max canopy and `annual_max_ge_10pct`.

## Acceptance Criteria

- The generated diagnostics cover Big Sur local, Monterey local,
  pooled-on-Big-Sur, and pooled-on-Monterey contexts.
- Binary support failures are separated from conditional amount failures and
  hurdle composition effects.
- Failure tables are broken down by label context, bathymetry/domain context,
  and computed spatial edge/interior context.
- The report or decision note explicitly states whether visual edge errors look
  consistent with boundary/blur effects, temporal annual-max mismatch, domain
  context, model underfit, or training-regime differences.
- Held-out test rows are used for diagnosis only. No thresholds, sample quotas,
  masks, labels, or model policy are tuned from these diagnostics.
- The output gives P2-14 closeout enough evidence to choose between regional
  expansion, temporal-label experiments, simple non-linear tabular modeling,
  ingestion/domain hardening, or evaluation-tooling work.

## Outcome

Completed via `kelp-aef analyze-model --config configs/big_sur_smoke.yaml`.

The diagnostic pass writes six train/evaluate contexts: the four required
local/pooled contexts plus reciprocal transfer sidecars. It records summary,
label, domain, spatial, model-context, edge-effect, temporal-label, and manifest
artifacts under `/Volumes/x10pro/kelp_aef/reports/tables/` and
`/Volumes/x10pro/kelp_aef/interim/`.

Headline results on the required 2022 retained-domain contexts:

- Big Sur local has `1,957` binary FNs, `1,460` binary FPs, `3,369`
  detected-positive amount-underprediction rows (`32.8%` of detected observed
  positives), `1,564` composition-shrinkage rows (`15.2%` of the same
  denominator), and `-2.84%` expected-value area bias.
- Pooled-on-Big-Sur has slightly fewer FPs (`1,278`) but more binary misses
  (`2,152`) and many more detected-positive amount-underprediction rows
  (`5,897`, `58.6%`), matching the weaker pooled Big Sur area calibration.
- Monterey local has `1,470` binary FNs, `954` binary FPs, and `4,267`
  detected-positive amount-underprediction rows; pooled-on-Monterey reduces FPs
  to `643` but increases FNs to `2,008`.
- Edge context is strong but now split: Big Sur local FNs are `6.1%` isolated
  positives and `91.3%` positive-edge cells; pooled-on-Big-Sur FNs are `5.4%`
  isolated positives and `92.8%` positive-edge cells. Big Sur local FPs are
  only `1.6%` isolated predicted positives, while `90.0%` are adjacent to or
  near observed positives.
- Depth/domain context concentrates active errors in retained `0_40m` cells,
  especially `(10, 20m]` and `(0, 10m]`; retained `40_60m` cells contribute
  many rows but essentially no binary FP/FN events in the primary contexts.
- Temporal context shows the largest positive amount residuals on persistent or
  intermittent Kelpwatch station rows, not on one-quarter spikes alone.

Interpretation: Phase 2 failures are not mostly off-domain leakage or a simple
binary-support collapse. The bigger unresolved issue is high/positive canopy
amount shrinkage after support is detected, with a clear kelp-mat edge/boundary
component and region/training-regime differences. Held-out 2022 diagnostics were
used only for failure analysis; no thresholds, sample quotas, masks, labels,
features, or model policy were tuned.

## Known Constraints / Non-Goals

- Do not download new source data.
- Do not switch the Phase 2 label target away from annual max in this task.
- Do not add bathymetry, edge-distance, component id, or label-persistence
  fields as model predictors.
- Do not select a new default model policy from held-out 2022 diagnostics.
- Do not collapse local and pooled contexts into one unlabeled aggregate.
- Keep this diagnostic report compact enough to support the closeout decision;
  detailed tables can live in the artifact index.
