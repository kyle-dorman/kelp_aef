# Task 56: Build Pooled Phase 2 Context Diagnostics

## Goal

Build report-visible diagnostics that make the pooled Monterey+Big Sur model
useful as the forward-looking Phase 2 evaluation surface.

The local/transfer/pooled comparison is now mostly a top-level gate. The deeper
analysis should focus on pooled evaluation on each region because that is the
workflow most likely to remain useful as the dataset grows and as individual
failure modes are fixed.

The main question is:

```text
For pooled Monterey+Big Sur models, which failures belong to binary support,
amount prediction, hurdle composition, observed-canopy regime, temporal
persistence, depth/elevation context, or prediction-distribution behavior?
```

Frame results as Kelpwatch-style annual maximum reproduction, not independent
field-truth biomass validation.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml`
- Pooled model prediction artifacts:
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_baseline_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_hurdle_full_grid_predictions.parquet`
- Current Phase 2 comparison and component-failure outputs:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_summary.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_component_failure_by_domain_context.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_temporal_label_context.csv`
- Existing full-grid label, CRM/domain, split, quarterly/persistence, and
  retained-domain columns already available to `analyze-model`.

Primary filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
training_regime = pooled_monterey_big_sur
evaluation_region in {big_sur, monterey}
```

## Outputs

Write report-visible sidecar tables under
`/Volumes/x10pro/kelp_aef/reports/tables/`, with paths recorded in the Phase 2
model-analysis manifest. Expected outputs:

- `monterey_big_sur_pooled_context_model_performance.csv`
- `monterey_big_sur_pooled_binary_context_diagnostics.csv`
- `monterey_big_sur_pooled_amount_context_diagnostics.csv`
- `monterey_big_sur_pooled_prediction_distribution_by_context.csv`
- `monterey_big_sur_pooled_context_diagnostics_manifest.json` under
  `/Volumes/x10pro/kelp_aef/interim/`

The exact file names can vary if they fit the existing report-output naming
better, but the outputs must be distinct from the six-context comparison
tables.

## Config File

Use `configs/big_sur_smoke.yaml` as the Phase 2 coordinating config. If new
paths are needed, add them under the existing report or
`training_regime_comparison` output block. Do not hard-code artifact paths in
implementation code.

## Plan / Spec Requirement

Before implementation, write a short implementation note in this task file or
the PR/commit message that confirms:

- which `analyze-model` report pass owns the pooled diagnostics;
- the exact pooled prediction files loaded;
- the context bins used for observed canopy, quarterly persistence, depth, and
  elevation;
- the shared denominator for amount-underprediction and composition-shrinkage
  rates;
- how binary, ridge, and hurdle metrics are aligned into one comparison;
- how prediction distributions are summarized without changing model policy.

## Required Analysis

Use the three primary model surfaces:

- `binary`: calibrated binary probability and validation-selected binary class
  for `annual_max_ge_10pct`;
- `ridge`: AEF ridge predicted canopy amount;
- `hurdle`: expected-value hurdle prediction from calibrated binary
  probability times conditional canopy amount.

For the binary model, report by context bin:

- row count;
- observed positive count and observed positive rate;
- predicted positive count and predicted positive rate;
- precision, recall, F1;
- false-positive count and rate;
- false-negative count and rate;
- mean and quantiles of calibrated predicted probability;
- probability margin from the selected validation threshold.

For ridge and hurdle amount surfaces, report by context bin:

- row count;
- observed mean and predicted mean;
- observed total area and predicted total area;
- area bias and area percent bias;
- mean residual and median residual;
- MAE and RMSE;
- prediction quantiles, including p50, p90, p95, and p99;
- clipping/saturation counts if predictions are clipped to the canopy range.

For component-failure rates, use directly comparable denominators:

- `amount_under_rate`: among observed-positive rows where binary support was
  detected, the share whose expected-value hurdle amount is too low;
- `composition_shrink_rate`: among the same observed-positive, support-detected
  rows, the share where probability composition materially shrinks a substantial
  conditional prediction;
- this intentionally makes composition shrink depend on the binary threshold so
  it can be compared directly to amount-underprediction.

Break every model surface down by:

- observed annual max bins: zero, `(1, 90]`, `(90, 225]`, `(225, 450]`,
  `(450, 810]`, `(810, 900]`;
- quarterly persistence or temporal-label class when available:
  `assumed_background`, no-quarter positive, one-quarter spike, intermittent,
  persistent, and any existing station-derived persistence labels;
- fine CRM depth bins, preferring `crm_depth_m_bin` such as `(0, 10m]`,
  `(10, 20m]`, `(20, 40m]`, `(40, 60m]` over broad `retained_depth_0_60m`;
- `elevation_bin`;
- binary-support outcome and component-failure class from Task 53.

Prediction-distribution diagnostics must answer:

- Are pooled hurdle predictions compressed relative to observed high-canopy
  rows?
- Does ridge or hurdle place too much probability/mass in low positive bins?
- Do high observed-canopy rows have a prediction distribution that supports a
  log or transformed-target follow-up?
- Are Big Sur and Monterey failures similar under the same pooled model, or are
  the distributions region-specific?

## Validation Command

Focused validation should include:

```bash
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

If new helper functions are added for pooled context bins or prediction
distribution summaries, include small synthetic tests for denominator handling,
empty bins, and binary/amount metric alignment.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Model context: pooled Monterey+Big Sur evaluated separately on each region.
- Primary year: 2022 held-out test rows.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: Kelpwatch-style annual max canopy and `annual_max_ge_10pct`.

## Acceptance Criteria

- The new tables contain pooled-on-Monterey and pooled-on-Big-Sur rows with
  explicit `evaluation_region`, `training_regime`, and model-surface columns.
- Binary, ridge, and hurdle performance can be compared across the same context
  bins.
- Amount-underprediction and composition-shrinkage rates use the same
  denominator and the definition is written into the report metadata or
  appendix.
- Fine depth/elevation bins are available for report interpretation; broad
  retained-depth mask reasons are not the only domain context.
- Prediction-distribution diagnostics give P2-12 enough evidence to decide
  whether transformed positive-canopy targets are worth implementing or
  whether the problem points more directly at model capacity.
- Held-out test rows are diagnostic only. No thresholds, sample quotas, masks,
  labels, features, or model policy are tuned from these diagnostics.

## Known Constraints / Non-Goals

- Do not retrain models in this task.
- Do not change the Phase 2 annual-max label target.
- Do not add bathymetry, elevation, temporal persistence, or edge context as
  predictors.
- Do not collapse Monterey and Big Sur pooled rows into one unlabeled aggregate.
- Do not overwrite the existing six-context comparison tables.
- Do not move or rewrite the whole report in this task; Task 58 owns the report
  structure.
