# Phase 1 Model And Domain Hardening

Status: closed on 2026-05-13.

## Goal

Harden the Monterey annual-max pipeline before any larger scale-up. Phase 1
answered four questions:

- Do AlphaEarth embeddings beat persistence, site memory, and geography?
- Does a physically plausible kelp-domain mask reduce full-grid false positives
  without dropping real Kelpwatch-supported canopy?
- Can an imbalance-aware model reduce background leakage without shrinking high
  canopy rows toward the mean?
- Can the report show each change clearly enough that every rerun is
  interpretable?

## Inputs

- Config: `configs/monterey_smoke.yaml`
- Region: Monterey Peninsula.
- Years: 2018-2022.
- Split: train 2018-2020, validate 2021, test 2022.
- Label input: Kelpwatch annual max canopy, `kelp_max_y` /
  `kelp_fraction_y`.
- Features: current AlphaEarth annual 64-band embeddings aggregated to the
  Kelpwatch 30 m grid.
- Domain-filter support inputs: bathymetry and DEM data for Monterey, stored
  under `/Volumes/x10pro/kelp_aef`.

## Closeout Outcome

The final Phase 1 model-policy decision is tracked in
`docs/phase1_closeout_model_policy_decision.md`.

Closed policy:

- Default data policy: `crm_stratified_mask_first_sample` inside
  `plausible_kelp_domain`.
- Selected AEF model policy: expected-value hurdle,
  `calibrated_probability_x_conditional_canopy`.
- Diagnostic support policy: hard-gated hurdle,
  `calibrated_hard_gate_conditional_canopy`.
- Final report snapshot:
  `docs/report_snapshots/monterey_phase1_closeout_model_analysis.md`.

Phase 1 established a reproducible AlphaEarth/Kelpwatch annual-max pipeline,
reference baselines, a retained-domain mask, calibrated binary support,
positive-only conditional canopy, hurdle composition, and a rerunnable report
harness. It did not establish independent biomass truth or solve high-canopy
amount underprediction.

## Outputs

Phase 1 kept the same artifact style as Phase 0:

- Explicit manifests for new domain-filter inputs and masks.
- Model prediction artifacts for each reference baseline and embedding model.
- Report tables that compare pixel skill, threshold skill, and area
  calibration.
- Residual maps and residual taxonomy tables that distinguish masked and
  unmasked behavior.
- A model-analysis report that can be rerun after each task to show the effect
  of the latest change.

Exact active file paths are declared in `configs/monterey_smoke.yaml`.

## In Scope

- Previous-year annual-max baseline.
- Station or grid-cell climatology baseline.
- Lat/lon/year-only geographic baseline.
- Bathymetry/DEM domain filtering and mask-aware diagnostics.
- Binary models derived from thresholding annual max.
- Hurdle or two-stage annual-max modeling.
- Historical comparison against capped-weight and stratified-background
  continuous models already completed as negative P1-22 tests.
- Validation-year calibration for thresholds, probabilities, and area totals.
- Report improvements that make each pipeline rerun interpretable.

## Out Of Scope

- Alternative temporal label inputs beyond annual max.
- Fall-only, winter-only, annual mean, or multi-season persistence targets.
- Full West Coast scale-up.
- Deep spatial models.
- New spectral products beyond the current AEF embeddings.
- Treating Kelpwatch-style label reproduction as independent biomass truth.
- Tuning model choices on the 2022 test split.

Bathymetry and DEM are allowed in Phase 1 as domain-filter and diagnostic
inputs. They should not become model predictors unless a later decision
explicitly changes that scope.

## Validation Loop

The closed Phase 1 pipeline preserves the ability to rerun the relevant stages
from the CLI. The default full loop is:

```bash
make check
uv run kelp-aef train-baselines --config configs/monterey_smoke.yaml
uv run kelp-aef predict-full-grid --config configs/monterey_smoke.yaml
uv run kelp-aef map-residuals --config configs/monterey_smoke.yaml
uv run kelp-aef analyze-model --config configs/monterey_smoke.yaml
```

Tasks that only touch one stage may run the relevant subset, but model, mask,
or calibration changes should end with an updated model-analysis report.

## Acceptance Criteria

Phase 1 closed after satisfying:

- Reference baselines are implemented and compared against AEF models.
- The report clearly separates pixel skill from full-grid area calibration.
- The domain mask is documented, aligned to the target grid, and evaluated both
  before and after retraining.
- At least one imbalance-aware model is evaluated against ridge, persistence,
  climatology, and geography.
- The selected expected-value hurdle policy beats the one-stage AEF ridge
  baseline and edges the previous-year reference on retained-domain RMSE and
  10% annual-max F1, while preserving its limitations in the report.
- The final Phase 1 run remains reproducible from config and CLI commands.

## Task Outline

The active task checklist lives in `docs/todo.md`. Keep each implementation
slice small enough that it can rerun the relevant pipeline and update the report
before moving to the next task.
