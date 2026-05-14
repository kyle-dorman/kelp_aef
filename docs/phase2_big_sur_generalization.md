# Phase 2 Big Sur Generalization

Status: selected on 2026-05-14; planning only. P2-01 has a numbered task plan;
later implementation task plans are still pending.

## Goal

Test whether the closed Phase 1 Monterey annual-max policy generalizes to the
neighboring Big Sur region before choosing a broader Phase 3 direction.

Phase 2 should start with the Phase 1 model policy frozen, then compare that
transfer result against region-specific and pooled training regimes. The point
is to see both whether the Monterey-trained policy transfers and how much
performance changes when Big Sur is included in training.

The Phase 2 decision question is:

```text
How large is the Big Sur performance gap between Monterey-trained transfer,
Big Sur-only training, and pooled Monterey+Big Sur training?
```

## Selected Region

Phase 2 starts with Big Sur as the neighboring generalization region.

User-provided AlphaEarth STAC feature:

- STAC item id: `8957`
- Region shorthand: `big_sur`
- CRS: `EPSG:32610`
- Bounding box:
  `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`
- Example 2022 AEF asset:
  `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`

The working assumption is that AEF and Kelpwatch coverage exist for this
region. Domain-source coverage is likely but must be verified quickly before
any model interpretation.

## Phase Boundary

Phase 2 is intentionally shorter than Phase 1.

In scope:

- Add Big Sur as a second small smoke/generalization region.
- Verify Big Sur AEF, Kelpwatch, and domain-source coverage before interpreting
  model metrics.
- Do a quick visual QA pass on Big Sur labels, AEF coverage, and domain context
  early, before model training or metric interpretation.
- Reuse the Phase 1 annual-max label input, years, split policy, plausible-kelp
  mask logic, CRM-stratified mask-first sampling policy, and selected AEF hurdle
  policy unless the first verification step proves they are not valid for Big
  Sur.
- Compare at least three training regimes when evaluating on Big Sur:
  Monterey-trained transfer, Big Sur-only training, and pooled Monterey+Big Sur
  training.
- Use validation-driven calibration and threshold selection inside each
  training regime where needed, while keeping held-out test rows for final
  evaluation.
- Generate Big Sur model-analysis outputs and compare them against Monterey.
- Generate a Big Sur interactive visualizer output.
- Leave room for visualizer changes such as region and year selection.
- Leave room for report changes needed to show multi-region comparisons
  clearly.

Out of scope:

- Full West Coast scale-up.
- Alternative temporal label targets such as annual mean, fall-only,
  winter-only, or multi-season persistence.
- Deep spatial models or 10 m label experiments.
- Adding CRM bathymetry, elevation, depth bins, or mask reasons as predictors.
- Choosing final thresholds, quotas, or model policy by tuning on held-out Big
  Sur test rows.
- Treating Kelpwatch-style reproduction as independent field-truth biomass
  validation.

## Expected Workflow

The implementation tasks should stay small and report-visible, but they are not
spelled out here as numbered task files yet.

The expected sequence is:

1. Create or generalize configuration for a `big_sur` run without breaking the
   closed Monterey config.
2. Materialize reviewable Big Sur source manifests for AEF, Kelpwatch, and
   domain sources. Confirm the exact AEF years and Kelpwatch overlap window.
3. Run an early visual QA pass for Big Sur labels, AEF coverage, and domain
   context so obvious source or alignment problems are caught before training.
4. Verify domain-source coverage and build the Big Sur plausible-kelp retained
   domain before training or reporting model quality.
5. Run the Phase 1 pipeline on Big Sur using the same annual-max target and
   validation/test split semantics where coverage permits.
6. Add Big Sur-only and pooled Monterey+Big Sur training comparisons, then
   evaluate each training regime on the Big Sur held-out split.
7. Refresh the report so Monterey and Big Sur can be compared side by side on
   the same policy rows and training-regime labels.
8. Generate the Big Sur results visualizer and decide whether the visualizer
   should support selecting region and year in one interface or should write
   separate region-specific viewers for now.
9. Close Phase 2 with a short decision note that recommends a Phase 3 path.

## Evaluation Questions

The primary comparison rows should mirror Phase 1:

```text
split = test
year = 2022, unless source verification changes the overlap window
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

The report should answer:

- How does the frozen Monterey-trained policy perform when evaluated on Big
  Sur?
- Does Big Sur-only training improve Big Sur held-out performance?
- Does pooled Monterey+Big Sur training improve Big Sur held-out performance?
- Is there a meaningful gap between Monterey-trained transfer, Big Sur-only
  training, and pooled training?
- Does the expected-value hurdle improve over AEF ridge in each training
  regime?
- Does the expected-value hurdle compete with previous-year annual max in Big
  Sur?
- Does the calibrated binary gate still have useful support skill for
  `annual_max_ge_10pct`?
- Does conditional canopy still underpredict high-canopy annual-max rows?
- Does the plausible-kelp domain retain Kelpwatch-positive rows in Big Sur?
- Do the early Big Sur label, AEF, and domain visuals look spatially coherent?
- Are Big Sur residuals concentrated in the same domain contexts as Monterey,
  or does a new failure mode appear?
- Does the visualizer make those failures inspectable without manual table
  digging?

## Visualizer And Report Allowance

Phase 2 should reserve explicit room for visualizer and report updates.

The current visualizer is useful for local qualitative inspection. Big Sur will
likely force two practical design decisions:

- whether the visualizer should write one region-specific HTML artifact per
  config, or a multi-region viewer with region and year selectors;
- whether report tables and figures should remain single-region outputs or grow
  a compact region-comparison section.

These are Phase 2 support changes, not side quests. The core standard is that a
Big Sur rerun should be inspectable in both the generated report and the local
visualizer before Phase 2 closes.

## Phase 3 Decision Gate

Phase 2 should close by choosing one Phase 3 direction:

- If the Phase 1 AEF hurdle policy generalizes to Big Sur and remains
  competitive with previous-year persistence, Phase 3 should broaden to a
  multi-region California or broader West Coast generalization phase with
  explicit regional holdouts.
- If Big Sur-only or pooled training closes most of the gap, Phase 3 should
  broaden to a multi-region training/evaluation design before full scale-up.
- If binary support generalizes but high-canopy amount prediction remains the
  main failure, Phase 3 should choose between a simple non-linear tabular model
  pass, such as random forest or gradient boosting, and deferred temporal-label
  experiments such as annual mean, fall-only, winter-only, or persistence
  targets. A small model-capacity pass is reasonable before deep spatial models
  if the failure looks like underfit rather than target mismatch.
- If Big Sur fails because domain-source coverage, mask retention, or
  source-alignment assumptions break, Phase 3 should harden ingestion and
  domain-source coverage before more modeling.
- If model metrics are acceptable but the report or visualizer cannot support
  multi-region inspection, Phase 3 should harden evaluation and review tooling
  before expanding geography.

## Acceptance Criteria

Phase 2 is ready to close when:

- Big Sur source and domain coverage have been verified from artifacts, not
  assumed from the STAC feature alone.
- Big Sur labels, AEF coverage, and domain context have passed an early visual
  QA check.
- The Phase 1 selected policy has been fairly evaluated as Monterey-trained
  transfer on Big Sur.
- Big Sur-only and pooled Monterey+Big Sur training regimes have been evaluated
  on Big Sur held-out rows.
- The generated report includes Big Sur, Monterey, and training-regime
  comparison context.
- The Big Sur visualizer exists, or a multi-region visualizer clearly supports
  Big Sur selection.
- The closeout decision states whether Phase 3 should be scale-up,
  non-linear tabular modeling, temporal-label exploration, ingestion/domain
  hardening, or evaluation tooling.
