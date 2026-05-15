# Task 60: Refine Pooled Context Diagnostic Plots

## Goal

After Task 59 makes Phase 2 report iteration fast, refine the
`Pooled Context Diagnostics` section so the plot tells the combined pooled-data
story clearly instead of splitting Big Sur and Monterey into separate panels.

The current metric-breakdown plot is useful as a first pass, but it is too busy
and still carries design choices that make interpretation harder:

- Big Sur and Monterey are split even though this section should ask what is
  happening across the combined pooled evaluation dataset.
- Context values are not ordered consistently or meaningfully.
- The expected-value hurdle panels add clutter here; the hurdle is already
  discussed in the amount/hurdle failure section.
- Some context families may be redundant or less useful in this chart.

The new chart should help answer:

```text
Across pooled Monterey+Big Sur evaluation rows, which context bins explain
binary support skill and continuous amount error?
```

Frame results as Kelpwatch-style annual maximum reproduction diagnostics, not
independent field-truth biomass validation.

## Dependency

Do this after Task 59:

- `tasks/59_cache_phase2_diagnostics_for_fast_report_iteration.md`

This task is intentionally report/design iteration. It should not be started
until cached/reused Phase 2 diagnostics make repeated report runs fast enough.

## Inputs

- Config:
  - `configs/big_sur_smoke.yaml`
- Report-generation code:
  - `src/kelp_aef/evaluation/model_analysis.py`
- Pooled-context diagnostic code:
  - `src/kelp_aef/evaluation/pooled_context.py`
- Cached or generated pooled-context outputs:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_context_model_performance.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_binary_context_diagnostics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_amount_context_diagnostics.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_pooled_prediction_distribution_by_context.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_pooled_context_diagnostics_manifest.json`
- Current report:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- Current pooled context figure:
  - `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_metric_breakdown.png`

Primary report filters remain:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
training_regime = pooled_monterey_big_sur
```

## Outputs

Regenerate the Phase 2 model-analysis report and the pooled context diagnostic
figure:

- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`
- `/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_metric_breakdown.png`

If the revised plot becomes too dense for one image, write one configured
figure per retained context family, for example:

```text
/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_observed_max_bin.png
/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_mean_canopy_bin.png
/Volumes/x10pro/kelp_aef/reports/figures/monterey_big_sur_pooled_context_quarterly_persistence.png
```

Keep all figure paths explicit in `configs/big_sur_smoke.yaml` if new outputs
are added.

## Config File

Use `configs/big_sur_smoke.yaml`.

Do not hard-code report figure paths. Add or rename configured output keys if
the plot is split into multiple files.

## Plan / Spec Requirement

Before implementation, write a short design note in this task file or PR
description that confirms:

- which context families remain in the main chart;
- which context families are dropped or moved to appendix and why;
- the exact metric definitions;
- the y-axis order for every retained context family;
- how Big Sur and Monterey rows are combined;
- whether annual mean canopy is computed from existing quarterly/annual label
  fields or added to the pooled-context diagnostic builder;
- whether the output is one large figure or multiple figures.

## Required Plot Changes

### Combine Regions

Merge Big Sur and Monterey pooled evaluation rows for this section.

The main `Pooled Context Diagnostics` plot should no longer split into Big Sur
and Monterey panels. It should aggregate context rows across both evaluation
regions for the pooled Monterey+Big Sur training regime.

If regional differences are still useful for audit, keep them in CSVs or an
appendix figure, not the main chart.

### Model Surfaces

Use only:

- Binary support: F1 by context value.
- Continuous amount model: RMSE by context value.

Skip the expected-value hurdle panels in this context diagnostic chart. The
hurdle remains relevant in the earlier `Pooled Amount And Hurdle Failures`
section, but it makes this chart harder to read.

Use clear titles:

```text
Binary F1
Ridge RMSE
```

Avoid long model labels in panel titles.

### Context Families

Review the current context families before plotting. Candidate main-chart
families:

- Observed annual max canopy bin.
- Annual mean canopy bin.
- Quarterly persistence.
- Previous-year class.
- Fine CRM depth bin.
- Possibly elevation bin, only if it adds information beyond CRM depth.
- Component failure class, but with shorter labels.

Likely remove from the main plot:

- Binary outcome. It is mostly a confusion-matrix partition, not a useful
  explanatory context chart here.

Open question to resolve during implementation:

- Do we need both `elevation_bin` and fine `crm_depth_m_bin`? If they are
  mostly redundant, keep the one that better explains kelp-domain context and
  move the other to appendix or CSV-only audit. If both are retained, explain
  the distinct interpretation in the report text.

### Annual Mean Canopy Bin

Add an annual mean canopy context family in addition to annual max canopy.

This should give another view of persistence/intensity:

- annual max canopy shows peak canopy;
- quarterly persistence shows how many quarters were positive;
- annual mean canopy shows whether the positive signal is sustained in amount,
  not just present.

Prefer deriving this from existing label fields if available, such as
`kelp_mean_area_y` or equivalent quarterly-derived annual mean columns. If the
pooled-context input frame does not carry the needed column, add it in the
pooled-context diagnostic builder and include it in cached diagnostics from
Task 59.

### Observed Annual Max Bin Semantics

Fix the annual max bin boundary labels and semantics.

The current observed canopy bin text is confusing around the binary threshold.
The `90 m2` threshold is the first positive value for `annual_max_ge_10pct`.
The low-positive bin should be labelled so it is clear that:

```text
0-89 m2 is below the 10% threshold
>=90 m2 is in the binary-positive target
```

Do not label a bin as `1-90` if `90` is included in the positive class and the
rest of that bin is below threshold. Use explicit inclusive/exclusive labels,
for example:

```text
0
1-89
90-224
225-449
450-809
810-899
900
```

or an equivalent scheme that avoids ambiguity.

### Y-Axis Ordering

Define explicit order functions per context family.

Do not rely on alphabetical order for context values. Suggested orders:

- Observed annual max bin: numeric low to high.
- Annual mean canopy bin: numeric low to high.
- Quarterly persistence: no quarter, one-quarter spike, intermittent, persistent,
  assumed/background or missing last.
- Previous-year class: stable/background, previous low canopy, lost 10 pct,
  new 10 pct, persistent 10 pct, missing/unknown last.
- Fine CRM depth bin: shallow to deeper water, with not-subtidal/zero and
  missing placed consistently at the edges.
- Elevation bin if retained: land/ambiguous/subtidal ordered physically, not
  alphabetically.
- Component failure class: near-correct first or last, then support miss,
  support leakage, amount underprediction, amount overprediction,
  composition shrinkage, high-confidence wrong, other/missing.

Add tests around these order functions so the chart does not regress into
alphabetical sorting.

### Label Cleanup

Shorten labels before plotting, especially component failure labels.

Examples:

```text
amount under
amount over low/zero
composition shrink
support miss
support leakage
near correct
high-conf wrong
other
```

Avoid long y-axis labels that make the figure unreadable.

## Suggested Implementation

- Add a combined-region aggregation mode for pooled context plotting in
  `model_analysis.py`.
- Prefer using existing pooled-context CSV rows where possible.
- If combined-region rows are not available, add combined rows in
  `pooled_context.py` or combine raw cached frames from Task 59 before
  aggregation.
- If adding annual mean canopy bins, add them to the pooled-context row-level
  diagnostic frame before context aggregation.
- Replace the current six-column region/model grid with a simpler layout:
  - one row per context family;
  - two panels per family: `Binary F1` and `Ridge RMSE`;
  - all rows combined across Big Sur and Monterey.
- If one figure is too tall, split by context family and embed several figures
  in the report.
- Update the report text so it says the section uses combined pooled evaluation
  rows, not region-separated panels.

## Validation Command

Use the cached report path from Task 59 once available. Expected validation:

```bash
uv run ruff check src/kelp_aef/evaluation/model_analysis.py src/kelp_aef/evaluation/pooled_context.py tests/test_model_analysis.py
uv run mypy src
uv run pytest tests/test_model_analysis.py
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml --reuse-phase2-diagnostics
git diff --check
```

If the Task 59 CLI shape differs, update the command here before starting.

Manually inspect:

- The Markdown `Pooled Context Diagnostics` section.
- The HTML rendering.
- The pooled context figure(s).

## Smoke-Test Region And Years

- Combined regions: Big Sur plus Monterey.
- Training regime: pooled Monterey+Big Sur.
- Primary year: 2022 held-out test rows.
- Evaluation scope: retained plausible-kelp-domain full grid.
- Target: Kelpwatch-style annual max canopy and `annual_max_ge_10pct`.

## Acceptance Criteria

- The main `Pooled Context Diagnostics` section combines Big Sur and Monterey
  pooled evaluation rows instead of splitting them into regional panels.
- The main context chart excludes expected-value hurdle panels.
- Binary context panels use F1.
- Continuous amount context panels use ridge RMSE.
- Context y-axis values use explicit, meaningful orders rather than alphabetical
  order.
- Observed annual max bin labels correctly handle the `90 m2` threshold.
- Annual mean canopy bin is added as a context family or a clear reason is
  documented if it cannot be derived from current inputs.
- Binary outcome is removed from the main chart unless implementation finds a
  stronger reason to keep it and documents that reason.
- Fine CRM depth and elevation are reviewed for redundancy; the report keeps
  only the useful main-chart view or clearly explains why both remain.
- Component failure labels are shortened enough to read in the figure.
- Report prose explains the combined pooled context chart without overstating
  ecological truth beyond Kelpwatch-style label reproduction.

## Known Constraints / Non-Goals

- Do not start this before Task 59 fast report iteration is available.
- Do not retrain models.
- Do not change model thresholds, masks, sample quotas, labels, features, or
  model policy.
- Do not remove detailed CSV diagnostics.
- Do not choose Phase 3 in this task.
- Do not optimize unrelated report sections unless they block this chart.
