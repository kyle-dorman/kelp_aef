# Task 51: Update Phase 2 Model-Analysis Report

## Goal

Update the generated model-analysis report so Phase 2 outcomes are visible as a
compact Monterey-vs-Big-Sur and training-regime comparison, without turning the
report into a chronological task log.

The report should answer the P2-09 question directly:

```text
Does Big Sur-only or pooled Monterey+Big Sur training improve Big Sur
performance, and is the answer different for binary support versus canopy
amount?
```

Frame all results as Kelpwatch-style annual maximum reproduction, not
independent field-truth biomass validation.

## Inputs

- Configs:
  - `configs/big_sur_smoke.yaml`
  - `configs/monterey_smoke.yaml` only as a source for existing Monterey
    comparison artifacts and labels.
- Current Phase 2 comparison artifacts:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_model_comparison.csv`
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - `/Volumes/x10pro/kelp_aef/interim/monterey_big_sur_training_regime_comparison_manifest.json`
- Binary support artifacts needed for the compact binary-transfer table:
  - `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_big_sur_transfer_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/big_sur_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - `/Volumes/x10pro/kelp_aef/processed/monterey_pooled_monterey_big_sur_binary_presence_full_grid_predictions.parquet`
  - The matching binary calibration model payloads and recommended thresholds
    declared by the Monterey-only, Big Sur-only, and pooled configs.
- Existing same-region report inputs produced by `analyze-model`, including
  retained-domain model comparison, binary threshold, class-balance, residual,
  and data-health tables.
- Task outcome notes:
  - `tasks/48_evaluate_monterey_transfer_on_big_sur.md`
  - `tasks/49_train_evaluate_big_sur_only_models.md`
  - `tasks/50_train_evaluate_pooled_monterey_big_sur_models.md`

Primary comparison filters:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
mask_status = plausible_kelp_domain
```

## Outputs

- Updated Phase 2 report artifacts from `configs/big_sur_smoke.yaml`:
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
  - `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`
- Report-visible Phase 2 comparison tables, either existing or newly emitted:
  - `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`
  - Optional, if needed for reproducibility:
    `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`
- Optional compact Phase 2 figures if they improve the report without adding
  chronology:
  - Training-regime area calibration by evaluation region.
  - Binary precision/recall/F1 by evaluation region and training regime.
- Updated `docs/todo.md` outcome for P2-09 after the report is regenerated.

Do not overwrite the closed Monterey Phase 1 report snapshot. Do not create a
Phase 2 closeout snapshot here unless the implementation discovers that P2-09
is also being used as the Phase 2 closeout; P2-11 owns closeout.

## Config File

Use `configs/big_sur_smoke.yaml` as the primary report config because it already
declares Phase 2 report output paths and the cross-regime comparison inputs.

If new report inputs are needed, add path-explicit entries under
`reports.outputs` or `training_regime_comparison` rather than hard-coding
artifact paths in the report builder.

## Plan / Spec Requirement

Before editing implementation code, write a short implementation note in this
task or the PR/commit message that confirms:

- The exact report sections being added or replaced.
- The source table for each headline metric.
- Whether binary support comparison is read from an emitted table or computed
  from binary full-grid predictions during `analyze-model`.
- The primary filters used for all headline rows.
- Whether any existing Phase 1 report sections are suppressed, renamed, or
  retained as appendix material in the Phase 2 report.

## Implementation Plan

## Implementation Note

This implementation keeps the existing Monterey Phase 1 report stable and adds
Phase 2 sections only when `training_regime_comparison` is configured, which is
the case for `configs/big_sur_smoke.yaml`.

- Added/replaced report sections: the Big Sur report title and executive
  summary become Phase 2-specific; a compact `Phase 2 Training-Regime Answer`
  section leads the report; the existing same-region Big Sur diagnostics remain
  as appendix/context material rather than a task chronology.
- Headline amount-calibration metrics come from
  `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_training_regime_primary_summary.csv`.
- Headline binary-support metrics are computed during `analyze-model` from the
  six configured full-grid binary prediction parquet paths and matching
  calibration payloads, then emitted to
  `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`.
- Primary filters for headline rows are `split = test`, `year = 2022`,
  `mask_status = plausible_kelp_domain`, `evaluation_scope =
  full_grid_masked`, and `label_source = all`.
- Existing Phase 1 sections are not suppressed for Monterey. In the Phase 2 Big
  Sur report, same-region scoreboards and residual diagnostics are retained as
  context/appendix material, while Phase 1 closeout language is not reused as
  the report lead.

1. Inspect the current Big Sur report generated by:

   ```bash
   uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
   ```

   Identify which existing Phase 1-style sections are still useful and which
   need Phase 2-specific wording.

2. Extend `src/kelp_aef/evaluation/model_analysis.py` in the smallest
   reasonable way.

   Prefer adding Phase 2 helpers and sections over rewriting the entire report
   harness. Keep the existing Phase 1 report path stable for
   `configs/monterey_smoke.yaml`.

3. Add a compact Phase 2 executive summary.

   It should lead with:

   - Big Sur-only training materially improves Big Sur canopy amount
     calibration over Monterey-only transfer.
   - Pooled Monterey+Big Sur training does not beat Big Sur-only for Big Sur
     amount calibration.
   - Binary support transfers relatively well compared with canopy amount.
   - Pooled binary support is more conservative: higher precision, lower
     recall, and lower predicted-positive rate than local models.

4. Add a training-regime comparison section sourced from
   `monterey_big_sur_training_regime_primary_summary.csv`.

   Show only the compact primary rows needed for interpretation:

   - Evaluation region.
   - Training regime.
   - Model name or composition policy.
   - F1 at `kelp_fraction_y >= 0.10`.
   - RMSE.
   - R2.
   - Area percent bias.

   At minimum include:

   - `ridge_regression`
   - `calibrated_probability_x_conditional_canopy`
   - `calibrated_hard_gate_conditional_canopy`

5. Add a binary support transfer section.

   Emit or compute a six-cell table for 2022 retained-domain full-grid binary
   support:

   | Evaluation region | Training regime | Binary F1 | Precision | Recall | AUPRC |
   | --- | --- | ---: | ---: | ---: | ---: |
   | Big Sur | Monterey-only transfer | `0.846453` | `0.829083` | `0.864566` | `0.897458` |
   | Big Sur | Big Sur-only | `0.857286` | `0.875459` | `0.839853` | `0.934899` |
   | Big Sur | Pooled Monterey+Big Sur | `0.854451` | `0.887361` | `0.823895` | `0.927124` |
   | Monterey | Big Sur-only transfer | `0.777221` | `0.866996` | `0.704293` | `0.824119` |
   | Monterey | Monterey-only | `0.852555` | `0.880181` | `0.826610` | `0.930024` |
   | Monterey | Pooled Monterey+Big Sur | `0.829966` | `0.909602` | `0.763152` | `0.920076` |

   The section should state that Monterey-to-Big-Sur binary transfer is decent,
   Big-Sur-to-Monterey binary transfer is weaker, and the larger Phase 2 issue
   is canopy amount calibration.

6. Keep the report compact.

   Avoid a task-by-task history of P2-01 through P2-08. Put detailed source
   paths and command history in the manifest or appendix, not in the narrative
   body.

7. Update report artifact manifests.

   The manifest should record:

   - Input comparison tables.
   - Binary full-grid prediction paths and calibration payload paths if the
     binary support table is computed during report generation.
   - Primary filters.
   - Output report paths.

8. Refresh generated outputs:

   ```bash
   uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
   ```

9. Inspect the rendered Markdown and HTML.

   Verify that the first screen of the report answers the Big Sur local vs
   pooled question and that the binary-support interpretation is not confused
   with hurdle amount calibration.

10. Record the outcome in `docs/todo.md`.

## Outcome

Completed on 2026-05-14 by extending `analyze-model` to switch into a Phase 2
report shape when `training_regime_comparison` is configured.

Generated/updated outputs from `configs/big_sur_smoke.yaml`:

- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.md`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.html`
- `/Volumes/x10pro/kelp_aef/reports/model_analysis/big_sur_phase2_model_analysis.pdf`
- `/Volumes/x10pro/kelp_aef/interim/big_sur_model_analysis_manifest.json`
- `/Volumes/x10pro/kelp_aef/reports/tables/monterey_big_sur_binary_support_primary_summary.csv`

Primary answer: Big Sur-only training is best for Big Sur canopy amount
calibration. The Big Sur-only expected-value hurdle has F1 `0.859563`, RMSE
`0.044095`, R2 `0.894855`, and area bias `-2.8414%`; pooled
Monterey+Big Sur has F1 `0.850056`, RMSE `0.050552`, R2 `0.861757`, and area
bias `-20.1402%`; Monterey-only transfer has F1 `0.849834`, RMSE `0.054901`,
R2 `0.837025`, and area bias `-22.1124%`.

Binary support is more transferable than canopy amount. Monterey-to-Big-Sur
binary F1 is `0.846453`, Big Sur-only is `0.857286`, and pooled is
`0.854451`. Pooled support is more conservative on Big Sur, with precision
`0.887361`, recall `0.823895`, and predicted-positive rate `0.041709`.

## Validation Command

For implementation:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run kelp-aef analyze-model --config configs/big_sur_smoke.yaml
git diff --check
```

If the implementation is docs/config-only, run:

```bash
git diff --check
```

and inspect the rendered Markdown diff.

## Smoke-Test Region And Years

- Regions: Monterey and Big Sur.
- Training regimes:
  - `monterey_only`
  - `big_sur_only`
  - `pooled_monterey_big_sur`
- Evaluation regions:
  - `monterey`
  - `big_sur`
- Label input: Kelpwatch-style annual maximum canopy.
- Primary year: 2022 held-out test split.
- Primary scope: retained plausible-kelp-domain full grid.

## Acceptance Criteria

- The report has a compact Phase 2 summary that answers whether Big Sur-only or
  pooled training improves Big Sur performance.
- The report separates binary support behavior from conditional canopy amount
  and hurdle area calibration.
- The headline Big Sur answer is explicit:
  - Big Sur-only is best for Big Sur amount calibration.
  - Pooled does not beat Big Sur-only for Big Sur amount calibration.
  - Pooled binary support is competitive but more conservative.
- The report includes a Monterey-vs-Big-Sur comparison table with canonical
  training-regime labels.
- All headline rows use the primary 2022 retained-domain filters.
- The generated manifest records the comparison input tables and report output
  paths.
- Existing Monterey Phase 1 closeout report paths and snapshots remain
  unchanged.
- `docs/todo.md` records the completed P2-09 outcome and points P2-10/P2-11 at
  any remaining visual QA or closeout needs.

## Known Constraints And Non-Goals

- Do not retrain models in this task.
- Do not tune thresholds or choose policies from 2022 held-out rows.
- Do not change the Phase 2 target away from Kelpwatch annual max.
- Do not add new model families or source inputs.
- Do not turn the report into a long chronology of Phase 2 tasks.
- Do not close Phase 2 here unless the user explicitly folds P2-11 into this
  task.
