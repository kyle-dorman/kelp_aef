# Task 48: Evaluate Monterey Transfer On Big Sur

## Goal

Evaluate the closed Monterey Phase 1 policy on Big Sur as the first Phase 2
transfer baseline. This task should answer how the Monterey-trained AEF ridge,
binary support, and hurdle policies perform when applied to Big Sur inputs
without using Big Sur labels to change model policy, calibration, or thresholds.

## Inputs

- Big Sur config: `configs/big_sur_smoke.yaml`.
- Monterey source config: `configs/monterey_smoke.yaml`.
- Big Sur model-input artifacts from P2-03c / Task 45:
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_full_grid_training_table.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_aligned_background_sample_training_table.masked.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_split_manifest.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask.parquet`
  - `/Volumes/x10pro/kelp_aef/interim/big_sur_plausible_kelp_domain_mask_manifest.json`
- Frozen Monterey Phase 1 model artifacts, including:
  - AEF ridge model.
  - Calibrated binary-presence model and frozen Monterey threshold selection.
  - Conditional positive-canopy model.
  - Existing reference-baseline configuration and reporting semantics.

## Outputs

- Transfer prediction sidecars under `/Volumes/x10pro/kelp_aef/processed/`,
  using names that cannot overwrite later Big Sur-trained outputs, for example:
  - `big_sur_monterey_transfer_baseline_full_grid_predictions.parquet`
  - `big_sur_monterey_transfer_binary_presence_full_grid_predictions.parquet`
  - `big_sur_monterey_transfer_hurdle_full_grid_predictions.parquet`
- Transfer manifests under `/Volumes/x10pro/kelp_aef/interim/`, recording:
  - Big Sur input artifact paths and checksums or mtimes.
  - Monterey model artifact paths.
  - Calibration and threshold source.
  - Feature schema checks.
  - Evaluation split, year, mask, and label-source filters.
- Big Sur transfer metric tables under `/Volumes/x10pro/kelp_aef/reports/tables/`,
  including:
  - AEF ridge metrics.
  - Expected-value hurdle metrics.
  - Hard-gated hurdle metrics.
  - Calibrated binary support metrics and area summaries.
  - Reference baseline metrics and area calibration.
  - A compact comparison table with `training_regime = monterey_transfer` and
    `model_origin_region = monterey`.

## Config File

Use `configs/big_sur_smoke.yaml` as the Big Sur evaluation config. If additional
paths are needed, add a clearly named transfer block or sidecar path section
rather than replacing Big Sur-training model paths already present in the file.

## Plan / Spec Requirement

Write a brief implementation plan before editing code. The plan should name the
CLI shape, files to change, output paths, and the exact frozen Monterey artifacts
that will be loaded. Keep the implementation package-backed and config-driven.

## Implementation Notes

1. Verify the Big Sur full-grid and masked sample artifacts exist and contain
   the feature columns expected by the Monterey model payloads.
2. Load frozen Monterey model artifacts directly; do not call a training command
   against Big Sur data.
3. Apply the Monterey AEF ridge model to Big Sur full-grid rows and evaluate
   held-out Big Sur labels.
4. Apply the Monterey binary model to Big Sur, then apply the frozen Monterey
   calibration model and frozen Monterey threshold selection.
5. Apply the Monterey conditional positive-canopy model to Big Sur and compose:
   - expected-value hurdle prediction;
   - hard-gated hurdle prediction using the frozen Monterey binary policy.
6. Compute Big Sur reference baselines for comparison, labeled as reference
   baselines rather than Big Sur-trained AEF models.
7. Emit report-visible tables with the same primary row semantics as Phase 1:
   `split = test`, `year = 2022`, `evaluation_scope = full_grid_masked`,
   `label_source = all`, and `mask_status = plausible_kelp_domain`, unless
   source verification requires a documented change.
8. Record the result in `docs/todo.md` after artifacts are regenerated and
   validated.

## Implementation Plan

- Add a package-backed transfer command:
  `kelp-aef evaluate-transfer --config configs/big_sur_smoke.yaml --source-config configs/monterey_smoke.yaml`.
- Add a `models.transfer.monterey` block in the Big Sur config for path-distinct
  Monterey-transfer prediction sidecars, manifests, metrics, area summaries, and
  compact comparison tables.
- Load the frozen Monterey ridge, binary logistic, Platt calibration, threshold,
  and conditional canopy artifacts directly from `configs/monterey_smoke.yaml`.
  Do not call any training command on Big Sur rows.
- Score the Big Sur full-grid inference table, apply the configured Big Sur
  plausible-kelp reporting mask, write transfer-labeled artifacts, and compute
  the primary `split = test`, `year = 2022`, `evaluation_scope =
  full_grid_masked`, `label_source = all` rows.
- Add a focused synthetic test for the transfer command, then run the real
  transfer command and record the outcome in `docs/todo.md`.

## Validation Command

For code changes, run the relevant subset plus the transfer command:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
uv run kelp-aef <transfer-evaluation-command> --config configs/big_sur_smoke.yaml --source-config configs/monterey_smoke.yaml
```

The exact CLI command can be a new transfer-specific command or a sidecar-aware
extension of existing prediction/evaluation commands, but it must be rerunnable
without retraining or overwriting Big Sur-trained artifacts.

## Smoke-Test Region And Years

- Region: Big Sur.
- Transfer source: Monterey Phase 1 closed policy.
- Primary evaluation year: 2022 held-out Big Sur rows.
- Evaluation scope: plausible-kelp-domain masked full grid.

## Acceptance Criteria

- Big Sur held-out performance is reported for:
  - Monterey-trained AEF ridge.
  - Monterey-trained calibrated binary support.
  - Monterey-trained expected-value hurdle.
  - Monterey-trained hard-gated hurdle.
  - Reference baselines.
- Transfer outputs are clearly labeled with `monterey_transfer` or equivalent
  provenance in filenames, manifests, and tables.
- The binary support results use the frozen Monterey calibration and threshold
  policy; Big Sur labels are not used for threshold selection.
- No Big Sur model training, Big Sur calibration fitting, or Big Sur-driven
  policy changes are introduced by this task.
- Big Sur-trained sidecar paths remain untouched for P2-07.
- Metrics include the same primary reporting filters used in the Phase 1
  closeout row semantics.

## Known Constraints And Non-Goals

- Do not tune the Monterey policy on Big Sur held-out labels.
- Do not use Big Sur validation or test performance to choose a new binary
  threshold.
- Do not compare against pooled Monterey + Big Sur training here; that belongs
  in P2-08.
- Do not make narrative report updates that belong in the Phase 2 synthesis
  task, except for adding concise task outcomes and generated metric artifacts.

## Outcome

Completed on 2026-05-14 with:

```bash
uv run kelp-aef evaluate-transfer --config configs/big_sur_smoke.yaml --source-config configs/monterey_smoke.yaml
```

Primary Big Sur transfer artifacts:

- `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_baseline_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_binary_presence_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/processed/big_sur_monterey_transfer_hurdle_full_grid_predictions.parquet`
- `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_monterey_transfer_primary_summary.csv`
- `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_monterey_transfer_binary_presence_metrics.csv`
- `/Volumes/x10pro/kelp_aef/interim/big_sur_monterey_transfer_eval_manifest.json`

Primary `test`/2022 `full_grid_masked`/`all` result:

- AEF ridge transfer: F1 `0.771748`, area bias `+5.5720%`.
- Expected-value hurdle transfer: F1 `0.849834`, area bias `-22.1124%`.
- Hard-gated hurdle transfer: F1 `0.849308`, area bias `-19.2801%`.
- Frozen Monterey calibrated binary support: threshold `0.37`, AUROC
  `0.992484`, AUPRC `0.897458`, precision `0.829083`, recall `0.864566`,
  F1 `0.846453`.

The transfer manifest records no Big Sur ridge refit, binary-model refit,
binary-calibrator refit, conditional-canopy refit, or use of Big Sur test rows
for training, calibration, or threshold selection.
