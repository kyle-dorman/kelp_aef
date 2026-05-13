# Phase 1 Stratified-Background Continuous Sweep Results

Status: recorded on 2026-05-13 before backing out experimental sweep code.

## Purpose

Task 37 tested whether a one-stage direct continuous AlphaEarth model could use
retained-domain stratified background weighting to reduce background leakage
without needing the two-stage hurdle composition.

The target, split, features, mask, and sample policy stayed fixed:

- Target: Kelpwatch-style annual max, `kelp_fraction_y = kelp_max_y / 900 m2`.
- Features: `A00-A63` only.
- Split: train 2018-2020, validation 2021, test 2022.
- Reporting scope: `full_grid_masked` inside `plausible_kelp_domain`.
- Sample policy: `crm_stratified_mask_first_sample`.
- CRM context was used only for weighting strata, not as model predictors.

## Models Tested

The first stratified-background model used:

```text
stratum_columns = year, label_source, domain_mask_reason, depth_bin
```

Kelpwatch-supported and positive annual-max rows kept `fit_weight = 1.0`.
Assumed-background rows were weighted so each retained background stratum
contributed the same total fit weight within each year.

That full equalization reduced leakage but badly hurt station skill, so a small
sweep tested:

- `stratum_balance_gamma`: partial movement from raw sample weights toward equal
  background-stratum totals.
- `background_weight_budget_multiplier`: optional cap on total background fit
  weight per year as a multiple of Kelpwatch-supported rows.

Sweep variants:

- `stratified-gamma-025`
- `stratified-gamma-050`
- `stratified-gamma-075`
- `stratified-gamma-025-bg5`
- `stratified-gamma-050-bg5`
- `stratified-gamma-050-bg2`

All direct continuous variants selected `alpha = 0.01` using 2021 validation
rows only.

## Primary 2022 Results

Rows below use:

```text
split = test
year = 2022
evaluation_scope = full_grid_masked
label_source = all
```

| Model | RMSE | R2 | F1 >=10% | Predicted area | Area bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expected-value hurdle | 0.0322 | 0.790 | 0.812 | 3.50 M m2 | -16.0% |
| Previous-year annual max | 0.0341 | 0.765 | 0.797 | 3.50 M m2 | -15.9% |
| AEF ridge regression | 0.0452 | 0.587 | 0.476 | 8.42 M m2 | +102.1% |
| Stratified gamma 0.50 + bg2 | 0.0466 | 0.560 | 0.449 | 8.55 M m2 | +105.5% |
| Stratified gamma 0.25 + bg5 | 0.0472 | 0.549 | 0.503 | 8.23 M m2 | +97.8% |
| Stratified gamma 0.50 + bg5 | 0.0472 | 0.548 | 0.511 | 8.20 M m2 | +96.9% |
| Capped-weight ridge | 0.0493 | 0.507 | 0.590 | 8.64 M m2 | +107.5% |
| Stratified gamma 0.25 | 0.0532 | 0.426 | 0.674 | 5.93 M m2 | +42.5% |
| Stratified gamma 0.50 | 0.0535 | 0.419 | 0.680 | 5.91 M m2 | +41.9% |
| Stratified gamma 0.75 | 0.0538 | 0.412 | 0.686 | 5.88 M m2 | +41.2% |
| Full stratified background | 0.0542 | 0.405 | 0.694 | 5.85 M m2 | +40.4% |

## Station Skill Check

Rows below use:

```text
split = test
year = 2022
evaluation_scope = kelpwatch_station_sample
label_source = kelpwatch_station
```

| Model | Station RMSE | Station R2 | F1 >=10% | Station area bias |
| --- | ---: | ---: | ---: | ---: |
| Stratified gamma 0.50 + bg2 | 0.1641 | 0.666 | 0.772 | -18.5% |
| AEF ridge regression | 0.1647 | 0.664 | 0.776 | -18.9% |
| Stratified gamma 0.25 + bg5 | 0.1795 | 0.600 | 0.759 | -23.7% |
| Stratified gamma 0.50 + bg5 | 0.1807 | 0.595 | 0.759 | -24.0% |
| Capped-weight ridge | 0.1969 | 0.519 | 0.752 | -29.1% |
| Stratified gamma 0.25 | 0.2305 | 0.341 | 0.746 | -49.3% |
| Stratified gamma 0.50 | 0.2322 | 0.331 | 0.744 | -49.8% |
| Stratified gamma 0.75 | 0.2341 | 0.321 | 0.743 | -50.3% |
| Full stratified background | 0.2360 | 0.309 | 0.743 | -50.9% |

## Leakage Check

Rows below use 2022 retained-domain assumed-background leakage.

| Model | Assumed-background predicted area | Predicted positive rate |
| --- | ---: | ---: |
| Full stratified background | 3.80 M m2 | 0.18% |
| Stratified gamma 0.75 | 3.81 M m2 | 0.21% |
| Stratified gamma 0.50 | 3.82 M m2 | 0.25% |
| Stratified gamma 0.25 | 3.82 M m2 | 0.28% |
| Stratified gamma 0.50 + bg5 | 5.03 M m2 | 1.52% |
| Stratified gamma 0.25 + bg5 | 5.06 M m2 | 1.59% |
| Stratified gamma 0.50 + bg2 | 5.16 M m2 | 2.23% |
| Capped-weight ridge | 5.69 M m2 | 0.87% |

## Interpretation

The sweep did not find a direct continuous stratified-background variant that
beats ridge or competes with the expected-value hurdle.

The tradeoff is consistent:

- Gamma-only variants keep the leakage reduction, but station skill and
  full-grid RMSE remain poor.
- Budgeted variants recover station skill, but they restore the full-grid
  overprediction failure.
- The best budgeted station result, `ridge_stratified_gamma_050_bg2`, has test
  station RMSE `0.1641`, essentially matching ridge, but its retained full-grid
  area bias is still `+105.5%`.
- The full stratified-background model improves area bias relative to ridge
  (`+40.4%` versus `+102.1%`) and improves F1 (`0.694` versus `0.476`), but its
  RMSE (`0.0542`) and station RMSE (`0.2360`) are worse.

Conclusion: this looks like a structural limitation of the one-stage direct
continuous objective under the current annual-max target, features, split, and
retained-domain mask. The expected-value hurdle remains the stronger Phase 1 AEF
candidate.

## Backout Note

The experimental sweep implementation can be backed out later without losing
the result, as long as this note, the task completion notes, and the generated
artifact paths remain available for audit.
