# Data And Artifact Contract

## Purpose

This document defines where data and generated artifacts live, which outputs are
expected from the Monterey feasibility spike, and what can be tracked in git.

The repo should remain focused on source code, configs, tests, and docs. Raw
data, generated tables, rasters, figures, and models should live outside the
repo unless explicitly promoted as tiny fixtures.

## Canonical Root

Canonical data and generated artifact root:

```text
/Volumes/x10pro/kelp_aef
```

Expected directory layout:

```text
/Volumes/x10pro/kelp_aef/
  raw/
    kelpwatch/
    aef/
    aef_samples/
    domain/
  geos/
  interim/
  processed/
  models/
  reports/
    figures/
    tables/
```

Create the directories with:

```bash
mkdir -p /Volumes/x10pro/kelp_aef/raw/kelpwatch \
  /Volumes/x10pro/kelp_aef/raw/aef \
  /Volumes/x10pro/kelp_aef/raw/aef_samples \
  /Volumes/x10pro/kelp_aef/geos \
  /Volumes/x10pro/kelp_aef/interim \
  /Volumes/x10pro/kelp_aef/processed \
  /Volumes/x10pro/kelp_aef/models \
  /Volumes/x10pro/kelp_aef/reports/figures \
  /Volumes/x10pro/kelp_aef/reports/tables
```

## Git Policy

Tracked in git:

- Source code under `src/`.
- Tests under `tests/`.
- Configs under `configs/`.
- Documentation under `docs/`.
- Small hand-authored fixtures only when needed for deterministic tests.

Not tracked in git:

- Raw Kelpwatch data.
- AlphaEarth samples, chips, rasters, Zarr stores, or Earth Engine exports.
- AlphaEarth tile footprint GeoJSONs under the external artifact root.
- Derived Parquet/CSV/JSON artifacts from normal pipeline runs.
- Generated maps, figures, and report tables from normal pipeline runs.
- Trained model files and serialized estimators.

The `.gitignore` file ignores repo-local `data/`, `reports/`, `models/`,
Earth Engine export folders, common raster/Zarr outputs, and common model
serialization formats. This is defensive: the intended artifact location is the
external root above, not the checkout.

## Visual QA Samples

No visual QA outputs are intentionally tracked yet.

If a tiny output needs to become a tracked reference sample, put it under a
purpose-specific fixture directory such as:

```text
tests/fixtures/
docs/qa_samples/
```

Tracked QA samples should be small, manually selected, and documented with:

- Source command or producer.
- Input config.
- Date generated.
- Reason it is useful to keep.

Do not track large rasters, Zarr stores, full-resolution maps, or model outputs
as visual QA samples.

## Monterey Feasibility Outputs

The first feasibility milestone produced these outputs under the external
artifact root. They are generated artifacts and should not be tracked in git.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| AEF tile footprint | `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson` | Single-footprint smoke geometry extracted from the configured AlphaEarth tile. | Not tracked |
| AEF catalog query result | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet` | STAC GeoParquet rows selected for the Monterey footprint and 2018-2022 years. | Not tracked |
| AEF catalog query summary | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json` | Human-readable counts, bounds, and selected asset hrefs from the catalog query. | Not tracked |
| AEF tile manifest | `/Volumes/x10pro/kelp_aef/interim/aef_monterey_tile_manifest.json` | Local and source URIs for the matching 2018-2022 AEF tiles. | Not tracked |
| Kelpwatch source manifest | `/Volumes/x10pro/kelp_aef/interim/kelpwatch_source_manifest.json` | Downloaded/source Kelpwatch files used by the smoke config. | Not tracked |
| Metadata summary | `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json` | CRS, bounds, variables, seasons, years, units, and missing-data notes for source inputs. | Not tracked |
| Annual labels | `/Volumes/x10pro/kelp_aef/interim/labels_annual.parquet` | Kelpwatch seasonal labels collapsed to the configured annual target. | Not tracked |
| Station aligned table | `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet` | Station-centered alignment artifact retained as QA/reference; not the primary Phase 0 model input after Task 11. | Not tracked |
| Full-grid aligned table | `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table.parquet` | Full AEF-aligned 30 m target grid with Kelpwatch-station and assumed-background rows. | Not tracked |
| Full-grid alignment manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_full_grid_training_table_manifest.json` | Row counts, label-source counts, assets, and alignment settings for the full-grid artifact. | Not tracked |
| Background sample table | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.parquet` | Background-inclusive sampled model input used by the Phase 0 ridge baseline. | Not tracked |
| Background sample manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table_manifest.json` | Sampling policy, schema, and sample count metadata. | Not tracked |
| Split manifest | `/Volumes/x10pro/kelp_aef/interim/split_manifest.parquet` | Train/validation/test assignments for year-holdout evaluation. | Not tracked |
| Baseline model | `/Volumes/x10pro/kelp_aef/models/baselines/ridge_kelp_fraction.joblib` | Serialized Phase 0 ridge model payload. | Not tracked |
| Sample predictions | `/Volumes/x10pro/kelp_aef/processed/baseline_sample_predictions.parquet` | No-skill and ridge predictions on the sampled model frame. | Not tracked |
| Full-grid predictions | `/Volumes/x10pro/kelp_aef/processed/baseline_full_grid_predictions.parquet` | Streamed ridge predictions for the full grid across 2018-2022. | Not tracked |
| Full-grid prediction manifest | `/Volumes/x10pro/kelp_aef/interim/baseline_full_grid_prediction_manifest.json` | Row count, part count, and label-source counts for full-grid predictions. | Not tracked |
| Baseline metrics | `/Volumes/x10pro/kelp_aef/reports/tables/baseline_metrics.csv` | No-skill and ridge metrics by split and label source. | Not tracked |
| Residual map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.png` | Three-panel observed, predicted, and residual map for the held-out year. | Not tracked by default |
| Interactive residual map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.html` | Local HTML map for exploring observed, predicted, and residual values. | Not tracked by default |
| Area bias by year | `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.csv` | Full-grid year-level area-bias summary. | Not tracked |
| Phase 0 report snapshot | `docs/report_snapshots/monterey_phase0_model_analysis.md` | Repo copy of the closed Phase 0 report. | Tracked |
| Phase 1 Markdown report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.md` | Generated Phase 1 closeout report with the final model-policy decision, model comparison, and data-health sections. | Not tracked |
| Phase 1 HTML report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.html` | Rendered Phase 1 closeout report with embedded figures and model comparison. | Not tracked |
| Phase 1 PDF report | `/Volumes/x10pro/kelp_aef/reports/model_analysis/monterey_phase1_model_analysis.pdf` | PDF Phase 1 closeout report with the same core model-comparison content. | Not tracked |
| Phase 1 closeout report snapshot | `docs/report_snapshots/monterey_phase1_closeout_model_analysis.md` | Repo copy of the final Phase 1 closeout report. | Tracked |
| Pixel skill and area calibration figure | `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_pixel_skill_area_calibration.png` | Two-panel Phase 1 model comparison showing 10% annual-max F1 and full-grid signed area bias for the primary report split/year. | Not tracked |

## Phase 0 Artifact Counts

The final Phase 0 run used the Monterey smoke config with 2018-2022:

- Full-grid aligned table: 37,291,805 rows.
- Per year full-grid rows: 7,428,207 `assumed_background` rows and 30,154
  `kelpwatch_station` rows.
- Background-inclusive model input sample: 1,400,809 rows.
- Baseline retained model rows after missing-feature drops: 1,342,631 rows.
- Full-grid prediction dataset: 37,291,805 rows in 430 Parquet parts.
- Phase 0 report primary map/report year: 2022 test split with 7,458,361
  full-grid prediction rows.

## Phase 0 Interpretation Notes

The current Phase 0 ridge baseline is trained on the background-inclusive sample
without sample expansion weights. This avoids the near-zero collapse observed
with population-expanded background weights, but it is still not calibrated on
the full grid. In the final report, station-supported Kelpwatch rows and
assumed-background rows are interpreted separately.

Phase 1 closed as Monterey annual-max model and domain hardening on
2026-05-13. The high-level artifact categories are defined in
`docs/phase1_model_domain_hardening.md`, and the final model-policy decision is
tracked in `docs/phase1_closeout_model_policy_decision.md`.

## Phase 1 Domain Source Outputs

These outputs support the Monterey domain-filter tasks. They are generated
artifacts and should not be tracked in git.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| NOAA CUSP query manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_query_manifest.json` | Selected CUSP regional package, local mirror path, source metadata, and Monterey coverage check. | Not tracked |
| NOAA CUSP source package | `/Volumes/x10pro/kelp_aef/raw/domain/noaa_cusp/West.zip` | NOAA NSDE West regional CUSP shoreline shapefile ZIP for shoreline-side support. | Not tracked |
| NOAA CUSP source manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_cusp_source_manifest.json` | Transfer status, remote/local file size, vector metadata, and source provenance for the CUSP package. | Not tracked |
| USGS 3DEP query manifest | `/Volumes/x10pro/kelp_aef/interim/usgs_3dep_query_manifest.json` | TNMAccess 3DEP 1/3 arc-second GeoTIFF product query, selected source URLs, local mirror paths, and Monterey coverage context. | Not tracked |
| USGS 3DEP source rasters | `/Volumes/x10pro/kelp_aef/raw/domain/usgs_3dep/` | Selected USGS 3DEP land DEM fallback GeoTIFFs for Monterey-side domain filtering support. | Not tracked |
| USGS 3DEP source manifest | `/Volumes/x10pro/kelp_aef/interim/usgs_3dep_source_manifest.json` | Transfer status, remote/local file size, raster metadata, and source provenance for selected 3DEP rasters. | Not tracked |
| NOAA CRM query manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_crm_query_manifest.json` | NOAA Coastal Relief Model California mosaic query against the configured target-grid footprint, including selected products or subsets and skipped products. | Not tracked |
| NOAA CRM source rasters or subsets | `/Volumes/x10pro/kelp_aef/raw/domain/noaa_crm/` | Selected NOAA CRM Southern California v2 and/or Volume 7 source files or clipped subsets for broad topo-bathy domain filtering. | Not tracked |
| NOAA CRM source manifest | `/Volumes/x10pro/kelp_aef/interim/noaa_crm_source_manifest.json` | Transfer status, remote/local file size, raster metadata, source provenance, vertical datum, and coverage checks for selected CRM sources. | Not tracked |
| Aligned NOAA CRM table | `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm.parquet` | One static target-grid row per AEF/Kelpwatch full-grid cell with CRM elevation/depth, product provenance, and optional CUDEM/3DEP QA samples. | Not tracked |
| Aligned NOAA CRM manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_noaa_crm_manifest.json` | Source manifests, product-boundary rule, row counts, coverage counts, CUSP vector validation, and output schema for the aligned CRM support layer. | Not tracked |
| Aligned NOAA CRM QA summary | `/Volumes/x10pro/kelp_aef/reports/tables/aligned_noaa_crm_summary.csv` | Tall QA table with CRM coverage, product counts, broad depth bins, and optional CUDEM/3DEP coverage counts. | Not tracked |
| Aligned domain-source comparison | `/Volumes/x10pro/kelp_aef/reports/tables/aligned_domain_source_comparison.csv` | CRM-vs-CUDEM and CRM-vs-3DEP overlap, elevation/depth differences, sign disagreements, and coverage fractions. | Not tracked |
| Plausible kelp domain mask | `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask.parquet` | First static CRM depth/elevation mask with one row per target-grid cell, retain/drop flag, reason codes, and depth/elevation QA bins. | Not tracked |
| Plausible kelp domain mask manifest | `/Volumes/x10pro/kelp_aef/interim/plausible_kelp_domain_mask_manifest.json` | Mask inputs, thresholds, rule precedence, row counts, Kelpwatch-positive retention counts, CUSP provenance, outputs, and schema. | Not tracked |
| Plausible kelp domain mask coverage summary | `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_summary.csv` | Retained and dropped cell counts by mask reason and CRM source product. | Not tracked |
| Plausible kelp domain mask Kelpwatch retention | `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_kelpwatch_retention.csv` | Kelpwatch-positive retained/dropped counts by year, label source, and mask reason. | Not tracked |
| Plausible kelp domain mask depth bins | `/Volumes/x10pro/kelp_aef/reports/tables/plausible_kelp_domain_mask_depth_bins.csv` | Mask coverage and Kelpwatch-positive counts by land, ambiguous-coast, shallow-depth, intermediate-depth, and deep-water bins. | Not tracked |
| Plausible kelp domain mask visual QA | `/Volumes/x10pro/kelp_aef/reports/figures/plausible_kelp_domain_mask_qa.png` | Three-panel visual QA showing mask reasons, CRM elevation context, and Kelpwatch-positive retained/dropped overlay. | Not tracked |

## Phase 1 Masked Reporting Outputs

These outputs apply the P1-12 plausible-kelp domain mask to full-grid reporting
and pair with the masked training sample used by the current baseline run.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Masked residual map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_observed_predicted_residual.masked.png` | Three-panel observed, predicted, and residual map restricted to retained plausible-kelp cells for the configured report year. | Not tracked |
| Masked residual interactive map | `/Volumes/x10pro/kelp_aef/reports/figures/ridge_2022_residual_interactive.masked.html` | Interactive full-grid residual review restricted to retained plausible-kelp cells. | Not tracked |
| Masked area bias by year | `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_year.masked.csv` | Ridge full-grid area-bias rows filtered to `mask_status = plausible_kelp_domain` and `evaluation_scope = full_grid_masked`. | Not tracked |
| Masked area bias by latitude band | `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_by_latitude_band.masked.csv` | Ridge full-grid area-bias rows by latitude band inside the retained plausible-kelp domain. | Not tracked |
| Masked reference baseline area calibration | `/Volumes/x10pro/kelp_aef/reports/tables/reference_baseline_area_calibration.masked.csv` | Compact reference-baseline and ridge area calibration rows computed inside the retained plausible-kelp domain. | Not tracked |
| Off-domain prediction leakage audit | `/Volumes/x10pro/kelp_aef/reports/tables/off_domain_prediction_leakage_audit.csv` | Separate diagnostic for predicted area on cells dropped by the domain mask, grouped by model, split, year, and mask reason. | Not tracked |

## Phase 1 Mask-Aware Residual Diagnostics

These outputs explain remaining residuals inside the retained plausible-kelp
domain. They do not change mask thresholds, model inputs, or fitted models.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Residual by domain context | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_domain_context.csv` | Retained-domain residual taxonomy by mask reason, depth/elevation bin, label source, observed canopy bin, and residual class. | Not tracked |
| Residual by mask reason | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_mask_reason.csv` | Retained-domain residual summary by static mask reason. | Not tracked |
| Residual by depth/elevation bin | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_residual_by_depth_bin.csv` | Retained-domain residual summary by CRM depth and elevation bins. | Not tracked |
| Top residuals with domain context | `/Volumes/x10pro/kelp_aef/reports/tables/top_residual_stations.domain_context.csv` | Highest underprediction and overprediction rows with mask reason, CRM depth/elevation, depth bin, elevation bin, label source, coordinates, and residual class. | Not tracked |
| Residual by domain context figure | `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_residual_by_domain_context.png` | Compact report figure showing mean residual by retained depth/elevation bin. | Not tracked |

## Phase 1 Class And Target-Balance Diagnostics

These outputs quantify annual-max target imbalance before changing thresholds,
objectives, model classes, sample weights, or label inputs.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Class balance by split | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_class_balance_by_split.csv` | Positive, high-canopy, saturated, zero, and assumed-background rates by data scope, split, year, label source, mask status, and evaluation scope. | Not tracked |
| Target balance by label source | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_target_balance_by_label_source.csv` | Source-level annual-max target distribution by label source and mask status. | Not tracked |
| Background rate summary | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_background_rate_summary.csv` | Compact assumed-background rate summary by data scope, split, year, mask status, and evaluation scope. | Not tracked |
| Class balance figure | `/Volumes/x10pro/kelp_aef/reports/figures/model_analysis_class_balance.png` | Compact report figure comparing zero, positive, high-canopy, and saturated rates for primary analysis scopes. | Not tracked |

## Phase 1 CRM-Stratified Default Sampling

Task 33 promoted CRM-stratified, mask-first sampling to the Monterey Phase 1
default masked model-input path. The sampler filters to the retained
plausible-kelp domain before applying retained-background quotas, keeps all
retained Kelpwatch-observed rows, and uses CRM mask reason and depth bin only as
sampling and diagnostic context. Historical `.crm_stratified` sidecar outputs
may remain on disk for audit, but they are not required by the default
workflow.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Default CRM-stratified masked sample | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet` | Default model-input sample built from retained full-grid rows, all retained Kelpwatch-observed rows, and retained assumed-background rows sampled by `domain_mask_reason` and `depth_bin`. | Not tracked |
| Default CRM-stratified sample manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json` | Active sample policy, mask-first flag, retained-domain population counts, sampled counts, quota-dropped counts, mask-dropped counts, deterministic seed, and feature-policy note. | Not tracked |
| Default CRM-stratified sample summary | `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.masked_summary.csv` | Population counts, sampled counts, effective fractions, and weights by year, label source, mask reason, and retained depth bin. | Not tracked |
| Historical CRM-stratified sidecar sample | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.crm_stratified.masked.parquet` | Disabled audit sample from the earlier sidecar experiment; retained on disk for comparison provenance only. | Not tracked |
| Historical CRM-stratified all-model comparison | `/Volumes/x10pro/kelp_aef/reports/tables/model_analysis_crm_stratified_all_models_comparison.csv` | Audit comparison artifact now populated by the active default policy plus shared observed-positive support rows. It is not a headline report section. | Not tracked |

## Phase 1 Balanced Binary Presence Model

These outputs train and report the first imbalance-aware binary model for the
validation-backed `annual_max_ge_10pct` target. They remain Kelpwatch-style
annual-max weak-label artifacts, not independent ecological presence truth.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Binary presence model | `/Volumes/x10pro/kelp_aef/models/binary_presence/logistic_annual_max_ge_10pct.joblib` | Serialized class-weighted logistic regression payload, feature list, target metadata, and validation-selected probability threshold. | Not tracked |
| Binary presence sample predictions | `/Volumes/x10pro/kelp_aef/processed/binary_presence_sample_predictions.parquet` | Row-level masked model-input sample probabilities, selected classes, split labels, label-source fields, and mask metadata. | Not tracked |
| Binary presence full-grid predictions | `/Volumes/x10pro/kelp_aef/processed/binary_presence_full_grid_predictions.parquet` | Streamed masked full-grid probability predictions and selected classes inside the retained plausible-kelp domain. | Not tracked |
| Binary presence metrics | `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_metrics.csv` | AUROC, AUPRC, precision, recall, F1, predicted-positive rates, and false-positive behavior by split, year, and label source. | Not tracked |
| Binary presence threshold selection | `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_threshold_selection.csv` | Validation-only diagnostic probability-threshold grid and selected max-F1 operating threshold. | Not tracked |
| Binary presence full-grid area summary | `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_full_grid_area_summary.csv` | Predicted positive count/rate/area and assumed-background leakage summaries for the masked full-grid scope. | Not tracked |
| Binary presence thresholded model comparison | `/Volumes/x10pro/kelp_aef/reports/tables/binary_presence_thresholded_model_comparison.csv` | Balanced binary model metrics alongside continuous baseline predictions thresholded at the same `kelp_fraction_y >= 0.10` target. | Not tracked |
| Binary presence prediction manifest | `/Volumes/x10pro/kelp_aef/interim/binary_presence_prediction_manifest.json` | Inputs, outputs, split policy, selected logistic settings, threshold policy, row counts, and mask/report scope metadata. | Not tracked |
| Binary presence precision-recall figure | `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_precision_recall.png` | Compact validation precision, recall, and F1 diagnostic by probability threshold. | Not tracked |
| Binary presence map figure | `/Volumes/x10pro/kelp_aef/reports/figures/binary_presence_2022_map.png` | Four-panel 2022 retained-domain map of observed binary target, predicted probability, selected class, and classification outcome. | Not tracked |

## Phase 1 Conditional Canopy Amount Model

These outputs train and report the first positive-only annual-max canopy amount
model. They evaluate Kelpwatch-style positive-cell canopy amount separately from
binary presence and do not compose a final full-grid hurdle prediction.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Conditional canopy model | `/Volumes/x10pro/kelp_aef/models/conditional_canopy/ridge_positive_annual_max.joblib` | Serialized positive-only ridge model payload, selected validation alpha, feature list, and support-policy metadata. | Not tracked |
| Conditional canopy sample predictions | `/Volumes/x10pro/kelp_aef/processed/conditional_canopy_sample_predictions.parquet` | Row-level conditional amount predictions for observed-positive sample rows plus calibrated likely-positive diagnostics. | Not tracked |
| Conditional canopy metrics | `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_metrics.csv` | Positive-cell canopy amount metrics by split, year, label source, support policy, and mask status. | Not tracked |
| Conditional canopy positive residuals | `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_positive_residuals.csv` | Residual summaries for observed-positive, high-canopy, and near-saturated annual-max bins. | Not tracked |
| Conditional canopy model comparison | `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_model_comparison.csv` | Apples-to-apples comparison between positive-only conditional ridge and the current ridge baseline on the same observed-positive rows. | Not tracked |
| Conditional canopy full-grid likely-positive summary | `/Volumes/x10pro/kelp_aef/reports/tables/conditional_canopy_full_grid_likely_positive_summary.csv` | Compact count-only diagnostic of retained full-grid rows that would receive a conditional amount under the current calibrated binary gate. | Not tracked |
| Conditional canopy residual figure | `/Volumes/x10pro/kelp_aef/reports/figures/conditional_canopy_positive_residuals.png` | Compact figure showing mean residual area across conditional positive-cell bins. | Not tracked |
| Conditional canopy manifest | `/Volumes/x10pro/kelp_aef/interim/conditional_canopy_manifest.json` | Inputs, outputs, selected alpha, support row counts, and non-composition QA notes for the conditional stage. | Not tracked |

## Phase 1 First Hurdle Model

These outputs compose the saved calibrated binary annual-max presence
probability with the saved positive-only conditional canopy amount model. They
do not retrain either upstream model, refit Platt calibration, or tune
thresholds on the 2022 test split.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Hurdle full-grid predictions | `/Volumes/x10pro/kelp_aef/processed/hurdle_full_grid_predictions.parquet` | Row-level retained-domain predictions for the expected-value primary policy and hard-gated diagnostic policy. | Not tracked |
| Hurdle prediction manifest | `/Volumes/x10pro/kelp_aef/interim/hurdle_prediction_manifest.json` | Inputs, loaded payloads, row counts, selected threshold, non-refit flags, and output paths for the composed hurdle stage. | Not tracked |
| Hurdle metrics | `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_metrics.csv` | Pixel and area metrics by split, year, label source, mask status, evaluation scope, and composition policy. | Not tracked |
| Hurdle area calibration | `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_area_calibration.csv` | Full-grid retained-domain area calibration rows for expected-value and hard-gated hurdle predictions. | Not tracked |
| Hurdle-vs-reference comparison | `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_model_comparison.csv` | Hurdle rows alongside ridge, previous-year, grid-cell climatology, geographic ridge, and train-mean reference rows. | Not tracked |
| Hurdle residual by observed bin | `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_residual_by_observed_bin.csv` | Residual summaries by observed annual-max canopy area bin for each composition policy. | Not tracked |
| Hurdle assumed-background leakage | `/Volumes/x10pro/kelp_aef/reports/tables/hurdle_assumed_background_leakage.csv` | Assumed-background predicted area and predicted-positive leakage diagnostics for each composition policy. | Not tracked |
| Hurdle 2022 map figure | `/Volumes/x10pro/kelp_aef/reports/figures/hurdle_2022_observed_predicted_residual.png` | Three-panel 2022 retained-domain expected-value map of observed, predicted, and residual area. | Not tracked |

## Historical P1-22 Direct-Continuous Failure Artifacts

These outputs are historical records from the failed P1-22 direct-continuous
experiments. Task 38 removed the active training, config, and report code paths,
but the generated files remain useful audit artifacts for the capped-weight,
cap-sweep, stratified-background, and stratified-background sweep results.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Capped-weight run | `/Volumes/x10pro/kelp_aef/{models/continuous_objective,processed,reports/tables,interim}/continuous_objective_capped_weight_*` | Historical cap-5 direct continuous ridge artifacts from P1-22a. The result failed to beat ridge or compete with the expected-value hurdle. | Not tracked |
| Cap-sweep variants | `/Volumes/x10pro/kelp_aef/{models/continuous_objective,processed,reports/tables,interim}/continuous_objective_cap_{1,2,10,20,100}_*` | Historical cap sweep around cap 5. No cap resolved the direct-continuous tradeoff. | Not tracked |
| Stratified-background run | `/Volumes/x10pro/kelp_aef/{models/continuous_objective,processed,reports/tables,interim}/continuous_objective_stratified_background_*` | Historical P1-22b stratum-balanced background-weighting artifacts. The run reduced leakage but worsened RMSE and station skill. | Not tracked |
| Stratified-background sweep variants | `/Volumes/x10pro/kelp_aef/{models/continuous_objective,processed,reports/tables,interim}/continuous_objective_stratified_gamma_*` | Historical gamma and background-budget sweep artifacts. No variant fixed the one-stage direct-continuous tradeoff. | Not tracked |

## Phase 1 Masked Training Outputs

These outputs apply the P1-12 plausible-kelp domain mask to the model-input
background sample while preserving the unmasked sample as a sidecar comparison
artifact.

| Output | Path | Purpose | Git |
| --- | --- | --- | --- |
| Masked background sample training table | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked.parquet` | Background-inclusive model-input sample filtered to `is_plausible_kelp_domain = true`, with mask metadata columns retained for audit. | Not tracked |
| Masked background sample manifest | `/Volumes/x10pro/kelp_aef/interim/aligned_background_sample_training_table.masked_manifest.json` | Source sample path, mask inputs, retained/dropped counts, dropped observed/positive counts, retained-domain population counts, and schema. | Not tracked |
| Masked background sample summary | `/Volumes/x10pro/kelp_aef/reports/tables/aligned_background_sample_training_table.masked_summary.csv` | Retained and dropped sample rows by year, label source, mask flag, and mask reason, including observed-positive counts. | Not tracked |
