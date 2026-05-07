# Product

## Goal

Build a reproducible pipeline that uses AlphaEarth/Satellite Embedding features
to learn Kelpwatch-style kelp canopy labels for the U.S. West Coast.

The first version should be framed as learning, improving, or reproducing
Kelpwatch-derived labels from AlphaEarth embeddings. Kelpwatch is a
Landsat-derived product, not independent field truth, so this project should not
claim to prove true kelp biomass.

## First Milestone

Start with a tiny feasibility spike before scaling.

```text
Kelpwatch label tile + AlphaEarth embedding tile -> aligned training table -> simple model -> map
```

Initial scope:

- Region: Monterey Peninsula, unless the project explicitly changes scope.
- Years: 2018-2022 for the first Monterey smoke test, using a year holdout.
- Label: Kelpwatch annual max canopy.
- Features: AlphaEarth annual 64-band embeddings aggregated from 10 m to the
  Kelpwatch 30 m grid.
- Models: logistic or ridge regression plus a tree model such as random forest.
- Split: year holdout.
- Outputs: predicted map, residual map, and area-bias summary.

## Success Criteria

The feasibility spike is useful when:

- AlphaEarth embeddings load for the selected region and years.
- Kelpwatch labels load for the same spatial/temporal scope.
- Coordinate alignment works and produces a grid-cell-by-year training table.
- A simple embedding model beats no-skill and geographic baselines.
- Maps and diagnostics look spatially plausible.

## Primary Data Products

Canonical data and generated artifact root:

```text
/Volumes/x10pro/kelp_aef
```

Expected first table product:

```text
one row = grid cell x year
columns = A00-A63 + lat + lon + year + state + label columns
```

Expected early artifacts:

- `/Volumes/x10pro/kelp_aef/raw/kelpwatch/`
- `/Volumes/x10pro/kelp_aef/raw/aef/`
- `/Volumes/x10pro/kelp_aef/raw/aef_samples/`
- `/Volumes/x10pro/kelp_aef/geos/monterey_aef_10n_8192_8192_footprint.geojson`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query.parquet`
- `/Volumes/x10pro/kelp_aef/interim/aef_monterey_catalog_query_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/metadata_summary.json`
- `/Volumes/x10pro/kelp_aef/interim/aligned_training_table.parquet`
- `/Volumes/x10pro/kelp_aef/reports/figures/sample_kelpwatch_vs_aef.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/predicted_map.png`
- `/Volumes/x10pro/kelp_aef/reports/figures/residual_map.png`
- `/Volumes/x10pro/kelp_aef/reports/tables/area_bias_summary.csv`

## Non-Goals

- Do not start with the whole West Coast.
- Do not bulk-download the full AlphaEarth embedding collection.
- Do not claim independent validation of real kelp biomass from Kelpwatch alone.
- Do not jump to spatial deep learning before simple tabular baselines work.
- Do not let one random-pixel split stand in for temporal or spatial
  generalization.

## Evaluation Priorities

Primary results should emphasize temporal and spatial holdouts because coastal
pixels are spatially autocorrelated.

Useful split families:

- Year holdout.
- Latitude band holdout.
- State or regional holdout.
- Random or block split only as a sanity check.

Useful metrics:

- Classification: AUROC, AUPRC, precision-recall at useful thresholds, and
  false positives near shoreline or water boundaries.
- Regression: RMSE, MAE, area bias, region-year total canopy error, and
  region/year correlations.
- Ecological products: annual max area error, fall persistence agreement,
  two-year fall change error, and collapse/recovery classification.
