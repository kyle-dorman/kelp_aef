# Task 42: Verify Big Sur Source Coverage And Early Visual QA

## P2 Mapping

P2-02: Verify Big Sur source coverage and early visual QA.

## Goal

Confirm that the configured Big Sur smoke region has usable AEF, Kelpwatch,
CRM/domain, and shoreline support before any model training or metric
interpretation.

This task should download the Big Sur AEF assets selected for the smoke years
if they are not already present. It should also produce comparable early visual
QA artifacts for Big Sur and Monterey so the two regions can be contrasted
before building Big Sur model-input artifacts.

## Inputs

- Phase 2 plan: `docs/phase2_big_sur_generalization.md`.
- Active checklist: `docs/todo.md`.
- Big Sur config: `configs/big_sur_smoke.yaml`.
- Monterey contrast config: `configs/monterey_smoke.yaml`.
- Big Sur config/source-manifest plan:
  `tasks/41_big_sur_config_source_manifest_plan.md`.
- Existing source commands:
  - `kelp-aef query-aef-catalog`.
  - `kelp-aef download-aef`.
  - `kelp-aef inspect-kelpwatch`.
  - `kelp-aef visualize-kelpwatch`.
  - `kelp-aef query-noaa-crm`.
  - `kelp-aef download-noaa-crm`.
  - `kelp-aef query-noaa-cudem`.
  - `kelp-aef download-noaa-cudem`.
  - `kelp-aef query-noaa-cusp`.
  - `kelp-aef download-noaa-cusp`.
  - `kelp-aef query-usgs-3dep`.
  - `kelp-aef download-usgs-3dep`.
- User-provided Big Sur AEF STAC context from the config:
  - STAC item id: `8957`.
  - CRS: `EPSG:32610`.
  - Bbox:
    `[-122.09641373617602, 35.51952415252234, -121.17627335446835, 36.26818075229042]`.
  - Example 2022 AEF asset:
    `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2022/10N/xaspzf5khdg4c5pbs-0000000000-0000008192.tiff`.
- Canonical artifact root: `/Volumes/x10pro/kelp_aef`.

## Outputs

Big Sur source and coverage outputs:

- Big Sur footprint GeoJSON:
  `/Volumes/x10pro/kelp_aef/geos/big_sur_aef_10n_0000_8192_footprint.geojson`.
- AEF catalog query:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_catalog_query.parquet`.
- AEF catalog query summary:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_catalog_query_summary.json`.
- AEF tile manifest after download:
  `/Volumes/x10pro/kelp_aef/interim/aef_big_sur_tile_manifest.json`.
- Kelpwatch source manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_kelpwatch_source_manifest.json`.
- Domain-source query/source manifests:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_crm_query_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_crm_source_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cudem_tile_query_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cudem_tile_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cusp_query_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_noaa_cusp_source_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_usgs_3dep_query_manifest.json`.
  `/Volumes/x10pro/kelp_aef/interim/big_sur_usgs_3dep_source_manifest.json`.
- Short coverage summary:
  `/Volumes/x10pro/kelp_aef/reports/tables/big_sur_source_coverage_summary.csv`.
- Source coverage manifest:
  `/Volumes/x10pro/kelp_aef/interim/big_sur_source_coverage_manifest.json`.
- Early visual QA artifacts:
  `/Volumes/x10pro/kelp_aef/reports/figures/big_sur_source_coverage_qa.png`.
  `/Volumes/x10pro/kelp_aef/reports/figures/big_sur_source_coverage_interactive_qa.html`.
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_big_sur_annual_max_qa.png`.
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_big_sur_quarterly_timeseries_qa.png`.
  `/Volumes/x10pro/kelp_aef/reports/figures/kelpwatch_big_sur_interactive_qa.html`.
  `/Volumes/x10pro/kelp_aef/reports/tables/kelpwatch_big_sur_source_qa.csv`.
  `/Volumes/x10pro/kelp_aef/interim/kelpwatch_big_sur_source_qa.json`.

Monterey contrast outputs:

- Verify existing Monterey Kelpwatch, AEF, and domain QA artifacts.
- If a Big Sur QA artifact has no comparable Monterey counterpart, regenerate
  or create the Monterey counterpart with `monterey` in the filename, for
  example:
  `/Volumes/x10pro/kelp_aef/reports/tables/monterey_source_coverage_summary.csv`.
  `/Volumes/x10pro/kelp_aef/reports/figures/monterey_source_coverage_qa.png`.
  `/Volumes/x10pro/kelp_aef/reports/figures/monterey_source_coverage_interactive_qa.html`.

## Config File

Primary config:

```bash
configs/big_sur_smoke.yaml
```

Contrast config:

```bash
configs/monterey_smoke.yaml
```

Keep both configs explicit. Do not introduce a multi-region config refactor in
this task unless a command cannot be made region-safe any other way.

## Plan/Spec Requirement

This task plan is the implementation spec for P2-02. Before editing code, write
a brief implementation note in the task outcome section that states:

- Which Big Sur source artifacts already exist.
- Whether the Big Sur AEF tile files are already present.
- Which Monterey contrast QA artifacts already exist and which will be
  regenerated.
- Any source that will be query-only or dry-run-only in this pass.

## Implementation Plan

1. Verify artifact state before downloading:
   - Check whether the Big Sur footprint exists.
   - Check whether `aef_big_sur_catalog_query.parquet`,
     `aef_big_sur_tile_manifest.json`, and the selected local AEF TIFF/VRT
     paths exist.
   - Check which Monterey QA artifacts already exist so P2-02 can create only
     missing contrast artifacts.
2. Materialize the Big Sur footprint if missing:
   - Prefer a one-feature EPSG:4326 polygon from `region.geometry.bbox` in
     `configs/big_sur_smoke.yaml`.
   - Record the footprint provenance in the Big Sur source coverage manifest.
   - Do not write the generated footprint into the git checkout.
3. Query Big Sur AEF coverage:
   - Run `query-aef-catalog` against `configs/big_sur_smoke.yaml`.
   - Verify selected years, selected asset hrefs, overlap fractions, and
     projection metadata in the query summary.
   - If fewer than the expected 2018-2022 years are selected, record the gap and
     stop before any model-input work.
4. Download Big Sur AEF assets only after query review:
   - Run `download-aef` for `configs/big_sur_smoke.yaml`.
   - This task may download the selected Big Sur smoke AEF assets, but it must
     not bulk-download the full AEF collection or any non-selected region.
   - Record local TIFF/VRT paths, file sizes, validation status, and missing
     years in `aef_big_sur_tile_manifest.json`.
5. Verify Kelpwatch support:
   - Run `inspect-kelpwatch` with `configs/big_sur_smoke.yaml`.
   - Reuse the existing local Kelpwatch NetCDF if valid; download only if the
     configured source manifest cannot be produced from an existing valid file.
   - Run or update `visualize-kelpwatch` so output names are region-scoped
     (`kelpwatch_big_sur_*`) and do not overwrite Monterey QA outputs.
6. Verify domain and shoreline support:
   - Run query-first domain-source commands for NOAA CRM, NOAA CUDEM, NOAA
     CUSP, and USGS 3DEP against `configs/big_sur_smoke.yaml`.
   - Use dry-run or manifest-only paths first where the command supports it.
   - Download or register CRM/CUSP/CUDEM/3DEP support only when needed to make
     the early coverage and visual QA honest.
   - Keep CUSP as shoreline-vector coverage/provenance QA, not as raster depth
     input.
7. Add a compact source coverage QA artifact if no existing command can produce
   it:
   - Prefer a package-backed module under `src/kelp_aef/viz/`.
   - It should read the region footprint plus AEF, Kelpwatch, CRM, CUDEM, CUSP,
     and 3DEP manifests.
   - It should write a CSV summary, JSON manifest, static figure, and optional
     local HTML view for both Big Sur and Monterey.
8. Create Monterey contrast QA artifacts when missing:
   - Re-run existing Monterey QA commands only when the corresponding artifact
     is missing or stale.
   - Do not modify Monterey model outputs or report snapshots.
9. Write a short P2-02 outcome section in this task file:
   - Source years found for AEF and Kelpwatch.
   - Local AEF download status.
   - Domain-source coverage status.
   - Kelpwatch-positive source coverage inside the Big Sur footprint.
   - Any source/alignment caveat that blocks P2-03.
10. Update `docs/todo.md` only after the coverage and visual QA artifacts are
    written and reviewed.

## Suggested Command Order

Safe preflight and query commands:

```bash
uv run kelp-aef query-aef-catalog --config configs/big_sur_smoke.yaml
uv run kelp-aef download-aef --config configs/big_sur_smoke.yaml --dry-run --skip-remote-checks --manifest-output /private/tmp/aef_big_sur_tile_manifest_dry_run.json
uv run kelp-aef inspect-kelpwatch --config configs/big_sur_smoke.yaml --dry-run --manifest-output /private/tmp/big_sur_kelpwatch_source_manifest_dry_run.json
uv run kelp-aef query-noaa-crm --config configs/big_sur_smoke.yaml --skip-remote-checks --manifest-output /private/tmp/big_sur_noaa_crm_query_manifest.json
uv run kelp-aef query-noaa-cusp --config configs/big_sur_smoke.yaml --skip-remote-checks --manifest-output /private/tmp/big_sur_noaa_cusp_query_manifest.json
uv run kelp-aef query-usgs-3dep --config configs/big_sur_smoke.yaml --dry-run --manifest-output /private/tmp/big_sur_usgs_3dep_query_manifest_dry_run.json
```

Real AEF download command after query review:

```bash
uv run kelp-aef download-aef --config configs/big_sur_smoke.yaml
```

Expected visual QA commands after manifests exist:

```bash
uv run kelp-aef visualize-kelpwatch --config configs/big_sur_smoke.yaml
uv run kelp-aef visualize-kelpwatch --config configs/monterey_smoke.yaml
```

Add a new source-coverage visual QA command only if the existing commands do
not already produce enough AEF/domain context for side-by-side review.

## Validation Command

For docs/config-only setup or task outcome updates:

```bash
git diff --check
rg -n "P2-02|big_sur_source_coverage|aef_big_sur|kelpwatch_big_sur|monterey_source_coverage" docs tasks configs
```

For code changes:

```bash
uv run ruff check .
uv run mypy src tests
uv run pytest
```

For the source/QA artifact run, use the relevant non-training command subset:

```bash
uv run kelp-aef query-aef-catalog --config configs/big_sur_smoke.yaml
uv run kelp-aef download-aef --config configs/big_sur_smoke.yaml
uv run kelp-aef inspect-kelpwatch --config configs/big_sur_smoke.yaml
uv run kelp-aef visualize-kelpwatch --config configs/big_sur_smoke.yaml
```

Add domain-source query/download validation commands as needed for the touched
source modules.

## Smoke-Test Region And Years

- Region: Big Sur.
- Region shorthand: `big_sur`.
- Planned years: 2018-2022, subject to source coverage confirmation.
- Primary visual QA year: 2022, with yearly Kelpwatch panels for all confirmed
  smoke years.
- Contrast region: Monterey Peninsula, same 2018-2022 window.

## Acceptance Criteria

- The task file records whether Big Sur AEF was already present or downloaded
  during P2-02.
- Big Sur AEF catalog query and tile manifest exist and list selected assets by
  year.
- Selected Big Sur AEF local files exist or missing years are explicitly
  recorded as blockers.
- Big Sur Kelpwatch source manifest and Kelpwatch visual QA artifacts exist.
- Big Sur domain-source query/source manifests record CRM, CUDEM, CUSP, and
  3DEP coverage status where relevant.
- A short coverage summary table and manifest make source years, source bounds,
  source status, and coverage gaps explicit.
- Big Sur and Monterey have comparable early QA figures or the missing
  Monterey counterpart is explicitly justified.
- Kelpwatch-positive source coverage inside the Big Sur footprint is recorded.
  Exact retained-domain mask retention can remain a P2-03 output if it requires
  aligned full-grid and mask artifacts.
- No model training, model evaluation, threshold tuning, or Big Sur performance
  interpretation occurs in this task.

## Known Constraints And Non-Goals

- Do not start full West Coast scale-up.
- Do not download AEF assets outside the Big Sur smoke selection.
- Do not build Big Sur aligned feature/label tables.
- Do not build the final Big Sur plausible-kelp domain mask or model-input
  sample; those belong to P2-03.
- Do not train, calibrate, compose, or evaluate Big Sur models.
- Do not tune thresholds or sample quotas from Big Sur held-out test rows.
- Do not add bathymetry, depth, elevation, or domain-source fields as model
  predictors.
- Do not overwrite Monterey artifacts; create region-scoped contrast outputs
  where needed.
