import csv
import json
from pathlib import Path

from kelp_aef import main


def test_visualize_source_coverage_creates_footprint_and_outputs(tmp_path: Path) -> None:
    """Verify source-coverage QA writes footprint, summary, figure, HTML, and manifest."""
    config_path = write_source_coverage_config(tmp_path)
    write_source_coverage_inputs(tmp_path)

    assert main(["visualize-source-coverage", "--config", str(config_path)]) == 0

    assert (tmp_path / "geos/big_sur_footprint.geojson").is_file()
    assert (tmp_path / "tables/big_sur_source_coverage_summary.csv").is_file()
    assert (tmp_path / "figures/big_sur_source_coverage_qa.png").is_file()
    assert (tmp_path / "figures/big_sur_source_coverage_interactive_qa.html").is_file()
    assert (tmp_path / "interim/big_sur_source_coverage_manifest.json").is_file()

    with (tmp_path / "tables/big_sur_source_coverage_summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    positive_row = next(row for row in rows if row["artifact"] == "positive_source_qa")
    assert positive_row["positive_count"] == "12"
    assert positive_row["valid_count"] == "20"

    html = (tmp_path / "figures/big_sur_source_coverage_interactive_qa.html").read_text()
    assert "Big Sur Source Coverage QA" in html
    assert "noaa_crm" in html

    manifest = json.loads((tmp_path / "interim/big_sur_source_coverage_manifest.json").read_text())
    assert manifest["region"]["slug"] == "big_sur"
    assert manifest["region"]["footprint_created"] is True


def write_source_coverage_config(tmp_path: Path) -> Path:
    """Write a minimal config for source-coverage QA tests."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
years:
  smoke: [2018, 2019]
region:
  name: big_sur
  crs: EPSG:4326
  geometry:
    path: {tmp_path / "geos/big_sur_footprint.geojson"}
    bbox: [-122.1, 35.5, -121.1, 36.3]
    source_stac_item_id: "8957"
    source_crs: EPSG:32610
    source_tile_uri: s3://example/aef.tiff
labels:
  target: kelp_max_y
  aggregation: annual_max
  paths:
    source_manifest: {tmp_path / "interim/big_sur_kelpwatch_source_manifest.json"}
features:
  paths:
    catalog_query_summary: {tmp_path / "interim/aef_big_sur_catalog_query_summary.json"}
    tile_manifest: {tmp_path / "interim/aef_big_sur_tile_manifest.json"}
domain:
  noaa_crm:
    query_manifest: {tmp_path / "interim/big_sur_noaa_crm_query_manifest.json"}
    source_manifest: {tmp_path / "interim/big_sur_noaa_crm_source_manifest.json"}
  noaa_cudem:
    query_manifest: {tmp_path / "interim/big_sur_noaa_cudem_tile_query_manifest.json"}
    tile_manifest: {tmp_path / "interim/big_sur_noaa_cudem_tile_manifest.json"}
  noaa_cusp:
    query_manifest: {tmp_path / "interim/big_sur_noaa_cusp_query_manifest.json"}
    source_manifest: {tmp_path / "interim/big_sur_noaa_cusp_source_manifest.json"}
  usgs_3dep:
    query_manifest: {tmp_path / "interim/big_sur_usgs_3dep_query_manifest.json"}
    source_manifest: {tmp_path / "interim/big_sur_usgs_3dep_source_manifest.json"}
reports:
  figures_dir: {tmp_path / "figures"}
  tables_dir: {tmp_path / "tables"}
""".lstrip()
    )
    return config_path


def write_source_coverage_inputs(tmp_path: Path) -> None:
    """Write tiny source manifests used by the source-coverage QA test."""
    local_aef = tmp_path / "raw/aef/2018.tif"
    local_aef.parent.mkdir(parents=True, exist_ok=True)
    local_aef.write_bytes(b"tiny")
    local_kelpwatch = tmp_path / "raw/kelpwatch/kelpwatch.nc"
    local_kelpwatch.parent.mkdir(parents=True, exist_ok=True)
    local_kelpwatch.write_bytes(b"tiny")

    write_json(
        tmp_path / "interim/aef_big_sur_catalog_query_summary.json",
        {
            "years": [2018, 2019],
            "raw_spatial_candidate_count": 2,
            "footprint_bounds": [-122.1, 35.5, -121.1, 36.3],
            "selected_assets": [{"year": 2018}, {"year": 2019}],
        },
    )
    write_json(
        tmp_path / "interim/aef_big_sur_tile_manifest.json",
        {
            "records": [
                {
                    "year": 2018,
                    "local_tiff_path": str(local_aef),
                    "catalog_bounds": [-122.1, 35.5, -121.1, 36.3],
                    "validation_status": "valid",
                }
            ]
        },
    )
    write_json(
        tmp_path / "interim/big_sur_kelpwatch_source_manifest.json",
        {
            "years": [2018, 2019],
            "transfer": {"local_path": str(local_kelpwatch), "status": "skipped_existing"},
            "source": {"bounds": {"west": -125.0, "south": 32.0, "east": -114.0, "north": 49.0}},
        },
    )
    (tmp_path / "tables").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tables/kelpwatch_big_sur_source_qa.csv").write_text(
        "year,pixel_count,valid_count,missing_count,zero_count,nonzero_count,min,median,p95,p99,max,aggregate_canopy_area\n"
        "2018,10,10,0,4,6,0,1,2,3,4,100\n"
        "2019,10,10,0,4,6,0,1,2,3,4,100\n"
    )

    write_domain_manifest_pair(tmp_path, "big_sur_noaa_crm", "selected_products", "product_id")
    write_domain_manifest_pair(tmp_path, "big_sur_noaa_cudem_tile", "selected_tiles", "tile_id")
    write_domain_manifest_pair(tmp_path, "big_sur_noaa_cusp", "selected_artifacts", "artifact_id")
    write_domain_manifest_pair(tmp_path, "big_sur_usgs_3dep", "selected_artifacts", "source_id")


def write_domain_manifest_pair(
    tmp_path: Path,
    stem: str,
    selected_key: str,
    id_key: str,
) -> None:
    """Write a tiny query/source manifest pair for one domain source."""
    local_path = tmp_path / f"raw/domain/{stem}.dat"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"tiny")
    record = {
        id_key: stem,
        "bounds": {"west": -122.2, "south": 35.4, "east": -121.0, "north": 36.4},
        "local_path": str(local_path),
        "transfer": {"status": "skipped_existing", "local_path": str(local_path)},
    }
    query_stem = stem.replace("_tile", "")
    write_json(
        tmp_path / f"interim/{stem}_query_manifest.json",
        {
            "query_status": "selected",
            selected_key: [record],
            "region": {"bounds": record["bounds"]},
        },
    )
    source_suffix = "tile_manifest" if "cudem" in stem else "source_manifest"
    write_json(
        tmp_path / f"interim/{query_stem}_{source_suffix}.json",
        {"records": [record], "dry_run": False},
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON fixture with parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
