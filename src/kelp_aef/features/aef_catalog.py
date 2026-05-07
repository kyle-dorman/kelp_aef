"""Query the AlphaEarth/AEF STAC GeoParquet catalog."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import geopandas as gpd  # type: ignore[import-untyped]
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from kelp_aef.config import load_yaml_config, require_mapping, require_string

DEFAULT_LAYER_NAME = "aef_index_stac_geoparquet"
DEFAULT_SELECTION_POLICY = "max_overlap_per_year"
DEFAULT_MIN_OVERLAP_FRACTION = 0.5

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AefCatalogQueryConfig:
    """Resolved config values needed by the AEF catalog query step."""

    config_path: Path
    catalog_url: str
    layer_name: str
    footprint_path: Path
    years: tuple[int, ...]
    output_parquet: Path
    output_summary: Path
    selection_policy: str
    min_footprint_overlap_fraction: float


@dataclass(frozen=True)
class CandidateFeature:
    """A catalog feature with derived query metadata."""

    feature: dict[str, Any]
    year: int
    asset_href: str
    overlap_area: float
    footprint_overlap_fraction: float


def load_aef_catalog_query_config(config_path: Path) -> AefCatalogQueryConfig:
    """Load AEF catalog query settings from the workflow config."""
    config = load_yaml_config(config_path)
    region = require_mapping(config.get("region"), "region")
    geometry = require_mapping(region.get("geometry"), "region.geometry")
    years_config = require_mapping(config.get("years"), "years")
    features = require_mapping(config.get("features"), "features")
    catalog = require_mapping(features.get("catalog"), "features.catalog")
    paths = require_mapping(features.get("paths"), "features.paths")

    raw_years = years_config.get("smoke")
    if not isinstance(raw_years, list) or not all(isinstance(year, int) for year in raw_years):
        msg = "config field must be a list of integer years: years.smoke"
        raise ValueError(msg)

    selection_policy = str(catalog.get("selection_policy", DEFAULT_SELECTION_POLICY))
    if selection_policy != DEFAULT_SELECTION_POLICY:
        msg = f"unsupported AEF catalog selection policy: {selection_policy}"
        raise ValueError(msg)

    return AefCatalogQueryConfig(
        config_path=config_path,
        catalog_url=require_string(catalog.get("url"), "features.catalog.url"),
        layer_name=str(catalog.get("layer_name", DEFAULT_LAYER_NAME)),
        footprint_path=Path(require_string(geometry.get("path"), "region.geometry.path")),
        years=tuple(raw_years),
        output_parquet=Path(
            require_string(paths.get("catalog_query"), "features.paths.catalog_query")
        ),
        output_summary=Path(
            require_string(
                paths.get("catalog_query_summary"), "features.paths.catalog_query_summary"
            )
        ),
        selection_policy=selection_policy,
        min_footprint_overlap_fraction=float(
            catalog.get("min_footprint_overlap_fraction", DEFAULT_MIN_OVERLAP_FRACTION)
        ),
    )


def query_aef_catalog(config_path: Path) -> int:
    """Run the AEF catalog query step."""
    query_config = load_aef_catalog_query_config(config_path)
    LOGGER.info("Loaded AEF catalog query config from %s", query_config.config_path)
    footprint = read_first_geometry(query_config.footprint_path)
    footprint_area = footprint.area
    if footprint_area <= 0:
        msg = f"footprint polygon must have positive area: {query_config.footprint_path}"
        raise ValueError(msg)

    raw_catalog = run_ogrinfo_query(
        query_config.catalog_url, query_config.layer_name, footprint.bounds
    )
    catalog_features = extract_catalog_features(raw_catalog)
    LOGGER.info("Found %s raw spatial AEF catalog candidates", len(catalog_features))
    selected, candidate_counts = select_catalog_features(
        catalog_features=catalog_features,
        footprint=footprint,
        footprint_area=footprint_area,
        years=query_config.years,
        min_footprint_overlap_fraction=query_config.min_footprint_overlap_fraction,
    )

    if not selected:
        msg = "AEF catalog query did not select any features"
        raise RuntimeError(msg)

    selected_geojson = build_selected_feature_collection(selected)
    write_query_outputs(
        selected_geojson=selected_geojson,
        selected=selected,
        candidate_counts=candidate_counts,
        query_config=query_config,
        footprint=footprint,
        raw_candidate_count=len(catalog_features),
    )

    LOGGER.info("query-aef-catalog: selected %s AEF assets", len(selected))
    LOGGER.info("catalog query: %s", query_config.output_parquet)
    LOGGER.info("summary: %s", query_config.output_summary)
    return 0


def run_ogrinfo_query(
    catalog_url: str, layer_name: str, bbox: tuple[float, float, float, float]
) -> dict[str, Any]:
    """Query the remote catalog with OGR and return its JSON output."""
    ogrinfo = shutil.which("ogrinfo")
    if ogrinfo is None:
        raise RuntimeError("ogrinfo is required for AEF catalog querying")

    source = f"/vsicurl/{catalog_url}"
    command = [
        ogrinfo,
        "-json",
        "-features",
        "-spat",
        str(bbox[0]),
        str(bbox[1]),
        str(bbox[2]),
        str(bbox[3]),
        source,
        layer_name,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return cast(dict[str, Any], json.loads(result.stdout))


def extract_catalog_features(ogrinfo_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract feature dictionaries from OGR's JSON response."""
    layers = ogrinfo_json.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("OGR JSON response does not contain layers")
    layer = require_mapping(layers[0], "layers[0]")
    features = layer.get("features")
    if not isinstance(features, list):
        raise ValueError("OGR JSON response does not contain layer features")
    return [require_mapping(feature, "feature") for feature in features]


def select_catalog_features(
    catalog_features: list[dict[str, Any]],
    footprint: BaseGeometry,
    footprint_area: float,
    years: tuple[int, ...],
    min_footprint_overlap_fraction: float,
) -> tuple[list[CandidateFeature], dict[int, int]]:
    """Select the max-overlap catalog feature for each configured year."""
    requested_years = set(years)
    candidates_by_year: dict[int, list[CandidateFeature]] = {year: [] for year in years}

    for feature in catalog_features:
        candidate = candidate_from_feature(feature, footprint, footprint_area)
        if candidate is None or candidate.year not in requested_years:
            continue
        if candidate.footprint_overlap_fraction < min_footprint_overlap_fraction:
            continue
        candidates_by_year[candidate.year].append(candidate)

    selected: list[CandidateFeature] = []
    for year in years:
        year_candidates = candidates_by_year[year]
        if not year_candidates:
            continue
        selected.append(
            max(
                year_candidates,
                key=lambda candidate: (
                    candidate.footprint_overlap_fraction,
                    candidate.overlap_area,
                ),
            )
        )

    candidate_counts = {year: len(candidates_by_year[year]) for year in years}
    return selected, candidate_counts


def candidate_from_feature(
    feature: dict[str, Any], footprint: BaseGeometry, footprint_area: float
) -> CandidateFeature | None:
    """Build a candidate feature with year and overlap metadata."""
    properties = require_mapping(feature.get("properties"), "feature.properties")
    year = year_from_datetime(properties.get("datetime"))
    asset_href = properties.get("assets.data.href")
    if year is None or not isinstance(asset_href, str):
        return None

    feature_geometry = shape(require_mapping(feature.get("geometry"), "feature.geometry"))
    overlap_area = feature_geometry.intersection(footprint).area
    if overlap_area <= 0:
        return None

    feature = json.loads(json.dumps(feature))
    properties = require_mapping(feature["properties"], "feature.properties")
    properties["query_year"] = year
    properties["asset_tiff_href"] = asset_href
    properties["asset_vrt_href"] = vrt_href_for_tiff(asset_href)
    properties["asset_vrt_href_source"] = "derived_from_tiff_href"
    properties["overlap_area_degrees2"] = overlap_area
    properties["footprint_overlap_fraction"] = overlap_area / footprint_area
    properties["selection_policy"] = DEFAULT_SELECTION_POLICY

    return CandidateFeature(
        feature=feature,
        year=year,
        asset_href=asset_href,
        overlap_area=overlap_area,
        footprint_overlap_fraction=overlap_area / footprint_area,
    )


def year_from_datetime(value: object) -> int | None:
    """Extract a year from OGR's datetime string."""
    if not isinstance(value, str):
        return None
    normalized = value.replace("/", "-").replace("+00", "+0000")
    for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(normalized, fmt).year
        except ValueError:
            continue
    return None


def vrt_href_for_tiff(href: str) -> str:
    """Return the expected VRT href for an AEF TIFF asset href."""
    if href.endswith(".tiff"):
        return f"{href[:-5]}.vrt"
    return f"{href}.vrt"


def write_query_outputs(
    selected_geojson: dict[str, Any],
    selected: list[CandidateFeature],
    candidate_counts: dict[int, int],
    query_config: AefCatalogQueryConfig,
    footprint: BaseGeometry,
    raw_candidate_count: int,
) -> None:
    """Write selected catalog rows as GeoParquet and a JSON summary."""
    query_config.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    query_config.output_summary.parent.mkdir(parents=True, exist_ok=True)
    query_config.output_parquet.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="kelp_aef_aef_catalog_") as temp_dir:
        selected_geojson_path = Path(temp_dir) / "selected.geojson"
        with selected_geojson_path.open("w") as file:
            json.dump(selected_geojson, file)

        ogr2ogr = shutil.which("ogr2ogr")
        if ogr2ogr is None:
            raise RuntimeError("ogr2ogr is required to write AEF catalog query GeoParquet")
        subprocess.run(
            [
                ogr2ogr,
                "-overwrite",
                "-f",
                "Parquet",
                str(query_config.output_parquet),
                str(selected_geojson_path),
            ],
            check=True,
        )

    summary = build_summary(
        selected=selected,
        candidate_counts=candidate_counts,
        query_config=query_config,
        footprint=footprint,
        raw_candidate_count=raw_candidate_count,
    )
    with query_config.output_summary.open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def build_selected_feature_collection(selected: list[CandidateFeature]) -> dict[str, Any]:
    """Build a GeoJSON feature collection for selected catalog rows."""
    return {
        "type": "FeatureCollection",
        "name": "aef_monterey_catalog_query",
        "features": [candidate.feature for candidate in selected],
    }


def build_summary(
    selected: list[CandidateFeature],
    candidate_counts: dict[int, int],
    query_config: AefCatalogQueryConfig,
    footprint: BaseGeometry,
    raw_candidate_count: int,
) -> dict[str, Any]:
    """Build a human-readable JSON summary for the query result."""
    selected_by_year = Counter(candidate.year for candidate in selected)
    return {
        "catalog_url": query_config.catalog_url,
        "layer_name": query_config.layer_name,
        "footprint_path": str(query_config.footprint_path),
        "footprint_bounds": footprint.bounds,
        "years": list(query_config.years),
        "selection_policy": query_config.selection_policy,
        "min_footprint_overlap_fraction": query_config.min_footprint_overlap_fraction,
        "raw_spatial_candidate_count": raw_candidate_count,
        "candidate_counts_by_year": {
            str(year): candidate_counts[year] for year in query_config.years
        },
        "selected_count": len(selected),
        "selected_counts_by_year": {
            str(year): selected_by_year.get(year, 0) for year in query_config.years
        },
        "selected_assets": [
            {
                "year": candidate.year,
                "asset_tiff_href": candidate.asset_href,
                "asset_vrt_href": vrt_href_for_tiff(candidate.asset_href),
                "overlap_area_degrees2": candidate.overlap_area,
                "footprint_overlap_fraction": candidate.footprint_overlap_fraction,
                "proj_epsg": candidate.feature["properties"].get("proj:epsg"),
            }
            for candidate in selected
        ],
        "output_parquet": str(query_config.output_parquet),
    }


def read_first_geometry(path: Path) -> BaseGeometry:
    """Read the first geometry from a vector file with GeoPandas."""
    geometries = gpd.read_file(path).geometry
    if geometries.empty:
        msg = f"GeoJSON does not contain features: {path}"
        raise ValueError(msg)
    geometry = cast(BaseGeometry | None, geometries.iloc[0])
    if geometry is None:
        msg = f"GeoJSON first feature does not contain a geometry: {path}"
        raise ValueError(msg)
    if geometry.is_empty:
        msg = f"GeoJSON feature contains an empty geometry: {path}"
        raise ValueError(msg)
    return geometry
