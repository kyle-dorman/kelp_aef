from shapely.geometry import Polygon

from kelp_aef.features.aef_catalog import (
    CandidateFeature,
    select_catalog_features,
    vrt_href_for_tiff,
    year_from_datetime,
)


def test_year_from_ogr_datetime() -> None:
    """Extract years from the datetime strings OGR returns for STAC rows."""
    assert year_from_datetime("2022/01/01 00:00:00+00") == 2022
    assert year_from_datetime("2022-01-01T00:00:00+0000") == 2022
    assert year_from_datetime(None) is None


def test_vrt_href_for_tiff() -> None:
    """Derive a sibling VRT asset href from a TIFF asset href."""
    assert vrt_href_for_tiff("s3://bucket/path/tile.tiff") == "s3://bucket/path/tile.vrt"


def test_select_catalog_features_uses_max_overlap_per_year() -> None:
    """Select the configured year's candidate with the largest footprint overlap."""
    footprint = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    full_overlap = _feature(
        year="2022",
        href="s3://bucket/a.tiff",
        ring=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
    )
    partial_overlap = _feature(
        year="2022",
        href="s3://bucket/b.tiff",
        ring=[(1.5, 0.0), (3.0, 0.0), (3.0, 2.0), (1.5, 2.0)],
    )

    selected, candidate_counts = select_catalog_features(
        catalog_features=[partial_overlap, full_overlap],
        footprint=footprint,
        footprint_area=footprint.area,
        years=(2022,),
        min_footprint_overlap_fraction=0.0,
    )

    assert candidate_counts == {2022: 2}
    assert [candidate.asset_href for candidate in selected] == ["s3://bucket/a.tiff"]
    assert isinstance(selected[0], CandidateFeature)


def _feature(year: str, href: str, ring: list[tuple[float, float]]) -> dict[str, object]:
    """Build a minimal STAC-like GeoJSON feature for catalog unit tests."""
    closed_ring = [[x, y] for x, y in [*ring, ring[0]]]
    return {
        "type": "Feature",
        "properties": {
            "datetime": f"{year}/01/01 00:00:00+00",
            "assets.data.href": href,
            "proj:epsg": "EPSG:32610",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [closed_ring],
        },
    }
