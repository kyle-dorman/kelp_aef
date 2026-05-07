import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr  # type: ignore[import-untyped]

from kelp_aef.labels import kelpwatch
from kelp_aef.labels.kelpwatch import (
    inspect_kelpwatch,
    parse_eml_source_metadata,
    parse_entity_id,
    parse_latest_revision,
)


def test_parse_latest_revision() -> None:
    """Parse the EDI latest-revision response."""
    assert parse_latest_revision("32\n") == 32


def test_parse_entity_id() -> None:
    """Parse the first EDI data entity id response line."""
    assert parse_entity_id("abc123\n") == "abc123"


def test_parse_eml_source_metadata_records_cumulative_revision() -> None:
    """Read cumulative Kelpwatch source metadata from EML XML."""
    source = parse_eml_source_metadata(
        metadata_xml=eml_xml(
            revision=32,
            object_name="LandsatKelpBiomass_2025_Q4_withmetadata.nc",
            size_bytes=123,
            begin_date="1984-03-23",
            end_date="2025-12-31",
        ),
        latest_revision_url="https://example.test/latest",
        entity_list_url="https://example.test/entities",
        metadata_url="https://example.test/metadata",
        checksum_url="https://example.test/checksum",
        revision=32,
        entity_id=kelpwatch.KELPWATCH_ENTITY_ID,
        pasta_checksum="sha1ish",
    )

    assert source.package_id == "knb-lter-sbc.74.32"
    assert source.object_name == "LandsatKelpBiomass_2025_Q4_withmetadata.nc"
    assert source.temporal_begin == "1984-03-23"
    assert source.temporal_end == "2025-12-31"
    assert source.size_bytes == 123
    assert source.eml_md5 == "fake-md5"


def test_inspect_kelpwatch_dry_run_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise Kelpwatch source discovery without downloading the NetCDF."""
    config_path = write_kelpwatch_config(tmp_path)
    manifest_path = tmp_path / "dry_manifest.json"
    install_fake_session(monkeypatch, object_name="kelp.nc", size_bytes=123)

    assert (
        inspect_kelpwatch(
            config_path,
            dry_run=True,
            manifest_output=manifest_path,
            skip_checksum=True,
        )
        == 0
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dry_run"] is True
    assert manifest["source"]["package_id"] == "knb-lter-sbc.74.32"
    assert manifest["transfer"]["status"] == "dry_run"
    assert manifest["netcdf"]["validation_status"] == "not_checked_dry_run"


def test_inspect_kelpwatch_existing_netcdf_updates_metadata_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Inspect a local tiny NetCDF and identify the annual-max label value field."""
    raw_dir = tmp_path / "raw/kelpwatch"
    local_path = raw_dir / "kelp.nc"
    write_tiny_kelpwatch_netcdf(local_path)
    config_path = write_kelpwatch_config(tmp_path)
    install_fake_session(monkeypatch, object_name="kelp.nc", size_bytes=local_path.stat().st_size)

    assert inspect_kelpwatch(config_path, skip_checksum=True) == 0

    manifest = json.loads((tmp_path / "interim/kelpwatch_manifest.json").read_text())
    metadata_summary = json.loads((tmp_path / "interim/metadata_summary.json").read_text())
    assert manifest["transfer"]["status"] == "skipped_existing"
    assert manifest["netcdf"]["validation_status"] == "valid"
    assert manifest["label_source"]["selected_variable"] == "kelp_area"
    assert metadata_summary["kelpwatch"]["label_source"]["selected_variable"] == "kelp_area"


def test_download_file_resumes_partial_after_stream_failure(tmp_path: Path) -> None:
    """Resume a partial Kelpwatch download after a broken streaming response."""
    local_path = tmp_path / "kelp.nc"
    session = DownloadSession(
        [
            StreamingResponse(
                chunks=[b"abc"],
                status_code=200,
                error=kelpwatch.requests.exceptions.ChunkedEncodingError("broken"),
            ),
            StreamingResponse(chunks=[b"def"], status_code=206),
        ]
    )

    kelpwatch.download_file(
        session,  # type: ignore[arg-type]
        "https://example.test/kelp.nc",
        local_path,
        timeout_seconds=30.0,
        chunk_size_bytes=3,
        expected_size_bytes=6,
    )

    assert local_path.read_bytes() == b"abcdef"
    assert session.calls[0]["headers"] is None
    assert session.calls[1]["headers"] == {"Range": "bytes=3-"}


def install_fake_session(
    monkeypatch: pytest.MonkeyPatch, *, object_name: str, size_bytes: int
) -> None:
    """Install a fake requests.Session factory for deterministic endpoint tests."""
    fake_session = FakeSession(
        {
            kelpwatch.latest_revision_url(): FakeResponse("32\n"),
            kelpwatch.data_entity_list_url(32): FakeResponse(f"{kelpwatch.KELPWATCH_ENTITY_ID}\n"),
            kelpwatch.package_metadata_url(32): FakeResponse(
                eml_xml(
                    revision=32,
                    object_name=object_name,
                    size_bytes=size_bytes,
                    begin_date="1984-03-23",
                    end_date="2025-12-31",
                )
            ),
            kelpwatch.data_checksum_url(32, kelpwatch.KELPWATCH_ENTITY_ID): FakeResponse(
                "sha1ish\n"
            ),
        }
    )
    monkeypatch.setattr(kelpwatch.requests, "Session", lambda: fake_session)


class FakeSession:
    """Small requests.Session test double keyed by URL."""

    def __init__(self, responses: dict[str, "FakeResponse"]) -> None:
        """Store fake responses by URL."""
        self.responses = responses

    def __enter__(self) -> "FakeSession":
        """Return this fake session as a context manager value."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the context manager without cleanup."""
        return None

    def get(self, url: str, **_kwargs: Any) -> "FakeResponse":
        """Return the fake response registered for a URL."""
        return self.responses[url]


class FakeResponse:
    """Small requests.Response test double with text and byte streaming support."""

    def __init__(self, text: str, content: bytes = b"") -> None:
        """Store fake text and byte content."""
        self.text = text
        self.content = content or text.encode()

    def __enter__(self) -> "FakeResponse":
        """Return this fake response as a context manager value."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the context manager without cleanup."""
        return None

    def raise_for_status(self) -> None:
        """No-op success status hook."""
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        """Yield fake response bytes in requested chunk sizes."""
        return [
            self.content[index : index + chunk_size]
            for index in range(0, len(self.content), chunk_size)
        ]


class DownloadSession:
    """Small session test double that returns streaming responses in sequence."""

    def __init__(self, responses: list["StreamingResponse"]) -> None:
        """Store queued streaming responses and observed call metadata."""
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> "StreamingResponse":
        """Return the next streaming response and record request kwargs."""
        self.calls.append({"url": url, "headers": kwargs.get("headers")})
        return self.responses.pop(0)


class StreamingResponse:
    """Streaming response test double that can fail after yielding chunks."""

    def __init__(
        self,
        *,
        chunks: list[bytes],
        status_code: int,
        error: Exception | None = None,
    ) -> None:
        """Store chunks, status code, and an optional streaming error."""
        self.chunks = chunks
        self.status_code = status_code
        self.error = error

    def __enter__(self) -> "StreamingResponse":
        """Return this fake response as a context manager value."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Exit the context manager without cleanup."""
        return None

    def raise_for_status(self) -> None:
        """No-op success status hook."""
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        """Yield chunks and optionally raise a streaming error."""
        del chunk_size
        yield from self.chunks
        if self.error is not None:
            raise self.error


def write_kelpwatch_config(tmp_path: Path) -> Path:
    """Write the minimal YAML config needed by the Kelpwatch inspector."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
years:
  smoke: [2018, 2019]
labels:
  target: kelp_max_y
  aggregation: annual_max
  paths:
    raw_dir: {tmp_path / "raw/kelpwatch"}
    source_manifest: {tmp_path / "interim/kelpwatch_manifest.json"}
reports:
  outputs:
    metadata_summary: {tmp_path / "interim/metadata_summary.json"}
""".lstrip()
    )
    return config_path


def write_tiny_kelpwatch_netcdf(path: Path) -> None:
    """Write a tiny NetCDF fixture with a kelp canopy area variable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "kelp_area": (
                ("time", "y", "x"),
                np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32),
                {"units": "m2", "long_name": "kelp canopy area"},
            ),
            "kelp_biomass": (
                ("time", "y", "x"),
                np.array([[[0.0, 10.0], [20.0, 30.0]]], dtype=np.float32),
                {"units": "kg", "long_name": "kelp canopy biomass"},
            ),
        },
        coords={
            "time": np.array([2018], dtype=np.int16),
            "x": np.array([-122.0, -121.9], dtype=np.float32),
            "y": np.array([36.5, 36.6], dtype=np.float32),
        },
        attrs={"crs": "EPSG:4326"},
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def eml_xml(
    *,
    revision: int,
    object_name: str,
    size_bytes: int,
    begin_date: str,
    end_date: str,
) -> str:
    """Build a small EML XML fixture with the Kelpwatch source fields."""
    return f"""
<eml:eml packageId="knb-lter-sbc.74.{revision}"
  xmlns:eml="eml://ecoinformatics.org/eml-2.2.0">
  <dataset>
    <alternateIdentifier system="https://doi.org">doi:fake</alternateIdentifier>
    <title>Kelpwatch fake metadata</title>
    <pubDate>2026-01-30</pubDate>
    <coverage>
      <geographicCoverage>
        <boundingCoordinates>
          <westBoundingCoordinate>-124.77</westBoundingCoordinate>
          <eastBoundingCoordinate>-114.04</eastBoundingCoordinate>
          <northBoundingCoordinate>48.40</northBoundingCoordinate>
          <southBoundingCoordinate>27.01</southBoundingCoordinate>
        </boundingCoordinates>
      </geographicCoverage>
      <temporalCoverage>
        <rangeOfDates>
          <beginDate><calendarDate>{begin_date}</calendarDate></beginDate>
          <endDate><calendarDate>{end_date}</calendarDate></endDate>
        </rangeOfDates>
      </temporalCoverage>
    </coverage>
    <otherEntity id="{kelpwatch.KELPWATCH_ENTITY_ID}" scope="document">
      <entityName>Satellite kelp biomass since 1984</entityName>
      <physical>
        <objectName>{object_name}</objectName>
        <size unit="byte">{size_bytes}</size>
        <authentication method="MD5">fake-md5</authentication>
        <dataFormat>
          <externallyDefinedFormat>
            <formatName>NetCDF</formatName>
          </externallyDefinedFormat>
        </dataFormat>
        <distribution>
          <online>
            <url function="download">
              {kelpwatch.data_download_url(revision, kelpwatch.KELPWATCH_ENTITY_ID)}
            </url>
          </online>
        </distribution>
      </physical>
    </otherEntity>
  </dataset>
</eml:eml>
""".strip()
