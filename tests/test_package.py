from pathlib import Path

import pytest

from kelp_aef import main
from kelp_aef.cli import COMMANDS, DEFAULT_CONFIG, build_parser


def test_cli_imports() -> None:
    """Verify the package entrypoint and expected command registry are importable."""
    assert callable(main)
    assert build_parser().prog == "kelp-aef"
    assert set(COMMANDS) == {
        "smoke",
        "query-aef-catalog",
        "download-aef",
        "query-noaa-cudem",
        "download-noaa-cudem",
        "query-noaa-cusp",
        "download-noaa-cusp",
        "query-noaa-crm",
        "download-noaa-crm",
        "query-usgs-3dep",
        "download-usgs-3dep",
        "inspect-kelpwatch",
        "visualize-kelpwatch",
        "fetch-aef-chip",
        "build-labels",
        "align",
        "align-full-grid",
        "train-baselines",
        "predict-full-grid",
        "map-residuals",
        "analyze-model",
    }


def test_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI help path exits cleanly and lists workflow commands."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: kelp-aef" in captured.out
    assert "inspect-kelpwatch" in captured.out
    assert "visualize-kelpwatch" in captured.out


def test_subcommand_accepts_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify scaffold commands accept an explicit config path and log it."""
    config_path = Path("configs/monterey_smoke.yaml")

    assert main(["smoke", "--config", str(config_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"smoke: using config {config_path}" in captured.err


def test_default_config_resolves_outside_repo(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the default config path remains absolute and logged after cwd changes."""
    monkeypatch.chdir(tmp_path)

    assert main(["smoke"]) == 0

    captured = capsys.readouterr()
    assert DEFAULT_CONFIG.is_absolute()
    assert captured.out == ""
    assert f"smoke: using config {DEFAULT_CONFIG}" in captured.err


def test_subcommand_log_level_can_suppress_info(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify subcommand-position log-level arguments can suppress progress logs."""
    assert main(["smoke", "--log-level", "ERROR"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
