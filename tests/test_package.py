from pathlib import Path

import pytest

from kelp_aef import main
from kelp_aef.cli import COMMANDS, build_parser


def test_cli_imports() -> None:
    assert callable(main)
    assert build_parser().prog == "kelp-aef"
    assert set(COMMANDS) == {
        "smoke",
        "inspect-kelpwatch",
        "fetch-aef-chip",
        "build-labels",
        "align",
    }


def test_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: kelp-aef" in captured.out
    assert "inspect-kelpwatch" in captured.out


def test_subcommand_accepts_config(capsys: pytest.CaptureFixture[str]) -> None:
    config_path = Path("configs/monterey_smoke.yaml")

    assert main(["smoke", "--config", str(config_path)]) == 0

    captured = capsys.readouterr()
    assert f"smoke: using config {config_path}" in captured.out
