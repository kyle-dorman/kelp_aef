"""Command-line interface for the kelp-aef workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/monterey_smoke.yaml"

COMMANDS: dict[str, str] = {
    "smoke": "Run the configured smoke workflow.",
    "inspect-kelpwatch": "Inspect Kelpwatch metadata for the configured region.",
    "fetch-aef-chip": "Fetch or stage AlphaEarth embedding samples for the configured region.",
    "build-labels": "Build derived Kelpwatch labels for the configured target.",
    "align": "Align AlphaEarth features and Kelpwatch labels into a training table.",
}


def validate_config_path(path: Path) -> Path:
    """Validate a parsed config path."""
    if not path.exists():
        msg = f"config path does not exist: {path}"
        raise argparse.ArgumentTypeError(msg)
    if not path.is_file():
        msg = f"config path is not a file: {path}"
        raise argparse.ArgumentTypeError(msg)
    return path


def existing_config_path(value: str) -> Path:
    """Parse and validate a config path argument."""
    return validate_config_path(Path(value))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="kelp-aef",
        description="Run AlphaEarth/Kelpwatch kelp mapping workflow steps.",
    )
    parser.add_argument("--version", action="version", version="kelp-aef 0.1.0")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for command, help_text in COMMANDS.items():
        subparser = subparsers.add_parser(command, help=help_text, description=help_text)
        subparser.add_argument(
            "--config",
            type=existing_config_path,
            default=DEFAULT_CONFIG,
            help="Path to a workflow config file. Defaults to the repo's "
            "configs/monterey_smoke.yaml.",
        )

    return parser


def run_scaffold_command(command: str, config_path: Path) -> int:
    """Run a CLI command scaffold until the pipeline step is implemented."""
    print(f"{command}: using config {config_path}")
    print("Pipeline implementation pending; see docs/todo.md for the next data-contract tasks.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kelp-aef CLI."""
    args = build_parser().parse_args(argv)

    command = args.command
    config_path = args.config

    if not isinstance(command, str):
        raise TypeError("parsed command must be a string")
    if not isinstance(config_path, Path):
        raise TypeError("parsed config path must be a pathlib.Path")

    config_path = validate_config_path(config_path)

    return run_scaffold_command(command, config_path)
