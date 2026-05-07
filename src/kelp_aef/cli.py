"""Command-line interface for the kelp-aef workflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from kelp_aef.features.aef_catalog import query_aef_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/monterey_smoke.yaml"
DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

LOGGER = logging.getLogger(__name__)

COMMANDS: dict[str, str] = {
    "smoke": "Run the configured smoke workflow.",
    "query-aef-catalog": "Query the AEF STAC catalog for configured assets.",
    "inspect-kelpwatch": "Inspect Kelpwatch metadata for the configured region.",
    "fetch-aef-chip": "Fetch or stage AlphaEarth embedding samples for the configured region.",
    "build-labels": "Build derived Kelpwatch labels for the configured target.",
    "align": "Align AlphaEarth features and Kelpwatch labels into a training table.",
}


def log_level_name(value: str) -> str:
    """Normalize and validate a CLI logging level name."""
    level_name = value.upper()
    if level_name not in LOG_LEVEL_NAMES:
        valid_levels = ", ".join(LOG_LEVEL_NAMES)
        msg = f"invalid log level {value!r}; choose one of: {valid_levels}"
        raise argparse.ArgumentTypeError(msg)
    return level_name


def configure_logging(level_name: str) -> None:
    """Configure root logging for a single CLI invocation."""
    logging.basicConfig(
        level=getattr(logging, level_name),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )


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
    parser.add_argument(
        "--log-level",
        type=log_level_name,
        default=DEFAULT_LOG_LEVEL,
        metavar="LEVEL",
        help="Logging level for CLI progress messages. Defaults to INFO.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for command, help_text in COMMANDS.items():
        subparser = subparsers.add_parser(command, help=help_text, description=help_text)
        subparser.add_argument(
            "--log-level",
            type=log_level_name,
            default=argparse.SUPPRESS,
            metavar="LEVEL",
            help="Logging level for CLI progress messages.",
        )
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
    LOGGER.info("%s: using config %s", command, config_path)
    LOGGER.warning(
        "Pipeline implementation pending; see docs/todo.md for the next data-contract tasks."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kelp-aef CLI."""
    args = build_parser().parse_args(argv)
    log_level = args.log_level

    command = args.command
    config_path = args.config

    if not isinstance(log_level, str):
        raise TypeError("parsed log level must be a string")
    if not isinstance(command, str):
        raise TypeError("parsed command must be a string")
    if not isinstance(config_path, Path):
        raise TypeError("parsed config path must be a pathlib.Path")

    configure_logging(log_level)
    config_path = validate_config_path(config_path)
    LOGGER.info("Starting %s with config %s", command, config_path)

    try:
        if command == "query-aef-catalog":
            exit_code = query_aef_catalog(config_path)
        else:
            exit_code = run_scaffold_command(command, config_path)
    except Exception:
        LOGGER.exception("Command failed: %s", command)
        raise

    LOGGER.info("Finished %s with exit code %s", command, exit_code)
    return exit_code
