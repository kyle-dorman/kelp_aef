"""Command-line interface for the kelp-aef workflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from kelp_aef.alignment.feature_label_table import align_features_labels
from kelp_aef.evaluation.baselines import train_baselines
from kelp_aef.features.aef_catalog import query_aef_catalog
from kelp_aef.features.aef_download import download_aef
from kelp_aef.labels.kelpwatch import inspect_kelpwatch
from kelp_aef.labels.kelpwatch_labels import build_annual_labels
from kelp_aef.labels.kelpwatch_visualize import visualize_kelpwatch
from kelp_aef.viz.residual_maps import map_residuals

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
    "download-aef": "Download selected AEF tile assets from the catalog query.",
    "inspect-kelpwatch": "Inspect Kelpwatch metadata for the configured region.",
    "visualize-kelpwatch": "Visualize downloaded Kelpwatch source data for QA.",
    "fetch-aef-chip": "Fetch or stage AlphaEarth embedding samples for the configured region.",
    "build-labels": "Build derived Kelpwatch labels for the configured target.",
    "align": "Align AlphaEarth features and Kelpwatch labels into a training table.",
    "train-baselines": "Train and evaluate first simple tabular baselines.",
    "map-residuals": "Map baseline predictions, residuals, and area bias.",
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


def positive_float(value: str) -> float:
    """Parse a positive floating-point CLI value."""
    parsed = float(value)
    if parsed <= 0:
        msg = f"value must be positive: {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        msg = f"value must be positive: {value}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


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
        if command == "download-aef":
            add_download_aef_arguments(subparser)
        if command == "inspect-kelpwatch":
            add_inspect_kelpwatch_arguments(subparser)
        if command == "visualize-kelpwatch":
            add_visualize_kelpwatch_arguments(subparser)
        if command == "align":
            add_align_arguments(subparser)

    return parser


def add_download_aef_arguments(parser: argparse.ArgumentParser) -> None:
    """Add downloader-specific options to the download-aef subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a download plan without downloading files or updating metadata_summary.json.",
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip remote HEAD checks; useful with --dry-run for fast local validation.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files even when matching local files already exist.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote checks and downloads.",
    )


def add_inspect_kelpwatch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Kelpwatch-specific options to the inspect-kelpwatch subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve source metadata and write a manifest without downloading NetCDF data.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the NetCDF even when a matching local file already exists.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip local MD5 checksum calculation after download or local-file reuse.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for source metadata requests and downloads.",
    )


def add_visualize_kelpwatch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Kelpwatch-specific options to the visualize-kelpwatch subcommand."""
    parser.add_argument(
        "--variable",
        default=None,
        help="Optional NetCDF variable name to visualize instead of the manifest-selected label.",
    )
    parser.add_argument(
        "--preview-max-pixels",
        type=positive_int,
        default=500_000,
        help="Maximum pixels per layer in the self-contained HTML preview.",
    )


def add_align_arguments(parser: argparse.ArgumentParser) -> None:
    """Add feature/label alignment-specific options to the align subcommand."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a small spatial subset and write configured fast-path artifacts.",
    )
    parser.add_argument(
        "--years",
        type=positive_int,
        nargs="+",
        default=None,
        help="Optional year override, for example: --years 2022.",
    )
    parser.add_argument(
        "--max-stations",
        type=positive_int,
        default=None,
        help="Optional cap on spatially selected Kelpwatch stations.",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=None,
        help="Optional parquet output path override.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary CSV output path override.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest JSON output path override.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Optional fast-path method comparison CSV output path override.",
    )


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
        elif command == "download-aef":
            exit_code = download_aef(
                config_path,
                dry_run=bool(args.dry_run),
                skip_remote_checks=bool(args.skip_remote_checks),
                manifest_output=args.manifest_output,
                timeout_seconds=float(args.timeout_seconds),
                force=bool(args.force),
            )
        elif command == "inspect-kelpwatch":
            exit_code = inspect_kelpwatch(
                config_path,
                dry_run=bool(args.dry_run),
                manifest_output=args.manifest_output,
                force=bool(args.force),
                skip_checksum=bool(args.skip_checksum),
                timeout_seconds=float(args.timeout_seconds),
            )
        elif command == "visualize-kelpwatch":
            exit_code = visualize_kelpwatch(
                config_path,
                variable=args.variable,
                preview_max_pixels=int(args.preview_max_pixels),
            )
        elif command == "build-labels":
            exit_code = build_annual_labels(config_path)
        elif command == "align":
            exit_code = align_features_labels(
                config_path,
                fast=bool(args.fast),
                years=tuple(args.years) if args.years is not None else None,
                max_stations=args.max_stations,
                output_table=args.output_table,
                summary_output=args.summary_output,
                manifest_output=args.manifest_output,
                comparison_output=args.comparison_output,
            )
        elif command == "train-baselines":
            exit_code = train_baselines(config_path)
        elif command == "map-residuals":
            exit_code = map_residuals(config_path)
        else:
            exit_code = run_scaffold_command(command, config_path)
    except Exception:
        LOGGER.exception("Command failed: %s", command)
        raise

    LOGGER.info("Finished %s with exit code %s", command, exit_code)
    return exit_code
