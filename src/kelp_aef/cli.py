"""Command-line interface for the kelp-aef workflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from kelp_aef.alignment.feature_label_table import align_features_labels
from kelp_aef.alignment.full_grid import align_full_grid, build_model_input_sample
from kelp_aef.domain.crm_alignment import align_noaa_crm
from kelp_aef.domain.domain_mask import build_domain_mask
from kelp_aef.domain.noaa_crm import download_noaa_crm, query_noaa_crm
from kelp_aef.domain.noaa_cudem import download_noaa_cudem, query_noaa_cudem
from kelp_aef.domain.noaa_cusp import download_noaa_cusp, query_noaa_cusp
from kelp_aef.domain.usgs_3dep import download_usgs_3dep, query_usgs_3dep
from kelp_aef.evaluation.baselines import (
    predict_full_grid,
    train_baselines,
    write_configured_split_manifest,
)
from kelp_aef.evaluation.binary_presence import calibrate_binary_presence, train_binary_presence
from kelp_aef.evaluation.conditional_canopy import train_conditional_canopy
from kelp_aef.evaluation.hurdle import compose_hurdle_model
from kelp_aef.evaluation.model_analysis import analyze_model, build_phase2_diagnostics
from kelp_aef.evaluation.pooled_regions import (
    build_pooled_region_sample,
    write_training_regime_comparison,
)
from kelp_aef.evaluation.transfer import evaluate_transfer
from kelp_aef.features.aef_catalog import query_aef_catalog
from kelp_aef.features.aef_download import download_aef
from kelp_aef.labels.kelpwatch import inspect_kelpwatch
from kelp_aef.labels.kelpwatch_labels import build_annual_labels
from kelp_aef.labels.kelpwatch_visualize import visualize_kelpwatch
from kelp_aef.viz.residual_maps import map_residuals
from kelp_aef.viz.results_visualizer import visualize_results
from kelp_aef.viz.source_coverage import visualize_source_coverage

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
    "query-noaa-cudem": "Query the NOAA CUDEM tile index for the configured region.",
    "download-noaa-cudem": "Download selected NOAA CUDEM tiles from the query manifest.",
    "query-noaa-cusp": "Query the NOAA CUSP shoreline source for the configured region.",
    "download-noaa-cusp": "Download selected NOAA CUSP shoreline sources from the query manifest.",
    "query-noaa-crm": "Query NOAA CRM topo-bathy sources for the configured region.",
    "download-noaa-crm": "Download selected NOAA CRM sources from the query manifest.",
    "align-noaa-crm": "Align NOAA CRM topo-bathy values to the target grid.",
    "build-domain-mask": "Build a plausible-kelp domain mask from aligned CRM support.",
    "query-usgs-3dep": "Query USGS 3DEP DEM sources for the configured region.",
    "download-usgs-3dep": "Download selected USGS 3DEP DEM sources from the query manifest.",
    "inspect-kelpwatch": "Inspect Kelpwatch metadata for the configured region.",
    "visualize-kelpwatch": "Visualize downloaded Kelpwatch source data for QA.",
    "visualize-source-coverage": "Summarize and visualize configured source coverage for QA.",
    "fetch-aef-chip": "Fetch or stage AlphaEarth embedding samples for the configured region.",
    "build-labels": "Build derived Kelpwatch labels for the configured target.",
    "align": "Align AlphaEarth features and Kelpwatch labels into a training table.",
    "align-full-grid": "Align AlphaEarth features with full-grid background labels.",
    "build-model-input-sample": "Build the retained-domain model-input sample.",
    "write-split-manifest": "Write train/validation/test row assignments without training.",
    "train-baselines": "Train and evaluate first simple tabular baselines.",
    "predict-full-grid": "Stream baseline predictions over the full-grid feature table.",
    "train-binary-presence": "Train the balanced binary annual-max presence model.",
    "calibrate-binary-presence": "Calibrate binary annual-max probabilities and thresholds.",
    "train-conditional-canopy": "Train the positive-only conditional canopy amount model.",
    "compose-hurdle-model": "Compose calibrated presence and conditional canopy predictions.",
    "evaluate-transfer": "Evaluate a frozen source-region policy on the target region.",
    "build-pooled-region-sample": "Build a pooled sample across configured regions.",
    "compare-training-regimes": "Combine local, transfer, and pooled training-regime summaries.",
    "build-phase2-diagnostics": "Build cached Phase 2 diagnostic frames and tables.",
    "map-residuals": "Map baseline predictions, residuals, and area bias.",
    "visualize-results": "Build a local interactive viewer for labels and model results.",
    "analyze-model": "Analyze model behavior and write the Phase 1 report.",
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
        if command == "query-noaa-cudem":
            add_query_noaa_cudem_arguments(subparser)
        if command == "download-noaa-cudem":
            add_download_noaa_cudem_arguments(subparser)
        if command == "query-noaa-cusp":
            add_query_noaa_cusp_arguments(subparser)
        if command == "download-noaa-cusp":
            add_download_noaa_cusp_arguments(subparser)
        if command == "query-noaa-crm":
            add_query_noaa_crm_arguments(subparser)
        if command == "download-noaa-crm":
            add_download_noaa_crm_arguments(subparser)
        if command == "align-noaa-crm":
            add_align_noaa_crm_arguments(subparser)
        if command == "build-domain-mask":
            add_build_domain_mask_arguments(subparser)
        if command == "query-usgs-3dep":
            add_query_usgs_3dep_arguments(subparser)
        if command == "download-usgs-3dep":
            add_download_usgs_3dep_arguments(subparser)
        if command == "inspect-kelpwatch":
            add_inspect_kelpwatch_arguments(subparser)
        if command == "visualize-kelpwatch":
            add_visualize_kelpwatch_arguments(subparser)
        if command == "align":
            add_align_arguments(subparser)
        if command == "align-full-grid":
            add_align_full_grid_arguments(subparser)
        if command == "build-model-input-sample":
            add_build_model_input_sample_arguments(subparser)
        if command == "predict-full-grid":
            add_predict_full_grid_arguments(subparser)
        if command == "evaluate-transfer":
            add_evaluate_transfer_arguments(subparser)
        if command == "build-phase2-diagnostics":
            add_build_phase2_diagnostics_arguments(subparser)
        if command == "analyze-model":
            add_analyze_model_arguments(subparser)

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


def add_download_noaa_cudem_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA Coastal DEM-specific options to the download-noaa-cudem subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build a source manifest plan without downloading data or updating "
            "metadata_summary.json."
        ),
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
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional NOAA CUDEM query manifest path override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the NetCDF even when a matching local file already exists.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote checks and downloads.",
    )


def add_query_noaa_cudem_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CUDEM tile-index query options to the query-noaa-cudem subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a query plan without downloading the tile index.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional query manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--tile-index-path",
        type=Path,
        default=None,
        help="Optional local CUDEM tile-index path override.",
    )
    parser.add_argument(
        "--download-index",
        action="store_true",
        help="Download the CUDEM tile-index ZIP if it is not already local.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the tile index even when a local file already exists.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote checks and downloads.",
    )


def add_query_noaa_cusp_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CUSP source-query options to the query-noaa-cusp subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a query plan without checking or downloading the CUSP source package.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional query manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip remote HEAD checks during non-dry-run queries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote checks.",
    )


def add_download_noaa_cusp_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CUSP source-download options to the download-noaa-cusp subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build a source manifest plan without downloading data or updating "
            "metadata_summary.json."
        ),
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip remote HEAD checks; useful for fast local validation.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional NOAA CUSP query manifest path override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the CUSP source package even when a local file already exists.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote checks and downloads.",
    )


def add_query_noaa_crm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CRM source-query options to the query-noaa-crm subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a query plan without checking remote CRM metadata.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional query manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip THREDDS/OPeNDAP metadata checks during non-dry-run queries.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for remote metadata requests.",
    )


def add_download_noaa_crm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CRM source-download options to the download-noaa-crm subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build a source manifest plan without downloading CRM data or updating "
            "metadata_summary.json."
        ),
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
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional NOAA CRM query manifest path override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the CRM source file even when a local file already exists.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=120.0,
        help="HTTP read timeout in seconds for remote checks and downloads.",
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=5,
        help="Maximum attempts per CRM source file, resuming partial downloads when possible.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=positive_float,
        default=5.0,
        help="Base retry backoff in seconds between failed CRM download attempts.",
    )


def add_align_noaa_crm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NOAA CRM alignment options to the align-noaa-crm subcommand."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the configured small target-grid window and fast-path artifacts.",
    )


def add_build_domain_mask_arguments(parser: argparse.ArgumentParser) -> None:
    """Add plausible-kelp domain-mask options to the build-domain-mask command."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the configured small target-grid window and fast-path artifacts.",
    )


def add_query_usgs_3dep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add USGS 3DEP source-query options to the query-usgs-3dep subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a query plan without calling TNMAccess or downloading DEM data.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional query manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_float,
        default=30.0,
        help="HTTP timeout in seconds for TNMAccess metadata requests.",
    )


def add_download_usgs_3dep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add USGS 3DEP source-download options to the download-usgs-3dep subcommand."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build a source manifest plan without downloading DEM data or updating "
            "metadata_summary.json."
        ),
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip remote HEAD checks; useful for fast local validation.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest output path, useful for dry-run plans.",
    )
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=None,
        help="Optional USGS 3DEP query manifest path override.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the 3DEP source raster even when a local file already exists.",
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


def add_align_full_grid_arguments(parser: argparse.ArgumentParser) -> None:
    """Add full-grid alignment options to the align-full-grid subcommand."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the configured small full-grid window and fast-path artifacts.",
    )


def add_build_model_input_sample_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-input sample options to the build-model-input-sample subcommand."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Build the configured fast retained-domain model-input sample.",
    )


def add_predict_full_grid_arguments(parser: argparse.ArgumentParser) -> None:
    """Add full-grid prediction options to the predict-full-grid subcommand."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Write full-grid predictions for the configured fast inference artifact.",
    )


def add_evaluate_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add transfer-evaluation options to the evaluate-transfer subcommand."""
    parser.add_argument(
        "--source-config",
        type=existing_config_path,
        required=True,
        help="Path to the frozen source-region workflow config.",
    )
    parser.add_argument(
        "--transfer-name",
        default=None,
        help="Optional models.transfer entry name. Defaults to a source-region match.",
    )


def add_build_phase2_diagnostics_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cache-building options to the build-phase2-diagnostics subcommand."""
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Phase 2 diagnostic frames even when the cache manifest is fresh.",
    )


def add_analyze_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model-analysis report iteration options."""
    parser.add_argument(
        "--reuse-phase2-diagnostics",
        action="store_true",
        default=None,
        help="Reuse a fresh Phase 2 diagnostics cache instead of rebuilding row annotations.",
    )
    parser.add_argument(
        "--refresh-phase2-diagnostics",
        action="store_true",
        help="Rebuild Phase 2 diagnostic frame caches before rendering the report.",
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
        elif command == "query-noaa-cudem":
            exit_code = query_noaa_cudem(
                config_path,
                dry_run=bool(args.dry_run),
                manifest_output=args.manifest_output,
                tile_index_path=args.tile_index_path,
                download_index=bool(args.download_index),
                timeout_seconds=float(args.timeout_seconds),
                force=bool(args.force),
            )
        elif command == "download-noaa-cudem":
            exit_code = download_noaa_cudem(
                config_path,
                dry_run=bool(args.dry_run),
                skip_remote_checks=bool(args.skip_remote_checks),
                manifest_output=args.manifest_output,
                query_manifest=args.query_manifest,
                timeout_seconds=float(args.timeout_seconds),
                force=bool(args.force),
            )
        elif command == "query-noaa-cusp":
            exit_code = query_noaa_cusp(
                config_path,
                dry_run=bool(args.dry_run),
                manifest_output=args.manifest_output,
                timeout_seconds=float(args.timeout_seconds),
                skip_remote_checks=bool(args.skip_remote_checks),
            )
        elif command == "download-noaa-cusp":
            exit_code = download_noaa_cusp(
                config_path,
                dry_run=bool(args.dry_run),
                skip_remote_checks=bool(args.skip_remote_checks),
                manifest_output=args.manifest_output,
                query_manifest=args.query_manifest,
                timeout_seconds=float(args.timeout_seconds),
                force=bool(args.force),
            )
        elif command == "query-noaa-crm":
            exit_code = query_noaa_crm(
                config_path,
                dry_run=bool(args.dry_run),
                manifest_output=args.manifest_output,
                timeout_seconds=float(args.timeout_seconds),
                skip_remote_checks=bool(args.skip_remote_checks),
            )
        elif command == "download-noaa-crm":
            exit_code = download_noaa_crm(
                config_path,
                dry_run=bool(args.dry_run),
                skip_remote_checks=bool(args.skip_remote_checks),
                manifest_output=args.manifest_output,
                query_manifest=args.query_manifest,
                timeout_seconds=float(args.timeout_seconds),
                force=bool(args.force),
                max_attempts=int(args.max_attempts),
                retry_backoff_seconds=float(args.retry_backoff_seconds),
            )
        elif command == "align-noaa-crm":
            exit_code = align_noaa_crm(config_path, fast=bool(args.fast))
        elif command == "build-domain-mask":
            exit_code = build_domain_mask(config_path, fast=bool(args.fast))
        elif command == "query-usgs-3dep":
            exit_code = query_usgs_3dep(
                config_path,
                dry_run=bool(args.dry_run),
                manifest_output=args.manifest_output,
                timeout_seconds=float(args.timeout_seconds),
            )
        elif command == "download-usgs-3dep":
            exit_code = download_usgs_3dep(
                config_path,
                dry_run=bool(args.dry_run),
                skip_remote_checks=bool(args.skip_remote_checks),
                manifest_output=args.manifest_output,
                query_manifest=args.query_manifest,
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
        elif command == "visualize-source-coverage":
            exit_code = visualize_source_coverage(config_path)
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
        elif command == "align-full-grid":
            exit_code = align_full_grid(config_path, fast=bool(args.fast))
        elif command == "build-model-input-sample":
            exit_code = build_model_input_sample(config_path, fast=bool(args.fast))
        elif command == "write-split-manifest":
            exit_code = write_configured_split_manifest(config_path)
        elif command == "train-baselines":
            exit_code = train_baselines(config_path)
        elif command == "predict-full-grid":
            exit_code = predict_full_grid(config_path, fast=bool(args.fast))
        elif command == "train-binary-presence":
            exit_code = train_binary_presence(config_path)
        elif command == "calibrate-binary-presence":
            exit_code = calibrate_binary_presence(config_path)
        elif command == "train-conditional-canopy":
            exit_code = train_conditional_canopy(config_path)
        elif command == "compose-hurdle-model":
            exit_code = compose_hurdle_model(config_path)
        elif command == "evaluate-transfer":
            exit_code = evaluate_transfer(
                config_path,
                source_config_path=args.source_config,
                transfer_name=args.transfer_name,
            )
        elif command == "build-pooled-region-sample":
            exit_code = build_pooled_region_sample(config_path)
        elif command == "compare-training-regimes":
            exit_code = write_training_regime_comparison(config_path)
        elif command == "build-phase2-diagnostics":
            exit_code = build_phase2_diagnostics(config_path, force=bool(args.force))
        elif command == "map-residuals":
            exit_code = map_residuals(config_path)
        elif command == "visualize-results":
            exit_code = visualize_results(config_path)
        elif command == "analyze-model":
            if args.reuse_phase2_diagnostics is True and bool(args.refresh_phase2_diagnostics):
                msg = (
                    "--reuse-phase2-diagnostics and --refresh-phase2-diagnostics "
                    "cannot be used together"
                )
                raise ValueError(msg)
            exit_code = analyze_model(
                config_path,
                reuse_phase2_diagnostics=args.reuse_phase2_diagnostics,
                refresh_phase2_diagnostics=bool(args.refresh_phase2_diagnostics),
            )
        else:
            exit_code = run_scaffold_command(command, config_path)
    except Exception:
        LOGGER.exception("Command failed: %s", command)
        raise

    LOGGER.info("Finished %s with exit code %s", command, exit_code)
    return exit_code
