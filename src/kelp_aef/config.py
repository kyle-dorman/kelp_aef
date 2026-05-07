"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file as a mapping."""
    with path.open() as file:
        loaded = yaml.safe_load(file)

    if not isinstance(loaded, dict):
        msg = f"config must be a YAML mapping: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def require_mapping(value: object, name: str) -> dict[str, Any]:
    """Validate a dynamic config value as a mapping."""
    if not isinstance(value, dict):
        msg = f"config field must be a mapping: {name}"
        raise ValueError(msg)
    return cast(dict[str, Any], value)


def require_string(value: object, name: str) -> str:
    """Validate a dynamic config value as a string."""
    if not isinstance(value, str):
        msg = f"config field must be a string: {name}"
        raise ValueError(msg)
    return value
