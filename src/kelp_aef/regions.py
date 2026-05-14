"""Region naming helpers shared by report and QA commands."""

from __future__ import annotations

import re

MONTEREY_CONFIG_NAME = "monterey_peninsula"
MONTEREY_OUTPUT_SLUG = "monterey"


def region_output_slug(region_name: str) -> str:
    """Return the stable artifact slug for a configured region name."""
    if region_name == MONTEREY_CONFIG_NAME:
        return MONTEREY_OUTPUT_SLUG
    slug = re.sub(r"[^a-z0-9]+", "_", region_name.lower()).strip("_")
    if not slug:
        msg = f"region name cannot be converted to an output slug: {region_name!r}"
        raise ValueError(msg)
    return slug


def region_display_name(region_name: str) -> str:
    """Return a compact display label for a configured region name."""
    if region_name == MONTEREY_CONFIG_NAME:
        return "Monterey"
    return " ".join(part.capitalize() for part in region_output_slug(region_name).split("_"))
