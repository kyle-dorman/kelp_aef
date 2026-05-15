"""Cache manifest helpers for Phase 2 report diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class Phase2DiagnosticsCacheConfig:
    """Configured Phase 2 diagnostic cache artifact paths."""

    component_frame_dir: Path
    pooled_context_frame_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class Phase2CacheFreshness:
    """Result of validating a Phase 2 diagnostics cache manifest."""

    fresh: bool
    reason: str
    manifest: dict[str, Any] | None


def path_metadata(path: Path, *, content_hash: bool = False) -> dict[str, object]:
    """Return stable path metadata for cache freshness checks."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        sizes = [item.stat().st_size for item in files]
        mtimes = [item.stat().st_mtime_ns for item in files]
        return {
            "path": str(path),
            "exists": True,
            "kind": "directory",
            "file_count": len(files),
            "size_bytes": int(sum(sizes)),
            "max_mtime_ns": int(max(mtimes)) if mtimes else int(path.stat().st_mtime_ns),
        }
    stat = path.stat()
    metadata: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "kind": "file",
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        metadata["sha256"] = file_sha256(path)
    return metadata


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for a small file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload_hash(payload: dict[str, object]) -> str:
    """Return a deterministic hash for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_cache_manifest(path: Path) -> dict[str, Any] | None:
    """Load a cache manifest JSON object when it exists."""
    if not path.exists():
        return None
    with path.open() as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        msg = f"phase2 diagnostics cache manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def validate_cache_manifest(
    manifest_path: Path,
    expected_freshness_hash: str,
    required_paths: list[Path],
) -> Phase2CacheFreshness:
    """Validate manifest hash and required cache/table paths."""
    manifest = load_cache_manifest(manifest_path)
    if manifest is None:
        return Phase2CacheFreshness(False, "missing_manifest", None)
    if str(manifest.get("freshness_hash", "")) != expected_freshness_hash:
        return Phase2CacheFreshness(False, "stale_freshness_hash", manifest)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        manifest["missing_required_paths"] = missing
        return Phase2CacheFreshness(False, "missing_required_paths", manifest)
    return Phase2CacheFreshness(True, "fresh", manifest)


def write_cache_manifest(path: Path, payload: dict[str, object]) -> None:
    """Write the Phase 2 diagnostics cache manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
