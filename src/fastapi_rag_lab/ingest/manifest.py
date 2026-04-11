"""Read and write the ingestion manifest at data/raw/manifest.json."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {
    "commit_sha",
    "ingested_at",
    "file_count",
    "parent_count",
    "child_count",
    "embedding_model",
    "embedding_dimension",
    "collection_name",
}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomic write of manifest dict to JSON file."""
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(manifest, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    tmp_path.rename(path)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and validate manifest from disk."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Has the ingestion pipeline been run?"
        )

    with open(path) as f:
        manifest = json.load(f)

    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Check that all required keys are present and have expected types."""
    missing = REQUIRED_KEYS - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest is missing required keys: {sorted(missing)}")

    int_fields = {"file_count", "parent_count", "child_count", "embedding_dimension"}
    for field in int_fields:
        if not isinstance(manifest[field], int):
            raise TypeError(
                f"Expected int for '{field}', got {type(manifest[field]).__name__}"
            )
