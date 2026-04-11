import json

import pytest

from fastapi_rag_lab.ingest.manifest import (
    read_manifest,
    validate_manifest,
    write_manifest,
)

VALID_MANIFEST = {
    "commit_sha": "abc123",
    "ingested_at": "2026-04-09T18:30:00+00:00",
    "file_count": 142,
    "parent_count": 487,
    "child_count": 1834,
    "embedding_model": "nomic-embed-text",
    "embedding_dimension": 768,
    "collection_name": "fastapi_docs_v1",
}


def test_roundtrip(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, VALID_MANIFEST)

    loaded = read_manifest(manifest_path)
    assert loaded == VALID_MANIFEST


def test_write_is_atomic(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, VALID_MANIFEST)

    raw = json.loads(manifest_path.read_text())
    assert raw["commit_sha"] == "abc123"


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        read_manifest(tmp_path / "nope.json")


def test_validate_missing_key():
    broken = {k: v for k, v in VALID_MANIFEST.items() if k != "commit_sha"}
    with pytest.raises(ValueError, match="commit_sha"):
        validate_manifest(broken)


def test_validate_wrong_type():
    broken = {**VALID_MANIFEST, "file_count": "not_an_int"}
    with pytest.raises(TypeError, match="file_count"):
        validate_manifest(broken)
