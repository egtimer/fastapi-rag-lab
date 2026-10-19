"""Best-effort provenance for benchmark runs.

Every probe catches its own failures and records ``None`` plus a reason, so a
missing service or tool never aborts a benchmark.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

from fastapi_rag_lab.ingest.embedder import EMBEDDING_MODEL
from fastapi_rag_lab.ingest.fetcher import PINNED_COMMIT_SHA
from fastapi_rag_lab.ingest.store import COLLECTION_NAME
from fastapi_rag_lab.retrieval.reranker import RERANKER_MODEL

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/manifest.json")
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
HTTP_TIMEOUT = 3.0
PACKAGES = ("torch", "sentence-transformers", "qdrant-client", "rank-bm25", "numpy")


def _fail(exc: BaseException) -> dict[str, Any]:
    return {"value": None, "reason": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    cwd = cwd or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip()), "reason": None}
    except Exception as exc:
        return {"commit": None, "dirty": None, "reason": _fail(exc)["reason"]}


def dataset_info(path: Path, entry_count: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "sha256": None,
        "entry_count": entry_count,
        "reason": None,
    }
    try:
        info["sha256"] = _sha256(path)
    except Exception as exc:
        info["reason"] = _fail(exc)["reason"]
    return info


def corpus_info(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    info: dict[str, Any] = {
        "pinned_commit": PINNED_COMMIT_SHA,
        "manifest_sha256": None,
        "manifest_commit": None,
        "doc_count": None,
        "parent_count": None,
        "chunk_count": None,
        "reason": None,
    }
    try:
        info["manifest_sha256"] = _sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info["manifest_commit"] = manifest.get("commit_sha")
        info["doc_count"] = manifest.get("file_count")
        info["parent_count"] = manifest.get("parent_count")
        info["chunk_count"] = manifest.get("child_count")
    except Exception as exc:
        info["reason"] = _fail(exc)["reason"]
    return info


def qdrant_info(
    url: str | None = None, collection: str = COLLECTION_NAME
) -> dict[str, Any]:
    url = (url or os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)).rstrip("/")
    info: dict[str, Any] = {
        "url": url,
        "collection": collection,
        "server_version": None,
        "point_count": None,
        "reason": None,
    }
    reasons = []
    try:
        resp = httpx.get(f"{url}/", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        info["server_version"] = resp.json().get("version")
    except Exception as exc:
        reasons.append(f"version: {_fail(exc)['reason']}")
    try:
        resp = httpx.get(f"{url}/collections/{collection}", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        info["point_count"] = resp.json()["result"]["points_count"]
    except Exception as exc:
        reasons.append(f"point_count: {_fail(exc)['reason']}")
    info["reason"] = "; ".join(reasons) or None
    return info


def ollama_info(
    host: str | None = None, model: str = EMBEDDING_MODEL
) -> dict[str, Any]:
    host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    info: dict[str, Any] = {
        "host": host,
        "embedding_model": model,
        "embedding_model_digest": None,
        "reason": None,
    }
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        names = {model, f"{model}:latest"}
        for m in resp.json().get("models", []):
            if m.get("name") in names or m.get("model") in names:
                info["embedding_model_digest"] = m.get("digest")
                break
        else:
            info["reason"] = f"model {model!r} not listed by {host}/api/tags"
    except Exception as exc:
        info["reason"] = _fail(exc)["reason"]
    return info


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception as exc:
            versions[pkg] = None
            versions[f"{pkg}_reason"] = _fail(exc)["reason"]
    return versions


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_model": None,
        "logical_cores": os.cpu_count(),
        "ram_bytes": None,
        "os": platform.platform(),
        "reasons": [],
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        else:
            info["reasons"].append("cpu_model: no 'model name' in /proc/cpuinfo")
    except Exception as exc:
        info["reasons"].append(f"cpu_model: {_fail(exc)['reason']}")
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["ram_bytes"] = int(line.split()[1]) * 1024
                break
        else:
            info["reasons"].append("ram_bytes: no MemTotal in /proc/meminfo")
    except Exception as exc:
        info["reasons"].append(f"ram_bytes: {_fail(exc)['reason']}")
    return info


def collect_provenance(
    *,
    dataset_path: Path,
    entry_count: int | None,
    argv: list[str] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Gather all provenance probes; never raises."""
    probes = {
        "git": lambda: git_info(),
        "dataset": lambda: dataset_info(dataset_path, entry_count),
        "corpus": corpus_info,
        "qdrant": qdrant_info,
        "ollama": ollama_info,
        "packages": package_versions,
        "hardware": hardware_info,
    }
    out: dict[str, Any] = {
        "command": " ".join(sys.argv if argv is None else argv),
        "started_at": started_at,
        "finished_at": finished_at or datetime.now(UTC).isoformat(),
        "reranker_model": RERANKER_MODEL,
    }
    for name, probe in probes.items():
        try:
            out[name] = probe()
        except Exception as exc:  # last-resort guard
            out[name] = _fail(exc)
    return out
