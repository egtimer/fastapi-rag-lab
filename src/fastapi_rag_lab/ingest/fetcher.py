"""Fetch FastAPI documentation markdown from a pinned commit."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

FASTAPI_REPO = "https://github.com/fastapi/fastapi.git"
PINNED_COMMIT_SHA = "eba8942c81dbf990d25fbae34e6601bdbc21e74b"

CLONE_DIR = Path("data/raw/fastapi")
DOCS_DIR = Path("data/raw/fastapi_docs")

logger = logging.getLogger(__name__)


def fetch_fastapi_docs(
    *,
    clone_dir: Path = CLONE_DIR,
    docs_dir: Path = DOCS_DIR,
    commit_sha: str = PINNED_COMMIT_SHA,
) -> list[Path]:
    """Clone FastAPI repo at a pinned SHA and extract English markdown docs."""
    clone_dir = clone_dir.resolve()
    docs_dir = docs_dir.resolve()

    if clone_dir.exists():
        _verify_existing_clone(clone_dir, commit_sha)
    else:
        _shallow_clone(clone_dir, commit_sha)

    return _extract_markdown(clone_dir, docs_dir)


def _verify_existing_clone(clone_dir: Path, expected_sha: str):
    head_sha = subprocess.run(
        ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if head_sha.returncode != 0:
        raise RuntimeError(
            f"{clone_dir} exists but is not a git repository. "
            "Delete it manually and re-run."
        )

    actual_sha = head_sha.stdout.strip()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{clone_dir} is at {actual_sha}, expected {expected_sha}. "
            "Delete the directory manually and re-run."
        )
    logger.info("Existing clone at %s matches pinned SHA", clone_dir)


def _shallow_clone(clone_dir: Path, commit_sha: str):
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            FASTAPI_REPO,
            str(clone_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set", "docs/en/docs"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    # Fetch the exact pinned commit
    result = subprocess.run(
        ["git", "-C", str(clone_dir), "fetch", "origin", commit_sha],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not fetch commit {commit_sha}. "
            f"Does it exist in the repo? git said: {result.stderr}"
        )

    result = subprocess.run(
        ["git", "-C", str(clone_dir), "checkout", commit_sha],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed: {result.stderr}")

    logger.info("Cloned FastAPI repo at %s", commit_sha)


def _extract_markdown(clone_dir: Path, docs_dir: Path) -> list[Path]:
    source_root = clone_dir / "docs" / "en" / "docs"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Expected docs at {source_root} but directory does not exist. "
            "The FastAPI repo structure may have changed."
        )

    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)

    markdown_files = sorted(source_root.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found under {source_root}")

    for md_file in markdown_files:
        relative = md_file.relative_to(source_root)
        dest = docs_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest)

    extracted = sorted(docs_dir.rglob("*.md"))
    logger.info("Extracted %d markdown files to %s", len(extracted), docs_dir)
    return [f.resolve() for f in extracted]
