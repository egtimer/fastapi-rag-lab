"""Orchestrate the four ingestion stages with Langfuse tracing."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from langfuse import Langfuse

from fastapi_rag_lab.ingest.chunker import Child, Parent, chunk_markdown
from fastapi_rag_lab.ingest.embedder import EMBEDDING_MODEL, embed_texts
from fastapi_rag_lab.ingest.fetcher import PINNED_COMMIT_SHA, fetch_fastapi_docs
from fastapi_rag_lab.ingest.manifest import write_manifest
from fastapi_rag_lab.ingest.store import (
    COLLECTION_NAME,
    VECTOR_DIMENSION,
    upsert_children,
)

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/manifest.json")


def run_ingest() -> None:
    """Run the full ingestion pipeline: fetch, chunk, embed, upsert."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    langfuse = Langfuse()

    with langfuse.start_as_current_observation(name="ingest_run"):
        try:
            markdown_paths = _stage_fetch(langfuse)
            all_parents, all_children = _stage_chunk(langfuse, markdown_paths)
            embeddings = _stage_embed(langfuse, all_children)
            upserted = _stage_upsert(langfuse, all_parents, all_children, embeddings)

            _write_manifest(
                file_count=len(markdown_paths),
                parent_count=len(all_parents),
                child_count=len(all_children),
            )

            logger.info(
                "Ingestion complete: %d files, %d parents, %d children, "
                "%d points upserted",
                len(markdown_paths),
                len(all_parents),
                len(all_children),
                upserted,
            )
        except Exception:
            logger.exception("Ingestion failed")
            raise
        finally:
            langfuse.flush()


def _stage_fetch(langfuse: Langfuse) -> list[Path]:
    with langfuse.start_as_current_observation(name="fetch"):
        t0 = time.monotonic()
        markdown_paths = fetch_fastapi_docs()
        elapsed = time.monotonic() - t0
        logger.info("Fetch: %d files in %.1fs", len(markdown_paths), elapsed)
        return markdown_paths


def _stage_chunk(
    langfuse: Langfuse, markdown_paths: list[Path]
) -> tuple[list[Parent], list[Child]]:
    with langfuse.start_as_current_observation(name="chunk"):
        t0 = time.monotonic()
        all_parents: list[Parent] = []
        all_children: list[Child] = []

        for md_path in markdown_paths:
            relative_name = md_path.name
            text = md_path.read_text(encoding="utf-8")
            parents, children = chunk_markdown(relative_name, text)
            all_parents.extend(parents)
            all_children.extend(children)

        elapsed = time.monotonic() - t0
        logger.info(
            "Chunk: %d parents, %d children in %.1fs",
            len(all_parents),
            len(all_children),
            elapsed,
        )
        return all_parents, all_children


def _stage_embed(langfuse: Langfuse, children: list[Child]) -> list[list[float]]:
    with langfuse.start_as_current_observation(name="embed"):
        t0 = time.monotonic()
        child_texts = [c.text for c in children]
        embeddings = embed_texts(child_texts)
        elapsed = time.monotonic() - t0
        logger.info(
            "Embed: %d vectors in %.1fs (%.1f texts/s)",
            len(embeddings),
            elapsed,
            len(embeddings) / elapsed if elapsed > 0 else 0,
        )
        return embeddings


def _stage_upsert(
    langfuse: Langfuse,
    parents: list[Parent],
    children: list[Child],
    embeddings: list[list[float]],
) -> int:
    with langfuse.start_as_current_observation(name="upsert"):
        t0 = time.monotonic()
        count = upsert_children(parents, children, embeddings)
        elapsed = time.monotonic() - t0
        logger.info("Upsert: %d points in %.1fs", count, elapsed)
        return count


def _write_manifest(file_count: int, parent_count: int, child_count: int) -> None:
    manifest = {
        "commit_sha": PINNED_COMMIT_SHA,
        "ingested_at": datetime.now(UTC).isoformat(),
        "file_count": file_count,
        "parent_count": parent_count,
        "child_count": child_count,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": VECTOR_DIMENSION,
        "collection_name": COLLECTION_NAME,
    }
    write_manifest(MANIFEST_PATH, manifest)
    logger.info("Manifest written to %s", MANIFEST_PATH)
