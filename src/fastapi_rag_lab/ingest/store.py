"""Upsert child embeddings and metadata into Qdrant."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

if TYPE_CHECKING:
    from fastapi_rag_lab.ingest.chunker import Child, Parent

logger = logging.getLogger(__name__)

COLLECTION_NAME = "fastapi_docs_v1"
VECTOR_DIMENSION = 768
UPSERT_BATCH_SIZE = 100


def upsert_children(
    parents: list[Parent],
    children: list[Child],
    embeddings: list[list[float]],
    *,
    collection_name: str = COLLECTION_NAME,
    qdrant_url: str | None = None,
) -> int:
    """Upsert child chunks with embeddings into Qdrant."""
    if len(children) != len(embeddings):
        raise ValueError(
            f"children ({len(children)}) and embeddings ({len(embeddings)}) "
            f"must have the same length"
        )

    if qdrant_url is None:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    parent_lookup = {p.parent_id: p for p in parents}
    client = QdrantClient(url=qdrant_url)
    _ensure_collection(client, collection_name)

    ingested_at = datetime.now(UTC).isoformat()
    points = []

    for child, embedding in zip(children, embeddings, strict=True):
        parent = parent_lookup[child.parent_id]
        point_id = _child_id_to_uuid(child.child_id)

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "parent_id": child.parent_id,
                    "parent_text": parent.text,
                    "child_text": child.text,
                    "source_file": parent.source_file,
                    "heading_path": parent.heading_path,
                    "child_index": child.child_index,
                    "ingested_at": ingested_at,
                },
            )
        )

    upserted = 0
    for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[batch_start : batch_start + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        upserted += len(batch)

    logger.info("Upserted %d points into %s", upserted, collection_name)
    return upserted


def _ensure_collection(client: QdrantClient, collection_name: str):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection %s", collection_name)


def _child_id_to_uuid(child_id: str) -> str:
    return str(uuid.UUID(child_id[:32].ljust(32, "0")))
