"""Dense retrieval via Qdrant vector search."""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient

from fastapi_rag_lab.ingest.embedder import embed_texts
from fastapi_rag_lab.ingest.store import COLLECTION_NAME

from .types import RetrievalCandidate

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Bi-encoder retrieval using Qdrant's nearest-neighbor search."""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection_name

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievalCandidate]:
        """Embed the query and search Qdrant for nearest neighbors."""
        query_vector = embed_texts([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievalCandidate(
                chunk_id=str(r.id),
                parent_id=r.payload["parent_id"],
                score=r.score,
                text=r.payload["child_text"],
                metadata={
                    "source_file": r.payload.get("source_file", ""),
                    "heading_path": r.payload.get("heading_path", []),
                    "child_index": r.payload.get("child_index", 0),
                    "parent_text": r.payload.get("parent_text", ""),
                },
            )
            for r in response.points
        ]
