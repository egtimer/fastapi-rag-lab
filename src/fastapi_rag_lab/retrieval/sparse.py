"""BM25 sparse retrieval over the chunk corpus."""

from __future__ import annotations

import logging
import os
import re

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from fastapi_rag_lab.ingest.store import COLLECTION_NAME

from .types import RetrievalCandidate

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric.

    Kept simple intentionally: BM25 is robust to basic tokenization and
    this preserves code identifiers like HTTPException as single tokens.
    """
    return re.findall(r"\w+", text.lower())


class SparseRetriever:
    """In-memory BM25 index over all child chunks in a Qdrant collection.

    The index is built once at construction time by scrolling the full
    collection. For ~2K chunks this takes about 1 second.
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection = collection_name
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []
        self._build_index()

    def _build_index(self):
        """Pull all chunks from Qdrant, tokenize, build BM25 index."""
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)

        all_points = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if offset is None:
                break

        self.chunks = [
            {
                "chunk_id": str(p.id),
                "parent_id": p.payload["parent_id"],
                "text": p.payload["child_text"],
                "metadata": {
                    "source_file": p.payload.get("source_file", ""),
                    "heading_path": p.payload.get("heading_path", []),
                    "child_index": p.payload.get("child_index", 0),
                    "parent_text": p.payload.get("parent_text", ""),
                },
            }
            for p in all_points
        ]

        tokenized_corpus = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d chunks", len(self.chunks))

    def retrieve(self, query: str, top_k: int = 30) -> list[RetrievalCandidate]:
        """Score all chunks against the query and return the top-K."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [
            RetrievalCandidate(
                chunk_id=self.chunks[i]["chunk_id"],
                parent_id=self.chunks[i]["parent_id"],
                score=float(scores[i]),
                text=self.chunks[i]["text"],
                metadata=self.chunks[i]["metadata"],
            )
            for i in top_indices
            if scores[i] > 0
        ]
