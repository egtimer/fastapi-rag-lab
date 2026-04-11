import httpx
import pytest
from qdrant_client import QdrantClient

from fastapi_rag_lab.ingest.chunker import Child, Parent
from fastapi_rag_lab.ingest.store import VECTOR_DIMENSION, upsert_children

TEST_COLLECTION = "test_store_integration"


def _qdrant_reachable() -> bool:
    try:
        resp = httpx.get("http://localhost:6333/collections", timeout=3)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _make_test_data(count: int = 3):
    parent = Parent(
        parent_id="parent-abc",
        text="Parent context about FastAPI routing.",
        source_file="tutorial/routing.md",
        heading_path=["Tutorial", "Routing"],
        token_count=10,
    )

    children = []
    embeddings = []
    for i in range(count):
        children.append(
            Child(
                child_id=f"{i:016x}",
                parent_id="parent-abc",
                text=f"Child chunk {i} about routing.",
                child_index=i,
                token_count=5,
            )
        )
        embeddings.append([float(i)] * VECTOR_DIMENSION)

    return [parent], children, embeddings


@pytest.mark.qdrant
@pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable")
class TestQdrantStore:
    def setup_method(self):
        self.client = QdrantClient(url="http://localhost:6333")
        collections = [c.name for c in self.client.get_collections().collections]
        if TEST_COLLECTION in collections:
            self.client.delete_collection(TEST_COLLECTION)

    def teardown_method(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if TEST_COLLECTION in collections:
            self.client.delete_collection(TEST_COLLECTION)

    def test_upsert_creates_collection_and_inserts_points(self):
        parents, children, embeddings = _make_test_data(3)

        count = upsert_children(
            parents,
            children,
            embeddings,
            collection_name=TEST_COLLECTION,
        )

        assert count == 3
        info = self.client.get_collection(TEST_COLLECTION)
        assert info.points_count == 3

    def test_upsert_is_idempotent(self):
        parents, children, embeddings = _make_test_data(2)

        upsert_children(parents, children, embeddings, collection_name=TEST_COLLECTION)
        upsert_children(parents, children, embeddings, collection_name=TEST_COLLECTION)

        info = self.client.get_collection(TEST_COLLECTION)
        assert info.points_count == 2
