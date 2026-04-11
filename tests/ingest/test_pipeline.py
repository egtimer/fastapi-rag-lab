"""Integration test for the full ingestion pipeline.

Runs against real Qdrant and Ollama, with a tiny fixture corpus
instead of the full FastAPI docs.
"""

import json

import httpx
import pytest

FIXTURE_MD_1 = """\
# Getting Started

## Installation

Install FastAPI with pip:

```bash
pip install fastapi[standard]
```

You also need an ASGI server like uvicorn.

## First Steps

Create a file `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "hello"}
```

Run it with `uvicorn main:app --reload`.
"""

FIXTURE_MD_2 = """\
# Path Parameters

## Declaring Path Parameters

You can declare path parameters with the same syntax used by Python format strings:

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of `item_id` will be passed as an argument.

## Data Validation

If you pass a string where an int is expected, FastAPI returns a validation error.
"""


def _services_reachable() -> bool:
    import os

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        ollama_ok = httpx.get(f"{ollama_host}/api/tags", timeout=5).status_code == 200
        qdrant_ok = (
            httpx.get("http://localhost:6333/collections", timeout=3).status_code == 200
        )
        return ollama_ok and qdrant_ok
    except httpx.HTTPError:
        return False


@pytest.mark.integration
@pytest.mark.ollama
@pytest.mark.qdrant
@pytest.mark.skipif(not _services_reachable(), reason="Ollama or Qdrant not reachable")
def test_pipeline_end_to_end(tmp_path, monkeypatch):
    docs_dir = tmp_path / "fastapi_docs"
    docs_dir.mkdir()

    (docs_dir / "getting_started.md").write_text(FIXTURE_MD_1)
    (docs_dir / "path_params.md").write_text(FIXTURE_MD_2)

    manifest_path = tmp_path / "manifest.json"
    test_collection = "test_pipeline_integration"

    fixture_paths = sorted(docs_dir.rglob("*.md"))

    monkeypatch.setattr("fastapi_rag_lab.ingest.pipeline.MANIFEST_PATH", manifest_path)

    from langfuse import Langfuse

    from fastapi_rag_lab.ingest.pipeline import (
        _stage_chunk,
        _stage_embed,
        _write_manifest,
    )
    from fastapi_rag_lab.ingest.store import upsert_children

    langfuse = Langfuse()

    with langfuse.start_as_current_observation(name="test_ingest"):
        all_parents, all_children = _stage_chunk(langfuse, fixture_paths)
        assert len(all_parents) >= 2
        assert len(all_children) >= len(all_parents)

        embeddings = _stage_embed(langfuse, all_children)
        assert len(embeddings) == len(all_children)

        count = upsert_children(
            all_parents,
            all_children,
            embeddings,
            collection_name=test_collection,
        )
        assert count == len(all_children)

    _write_manifest(
        file_count=2,
        parent_count=len(all_parents),
        child_count=len(all_children),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["file_count"] == 2
    assert manifest["parent_count"] >= 2
    assert manifest["child_count"] >= manifest["parent_count"]
    assert manifest["embedding_model"] == "nomic-embed-text"

    langfuse.flush()

    # cleanup test collection
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333")
    client.delete_collection(test_collection)
