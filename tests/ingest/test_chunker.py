import pytest

from fastapi_rag_lab.ingest.chunker import chunk_markdown, token_count

FIXTURE_MARKDOWN = """\
# Tutorial

Introduction text for the tutorial section.

## Body Parameters

When you need to send a request body, you use `Body`.

Here is an example:

```python
from fastapi import Body, FastAPI

app = FastAPI()

@app.post("/items/")
async def create_item(name: str = Body(), price: float = Body()):
    return {"name": name, "price": price}
```

The `Body` function works similarly to `Query` and `Path`.

## Multiple Parameters

You can declare multiple body parameters.

### Mixed Parameters

You can mix path, query, and body parameters freely.

FastAPI will know which is which based on the function parameter types.
Each parameter that is a Pydantic model will be interpreted as a request body.

### Singular Values in Body

By default, singular values are interpreted as query parameters.
To force them to be body parameters, use `Body()` explicitly.

## Validation

FastAPI uses Pydantic for validation. If the data does not match,
a clear error response is returned automatically.

### Field Validation

You can add extra validation to body fields using Pydantic's `Field`.

### Nested Models

Pydantic models can be nested to represent complex data structures.
This is one of the most powerful features of FastAPI's data handling.
"""

LARGE_CODE_BLOCK = "```python\n" + ("x = 1\n" * 1500) + "```\n"
MASSIVE_CODE_MARKDOWN = f"# Big\n\n## Section\n\n{LARGE_CODE_BLOCK}\n"


def test_chunk_produces_parents_with_heading_paths():
    parents, _ = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)

    assert len(parents) >= 3
    heading_texts = [p.heading_path[-1] if p.heading_path else "" for p in parents]
    assert "Body Parameters" in heading_texts
    assert "Multiple Parameters" in heading_texts
    assert "Validation" in heading_texts


def test_chunk_produces_children_for_each_parent():
    parents, children = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)

    parent_ids = {p.parent_id for p in parents}
    child_parent_ids = {c.parent_id for c in children}

    assert child_parent_ids.issubset(parent_ids)
    assert len(children) >= len(parents)


def test_children_do_not_split_code_blocks():
    parents, children = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)

    code_snippet = "async def create_item"
    code_children = [c for c in children if code_snippet in c.text]

    assert len(code_children) >= 1
    for child in code_children:
        assert "```python" in child.text or "```" in child.text


def test_child_token_counts_within_tolerance():
    _, children = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)

    for child in children:
        actual_tokens = token_count(child.text)
        assert child.token_count == actual_tokens
        assert actual_tokens <= 500, (
            f"Child has {actual_tokens} tokens, expected <= 500"
        )


def test_deterministic_ids():
    parents_a, children_a = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)
    parents_b, children_b = chunk_markdown("tutorial.md", FIXTURE_MARKDOWN)

    assert [p.parent_id for p in parents_a] == [p.parent_id for p in parents_b]
    assert [c.child_id for c in children_a] == [c.child_id for c in children_b]


def test_massive_code_block_raises():
    with pytest.raises(ValueError, match="embedding limit"):
        chunk_markdown("huge.md", MASSIVE_CODE_MARKDOWN)


def test_empty_markdown_produces_nothing():
    parents, children = chunk_markdown("empty.md", "")
    assert parents == []
    assert children == []
