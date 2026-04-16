"""Golden dataset loader and schema."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class GoldenEntry(BaseModel):
    """A single hand-curated query with ground truth.

    `expected_sources` are relative paths under `data/raw/fastapi_docs/`
    (e.g. ``tutorial/handling-errors.md``) — they match the ``source_file``
    field stored in the Qdrant payload during ingestion.

    `reference_answer` is required so RAGAS context_precision / context_recall
    have a ground-truth comparator. `expected_answer_keywords` is a lighter
    sanity check used by the runner for quick keyword coverage stats.
    """

    id: str
    query: str
    reference_answer: str
    expected_answer_keywords: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    category: str


class GoldenDataset(BaseModel):
    entries: list[GoldenEntry]


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and validate a golden dataset JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"entries": raw}
    return GoldenDataset.model_validate(raw)
