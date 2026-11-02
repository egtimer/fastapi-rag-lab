"""Read an items run (``items.jsonl``) without importing the retrieval stack.

``answers.py`` writes items and needs Ollama, Qdrant and the reranker; the
judge, the labelling sheet and the agreement report only read them, and must
run on a machine with none of that installed (a labeller's laptop, CI).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def read_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
