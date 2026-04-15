"""Prompt templates for the query API. Versioned in code, not inline."""

QUERY_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about FastAPI documentation.
Use the following retrieved context to answer the question. If the context
doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""


def build_prompt(query: str, context_blocks: list[tuple[int, str]]) -> str:
    """Build the full prompt from query and numbered context blocks.

    Each context_block is (source_number, parent_text).
    """
    context_parts = [
        f"[Source {num}]\n{text}" for num, text in context_blocks
    ]
    context = "\n\n".join(context_parts)
    return QUERY_SYSTEM_PROMPT.format(context=context, query=query)
