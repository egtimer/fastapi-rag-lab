"""POST /query endpoint with SSE streaming and Langfuse tracing."""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from fastapi_rag_lab.api.llm.ollama_client import OllamaClient, OllamaGenerationError
from fastapi_rag_lab.api.prompt import build_prompt
from fastapi_rag_lab.api.schemas.query import (
    Citation,
    QueryRequest,
    QueryResponse,
    StreamChunk,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_citations(results: list) -> list[Citation]:
    """Extract citations from reranked retrieval results."""
    citations = []
    for result in results:
        excerpt = result.chunk_text[:200] if result.chunk_text else ""
        citations.append(
            Citation(
                source_id=result.chunk_id,
                parent_id=result.parent_id,
                relevance_score=round(result.rerank_score, 4),
                excerpt=excerpt,
                source_file=result.metadata.get("source_file", ""),
                heading_path=result.metadata.get("heading_path", []),
            )
        )
    return citations


@router.post("/query")
async def query_endpoint(body: QueryRequest, request: Request):
    """Answer a question using hybrid retrieval + LLM generation."""
    t0 = time.monotonic()
    langfuse = request.app.state.langfuse

    trace = langfuse.trace(
        name="query",
        input={"query": body.query, "top_k": body.top_k},
    )

    # -- Retrieval --
    retrieval_span = trace.span(
        name="retrieval",
        input={"query": body.query, "top_k": body.top_k},
    )
    try:
        retriever = request.app.state.retriever
        results = retriever.retrieve(body.query, final_k=body.top_k)
    except Exception as exc:
        retrieval_span.end(output={"error": str(exc)})
        langfuse.flush()
        raise
    retrieval_span.end(output={"result_count": len(results)})

    citations = _build_citations(results)
    context_blocks = [
        (i + 1, r.parent_text) for i, r in enumerate(results)
    ]
    prompt = build_prompt(body.query, context_blocks)

    model = body.model or OllamaClient().model
    llm = OllamaClient(model=model)

    if body.stream:
        return _stream_response(llm, prompt, citations, trace, langfuse, t0)

    # -- Non-streaming path --
    generation_span = trace.span(
        name="generation",
        input={"model": model, "prompt_length": len(prompt)},
    )
    try:
        answer = await llm.generate(prompt)
    except OllamaGenerationError as exc:
        generation_span.end(output={"error": str(exc)})
        langfuse.flush()
        return QueryResponse(
            answer=f"Generation failed: {exc}",
            citations=citations,
            latency_ms=int((time.monotonic() - t0) * 1000),
            trace_id=trace.id,
        )
    generation_span.end(output={"answer_length": len(answer)})

    latency_ms = int((time.monotonic() - t0) * 1000)
    trace.update(output={"answer_length": len(answer), "latency_ms": latency_ms})
    langfuse.flush()

    return QueryResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms,
        trace_id=trace.id,
    )


def _stream_response(llm, prompt, citations, trace, langfuse, t0):
    """Return a StreamingResponse that emits SSE events."""

    async def event_generator():
        generation_span = trace.span(
            name="generation",
            input={"model": llm.model, "prompt_length": len(prompt)},
        )
        answer_tokens: list[str] = []

        try:
            async for token in llm.generate_stream(prompt):
                answer_tokens.append(token)
                chunk = StreamChunk(type="token", content=token)
                yield f"event: token\ndata: {chunk.model_dump_json()}\n\n"
        except OllamaGenerationError as exc:
            error_chunk = StreamChunk(type="error", content=str(exc))
            yield f"event: error\ndata: {error_chunk.model_dump_json()}\n\n"
            generation_span.end(output={"error": str(exc)})
            langfuse.flush()
            return

        answer_length = sum(len(t) for t in answer_tokens)
        generation_span.end(output={"answer_length": answer_length})

        citations_chunk = StreamChunk(
            type="citations",
            content=[c.model_dump() for c in citations],
        )
        yield f"event: citations\ndata: {json.dumps(citations_chunk.model_dump())}\n\n"

        latency_ms = int((time.monotonic() - t0) * 1000)
        done_chunk = StreamChunk(
            type="done",
            content={"latency_ms": latency_ms, "trace_id": trace.id},
        )
        yield f"event: done\ndata: {json.dumps(done_chunk.model_dump())}\n\n"

        trace.update(
            output={
                "answer_length": answer_length,
                "latency_ms": latency_ms,
            }
        )
        langfuse.flush()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
