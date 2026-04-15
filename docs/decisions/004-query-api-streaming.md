# ADR 004: Query API with SSE Streaming

## Status
Accepted -- 2026-04-15

## Context

With hybrid retrieval in place (ADR 003), the next step is exposing it through
an API endpoint that retrieves context, generates an answer via LLM, and returns
it to the user. The key design questions are:

1. How to deliver tokens to the client (SSE vs WebSockets vs polling).
2. Whether to stream by default or serve complete responses.
3. How to handle citations alongside streamed content.
4. How to structure Langfuse tracing for the full query lifecycle.
5. How to version prompt templates.

## Decision

### SSE over WebSockets

POST /query returns a `text/event-stream` response by default. Each token is a
separate SSE event, followed by a `citations` event and a `done` event with
latency metrics.

SSE is HTTP-native: works through standard proxies and load balancers, needs no
connection upgrade, and every HTTP client already supports it. WebSockets would
add connection management complexity for a unidirectional data flow (server to
client). We don't need bidirectional communication -- the user sends one query
and receives one streamed answer.

The endpoint also supports `"stream": false` for programmatic consumers who
prefer a single JSON response. Streaming is the default because perceived
latency matters more than total latency for interactive use.

### Streaming-first design

First-token-time is the metric that matters for user experience. With
`gemma3:4b` on local Ollama, total generation takes 5-15 seconds depending on
answer length. Streaming delivers the first token in ~200ms after retrieval
completes, making the system feel responsive while the model is still working.

### Citations as final SSE event

Citations are derived from retrieval results, not from the generation itself.
They're known before generation starts. Two options:

1. Send citations first, then stream tokens.
2. Stream tokens first, send citations after.

Option 2 is better for the client: tokens start flowing immediately, and the
client can render citations after the answer is complete. Inline citation
markers like `[Source 1]` appear in the generated text naturally because the
prompt template includes `[Source N]` prefixes in the context.

### SSE event format

```
event: token
data: {"type": "token", "content": "FastAPI"}

event: citations
data: {"type": "citations", "content": [{"source_id": "...", ...}]}

event: done
data: {"type": "done", "content": {"latency_ms": 1234, "trace_id": "..."}}

event: error
data: {"type": "error", "content": "Ollama generation timed out..."}
```

Each event has a `type` field so clients can dispatch on it without parsing the
SSE event name. The `error` event terminates the stream early.

### Langfuse span structure

Each query creates a trace with two child spans:

```
query (trace)
  |-- retrieval (span): hybrid retrieval pipeline
  |-- generation (generation): LLM token generation
```

For streaming responses, spans use `start_observation` (explicit lifecycle)
rather than context managers, because the async generator outlives the endpoint
function. Each span is explicitly `.end()`-ed when its work completes.

When Langfuse is not configured (no API keys), tracing degrades gracefully --
spans are created as no-ops and a UUID is generated as a fallback trace ID.

### Prompt template versioning

The prompt template lives in `src/fastapi_rag_lab/api/prompt.py` as a module
constant, not inline in the endpoint. This keeps it visible in diffs and
git-blameable. No external prompt management system -- at this scale, code is
the version control.

The template uses `[Source N]` markers in the context so the model naturally
references them in its answer, creating a link between generated text and
citation metadata.

## Consequences

### Positive

- Sub-200ms first-token delivery for interactive use.
- Standard HTTP -- works with curl, browser EventSource, any HTTP client.
- Non-streaming mode available for batch/programmatic use.
- Full observability: every query has a Langfuse trace with retrieval + generation spans.
- Prompt template changes are tracked in git like any other code change.

### Negative

- SSE is unidirectional. If we later need follow-up questions in the same
  context, we'd need to add session state or switch to WebSockets.
- Streaming makes error handling more complex: errors mid-stream are delivered
  as SSE events rather than HTTP status codes.
- The 120s Ollama timeout is generous. In production, a circuit breaker or
  shorter timeout with retry would be appropriate.

### Tradeoffs accepted

- We optimize for PERCEIVED LATENCY over implementation simplicity. A
  non-streaming endpoint would be 30 lines simpler.
- We accept that citations are static (from retrieval) rather than dynamic
  (from generation). The model might reference context it wasn't given, and
  we won't catch that here -- that's what the evaluation pipeline is for.

## Alternatives Considered

1. **WebSockets**: More complex, bidirectional capability we don't need.
2. **Long polling**: Higher latency, more complex client code.
3. **Non-streaming only**: Simpler but 5-15s wait with no feedback.
4. **Inline citations**: Parsing model output for citation markers is fragile
   and model-dependent. Separate metadata is cleaner.
5. **External prompt management (Langfuse prompts)**: Unnecessary complexity
   at this scale. Code versioning is sufficient.
