# ADR 002: Ingestion Pipeline for FastAPI Documentation

**Status**: Accepted
**Date**: 2026-04-09
**Context**: Phase 2 of fastapi-rag-lab

## Context

Phase 1 left a working scaffold: FastAPI skeleton, Qdrant running in Docker, Ollama on the host via gateway IP, Langfuse for observability, CI green. The chunking strategy was already decided in ADR 001 (parent-child, 256/1024 tokens). What Phase 2 has to answer is narrower and more practical: where does the corpus come from, how does it flow from raw source to indexed vectors, and what guarantees does the pipeline offer so that retrieval quality can be measured later without re-ingesting everything every time an experiment changes.

The corpus for this lab is the official FastAPI documentation. It is public, non-trivial in size (a few hundred markdown files), has real heading structure, includes code blocks that matter for retrieval, and the domain is one I know well enough to judge retrieval quality by eye during development. It is the right training ground for the retrieval work that comes later.

## Decision

The ingestion pipeline is a four-stage flow, each stage owning one responsibility, each stage independently testable, each stage idempotent enough that re-running it on the same input produces the same output.

**Stage 1 — Fetch.** Clone the `fastapi/fastapi` repository into `data/raw/fastapi/` via shallow git clone pinned to a specific commit SHA recorded in a manifest file. Extract only `docs/en/docs/**/*.md` into `data/raw/fastapi_docs/`, preserving the directory structure so that file paths can be used later as part of the source metadata. No GitHub API calls. No rate limits. Reproducible across machines because the commit SHA is fixed.

**Stage 2 — Parse and chunk.** Each markdown file is parsed with `markdown-it-py` into a token stream. A custom walker reconstructs the heading hierarchy (H1 → H2 → H3) and segments the file into parent blocks bounded by H2 or H3 headings, whichever produces parent sizes closest to but not exceeding 1024 tokens (tokenized with `tiktoken` cl100k_base). Each parent is then split into children of roughly 256 tokens with a small overlap of 32 tokens. Children carry a reference to their parent_id. Code blocks are never split across chunk boundaries; if a code block would cross a boundary, the chunk is grown or shrunk to keep the block intact. This follows directly from ADR 001 and makes the parent-child pattern concrete.

**Stage 3 — Embed.** Only children are embedded. Parents exist to provide context at generation time, not to be retrieved. Embeddings come from `nomic-embed-text` via Ollama running on the host, accessed through the gateway IP resolved at runtime from `$OLLAMA_HOST`. Embedding is batched (batch size 16 initially, tunable) and runs sequentially rather than in parallel because local Ollama becomes the bottleneck and throwing more concurrency at it hurts rather than helps.

**Stage 4 — Upsert.** Embeddings and metadata are upserted into a Qdrant collection named `fastapi_docs_v1`. The payload on each point contains: `parent_id`, `parent_text`, `child_text`, `source_file`, `heading_path`, `child_index`, and `ingested_at`. The collection is created on first run with the correct vector dimension (768 for nomic-embed-text) and a cosine distance metric. Re-running ingestion on the same commit SHA is a no-op: points are upserted with deterministic IDs derived from `hash(source_file + child_index)`, so Qdrant overwrites rather than duplicates.

**Observability.** The entire run is wrapped in a Langfuse trace with spans per stage and per file. Counts, token totals, embedding duration, and Qdrant upsert duration are recorded. The first goal of observability here is not beautiful dashboards, it is being able to answer "what changed between yesterday's run and today's" when retrieval quality moves.

**Manifest.** Every run writes `data/raw/manifest.json` containing: git commit SHA, file count, chunk counts (parents and children), embedding model name, embedding dimension, and ingestion timestamp. The manifest is the contract between ingestion and the downstream retrieval/eval code. If the manifest does not exist or does not match, downstream code refuses to run.

## Alternatives considered

**Fetch via GitHub API.** Rejected. Rate limits make it fragile, authentication adds config, and there is no upside over a shallow clone for a public repo.

**Chunking by plain token count, no parent-child.** Rejected in ADR 001, but worth naming again: losing heading structure would make retrieval snippets harder to read and harder to cite, and would undermine the ability to return parent context at generation time. The simplicity gain is not worth the quality loss.

**Embed parents and children both.** Rejected. Parents exist to give the LLM enough context to reason, not to be retrieved. Embedding them doubles vector store size, increases ingestion time, and offers no retrieval benefit because children already represent the same content at finer resolution. If this turns out to be wrong during eval, it is a cheap reversal: add one upsert call in stage 4.

**Parallel embedding across workers.** Rejected for now. Local Ollama saturates on a single request; adding workers just increases contention. If I later move to a hosted embedder (Cohere, OpenAI), this decision gets revisited.

**Neo4j or another graph store for parent-child relationships.** Rejected. A single integer field `parent_id` in the Qdrant payload plus a parent lookup table in SQLite (or a JSON file for this scale) is sufficient. Adding Neo4j would be architecture theater.

## Consequences

**Good.** Each stage is testable in isolation with small markdown fixtures. Re-running is cheap. The manifest makes experiment diffs legible. The heading-bounded parent split means retrieved snippets align with semantic units a reader would recognize, which matters when I start doing eyeball evaluation on bad results.

**Acceptable costs.** Chunking logic is more complex than flat windowed chunking, perhaps 2-3x the code. Storage is larger than strict flat chunking because parent_text is denormalized onto every child payload (trade-off: faster retrieval, no second lookup, at the cost of duplication). Initial ingestion run on the full corpus will take several minutes because of Ollama throughput, not because of the rest of the pipeline.

**Unknowns I will find out by running it.** What the actual parent-to-child ratio is for this corpus (affects storage). Whether heading-based segmentation produces balanced parents or whether some sections will blow past 1024 tokens and force awkward fallback splits. Whether tiktoken's cl100k_base is a reasonable proxy for nomic-embed-text's tokenization (probably not exactly, but close enough for chunk sizing). Whether the code-block-never-split rule produces any parent that is too large to embed (nomic has a 2048 token limit; if a single code block is larger than that, the pipeline needs to fail loudly rather than silently truncate).

## Open questions (not blocking Phase 2)

- **Reranker stage.** Deferred to Phase 3. The decision tree: run naive dense retrieval first, measure, add BM25 hybrid if precision is poor, add bge-reranker if hybrid is still not enough. No point pre-wiring all of it.
- **Multi-version docs.** The FastAPI docs have translations and version branches. Phase 2 is English only, pinned commit. Multi-version support is a later concern.
- **Incremental re-ingestion.** Current design re-processes everything on every run. For a lab corpus this size it is fast enough. If the corpus grows, incremental ingestion driven by file hash comparison becomes worth the complexity.

## Success criteria for Phase 2

A developer can run `python -m fastapi_rag_lab.ingest` and within a few minutes end up with: a populated Qdrant collection, a manifest file on disk, a Langfuse trace visible in the UI, and a smoke test that asserts the collection has a plausible number of points and that a trivial query returns something non-empty. That is the entire bar. Retrieval quality is Phase 3's job.
