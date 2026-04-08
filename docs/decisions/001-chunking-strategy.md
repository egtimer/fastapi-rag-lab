# ADR 001: Parent-Child Chunking

**Status:** Accepted
**Date:** 2026-04
**Context:** Initial architecture

## Context

RAG quality is bounded by the chunking strategy. The common defaults —
fixed-size chunks of 512 or 1024 tokens with some overlap — are easy to
implement and usually wrong. They break semantic units at arbitrary
boundaries, which means the retriever sees fragments and the generator
sees fragments, and neither has enough context to do its job well.

The FastAPI documentation has a predictable structure: each page is a
markdown file with headings, code blocks, and prose. Questions users
actually ask about FastAPI tend to fall into two categories:

1. **Precise lookups.** "What's the signature of `Depends`?" The answer
   is a single sentence or code snippet. Retrieval needs to be surgical.
2. **Conceptual questions.** "How does dependency injection work in
   FastAPI?" The answer is a section, not a sentence. Retrieval needs to
   return enough surrounding context for the generator to explain it.

A single chunk size cannot serve both. Small chunks improve precision but
lose context. Large chunks preserve context but dilute retrieval signals.

## Decision

Use a **parent-child chunking strategy**:

- **Child chunks** of ~256 tokens, indexed in the vector database. These
  are what the retriever searches over. They are small enough to give
  clean semantic signal for short queries.
- **Parent chunks** of ~1024 tokens, stored keyed by parent ID. These are
  retrieved after the child match and passed to the generator. They
  provide enough surrounding context for conceptual questions.

Chunking respects markdown structure. Boundaries prefer to fall at:

1. Section headers (highest priority)
2. Paragraph breaks
3. Sentence boundaries
4. Token count (fallback only)

Code blocks are never split. A code block longer than the target parent
size stays intact and becomes an oversized parent. This is a deliberate
trade-off against uniform chunk sizes.

Each child stores a reference to its parent in metadata: chunk_id,
parent_id, source_file, section_path, chunk_type (prose or code), and
token_count.

## Alternatives considered

### Fixed-size chunks (512 tokens, 50 token overlap)

The default for most RAG tutorials. Simple to implement and fast to
ingest. I rejected it because the FastAPI docs have enough structure that
ignoring it is measurably wasteful. Headings, parameter tables, and code
blocks all get chopped mid-thought with fixed sizes.

### Semantic chunking (embedding similarity between sentences)

Chunks formed by clustering adjacent sentences with high embedding
similarity. I rejected it for this project because:

1. It's slow to ingest: every sentence has to be embedded before chunks
   can be formed.
2. It's hard to evaluate: chunk boundaries depend on the embedding model
   used at ingestion time, which couples ingestion to a specific
   embedder.
3. The FastAPI docs already have explicit semantic structure (markdown
   headings). Using markdown is more reliable than trying to rediscover
   the structure from embeddings.

It's a technique I'd reconsider for unstructured text like legal PDFs or
transcripts.

### Pure hierarchical chunking (whole sections as chunks)

Retrieval returns an entire section. Context is perfect, precision is
terrible: a query about `Depends()` retrieves the entire Dependencies
chapter, and the LLM has to find the relevant sentence inside 4000
tokens of context.

## Consequences

**Positive:**

- Retrieval can be precise (small child) without losing context in
  generation (large parent).
- Respects the natural structure of the source documents.
- Each child has a clear provenance (file + section path), which makes
  citation generation straightforward.

**Negative:**

- Storage overhead: each token effectively lives in both a child chunk
  (indexed) and a parent chunk (stored). This roughly doubles storage
  compared to a single-size strategy. For a corpus the size of FastAPI's
  docs this is negligible; for larger corpora it matters.
- Ingestion is more complex. The chunker has to handle markdown parsing,
  section tracking, and parent-child linkage. More code, more room for
  bugs. Paid for by tests on known-tricky markdown inputs.
- The 256 / 1024 token sizes are defensible but not optimal. The "right"
  sizes depend on the embedder, the generator's context window, and the
  question distribution. I picked these as a starting point based on
  prior experience. They should be revisited after the first benchmark
  run.

## Open questions

- Should the chunker treat `tutorial/` differently from `reference/`?
  Tutorial pages are prose-heavy; reference pages are code-heavy. A
  chunker tuned for prose may underperform on reference.
- At what point does an oversized code block become its own problem?
  Should we summarise very long code blocks as a separate retrievable
  unit?

These will be addressed once the eval pipeline is in place and we have
numbers to argue with.

## References

- LlamaIndex blog on parent-child chunking (inspiration, not dependency)
- Pinecone: "Chunking Strategies for LLM Applications"
- Personal experience from production RAG systems over regulatory
  documents. Those systems validated the general principle; this project
  applies it to a different domain.
