# ADR 003: Hybrid Retrieval with RRF and Cross-Encoder Reranking

## Status
Accepted -- 2026-04-13

## Context

After completing the ingestion pipeline (ADR 002), the next step is retrieval.
Pure dense retrieval (semantic similarity via embeddings) has known limitations:

1. **Misses exact keyword matches**: queries with specific function names, error
   codes, or rare terminology often fail because embeddings normalize semantic
   meaning, not lexical surface forms.

2. **Vocabulary mismatch**: when query and document use different words for the
   same concept, dense retrieval handles this well -- but when they use the SAME
   rare word, dense can rank semantically similar but lexically different
   documents higher than the exact match.

3. **No score interpretability**: cosine similarity scores between embeddings
   don't have clear thresholds for "good enough" retrieval.

## Decision

Implement hybrid retrieval combining:

1. **Dense retrieval** (Qdrant + nomic-embed-text): semantic search, top-30
2. **Sparse retrieval** (BM25 via rank_bm25): keyword search, top-30
3. **Reciprocal Rank Fusion (RRF)** with k=60: combine rankings -> top-15
4. **Cross-encoder rerank** (BAAI/bge-reranker-base): final precision -> top-5

## Rationale

### Why hybrid (not pure dense)?

Microsoft's Azure Cognitive Search, Pinecone, Vespa, and Weaviate all converge
on hybrid as the default for production RAG. Empirical benchmarks (BEIR, MTEB)
show hybrid consistently outperforms pure dense, especially on out-of-domain
queries.

### Why RRF (not weighted score normalization)?

- Score-based fusion requires normalizing scores from different distributions
  (cosine similarity 0-1 vs BM25 unbounded). Normalization is dataset-dependent
  and fragile.
- RRF only uses ranks, eliminating the normalization problem.
- k=60 is from the original RRF paper (Cormack et al., 2009). It dampens
  contribution from low-rank results without overweighting top-1.

### Why cross-encoder rerank (not just RRF top-K)?

- Bi-encoders (used in dense retrieval) encode query and passage independently.
  Fast but lossy.
- Cross-encoders process [query, passage] together with full attention. ~10-15%
  better NDCG@10 in benchmarks but ~50ms latency per pair.
- Solution: use bi-encoder + BM25 for recall (top-15), then cross-encoder on
  the small candidate set for precision (top-5). Best of both.

### Why bge-reranker-base (not large)?

- 278M params, ~50ms latency on CPU per pair -> 750ms total for top-15.
- Quality within 2-3% of bge-reranker-large (560M params, 150ms/pair).
- Production sweet spot for sub-second total retrieval latency.

### Why rank_bm25 (not pyserini)?

- Pure Python, no Java dependencies, simple deployment.
- Sufficient for our corpus size (~2K chunks). Pyserini becomes worth it at
  millions of chunks where Lucene's optimizations matter.
- Building the index in memory at startup is fast (~1s for 2K chunks).

### Why retrieve top-30 each, fuse to top-15, rerank to top-5?

- Top-30 gives RRF enough candidates to find consensus across retrievers.
- Top-15 fused candidates is a manageable rerank batch (~750ms latency).
- Top-5 final results fits typical context windows for downstream LLM.

## Consequences

### Positive

- Retrieval works across diverse query types (semantic + keyword).
- Sub-second end-to-end latency (acceptable for interactive use).
- Pipeline is composable: each component can be swapped/A-B tested.
- Reranker provides interpretable relevance scores in [-10, +10] range.

### Negative

- Memory footprint: BM25 index in memory + reranker model loaded (~500MB total).
- Cold start: BM25 index build at startup adds ~1-2s. Acceptable for server
  applications, problematic for serverless/lambda.
- Increased complexity vs pure dense: 4 components instead of 1.

### Tradeoffs accepted

- We optimize for retrieval QUALITY over latency at this scale. If we needed
  <100ms total, we would skip reranking and accept lower precision.
- We optimize for SIMPLICITY over scalability with rank_bm25. Migration to
  pyserini is straightforward when corpus exceeds ~100K chunks.

## Alternatives Considered

1. **Pure dense + larger top-K**: simpler but worse on keyword queries.
2. **Dense + BM25 with weighted score normalization**: requires hyperparameter
   tuning per domain, fragile.
3. **ColBERT (late interaction)**: better than bi-encoder, but heavier infra
   (per-token vectors) and not justified at our scale.
4. **Larger reranker (bge-reranker-large)**: 3x slower for marginal quality gain.

## References

- Cormack et al., 2009. "Reciprocal Rank Fusion outperforms Condorcet and
  individual Rank Learning Methods"
- BEIR benchmark: https://github.com/beir-cellar/beir
- bge-reranker-base: https://huggingface.co/BAAI/bge-reranker-base
