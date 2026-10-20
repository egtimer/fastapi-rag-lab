# ADR 007: Pin and Record the Retrieval Runtime

## Status
Proposed -- 2026-10-20 (the evidence section is completed once the
2026-10-20 runs are committed)

## Context

The golden-set benchmark committed on 2026-04-17
(`reports/2026-04-17-golden30-benchmark`) did not reproduce in October with
the same code path, the same pinned corpus
(`eba8942c81dbf990d25fbae34e6601bdbc21e74b`) and the same embedding model
(`nomic-embed-text`, digest `0a109f42…`). The dense source lists differed on
most questions.

The model digest identifies the weights, not the vectors. The Ollama server
that runs the model also shapes them. The Ollama v0.30.0 release notes
(2026-05-13) say that `nomic-embed-text` "now converts inputs to lowercase per
the model card where prior Ollama versions incorrectly preserved mixed case".
April's run did not record the Ollama version.

The same kind of drift can come from the other two moving parts of retrieval:

- **Reranker.** `BAAI/bge-reranker-base` was loaded from the Hugging Face
  `main` branch, so a new upload would change `hybrid_rerank` scores without
  any change in this repository.
- **Qdrant.** The client (`qdrant-client` 1.17.1, from `uv.lock`) and the
  server (`qdrant/qdrant:v1.12.4`) are further apart than the client supports:
  it warns that major versions should match and minor versions should differ
  by at most one. Ingestion and search work, but the pair is outside the range
  the client supports.

## Decision

**Ollama.** Pin the server to 0.35.1 (the `docker compose` service and the CI
workflow). 0.35.1 lowercases inputs as the model card asks, and it can pull the
current `gemma3:4b`. Ingestion stores the Ollama version and the model digest
in every point and in the manifest. The API and the benchmark refuse to query
an index built by a different runtime (`--allow-runtime-mismatch` overrides
this for the benchmark and is recorded in the report).

**Reranker.** Pin `BAAI/bge-reranker-base` to the Hugging Face commit
`2cfc18c9415c912f9d8155881c133215df768a70` (`RERANKER_REVISION` in
`retrieval/reranker.py`), the revision in the model cache when the 2026-10-20
runs started (earlier runs did not record theirs). Every
benchmark report records it as `provenance.reranker_revision`.

**Qdrant.** No version change during Phase B, so that runs stay comparable.
Every report already records both sides: `provenance.qdrant.server_version`
and `provenance.packages.qdrant-client`. Aligning them (a server within one
minor version of the client, or an older client) is pending. It needs a fresh
ingest and a rerun of the reference benchmarks to show that results do not
change.

## Evidence

To be completed from the 2026-10-20 reports: the April check on Ollama 0.20.7
compared per question with `reports/2026-04-17-golden30-benchmark`, and the
reference runs on 0.35.1 compared with the 0.5.7 runs of 2026-10-19.

## Consequences

- A benchmark number is tied to a runtime: Ollama version, model digest,
  reranker revision and Qdrant versions are all in the report's provenance.
- Upgrading Ollama, the reranker revision or Qdrant is a deliberate change that
  needs a re-ingest (for Ollama), new reference runs and a new comparison.
- A host Ollama at another version cannot serve an index built on 0.35.1; it
  fails loudly instead of returning quietly different results.
- April's numbers stay in `reports/` as a historical run; they are not
  comparable with runs on the pinned runtime.
