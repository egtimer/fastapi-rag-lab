# fastapi-rag-lab

A RAG system over the FastAPI documentation with a full evaluation pipeline.

I built this to test chunking strategies, hybrid retrieval, and hallucination
evaluation against a corpus I actually know well. Most RAG tutorials stop at
"it works on my demo query". This one measures whether it works on 100
questions and tells you honestly when it doesn't.

Everything runs locally: Ollama for embeddings and generation, Qdrant for
vector storage, Langfuse for traces. No API keys, no vendor lock-in. If you
have Docker and Ollama, you can reproduce every number in the benchmarks.

## Status

Work in progress. Current phase: initial setup and ingestion pipeline.

## Why this project exists

I've been building production RAG systems for clients over the last year, and
most of what I've learned does not fit in a tutorial. The interesting problems
are in the parts nobody writes blog posts about:

- Chunking that preserves semantic boundaries without losing context.
- Measuring retrieval quality before measuring generation quality.
- Knowing when a bad answer is the retriever's fault, the chunker's fault,
  or the model's fault.
- Catching hallucinations before they reach the user.

This repository is where I work those problems out on public data, so I can
talk about them openly.

## What's in here (planned)

- Parent-child chunking over FastAPI's markdown documentation.
- Hybrid retrieval: BM25 + dense with Reciprocal Rank Fusion.
- Optional reranking with `bge-reranker-base`.
- Generation with `gemma3:4b` via Ollama.
- Evaluation pipeline: RAGAS metrics plus a custom citation accuracy check
  against a hand-built gold dataset of ~100 real FastAPI questions.
- Observability via self-hosted Langfuse.
- FastAPI endpoint that ties the whole thing together.

## Stack

- Python 3.11+
- FastAPI
- Qdrant (self-hosted via Docker)
- Ollama: `nomic-embed-text`, `gemma3:4b` (running on the host, not in
  the compose stack — see ADR 002 once it exists)
- Langfuse (self-hosted via Docker)
- RAGAS for evaluation metrics
- `uv` for dependency management
- No LangChain, no LlamaIndex. Orchestration is hand-written.

## Running it locally

### Prerequisites

1. Docker and Docker Compose (for Qdrant and Langfuse)
2. [Ollama](https://ollama.com/) running on the host with `nomic-embed-text` pulled:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Python 3.11+ with [uv](https://docs.astral.sh/uv/)

### Start services

```bash
docker compose up -d
uv sync
```

### Running ingestion

The ingestion pipeline clones the FastAPI docs at a pinned commit, chunks them
using a heading-aware parent-child strategy, embeds children via Ollama, and
upserts everything into Qdrant.

```bash
# If Ollama is on the Windows host (WSL setup), set the gateway IP:
export OLLAMA_HOST=http://$(cat /proc/net/route | awk '/00000000.*00000000/ {print $3}' | head -1 | sed 's/../0x&\n/g' | tac | xargs printf "%d.%d.%d.%d\n"):11434

# Or if Ollama is running locally:
export OLLAMA_HOST=http://localhost:11434

# Langfuse credentials (matches docker-compose.yml defaults):
export LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
export LANGFUSE_SECRET_KEY=sk-lf-local-dev
export LANGFUSE_HOST=http://localhost:3000

python -m fastapi_rag_lab.ingest
```

After a successful run:
- `data/raw/manifest.json` contains ingestion metadata (file counts, timestamps)
- Qdrant dashboard at http://localhost:6333/dashboard shows the `fastapi_docs_v1` collection
- Langfuse at http://localhost:3000 shows the `ingest_run` trace with per-stage spans

## Design decisions

Significant architectural choices are documented in
[`docs/decisions/`](docs/decisions/). Start with
[001-chunking-strategy.md](docs/decisions/001-chunking-strategy.md) if you
want to understand why the chunker looks the way it does.

## Known limitations

This section will grow as the project does. Things I already know will be
limitations:

- The gold dataset will be hand-built and small (target: 80-120 queries).
  Statistical claims will be limited accordingly.
- Running everything locally on a single machine means latency numbers
  depend heavily on hardware. I'll document the machine used for benchmarks.
- `gemma3:4b` is a small model. A larger model would likely improve
  generation quality but the point here is reproducibility, not peak
  accuracy.
- No multilingual support. FastAPI docs are English, the retriever is
  English-only.
- Ingestion re-processes everything on each run. For a corpus this size
  that's fine (a few minutes). Incremental re-ingestion is not worth the
  complexity yet.
- `tiktoken` cl100k_base is a proxy for nomic-embed-text's actual tokenizer.
  Close enough for chunk sizing, but token counts in the manifest are
  approximate.
- Heading-based parent segmentation can produce unbalanced parents. Some
  FastAPI doc sections are much longer than 1024 tokens and get accepted
  as oversized rather than split mid-paragraph.

## License

MIT.
