# ADR 005: Evaluation Strategy

## Status
Accepted -- 2026-04-16

## Context

ADRs 001-004 cover ingestion, hybrid retrieval, and the streaming query API.
What they do not cover is the question every RAG project eventually has to
answer: *is this thing actually good?* "Good" needs a definition, and the
definition needs to be reproducible, version-controlled, and runnable on demand
without paying for a hosted eval service.

Concretely we want to answer four questions per query:

1. Is the generated answer **grounded** in the retrieved context, or did the
   model hallucinate?
2. Does the generated answer actually **address** the question?
3. Was the **retrieval** good enough -- did it surface the chunks needed to
   answer?
4. Did we **cite** the right sources?

The first three are well-trodden territory; the fourth is specific to this
project because every `/query` response includes citations and we need to
measure whether those citations are correct.

## Decision

Use **RAGAS** for the standard generation/retrieval-quality metrics and a
**custom citation_accuracy** metric for source attribution. Run them against
a hand-curated **golden dataset of 30 queries** spanning five categories of
real FastAPI questions. Drive the entire pipeline with the same Ollama models
that serve `/query` so the eval mirrors production behaviour.

### Why RAGAS over alternatives

We considered three options:

- **RAGAS**: industry-standard, four-metric suite explicitly built for RAG,
  metrics are well-documented and the underlying prompts are inspectable.
- **TruLens**: similar metric set but heavier dependency footprint and
  pulls in observability infrastructure we already have via Langfuse.
- **Custom only**: writing our own faithfulness scorer is a six-month
  project in itself; we'd be reinventing nontrivial NLI prompts.

RAGAS gives us a defensible baseline number for free. The cost is a chunky
transitive dependency on LangChain (RAGAS uses langchain-core for its
prompt scaffolding). We accept this: the **"no LangChain" rule from the
project standards applies to the request hot path** (orchestration), not
to eval-time tooling that runs once per release.

### The four RAGAS metrics

We use exactly four `ragas.metrics.collections` metrics, picked to cover one
failure mode each:

| Metric | What it detects |
|---|---|
| `Faithfulness` | answer claims not supported by retrieved context (hallucination) |
| `AnswerRelevancy` | answer drifts off-topic from the question |
| `ContextPrecision` | retriever returned irrelevant chunks alongside the right ones |
| `ContextRecall` | retriever missed chunks needed to answer the question |

Faithfulness is the one we care about most -- a confidently wrong answer is
the worst RAG failure mode and the one users will not catch. The
context_precision/recall pair tells us *which side of the pipeline* to fix
when faithfulness drops: low context_recall is a retrieval problem; high
context_recall but low faithfulness is a generation problem.

### Why a custom citation_accuracy metric

RAGAS does not measure source attribution. The `/query` endpoint returns
`citations` derived from retrieval results, and we want a direct number for
"of the sources we returned, how many were the right ones." The metric is
F1 over the set of returned source IDs vs. the set of expected source IDs:

```python
citation_accuracy(returned_sources, expected_sources)
# -> {"precision": ..., "recall": ..., "f1": ...}
```

Source IDs are the relative paths under `data/raw/fastapi_docs/`
(e.g. `tutorial/handling-errors.md`) -- the same `source_file` field stored
in the Qdrant payload during ingestion. Path-level matching is the right
granularity for now: chunk-level matching is too brittle (chunk boundaries
shift on ingestion changes) and document-level matching matches the user
mental model ("did it cite the right doc?").

### Thresholds

The runner pass/fail thresholds are:

| Metric | Threshold | Rationale |
|---|---|---|
| `faithfulness` | 0.75 | Production RAG papers report 0.75-0.90 with strong models. With `gemma3:4b` we expect to land near the floor; below 0.75 means the model is making up facts more than 1 in 4 claims. |
| `answer_relevancy` | 0.70 | Generous floor -- even an off-topic answer scores ~0.5 because the question paraphrase is similar in embedding space. |
| `context_precision` | 0.70 | Hybrid + rerank should easily clear this; below means the reranker is not separating signal from noise. |
| `context_recall` | 0.65 | Recall is harder than precision when each query maps to one or two source files; 0.65 means at least two of three queries fully recall their source. |
| `citation_f1` | 0.60 | One-source-per-query queries with `final_k=5` cap precision at 0.20, so the F1 floor reflects "the right doc is in the top 5 most of the time". |

These are **opinionated starting thresholds, not industry standards**. They
exist to flag regressions, not to claim absolute quality. Expect to revisit
them after the first three eval runs.

If a metric has zero successful samples (all `None` due to LLM structured-
output failures), the threshold is treated as failing. Silent zero coverage
is exactly the failure mode we want to catch.

### Golden dataset construction

30 queries, five categories, all hand-written by reading the actual FastAPI
markdown corpus at the pinned commit (see `fetcher.py`):

- **factual (8)**: short definitional answers (e.g. *"What is a path
  parameter in FastAPI?"*).
- **conceptual (6)**: broader explanations (e.g. *"How does dependency
  injection work?"*).
- **code (6)**: how-do-I queries that should produce code-shaped answers.
- **error_handling (5)**: HTTPException, custom handlers, validation
  override, etc.
- **advanced (5)**: pydantic-settings, lifespan, root_path, GZip,
  OAuth2+JWT.

Each entry carries:

- `query` -- the user question.
- `reference_answer` -- a 1-3 sentence ground truth answer used by
  `context_precision` and `context_recall`.
- `expected_answer_keywords` -- a small list of terms a correct answer
  should contain (used for quick keyword-coverage stats; not currently
  threshold-gated).
- `expected_sources` -- relative paths under `data/raw/fastapi_docs/`
  matching `source_file` in the Qdrant payload.

A test (`test_golden_dataset_expected_sources_resolve`) asserts that every
`expected_source` path actually exists in the corpus. If the FastAPI repo
is re-pinned to a newer commit and a doc moves, the test fails immediately
and the dataset gets updated with intent.

The dataset is **deliberately small**. 30 queries is enough to see large
regressions, not enough for fine-grained statistical claims about model
quality. Phase 5.2 will scale it toward 80-120 queries; until then,
treat aggregate numbers as directional, not absolute.

### Driving with Ollama

RAGAS' `ragas.llms.llm_factory` and `ragas.embeddings.OpenAIEmbeddings` both
accept any OpenAI-compatible client. We point them at Ollama's `/v1`
endpoint with a single `AsyncOpenAI` client (both LLM and embeddings use
`agenerate`/`aembed_text` from inside RAGAS' `.ascore()`). This means **the
exact same models that serve `/query` (`gemma3:4b` + `nomic-embed-text`)
also judge their own output**. That's a deliberate trade-off:

- *Pro*: no additional infrastructure, no API keys, fully reproducible.
- *Con*: a small judge model is a noisy judge. Faithfulness and
  context_precision in particular benefit from a stronger judge.

When this gets in the way, the backend is swappable -- the
`build_ollama_backend()` factory is one of two callers of `llm_factory`,
and a future ADR can introduce a different judge (`gpt-4o-mini`,
`claude-haiku`, or a stronger local model) for nightly runs.

### Per-metric error handling

`run_ragas_metrics()` computes each metric independently and traps
exceptions per-metric. Small models occasionally emit invalid JSON for
RAGAS' Instructor-driven structured outputs; rather than aborting the
whole run, the offending metric returns `None` and is excluded from the
mean (with the success count reported as `*_n` in the aggregate). A
single bad query should not cost a 10-minute run.

### When to re-run

- After ingestion changes (new corpus pin, chunker tuning, embedding
  model swap).
- After retrieval changes (reranker swap, RRF tuning, new sparse weights).
- After LLM swap or prompt template edit.
- Monthly as a baseline drift check, even if nothing changed.

CI is **not** wired to run evals on every commit. The runtime (5-10 min on
a 30-query dataset with `gemma3:4b`) and the local-Ollama dependency make
this prohibitive in GitHub Actions. A manual-dispatch workflow may land
once the CI billing issue is resolved (see roadmap), gated behind
`workflow_dispatch`.

## Consequences

### Positive

- A single command produces a reproducible quality number, written to
  `tests/eval/results/eval_<timestamp>.json` for diffing.
- Per-metric coverage means we can tell "the metric is bad" from "the
  judge model couldn't compute the metric" -- two very different
  failure modes.
- Citation accuracy is measured directly, not inferred from generation
  quality.
- The runner is sync and self-contained -- one entry point, no
  framework magic.

### Negative

- LangChain enters the dependency graph as a transitive dep of RAGAS.
  Acceptable because it lives entirely in eval-time code; the request
  path stays clean.
- Using a small local model as both producer and judge inflates
  agreement scores. Numbers should be read as relative (run-to-run
  comparison), not absolute (vs. published benchmarks).
- 30 queries is too small for statistical significance on subgroup
  analysis (e.g. "is the retriever worse on `error_handling` queries
  than on `factual` queries?"). Categories are reported but not
  threshold-gated.

### Tradeoffs accepted

- We optimise for **reproducibility on a developer laptop** over
  judge-model quality. Anyone with Docker + Ollama can rerun the
  whole suite.
- We measure **what we ship** (`gemma3:4b` answers + hybrid retrieval)
  rather than what an idealised system might produce. This is the
  right baseline for catching regressions.

## Alternatives Considered

1. **TruLens**: comparable metrics, larger surface area, redundant with
   Langfuse for tracing. Rejected.
2. **Hand-rolled NLI faithfulness scorer**: large investment for marginal
   gain over RAGAS' battle-tested prompts. Rejected.
3. **LLM-as-judge with GPT-4o-mini**: higher quality numbers, but
   reintroduces an external dependency and a per-run cost. Deferred to
   a future ADR.
4. **Pairwise preference eval (RankNet style)**: useful for ranking
   prompt variants against each other, but doesn't give absolute quality
   numbers. Considered for Phase 5.2.
5. **CI on every PR**: would catch regressions earlier but the runtime
   makes it infeasible until evals are sub-minute. Will revisit when
   the dataset and judge model stabilise.
