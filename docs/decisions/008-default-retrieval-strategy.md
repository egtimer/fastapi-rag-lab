# ADR 008: Plain Hybrid as the Default Retrieval Strategy

## Status
Accepted -- 2026-10-20

## Context

ADR 003 put a cross-encoder (`BAAI/bge-reranker-base`, pinned per ADR 007)
after the RRF fusion of dense and BM25 results, and the API reranked every
query. The April benchmark already hinted that the reranker did not help, but
it had no confidence intervals and only 30 questions.

The decision is made on the Stack Overflow **dev** split (143 questions), on
the reference runtime (Ollama 0.35.1), before the test split is run. The
test split is run once afterwards to report the chosen configuration; nothing
is changed after seeing it.

Run: `reports/2026-10-20-so-dev-ollama-0.35.1`, made with

```bash
uv run --frozen python tests/eval/run_benchmarks.py --dataset tests/eval/stackoverflow/dev.json \
  --strategies sparse,dense,hybrid,hybrid_rerank --top-k 3,5,10 \
  --run-id 2026-10-20-so-dev-ollama-0.35.1 --warmup 1 --resume
```

Confidence intervals are 95% percentile bootstrap over questions (10,000
resamples, seed 0), from the run's `analysis.json`. Labels are the docs pages
that Stack Overflow answerers linked, so absolute recall is low and noisy; the
paired differences between strategies on the same questions are what matter.

## Results (dev, n = 143)

| Config | Recall@k | MRR | nDCG@k | p50 / p95 latency |
|---|---|---|---|---|
| hybrid, k=3 | 0.234 [0.174, 0.297] | 0.260 [0.197, 0.325] | 0.219 [0.166, 0.276] | 82 / 104 ms |
| hybrid + rerank, k=3 | 0.209 [0.149, 0.272] | 0.199 [0.142, 0.258] | 0.181 [0.130, 0.234] | 6,771 / 9,707 ms |
| hybrid, k=5 | 0.284 [0.220, 0.350] | 0.278 [0.215, 0.343] | 0.239 [0.184, 0.294] | 84 / 103 ms |
| hybrid + rerank, k=5 | 0.279 [0.214, 0.346] | 0.226 [0.169, 0.283] | 0.213 [0.163, 0.266] | 6,805 / 9,698 ms |
| hybrid, k=10 | 0.371 [0.303, 0.440] | 0.296 [0.235, 0.360] | 0.272 [0.220, 0.327] | 82 / 103 ms |
| hybrid + rerank, k=10 | 0.383 [0.316, 0.453] | 0.248 [0.194, 0.304] | 0.254 [0.204, 0.305] | 6,800 / 9,718 ms |

Paired differences, hybrid + rerank minus hybrid (wins / losses / ties for the
reranker):

| k | Recall@k | MRR | nDCG@k |
|---|---|---|---|
| 3 | −0.026 [−0.084, +0.034] (12/21/110) | −0.061 [−0.121, +0.000] (15/31/97) | −0.039 [−0.088, +0.013] (16/31/96) |
| 5 | −0.005 [−0.057, +0.046] (11/14/118) | −0.052 [−0.105, +0.001] (19/36/88) | −0.025 [−0.067, +0.018] (20/36/87) |
| 10 | +0.013 [−0.023, +0.049] (9/5/129) | −0.048 [−0.098, +0.001] (22/43/78) | −0.019 [−0.052, +0.015] (24/45/74) |

The golden set on the same runtime (`reports/2026-10-20-golden30-ollama-0.35.1`,
n = 30) points the same way, more strongly: at k=5 the reranker changes
Recall@5 by +0.000 [−0.100, +0.100] and MRR by −0.138 [−0.256, −0.025].

Latency was measured on the benchmark machine recorded in the provenance: AMD
EPYC 7763, 4 logical cores, 16 GB RAM, CPU only. The reranker scores 15
candidates per query with a cross-encoder on the CPU; dense and hybrid stay
around 100 ms at p95.

## Decision

The API default is `hybrid` (dense + BM25 with RRF, no reranker), in
`RAG_RETRIEVAL_STRATEGY` and in `HybridRetriever`. `hybrid_rerank` stays
available with `RAG_RETRIEVAL_STRATEGY=hybrid_rerank`. Citations report the
score that ordered the results: RRF for `hybrid`, the cross-encoder score for
`hybrid_rerank`.

## Why

This is a negative result. The reranker did not improve any metric on dev: no
paired difference has a confidence interval above zero, and MRR is lower at
every k, with intervals that end at or just above zero (+0.001 at most). On
the golden set the MRR loss is outside the noise. It costs about 80 times the
latency (p50 6.8 s against 82 ms on this machine).

It is not evidence that cross-encoders do not work for this corpus. A larger or
domain-tuned reranker, a GPU, or a different candidate count were not tested.
The finding is about this model, this configuration and these labels.

## Test split, after the decision

The sections above were written and committed before the test split was run.
The test split (221 questions) was then run once, with the same grid and the
code of that commit (`reports/2026-10-20-so-test-ollama-0.35.1`). Nothing was
changed after seeing it.

Paired differences on test, hybrid + rerank minus hybrid:

| k | Recall@k | MRR | nDCG@k |
|---|---|---|---|
| 3 | +0.015 [−0.023, +0.054] (18/17/186) | +0.025 [−0.020, +0.072] (31/27/163) | +0.017 [−0.021, +0.057] (29/29/163) |
| 5 | +0.009 [−0.026, +0.043] (14/12/195) | +0.024 [−0.016, +0.065] (40/32/149) | +0.015 [−0.017, +0.047] (40/33/148) |
| 10 | +0.013 [−0.020, +0.046] (13/11/197) | +0.025 [−0.014, +0.064] (47/44/130) | +0.016 [−0.013, +0.046] (45/46/130) |

The sign of the MRR difference flips between dev and test, and every interval
on both splits contains zero. Test does not confirm that the reranker hurts on
Stack Overflow questions; together, the two splits show no detectable effect in
either direction. The golden set still shows an MRR loss. What holds on every
dataset is the latency: p50 6,786 ms with the reranker against 78 ms without on
test. With no demonstrated quality gain, plain `hybrid` stays the default.

## Consequences

- The default query path no longer loads the cross-encoder, so the API starts
  faster and uses less memory.
- `relevance_score` in citations is on a different scale for each strategy
  (RRF scores are around 0.01-0.03); clients should not compare them across
  strategies.
- The weekly retrieval eval keeps benchmarking `hybrid`; a reranker can be
  re-evaluated with `--strategies hybrid,hybrid_rerank` on dev before any
  change of default.
