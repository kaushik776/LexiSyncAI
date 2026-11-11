# Latency and cost report

### Per question type (full system)

| Type | Questions | Mean s | Max s | Mean LLM calls | Mean tokens in | Mean tokens out |
|---|---|---|---|---|---|---|
| factual | 13 | 10.1 | 27.5 | 1.5 | 3085 | 181 |
| multihop | 6 | 22.3 | 37.5 | 3.7 | 6710 | 451 |
| unanswerable | 4 | 3.2 | 6.7 | 0.5 | 1380 | 45 |

### Where the time goes (mean seconds per question)

| Retrieval (BM25 + dense) | Reranking | LLM | Other (verifier, planner logic) | Total |
|---|---|---|---|---|
| 0.10 | 0.89 | 10.71 | 0.39 | 12.09 |

Tokens for all 23 questions: 85,883 in, 5,237 out. Money cost: 0 (local models). With a paid API, cost = tokens x its price per token.

### Optimisations measured

| Optimisation | Before | After | Speed-up | Results unchanged? |
|---|---|---|---|---|
| Reranker fp32 -> fp16 (per question, 40 candidates) | 1.49 s | 0.45 s | 3.3x | top-5 overlap 100% |
| NLI verifier fp32 -> bf16 (20 premise/claim pairs) | 0.41 s | 0.16 s | 2.6x | same label 100% |
