Retrieval evaluation on 19 answerable questions (embedder BAAI/bge-small-en-v1.5, reranker BAAI/bge-reranker-v2-m3).

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR | Multi-hop all@5 | ms/query |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.658 | 0.868 | 0.974 | 1.000 | 0.860 | 0.833 | 3 |
| Dense only | 0.605 | 0.868 | 0.895 | 0.974 | 0.815 | 0.667 | 24 |
| Hybrid (BM25 + dense, RRF) | 0.605 | 0.842 | 0.921 | 1.000 | 0.825 | 0.833 | 25 |
| Hybrid + clause-reference lookup | 0.684 | 0.763 | 0.842 | 1.000 | 0.815 | 0.667 | 27 |
| Dense + cross-encoder rerank | 0.868 | 0.921 | 1.000 | 1.000 | 1.000 | 1.000 | 511 |
| Hybrid + cross-encoder rerank | 0.868 | 0.921 | 1.000 | 1.000 | 1.000 | 1.000 | 516 |
| Hybrid + refs + rerank (final) | 0.868 | 0.921 | 1.000 | 1.000 | 1.000 | 1.000 | 519 |
