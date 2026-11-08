Retrieval evaluation on 19 answerable questions (embedder BAAI/bge-small-en-v1.5, reranker BAAI/bge-reranker-v2-m3).

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR | Multi-hop all@5 | ms/query |
|---|---|---|---|---|---|---|---|
| BM25 only | 0.658 | 0.868 | 0.974 | 1.000 | 0.860 | 0.833 | 2 |
| Dense only | 0.579 | 0.737 | 0.868 | 0.974 | 0.797 | 0.500 | 20 |
| Hybrid (BM25 + dense, RRF) | 0.789 | 0.842 | 0.921 | 1.000 | 0.930 | 0.833 | 23 |
| Hybrid + clause-reference lookup | 0.711 | 0.763 | 0.842 | 1.000 | 0.842 | 0.667 | 21 |
| Dense + cross-encoder rerank | 0.763 | 0.974 | 1.000 | 1.000 | 0.947 | 1.000 | 497 |
| Hybrid + cross-encoder rerank | 0.816 | 0.974 | 1.000 | 1.000 | 0.974 | 1.000 | 504 |
| Hybrid + refs + rerank (final) | 0.816 | 0.974 | 1.000 | 1.000 | 0.974 | 1.000 | 507 |
