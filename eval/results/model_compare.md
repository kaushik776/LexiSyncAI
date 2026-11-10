| Setup | Question | Badge | Key facts | Total s | LLM s | LLM calls | Tokens out |
|---|---|---|---|---|---|---|---|
| qwen2.5:7b (current) | F04 (factual) | HIGH | 1.00 | 21.5 | 13.158 | 2 | 215 |
| qwen2.5:7b (current) | M01 (multihop) | MEDIUM | 0.50 | 28.4 | 25.915 | 4 | 560 |
| qwen2.5:7b (current) | U03 (unanswerable) | INSUFFICIENT EVIDENCE | - | 6.4 | 5.789 | 1 | 63 |
| qwen3:8b, thinking off | F04 (factual) | HIGH | 1.00 | 42.3 | 40.487 | 2 | 221 |
| qwen3:8b, thinking off | M01 (multihop) | MEDIUM | 0.50 | 65.5 | 63.804 | 2 | 331 |
| qwen3:8b, thinking off | U03 (unanswerable) | INSUFFICIENT EVIDENCE | - | 25.8 | 25.282 | 1 | 108 |
| qwen3:8b, thinking on | F04 (factual) | HIGH | 1.00 | 136.9 | 135.066 | 2 | 1168 |
| qwen3:8b, thinking on | M01 (multihop) | HIGH | 0.50 | 168.9 | 167.158 | 2 | 1220 |
| qwen3:8b, thinking on | U03 (unanswerable) | INSUFFICIENT EVIDENCE | - | 248.4 | 246.588 | 3 | 1489 |

| Setup | Mean total s | Mean tokens out | Correct abstention |
|---|---|---|---|
| qwen2.5:7b (current) | 18.8 | 279 | 3/3 |
| qwen3:8b, thinking off | 44.5 | 220 | 3/3 |
| qwen3:8b, thinking on | 184.7 | 1292 | 3/3 |
