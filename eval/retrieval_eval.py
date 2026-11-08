"""Retrieval evaluation: recall@k per configuration (includes the mandatory
dense-only vs hybrid comparison).

A gold evidence item counts as retrieved at k if one of the top-k chunks comes from the
same document and covers one of its pages. Per question, recall = share of its evidence
items retrieved; we average over the answerable questions.

    python -m eval.retrieval_eval
"""
from __future__ import annotations

import json
import sys
import time

from src import config
from src.indexer import load_index
from src.reranker import Reranker
from src.retriever import HybridRetriever

KS = (1, 3, 5, 10)
CONFIGS = [   # (name, retrieval mode, rerank?)
    ("BM25 only", "bm25", False),
    ("Dense only", "dense", False),
    ("Hybrid (BM25 + dense, RRF)", "hybrid", False),
    ("Hybrid + clause-reference lookup", "hybrid+refs", False),
    ("Dense + cross-encoder rerank", "dense", True),
    ("Hybrid + cross-encoder rerank", "hybrid", True),
    ("Hybrid + refs + rerank (final)", "hybrid+refs", True),
]


def covers(chunk, ev) -> bool:
    return chunk.doc_id == ev["doc_id"] and any(chunk.page_start <= p <= chunk.page_end for p in ev["pages"])


def evaluate_config(retriever, reranker, questions, mode, rerank) -> dict:
    per_q, latencies = [], []
    for q in questions:
        t = time.time()
        if rerank:
            hits = reranker.rerank(q["question"], retriever.search(q["question"], k=config.CANDIDATES, mode=mode),
                                   k=max(KS))
        else:
            hits = retriever.search(q["question"], k=max(KS), mode=mode)
        latencies.append(time.time() - t)
        chunks = [h.chunk for h in hits]
        row = {"id": q["id"], "type": q["type"]}
        for k in KS:
            found = [any(covers(c, ev) for c in chunks[:k]) for ev in q["evidence"]]
            row[f"recall@{k}"] = sum(found) / len(found)
            row[f"complete@{k}"] = float(all(found))
        first = next((r for r, c in enumerate(chunks, 1) if any(covers(c, ev) for ev in q["evidence"])), None)
        row["rr"] = 1 / first if first else 0.0
        per_q.append(row)

    def mean(key, rows):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None

    result = {f"recall@{k}": mean(f"recall@{k}", per_q) for k in KS}
    result["MRR"] = mean("rr", per_q)
    multihop = [r for r in per_q if r["type"] == "multihop"]
    result.update({f"multihop_all_evidence@{k}": mean(f"complete@{k}", multihop) for k in (5, 10)})
    result["ms_per_query"] = round(1000 * sum(latencies) / len(latencies), 1)
    result["per_question"] = per_q
    return result


def to_markdown(results: dict) -> str:
    head = "| Configuration | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | Multi-hop all@5 | ms/query |"
    lines = [head, "|" + "---|" * (len(KS) + 4)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS)
                     + f" | {r['MRR']:.3f} | {r['multihop_all_evidence@5']:.3f} | {r['ms_per_query']:.0f} |")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    questions = [q for q in json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
                 if q["evidence"]]
    retriever = HybridRetriever(load_index("default"))
    reranker = Reranker()
    retriever.search("warm up", k=1, mode="dense")             # load models before timing
    reranker.scores("warm up", ["warm up"])

    results = {}
    for name, mode, rerank in CONFIGS:
        results[name] = evaluate_config(retriever, reranker, questions, mode, rerank)
        r = results[name]
        print(f"{name:<36} R@1 {r['recall@1']:.3f}  R@5 {r['recall@5']:.3f}  R@10 {r['recall@10']:.3f}  "
              f"MRR {r['MRR']:.3f}  {r['ms_per_query']:.0f} ms")

    md = to_markdown(results)
    print("\n" + md)
    out = config.ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "retrieval_eval.md").write_text(
        f"Retrieval evaluation on {len(questions)} answerable questions "
        f"(embedder {config.EMBED_MODEL}, reranker {config.RERANK_MODEL}).\n\n{md}\n", encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
