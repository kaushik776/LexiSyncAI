"""Bonus feature 11 - latency and cost report.

Part 1 (from eval/results/answer_eval.json, the full system on the 23 questions):
  seconds and tokens per question, by question type and by pipeline stage.
  Cost: everything runs locally (Ollama + open models), so the money cost is 0; tokens are
  reported so the cost of a paid API can be estimated from its price per token.

Part 2 (measured now): two optimisations, before vs after, on the same inputs
  * reranker bge-reranker-v2-m3 in fp32 vs fp16 (40 candidates per question)
  * NLI verifier DeBERTa-v3 in fp32 vs bf16 (claims from the evaluation)
  with speed and whether the results stay the same.

    python -m eval.latency_report        # run with the laptop on mains power
"""
from __future__ import annotations

import json
import statistics
import sys
import time

from src.offline import use_offline_models_if_cached

use_offline_models_if_cached()

import torch  # noqa: E402

from src import config  # noqa: E402
from src.indexer import load_index  # noqa: E402
from src.retriever import HybridRetriever  # noqa: E402

OUT = config.ROOT / "eval" / "results"


def part1() -> str:
    data = json.loads((OUT / "answer_eval.json").read_text(encoding="utf-8"))["full"]["rows"]
    lines = ["### Per question type (full system)", "",
             "| Type | Questions | Mean s | Max s | Mean LLM calls | Mean tokens in | Mean tokens out |",
             "|---|---|---|---|---|---|---|"]
    for t in ("factual", "multihop", "unanswerable"):
        rows = [r for r in data if r["type"] == t]
        lines.append(f"| {t} | {len(rows)} | {statistics.mean(r['seconds'] for r in rows):.1f} | "
                     f"{max(r['seconds'] for r in rows):.1f} | {statistics.mean(r['llm_calls'] for r in rows):.1f} | "
                     f"{statistics.mean(r['prompt_tokens'] for r in rows):.0f} | "
                     f"{statistics.mean(r['completion_tokens'] for r in rows):.0f} |")
    stages = [r["timings"] for r in data if "timings" in r]
    if stages:
        lines += ["", "### Where the time goes (mean seconds per question)", "",
                  "| Retrieval (BM25 + dense) | Reranking | LLM | Other (verifier, planner logic) | Total |",
                  "|---|---|---|---|---|"]
        ret = statistics.mean(s.get("retrieval_s", 0) for s in stages)
        rer = statistics.mean(s.get("rerank_s", 0) for s in stages)
        llm = statistics.mean(s.get("llm_s", 0) for s in stages)
        tot = statistics.mean(s.get("total_s", 0) for s in stages)
        lines.append(f"| {ret:.2f} | {rer:.2f} | {llm:.2f} | {tot - ret - rer - llm:.2f} | {tot:.2f} |")
    total_in = sum(r["prompt_tokens"] for r in data)
    total_out = sum(r["completion_tokens"] for r in data)
    lines += ["", f"Tokens for all {len(data)} questions: {total_in:,} in, {total_out:,} out. "
              "Money cost: 0 (local models). With a paid API, cost = tokens x its price per token."]
    return "\n".join(lines)


def _time(fn, repeat: int = 3) -> float:
    fn()                                         # warm-up
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t) / repeat


def part2() -> str:
    from sentence_transformers import CrossEncoder
    questions = [q["question"] for q in json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))][:8]
    retriever = HybridRetriever(load_index("default"))
    pairs = [(q, h.chunk.context_text) for q in questions for h in retriever.search(q, k=config.CANDIDATES)]
    rows = ["| Optimisation | Before | After | Speed-up | Results unchanged? |", "|---|---|---|---|---|"]

    def run_reranker(dtype):
        kw = {"torch_dtype": dtype} if dtype else {}
        m = CrossEncoder(config.RERANK_MODEL, max_length=512, device="cuda", model_kwargs=kw)
        secs = _time(lambda: m.predict(pairs, batch_size=16, show_progress_bar=False), repeat=1) / len(questions)
        scores = m.predict(pairs, batch_size=16, show_progress_bar=False)
        del m
        torch.cuda.empty_cache()
        return secs, scores

    t32, s32 = run_reranker(None)
    t16, s16 = run_reranker(torch.float16)
    n = config.CANDIDATES
    same_top5 = statistics.mean(len(set(sorted(range(n), key=lambda i: -s32[k * n + i])[:5]) &
                                    set(sorted(range(n), key=lambda i: -s16[k * n + i])[:5])) / 5
                                for k in range(len(questions)))
    rows.append(f"| Reranker fp32 -> fp16 (per question, {n} candidates) | {t32:.2f} s | {t16:.2f} s | "
                f"{t32 / t16:.1f}x | top-5 overlap {same_top5:.0%} |")

    claims = [("The controller shall notify the breach within 72 hours.", "Article 33"),
              ("The time limit for the short-term right to reject is 30 days.", "Section 22"),
              ("The agreement is governed by the laws of the State of Delaware.", "Section 8.9")]
    chunks = load_index("default").chunks
    nli_pairs = [(c.text[:1500], claim) for claim, sid in claims for c in chunks if c.section_id == sid]

    def run_nli(dtype):
        kw = {"torch_dtype": dtype} if dtype else {}
        m = CrossEncoder(config.NLI_MODEL, device="cuda", model_kwargs=kw)
        secs = _time(lambda: m.predict(nli_pairs * 4, batch_size=32, show_progress_bar=False))
        probs = m.predict(nli_pairs, apply_softmax=True, show_progress_bar=False)
        del m
        torch.cuda.empty_cache()
        return secs, probs.argmax(axis=1)

    n32, l32 = run_nli(None)
    n16, l16 = run_nli(torch.bfloat16)
    agree = float((l32 == l16).mean())
    rows.append(f"| NLI verifier fp32 -> bf16 ({len(nli_pairs) * 4} premise/claim pairs) | {n32:.2f} s | {n16:.2f} s | "
                f"{n32 / n16:.1f}x | same label {agree:.0%} |")
    return "### Optimisations measured\n\n" + "\n".join(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = "# Latency and cost report\n\n" + part1() + "\n\n" + part2() + "\n"
    print(report)
    (OUT / "latency_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
