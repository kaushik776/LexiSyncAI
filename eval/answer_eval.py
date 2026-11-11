"""End-to-end answer evaluation.

For every question in eval/questions.json and every configuration it records the answer,
badge, citations, retries, latency and tokens, and computes:

  * key-fact recall     automatic proxy for correctness: share of the expected key facts
                        (figures, terms) that appear in the answer. Final correctness is
                        graded by hand in answers_for_grading.csv (column "correct").
  * citation precision  share of cited sources whose page holds gold evidence (page level)
  * evidence recall     share of gold evidence items that the citations cover
  * abstention          unanswerable questions: correctly abstained?
                        answerable questions: wrongly abstained?
  * latency, LLM calls and tokens per question

Configurations:
  full            hybrid retrieval + rerank + planner + verifier      (the system)
  no_retrieval    LLM answers from its own knowledge                  (mandatory comparison)
  dense_only      dense retrieval instead of hybrid                   (mandatory comparison)
  no_verifier     full system without the verifier loop               (ablation)
  text_only       full system without table chunks                    (ablation)

    python -m eval.answer_eval                      # all configurations
    python -m eval.answer_eval full no_retrieval    # some of them
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time

from src import config
from src.indexer import load_index
from src.llm import LLM
from src.pipeline import Pipeline

CONFIGS = {
    "full": dict(),
    "no_retrieval": dict(use_retrieval=False),
    "dense_only": dict(retrieval_mode="dense"),
    "no_verifier": dict(use_verifier=False),
    "text_only": dict(exclude_kinds=("table",)),
}
OUT = config.ROOT / "eval" / "results"


def key_fact_recall(answer: str, facts: list[str]) -> float | None:
    if not facts:
        return None
    return sum(bool(re.search(f, answer, re.IGNORECASE)) for f in facts) / len(facts)


def score_question(q: dict, a) -> dict:
    cited = a.cited_sources
    gold = q["evidence"]

    def hits_gold(src):
        return any(src.chunk.doc_id == ev["doc_id"] and
                   any(src.chunk.page_start <= p <= src.chunk.page_end for p in ev["pages"]) for ev in gold)

    row = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "answer": a.answer, "badge": a.badge, "abstained": a.abstained,
        "citations": "; ".join(f"[{s.marker}] {s.chunk.citation}" for s in cited),
        "cited_chunk_ids": [s.chunk.chunk_id for s in cited],
        "timings": a.timings,
        "key_fact_recall": key_fact_recall(a.answer if not a.abstained else "", q.get("key_facts", [])),
        "llm_calls": a.usage.get("llm_calls", 0), "prompt_tokens": a.usage.get("prompt_tokens", 0),
        "completion_tokens": a.usage.get("completion_tokens", 0), "seconds": a.timings.get("total_s", 0),
        "retries": sum(s["action"] == "redraft" for s in a.retry_log),
        "verifier_rejections": sum(s["action"] == "verify" and s["supported"] < s["claims"] for s in a.retry_log),
        "planned": bool(a.sub_questions),
        "retry_log": a.retry_log,
    }
    if gold:
        row["citation_precision"] = sum(hits_gold(s) for s in cited) / len(cited) if cited else None
        row["evidence_recall"] = sum(
            any(s.chunk.doc_id == ev["doc_id"] and any(s.chunk.page_start <= p <= s.chunk.page_end
                                                        for p in ev["pages"]) for s in cited)
            for ev in gold) / len(gold)
    return row


def summarise(rows: list[dict]) -> dict:
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "questions": len(rows),
        "key_fact_recall": mean(r["key_fact_recall"] for r in answerable),
        "key_fact_recall_factual": mean(r["key_fact_recall"] for r in answerable if r["type"] == "factual"),
        "key_fact_recall_multihop": mean(r["key_fact_recall"] for r in answerable if r["type"] == "multihop"),
        "citation_precision": mean(r.get("citation_precision") for r in answerable),
        "evidence_recall": mean(r.get("evidence_recall") for r in answerable),
        "abstention_accuracy_unanswerable": mean(float(r["abstained"]) for r in unanswerable),
        "false_abstention_rate": mean(float(r["abstained"]) for r in answerable),
        "mean_seconds": mean(r["seconds"] for r in rows),
        "p95_seconds": round(sorted(r["seconds"] for r in rows)[int(0.95 * (len(rows) - 1))], 2),
        "mean_llm_calls": mean(r["llm_calls"] for r in rows),
        "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rows),
        "mean_completion_tokens": mean(r["completion_tokens"] for r in rows),
        "questions_with_retry": sum(r["retries"] > 0 for r in rows),
        "badges": {b: sum(r["badge"] == b for r in rows) for b in sorted({r["badge"] for r in rows})},
    }


def run(name: str, questions: list[dict], index, llm) -> dict:
    opts = dict(CONFIGS[name])
    pipe_opts = {k: opts.pop(k) for k in ("exclude_kinds", "retrieval_mode") if k in opts}
    pipe = Pipeline(index, llm, **pipe_opts)
    rows = []
    for q in questions:
        t = time.time()
        a = pipe.ask(q["question"], **opts)
        rows.append(score_question(q, a))
        r = rows[-1]
        print(f"  {name:<12} {q['id']} [{a.badge:<21}] kf={r['key_fact_recall']} "
              f"cp={r.get('citation_precision')} {time.time() - t:5.1f}s  {a.answer[:80]!r}")
    return {"summary": summarise(rows), "rows": rows}


def to_markdown(results: dict) -> str:
    cols = [("key_fact_recall", "Key facts"), ("key_fact_recall_multihop", "Key facts (multi-hop)"),
            ("citation_precision", "Citation precision"), ("evidence_recall", "Evidence recall"),
            ("abstention_accuracy_unanswerable", "Abstention acc."), ("false_abstention_rate", "False abstention"),
            ("mean_seconds", "Mean s"), ("mean_prompt_tokens", "Tokens in"), ("mean_completion_tokens", "Tokens out")]
    lines = ["| Configuration | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, r in results.items():
        s = r["summary"]
        lines.append(f"| {name} | " + " | ".join("-" if s[c] is None else f"{s[c]}" for c, _ in cols) + " |")
    return "\n".join(lines)


def main(names: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    index, llm = load_index("default"), LLM()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    if (OUT / "answer_eval.json").exists():          # keep earlier configurations when running a subset
        results = json.loads((OUT / "answer_eval.json").read_text(encoding="utf-8"))
    for name in names or list(CONFIGS):
        print(f"\n=== {name} ===")
        results[name] = run(name, questions, index, llm)
        print(json.dumps(results[name]["summary"], indent=1))
        (OUT / "answer_eval.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    md = to_markdown({k: results[k] for k in CONFIGS if k in results})
    (OUT / "answer_eval.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)

    if "full" in results:                               # sheet for manual correctness grading
        with open(OUT / "answers_for_grading.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "question", "expected_answer", "system_answer", "badge", "citations",
                        "key_fact_recall", "correct (1 / 0.5 / 0)", "notes"])
            expected = {q["id"]: q["answer"] for q in questions}
            for r in results["full"]["rows"]:
                w.writerow([r["id"], r["type"], r["question"], expected[r["id"]], r["answer"], r["badge"],
                            r["citations"], r["key_fact_recall"], "", ""])
        print(f"\nManual grading sheet: {OUT / 'answers_for_grading.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
