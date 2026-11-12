"""Scanned vs original: does OCR hurt retrieval and answers?

The "scanned_pdf" case holds the same 20-page SEC exhibit twice: a scanned copy (read with
OCR) and the digital original. The questions in eval/scanned_questions.json are asked
twice, once searching only the scanned copy and once only the original. The gold pages
are the same for both, so any difference comes from the OCR text.

  Part 1  retrieval recall@k and MRR (same method as eval/retrieval_eval.py, no LLM)
  Part 2  answers with the full system (same scoring as eval/answer_eval.py): key facts,
          citation precision, evidence recall, abstention on unanswerable questions, latency

    python -m eval.scanned_eval                  # both parts (needs Ollama)
    python -m eval.scanned_eval --retrieval-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from eval.answer_eval import score_question, summarise
from eval.retrieval_eval import KS, evaluate_config
from src import config
from src.indexer import load_index
from src.ingest import ingest_pdf
from src.llm import LLM
from src.pipeline import Pipeline
from src.reranker import Reranker
from src.retriever import HybridRetriever

CASE = "scanned_pdf"
DOCS = {"Original (digital)": "EX_10_3_original", "Scanned (OCR)": "EX_10_3_scanned"}
ORIGINAL_PDF = config.DATA_DIR / "scanned" / "EX-10.3_original.pdf"
QUESTIONS = config.ROOT / "eval" / "scanned_questions.json"
OUT = config.ROOT / "eval" / "results"
RETRIEVAL_CONFIGS = [   # (name, retrieval mode, rerank?)
    ("BM25 only", "bm25", False),
    ("Dense only", "dense", False),
    ("Hybrid (BM25 + dense, RRF)", "hybrid", False),
    ("Hybrid + refs + rerank (final)", "hybrid+refs", True),
]


def _norm(s: str) -> str:
    s = s.replace(" ", " ").replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def load_questions() -> list[dict]:
    """Load the questions and check that every evidence phrase is on its page in the original."""
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    pages = {p.number: _norm(" ".join(b.text for b in p.blocks))
             for p in ingest_pdf(ORIGINAL_PDF, use_ocr=False).pages}
    for q in questions:
        for ev in q["evidence"]:
            text = " ".join(pages[n] for n in ev["pages"])
            if _norm(ev["phrase"]) not in text:
                sys.exit(f"{q['id']}: phrase not found on page(s) {ev['pages']}: {ev['phrase']!r}")
    return questions


def for_doc(questions: list[dict], doc_id: str) -> list[dict]:
    """The same questions with the gold evidence placed in one document of the case."""
    return [{**q, "evidence": [{**ev, "doc_id": doc_id} for ev in q["evidence"]]} for q in questions]


class DocRetriever:
    """Searches only one document of the case (the app's per-document filter)."""

    def __init__(self, retriever: HybridRetriever, doc_id: str):
        self.retriever, self.doc_id = retriever, doc_id

    def search(self, query, k=config.TOP_K, mode="hybrid+refs", **kw):
        return self.retriever.search(query, k=k, mode=mode, doc_ids=[self.doc_id], **kw)


def retrieval_part(questions: list[dict], index) -> dict:
    retriever, reranker = HybridRetriever(index), Reranker()
    retriever.search("warm up", k=1, mode="dense")
    reranker.scores("warm up", ["warm up"])
    answerable = [q for q in questions if q["evidence"]]
    results = {}
    for label, doc_id in DOCS.items():
        results[label] = {}
        for name, mode, rerank in RETRIEVAL_CONFIGS:
            r = evaluate_config(DocRetriever(retriever, doc_id), reranker, for_doc(answerable, doc_id), mode, rerank)
            results[label][name] = r
            print(f"{label:<20} {name:<32} R@1 {r['recall@1']:.3f}  R@5 {r['recall@5']:.3f}  MRR {r['MRR']:.3f}")
    return results


def answer_part(questions: list[dict], index) -> dict:
    pipe = Pipeline(index, LLM())
    results = {}
    for label, doc_id in DOCS.items():
        rows = []
        for q in for_doc(questions, doc_id):
            a = pipe.ask(q["question"], doc_ids=[doc_id])
            rows.append(score_question(q, a))
            r = rows[-1]
            print(f"  {label:<20} {q['id']} [{a.badge:<12}] kf={r['key_fact_recall']} "
                  f"cp={r.get('citation_precision')} {r['seconds']:5.1f}s  {a.answer[:70]!r}")
        results[label] = {"summary": summarise(rows), "rows": rows}
    return results


def to_markdown(questions: list[dict], retrieval: dict, answers: dict | None) -> str:
    n_ans = sum(bool(q["evidence"]) for q in questions)
    lines = [f"The same {len(questions)} questions ({n_ans} answerable, {len(questions) - n_ans} unanswerable) "
             "asked of each copy of the SEC exhibit, searching only that copy. Gold pages are identical, so the "
             "difference between the rows is the effect of OCR.", "", "**Retrieval** (no LLM)", "",
             "| Copy | Configuration | " + " | ".join(f"R@{k}" for k in KS) + " | MRR | Multi-hop all@5 |",
             "|" + "---|" * (len(KS) + 4)]
    for label, configs in retrieval.items():
        for name, r in configs.items():
            lines.append(f"| {label} | {name} | " + " | ".join(f"{r[f'recall@{k}']:.3f}" for k in KS)
                         + f" | {r['MRR']:.3f} | {r['multihop_all_evidence@5']:.3f} |")
    if answers:
        cols = [("key_fact_recall", "Key facts"), ("key_fact_recall_multihop", "Key facts (multi-hop)"),
                ("citation_precision", "Citation precision"), ("evidence_recall", "Evidence recall"),
                ("abstention_accuracy_unanswerable", "Abstention acc."),
                ("false_abstention_rate", "False abstention"), ("mean_seconds", "Mean s")]
        lines += ["", "**Answers** (full system: hybrid retrieval, rerank, planner, verifier)", "",
                  "| Copy | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for label, r in answers.items():
            s = r["summary"]
            lines.append(f"| {label} | " + " | ".join("-" if s[c] is None else f"{s[c]}" for c, _ in cols) + " |")
        lines += ["", "Per question (key-fact recall and badge):", "",
                  "| Question | " + " | ".join(answers) + " |", "|---|" + "---|" * len(answers)]
        by_label = {label: {row["id"]: row for row in r["rows"]} for label, r in answers.items()}
        for q in questions:
            cells = []
            for label in answers:
                row = by_label[label][q["id"]]
                kf = "abstained" if row["abstained"] else ("-" if row["key_fact_recall"] is None
                                                          else f"{row['key_fact_recall']:.2f}")
                cells.append(f"{kf} ({row['badge']})")
            lines.append(f"| {q['id']} {q['question'][:70]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="skip the answer part (no LLM needed)")
    args = ap.parse_args()
    questions = load_questions()
    index = load_index(CASE)
    retrieval = retrieval_part(questions, index)
    answers = None if args.retrieval_only else answer_part(questions, index)
    if answers:
        for label, r in answers.items():
            print(label, json.dumps(r["summary"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scanned_eval.json").write_text(json.dumps({"retrieval": retrieval, "answers": answers}, indent=1,
                                                      ensure_ascii=False), encoding="utf-8")
    (OUT / "scanned_eval.md").write_text(to_markdown(questions, retrieval, answers), encoding="utf-8")
    print(f"Saved {OUT / 'scanned_eval.md'} and scanned_eval.json")


if __name__ == "__main__":
    main()
