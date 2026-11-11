"""Bonus feature 4 - citation precision at region level.

The app highlights a box on the page for every cited passage. Page-level precision asks
"is the cited page a gold page?"; region-level precision asks the stricter question
"does the highlighted box actually cover the gold evidence text?".

For each answer of the full system (eval/results/answer_eval.json) and each cited passage:
  * the gold evidence phrase is located on its page (PyMuPDF text search -> its boxes),
  * the citation is a region hit if its highlight box covers >= 50% of the phrase's area.
We also report how much of the page the highlight covers (smaller = more precise).

    python -m eval.answer_eval full     # first, so the answers are fresh
    python -m eval.region_eval
"""
from __future__ import annotations

import json
import sys

import pymupdf

from src import config
from src.indexer import load_index

OUT = config.ROOT / "eval" / "results"


def phrase_boxes(pdf: pymupdf.Document, page_no: int, phrase: str) -> list[pymupdf.Rect]:
    page = pdf[page_no - 1]
    words = phrase.split()
    for n in (len(words), 8, 5):                     # long phrases may break across lines: retry shorter
        found = page.search_for(" ".join(words[:n]))
        if found:
            return found
    return []


def covered(box: list[float], targets: list[pymupdf.Rect]) -> float:
    area = sum(t.get_area() for t in targets) or 1.0
    r = pymupdf.Rect(*box)
    return sum((r & t).get_area() for t in targets) / area


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads((OUT / "answer_eval.json").read_text(encoding="utf-8"))["full"]["rows"]
    questions = {q["id"]: q for q in json.loads((config.ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))}
    index = load_index("default")
    chunks = {c.chunk_id: c for c in index.chunks}
    paths = {d["doc_id"]: d.get("path") or str(config.DATA_DIR / d["file"]) for d in index.documents}
    pdfs = {k: pymupdf.open(v) for k, v in paths.items()}

    cited_total = page_hits = region_hits = 0
    area_share = []
    per_q = []
    for r in rows:
        q = questions[r["id"]]
        if not q["evidence"] or r["abstained"] or "cited_chunk_ids" not in r:
            continue
        q_page = q_region = 0
        for cid in r["cited_chunk_ids"]:
            c = chunks.get(cid)
            if c is None:
                continue
            cited_total += 1
            page_hit = region_hit = False
            for ev in q["evidence"]:
                if ev["doc_id"] != c.doc_id:
                    continue
                for span in c.spans:
                    if span["page"] not in ev["pages"]:
                        continue
                    page_hit = True
                    targets = phrase_boxes(pdfs[c.doc_id], span["page"], ev["phrase"])
                    if targets and covered(span["bbox"], targets) >= 0.5:
                        region_hit = True
            for span in c.spans:
                pg = pdfs[c.doc_id][span["page"] - 1].rect
                area_share.append(pymupdf.Rect(*span["bbox"]).get_area() / pg.get_area())
            page_hits += page_hit
            region_hits += region_hit
            q_page += page_hit
            q_region += region_hit
        per_q.append({"id": r["id"], "cited": len(r["cited_chunk_ids"]), "page_hits": q_page, "region_hits": q_region})

    summary = {"cited_passages": cited_total,
               "page_precision": round(page_hits / cited_total, 3) if cited_total else None,
               "region_precision": round(region_hits / cited_total, 3) if cited_total else None,
               "mean_highlight_share_of_page": round(sum(area_share) / len(area_share), 3) if area_share else None}
    md = ("Citation precision at page and region level (answers of the full system).\n\n"
          "| Cited passages | Page-level precision | Region-level precision | Highlight covers (mean share of page) |\n"
          "|---|---|---|---|\n"
          f"| {summary['cited_passages']} | {summary['page_precision']} | {summary['region_precision']} | "
          f"{summary['mean_highlight_share_of_page']} |\n")
    print(md)
    (OUT / "region_eval.md").write_text(md, encoding="utf-8")
    (OUT / "region_eval.json").write_text(json.dumps({"summary": summary, "questions": per_q}, indent=1),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
