"""Resolve each evidence phrase in questions_source.json to the page(s) it is printed on,
and write eval/questions.json (the ground truth used by all evaluations).

Pages are found in the ingested text, so they match the page numbers the system cites.
If a phrase is not found, the script stops: the ground truth must be exact.

    python -m eval.build_questions
"""
from __future__ import annotations

import json
import re
import sys

from src import config
from src.ingest import ingest_pdf, make_doc_id

HERE = config.ROOT / "eval"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    source = json.loads((HERE / "questions_source.json").read_text(encoding="utf-8"))
    pages_text: dict[str, list[tuple[int, str]]] = {}
    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        doc = ingest_pdf(pdf, use_ocr=False)
        pages_text[make_doc_id(pdf)] = [(p.number, _norm(" ".join(b.text for b in p.blocks)))
                                        for p in doc.pages]
    # Text that runs across a page break is matched on the joined text of two pages.
    for q in source:
        for ev in q["evidence"]:
            phrase = _norm(ev["phrase"])
            pages = [n for n, t in pages_text[ev["doc_id"]] if phrase in t]
            if ev.get("pages"):
                # Pages pinned by hand when the phrase also occurs elsewhere: check them.
                wrong = [n for n in ev["pages"] if n not in pages]
                if wrong:
                    sys.exit(f"{q['id']}: phrase not on pinned page(s) {wrong}")
                pages = ev["pages"]
            if not pages:
                texts = pages_text[ev["doc_id"]]
                pages = [texts[i][0] for i in range(len(texts) - 1)
                         if phrase in texts[i][1] + " " + texts[i + 1][1]]
            if not pages:
                sys.exit(f"{q['id']}: phrase not found in {ev['doc_id']}: {phrase!r}")
            ev["pages"] = pages
            print(f"{q['id']} {ev['doc_id']:<28} p.{pages}  {ev['ref']}")
    out = HERE / "questions.json"
    out.write_text(json.dumps(source, indent=1, ensure_ascii=False), encoding="utf-8")
    counts = {t: sum(q["type"] == t for q in source) for t in ("factual", "multihop", "unanswerable")}
    print(f"\nWrote {len(source)} questions {counts} to {out}")


if __name__ == "__main__":
    main()
