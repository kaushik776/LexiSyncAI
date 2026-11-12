"""OCR accuracy on a real scanned document, checked against its original digital version.

The "scanned_pdf" case holds the same SEC exhibit twice (data/scanned/):
  * EX-10.3_scanned.pdf   every page is an image, so the app must read it with OCR
  * EX-10.3_original.pdf  the digital original, whose text is the ground truth

Both have the same 20 pages. For each page we compare the text the app gets from the
scanned page (OCR) with the text it gets from the original page, after the same cleaning
(headers and footers removed), and report:

  * detected as scanned   the app noticed the page is an image and switched to OCR
  * character accuracy    1 - character error rate (edit distance / length of the original)
  * word accuracy         1 - word error rate
  * OCR confidence        the OCR engine's own confidence (needs no ground truth)

    python -m eval.ocr_eval
"""
from __future__ import annotations

import json
import re
import sys
import time

from src import config
from src.ingest import ingest_pdf

FOLDER = config.DATA_DIR / "scanned"
SCANNED = FOLDER / "EX-10.3_scanned.pdf"
ORIGINAL = FOLDER / "EX-10.3_original.pdf"
OUT = config.ROOT / "eval" / "results"


def _normalise(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def page_text(page) -> str:
    return _normalise(" ".join(b.text for b in page.blocks))


def evaluate() -> dict:
    t0 = time.time()
    scanned = ingest_pdf(SCANNED, detect_tables=False, use_ocr=True)
    ocr_seconds = time.time() - t0
    original = ingest_pdf(ORIGINAL, detect_tables=False, use_ocr=False)
    if scanned.n_pages != original.n_pages:
        sys.exit(f"Page counts differ: scanned {scanned.n_pages}, original {original.n_pages}")

    rows = []
    for s, o in zip(scanned.pages, original.pages):
        ocr, truth = page_text(s), page_text(o)
        cer = edit_distance(ocr, truth) / max(1, len(truth))
        wer = edit_distance(ocr.split(), truth.split()) / max(1, len(truth.split()))
        rows.append({
            "page": s.number, "detected_as_scanned": s.is_scanned, "ocr_engine": s.ocr_engine or "none",
            "ocr_confidence": s.ocr_confidence, "char_accuracy": round(max(0.0, 1 - cer), 4),
            "word_accuracy": round(max(0.0, 1 - wer), 4), "words_original": len(truth.split()),
            "words_ocr": len(ocr.split()),
        })
    n = len(rows)
    total_chars = sum(len(page_text(o)) for o in original.pages)
    for r in rows:
        if r["ocr_confidence"] is not None:
            r["ocr_confidence"] = round(float(r["ocr_confidence"]), 4)
    confs = [r["ocr_confidence"] for r in rows if r["ocr_confidence"] is not None]
    summary = {
        "document": SCANNED.name, "ground_truth": ORIGINAL.name, "pages": n,
        "detected_as_scanned": sum(r["detected_as_scanned"] for r in rows),
        "ocr_engines": sorted({r["ocr_engine"] for r in rows}),
        "mean_char_accuracy": round(sum(r["char_accuracy"] for r in rows) / n, 4),
        "mean_word_accuracy": round(sum(r["word_accuracy"] for r in rows) / n, 4),
        "worst_page_word_accuracy": min(r["word_accuracy"] for r in rows),
        "mean_ocr_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "seconds_per_page": round(ocr_seconds / n, 2),
        "original_chars": total_chars,
    }
    return {"summary": summary, "pages": rows}


def to_markdown(result: dict) -> str:
    s = result["summary"]
    lines = [
        f"OCR on a real scanned document: `{s['document']}` ({s['pages']} image-only pages), "
        f"checked against its digital original `{s['ground_truth']}`.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pages detected as scanned | {s['detected_as_scanned']}/{s['pages']} |",
        f"| OCR engine | {', '.join(s['ocr_engines'])} |",
        f"| Character accuracy (mean per page) | {s['mean_char_accuracy']:.1%} |",
        f"| Word accuracy (mean per page) | {s['mean_word_accuracy']:.1%} |",
        f"| Worst page, word accuracy | {s['worst_page_word_accuracy']:.1%} |",
        f"| OCR engine confidence (mean) | "
        f"{'-' if s['mean_ocr_confidence'] is None else format(s['mean_ocr_confidence'], '.1%')} |",
        f"| OCR time per page | {s['seconds_per_page']} s |",
        "",
        "Per page:",
        "",
        "| Page | Scanned? | Engine | Char accuracy | Word accuracy | Confidence | Words (original / OCR) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["pages"]:
        conf = "-" if r["ocr_confidence"] is None else f"{r['ocr_confidence']:.1%}"
        lines.append(f"| {r['page']} | {'yes' if r['detected_as_scanned'] else 'no'} | {r['ocr_engine']} | "
                     f"{r['char_accuracy']:.1%} | {r['word_accuracy']:.1%} | {conf} | "
                     f"{r['words_original']} / {r['words_ocr']} |")
    lines += ["", "Page 4 is the signature page: few words in a two-column layout, so the OCR reads them in a "
              "different order and every short name that differs counts as an error.",
              "", "Accuracy compares the text after the same cleaning on both versions (headers and footers "
              "removed, spaces and quote marks normalised), so it measures what retrieval and the answers see."]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    result = evaluate()
    for r in result["pages"]:
        print(f"p.{r['page']:<3} scanned={r['detected_as_scanned']} engine={r['ocr_engine']:<9} "
              f"char={r['char_accuracy']:.3f} word={r['word_accuracy']:.3f} conf={r['ocr_confidence']}")
    print("\nSUMMARY:", result["summary"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ocr_eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT / "ocr_eval.md").write_text(to_markdown(result), encoding="utf-8")
    print(f"Saved {OUT / 'ocr_eval.md'} and ocr_eval.json")


if __name__ == "__main__":
    main()
