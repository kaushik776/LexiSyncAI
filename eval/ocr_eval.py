"""Bonus feature 1 - OCR accuracy on a sample.

We take digital pages (whose true text we know), render them to images to simulate a
scan, build an image-only PDF, run our ingestion (which detects the pages as scanned and
OCRs them), and compare the OCR text with the original text.

Metrics:
  * character accuracy = 1 - character error rate (edit distance / length of the truth)
  * word accuracy      = 1 - word error rate

    python -m eval.ocr_eval                          # default sample
    python -m eval.ocr_eval data/GDPR.pdf 5 83 --dpi 150
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import pymupdf

from src import config
from src.ingest import ingest_pdf

DEFAULT_SAMPLE = [("GDPR.pdf", [5, 83]), ("EX-10.2.pdf", [6]), ("UK_Consumer_Rights_Act_2015.pdf", [21])]


def _normalise(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def page_truth(pdf_path: Path, page_no: int) -> str:
    """The digital page's text, cleaned by the same ingestion (headers/footers removed)."""
    doc = ingest_pdf(pdf_path, detect_tables=False, use_ocr=False)
    return _normalise(" ".join(b.text for b in doc.pages[page_no - 1].blocks))


def make_scanned_pdf(pdf_path: Path, pages: list[int], dpi: int, out: Path) -> None:
    src = pymupdf.open(pdf_path)
    dst = pymupdf.open()
    for n in pages:
        pix = src[n - 1].get_pixmap(dpi=dpi)
        page = dst.new_page(width=src[n - 1].rect.width, height=src[n - 1].rect.height)
        page.insert_image(page.rect, pixmap=pix)      # image only, no text layer
    dst.save(out)


def evaluate(sample: list[tuple[str, list[int]]], dpi: int) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, pages in sample:
            pdf = config.DATA_DIR / name if not Path(name).exists() else Path(name)
            scanned = Path(tmp) / f"scanned_{pdf.stem}.pdf"
            make_scanned_pdf(pdf, pages, dpi, scanned)
            t0 = time.time()
            doc = ingest_pdf(scanned, detect_tables=False, use_ocr=True)
            secs = (time.time() - t0) / len(pages)
            for i, n in enumerate(pages):
                p = doc.pages[i]
                truth = page_truth(pdf, n)
                ocr = _normalise(" ".join(b.text for b in p.blocks))
                cer = edit_distance(list(ocr), list(truth)) / max(1, len(truth))
                wer = edit_distance(ocr.split(), truth.split()) / max(1, len(truth.split()))
                rows.append({
                    "document": pdf.stem, "page": n, "detected_as_scanned": p.is_scanned,
                    "ocr_engine": p.ocr_engine or "none", "ocr_confidence": p.ocr_confidence,
                    "char_accuracy": round(1 - cer, 4), "word_accuracy": round(1 - wer, 4),
                    "seconds_per_page": round(secs, 2), "truth_chars": len(truth),
                })
    n = len(rows)
    summary = {
        "dpi": dpi,
        "pages": n,
        "detected_as_scanned": sum(r["detected_as_scanned"] for r in rows),
        "mean_char_accuracy": round(sum(r["char_accuracy"] for r in rows) / n, 4),
        "mean_word_accuracy": round(sum(r["word_accuracy"] for r in rows) / n, 4),
        "mean_seconds_per_page": round(sum(r["seconds_per_page"] for r in rows) / n, 2),
    }
    return {"summary": summary, "pages": rows}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", help="PDF to sample (default: built-in sample)")
    ap.add_argument("pages", nargs="*", type=int)
    ap.add_argument("--dpi", type=int, default=150, help="scan resolution to simulate")
    args = ap.parse_args()
    sample = [(args.pdf, args.pages or [1])] if args.pdf else DEFAULT_SAMPLE

    result = evaluate(sample, args.dpi)
    for r in result["pages"]:
        print(f"{r['document']:<32} p.{r['page']:<4} scanned={r['detected_as_scanned']} "
              f"engine={r['ocr_engine']:<9} char_acc={r['char_accuracy']:.3f} "
              f"word_acc={r['word_accuracy']:.3f} conf={r['ocr_confidence']} {r['seconds_per_page']}s")
    print("\nSUMMARY:", result["summary"])
    out = config.ROOT / "eval" / "results" / "ocr_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
