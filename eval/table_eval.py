"""Bonus feature 2 - table extraction accuracy against a manual ground truth.

The tables in eval/tables_ground_truth.json were typed by hand from the page images.
For each one we compare the extracted table with the ground truth:

  * shape match     same number of rows and columns
  * cell accuracy   share of cells that match exactly (after normalising spaces/quotes)
  * cell similarity mean character similarity of the cells (1.0 = identical)
  * numeric cells   share of cells holding a number that match exactly (amounts matter most)

    python -m eval.table_eval
"""
from __future__ import annotations

import difflib
import json
import re
import sys

from src import config
from src.ingest import ingest_pdf

HERE = config.ROOT / "eval"


def norm(cell: str) -> str:
    cell = cell.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", cell).strip().lower()


def compare(pred: list[list[str]], gold: list[list[str]]) -> dict:
    shape_ok = len(pred) == len(gold) and all(len(p) == len(g) for p, g in zip(pred, gold))
    cells = numeric = exact = exact_num = 0
    sim = 0.0
    for r, gold_row in enumerate(gold):
        for c, g in enumerate(gold_row):
            p = pred[r][c] if r < len(pred) and c < len(pred[r]) else ""
            same = norm(p) == norm(g)
            cells += 1
            exact += same
            sim += difflib.SequenceMatcher(None, norm(p), norm(g)).ratio() if (p or g) else 1.0
            if re.search(r"\d", g):
                numeric += 1
                exact_num += same
    return {"shape_match": shape_ok, "cells": cells, "cell_accuracy": round(exact / cells, 3),
            "cell_similarity": round(sim / cells, 3),
            "numeric_cell_accuracy": round(exact_num / numeric, 3) if numeric else None}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.loads((HERE / "tables_ground_truth.json").read_text(encoding="utf-8"))
    docs, rows = {}, []
    for t in truth:
        if t["doc"] not in docs:
            docs[t["doc"]] = ingest_pdf(config.DATA_DIR / t["doc"], use_ocr=False)
        found = [tab for tab in docs[t["doc"]].pages[t["page"] - 1].tables]
        if not found:
            rows.append({"doc": t["doc"], "page": t["page"], "detected": False})
            continue
        result = compare(found[0].rows, t["rows"])
        rows.append({"doc": t["doc"], "page": t["page"], "detected": True,
                     "rows_pred": len(found[0].rows), "rows_gold": len(t["rows"]), **result})
    detected = [r for r in rows if r["detected"]]
    total_cells = sum(r["cells"] for r in detected)
    summary = {
        "tables": len(rows), "detected": len(detected),
        "shape_match": sum(r["shape_match"] for r in detected),
        "cell_accuracy": round(sum(r["cell_accuracy"] * r["cells"] for r in detected) / total_cells, 3),
        "cell_similarity": round(sum(r["cell_similarity"] * r["cells"] for r in detected) / total_cells, 3),
        "numeric_cell_accuracy": round(sum((r["numeric_cell_accuracy"] or 0) for r in detected) /
                                       max(1, sum(r["numeric_cell_accuracy"] is not None for r in detected)), 3),
    }
    lines = ["| Table | Detected | Rows (pred/gold) | Shape match | Cell accuracy | Cell similarity | Numeric cells |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["detected"]:
            lines.append(f"| {r['doc']} p.{r['page']} | no | - | - | - | - | - |")
            continue
        lines.append(f"| {r['doc']} p.{r['page']} | yes | {r['rows_pred']}/{r['rows_gold']} | "
                     f"{'yes' if r['shape_match'] else 'no'} | {r['cell_accuracy']:.3f} | {r['cell_similarity']:.3f} | "
                     f"{r['numeric_cell_accuracy']} |")
    lines.append(f"| **All** | {summary['detected']}/{summary['tables']} | | {summary['shape_match']}/{summary['detected']} | "
                 f"**{summary['cell_accuracy']:.3f}** | {summary['cell_similarity']:.3f} | {summary['numeric_cell_accuracy']} |")
    md = "\n".join(lines)
    print(md)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "table_eval.json").write_text(json.dumps({"summary": summary, "tables": rows}, indent=1), encoding="utf-8")
    (out / "table_eval.md").write_text("Table extraction vs manual ground truth (eval/tables_ground_truth.json).\n\n"
                                       + md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
