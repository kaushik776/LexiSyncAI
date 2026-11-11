"""Bonus feature 8 - how well does the autoencoder find unusual pages?

There is no hand-labelled "anomaly" set, so we use pages that are known to be special
from ingestion as weak labels: scanned, blank, table-of-contents and table pages, and
each document's first page (cover/title). We report the share of flagged pages that are
special (precision), the share of special pages flagged (recall), and list every flagged
page with the network's reason, so the result can be checked by eye.

    python -m eval.anomaly_eval
"""
from __future__ import annotations

import json
import sys

from src import config
from src.anomaly import detect
from src.ingest import ingest_pdf

# Pages checked by hand after a first run (not known special from ingestion, but unusual):
CHECKED_BY_HAND = {
    ("EX_10_2", 23): "signature page", ("EX_10_2", 24): "signature page",
    ("EX_10_2", 7): "nearly empty page", ("EX_10_2", 8): "nearly empty page",
    ("GDPR", 88): "signing page ('Done at Brussels')", ("UK_Consumer_Rights_Act_2015", 64): "nearly empty last page of the Act",
    ("UK_Consumer_Rights_Act_2015", 87): "page leading into the enforcer table",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    docs = [ingest_pdf(p, use_ocr=False) for p in sorted(config.DATA_DIR.glob("*.pdf"))]
    special = {}
    for d in docs:
        for p in d.pages:
            why = [w for w, cond in [("scanned", p.is_scanned), ("blank", p.is_blank),
                                     ("contents", any(b.kind == "toc" for b in p.blocks)),
                                     ("table", bool(p.tables)), ("first page", p.number == 1)] if cond]
            if why:
                special[(d.doc_id, p.number)] = why
    flagged = detect(docs)
    hits = [a for a in flagged if (a.doc_id, a.page) in special]
    hand = [a for a in flagged if (a.doc_id, a.page) in special or (a.doc_id, a.page) in CHECKED_BY_HAND]
    n_pages = sum(len(d.pages) for d in docs)
    summary = {"pages": n_pages, "special_pages": len(special), "flagged": len(flagged),
               "precision": round(len(hits) / len(flagged), 3) if flagged else None,
               "recall": round(len(hits) / len(special), 3) if special else None,
               "precision_incl_hand_checked": round(len(hand) / len(flagged), 3) if flagged else None}
    lines = ["| Document | Page | Score | Known special? | Network's reason |", "|---|---|---|---|---|"]
    for a in flagged:
        known = ", ".join(special.get((a.doc_id, a.page), [])) or CHECKED_BY_HAND.get((a.doc_id, a.page), "-")
        lines.append(f"| {a.doc_id} | {a.page} | {a.score} | {known} | {a.reason} |")
    md = (f"Autoencoder page-anomaly detector on {n_pages} pages ({len(special)} known special pages).\n\n"
          f"Flagged {summary['flagged']} pages: precision {summary['precision']} against the automatic labels, "
          f"{summary['precision_incl_hand_checked']} including pages checked by hand; recall {summary['recall']}.\n\n"
          + "\n".join(lines))
    print(md)
    out = config.ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    (out / "anomaly_eval.md").write_text(md + "\n", encoding="utf-8")
    (out / "anomaly_eval.json").write_text(json.dumps({"summary": summary, "flagged": [a.__dict__ for a in flagged]},
                                                      indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
