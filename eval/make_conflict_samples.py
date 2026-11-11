"""Create two versions of a fictional supply agreement for demonstrating conflict detection
(bonus feature 9). They agree on most terms but differ on the liability cap and the term.

    python -m eval.make_conflict_samples       # writes samples/conflict/*.pdf

Then create a case from both PDFs in the app and use "Compare documents", e.g.
"What is the cap on the Supplier's liability?" (conflict) or "Which law governs the
agreement?" (agree). The parties and terms are invented for this demo.
"""
from __future__ import annotations

import pymupdf

from src import config

COMMON = {
    "governing": "12.1 Governing Law. This Agreement and any dispute arising out of it shall be governed by "
                 "the laws of England and Wales.",
    "payment": "5.1 Payment Terms. The Buyer shall pay each undisputed invoice within thirty (30) days of "
               "receipt of the invoice.",
    "confidential": "9.1 Confidentiality. Each Party shall keep confidential all information received from the "
                    "other Party and shall not disclose it to any third party without prior written consent.",
}
VERSIONS = {
    "Supply_Agreement_2023.pdf": {
        "title": "SUPPLY AGREEMENT",
        "intro": "THIS SUPPLY AGREEMENT is made on 1 March 2023 between Northbridge Components Ltd (the "
                 "\"Supplier\") and Harlow Manufacturing plc (the \"Buyer\") (each a \"Party\").",
        "term": "3.1 Term. This Agreement shall commence on the Effective Date and continue for an initial term "
                "of three (3) years, unless terminated earlier in accordance with Clause 11.",
        "cap": "7.2 Limitation of Liability. The aggregate liability of the Supplier under or in connection with "
               "this Agreement shall not exceed EUR 2,000,000 in any contract year.",
    },
    "Supply_Agreement_2024_Amended.pdf": {
        "title": "AMENDED AND RESTATED SUPPLY AGREEMENT",
        "intro": "THIS AMENDED AND RESTATED SUPPLY AGREEMENT is made on 15 April 2024 between Northbridge "
                 "Components Ltd (the \"Supplier\") and Harlow Manufacturing plc (the \"Buyer\") (each a "
                 "\"Party\") and replaces the supply agreement dated 1 March 2023.",
        "term": "3.1 Term. This Agreement shall commence on the Effective Date and continue for an initial term "
                "of five (5) years, unless terminated earlier in accordance with Clause 11.",
        "cap": "7.2 Limitation of Liability. The aggregate liability of the Supplier under or in connection with "
               "this Agreement shall not exceed EUR 2,500,000 in any contract year.",
    },
}


def build(path, v: dict) -> None:
    sections = [
        ("ARTICLE 1 DEFINITIONS", ["1.1 Definitions. In this Agreement \"Products\" means the components listed in "
                                   "Schedule 1 and \"Effective Date\" means the date of this Agreement."]),
        ("ARTICLE 3 TERM", [v["term"]]),
        ("ARTICLE 5 PRICE AND PAYMENT", [COMMON["payment"], "5.2 Late Payment. Interest shall accrue on overdue "
                                         "sums at four per cent per year above the base rate."]),
        ("ARTICLE 7 LIABILITY", ["7.1 Indemnity. The Supplier shall indemnify the Buyer against losses arising "
                                 "from defective Products.", v["cap"]]),
        ("ARTICLE 9 CONFIDENTIALITY", [COMMON["confidential"]]),
        ("ARTICLE 12 GOVERNING LAW", [COMMON["governing"], "12.2 Jurisdiction. The courts of England and Wales "
                                      "shall have exclusive jurisdiction."]),
    ]
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    page.insert_textbox(pymupdf.Rect(72, y, 523, y + 30), v["title"], fontsize=14, fontname="hebo", align=1)
    y += 45
    page.insert_textbox(pymupdf.Rect(72, y, 523, y + 70), v["intro"], fontsize=10)
    y += 75
    for heading, paras in sections:
        if y > 680:
            page, y = doc.new_page(), 72
        page.insert_textbox(pymupdf.Rect(72, y, 523, y + 20), heading, fontsize=11, fontname="hebo")
        y += 22
        for para in paras:
            page.insert_textbox(pymupdf.Rect(72, y, 523, y + 60), para, fontsize=10)
            y += 14 * (1 + len(para) // 95) + 10
    doc.save(path)


def main() -> None:
    out = config.ROOT / "samples" / "conflict"
    out.mkdir(parents=True, exist_ok=True)
    for name, v in VERSIONS.items():
        build(out / name, v)
        print("wrote", out / name)


if __name__ == "__main__":
    main()
