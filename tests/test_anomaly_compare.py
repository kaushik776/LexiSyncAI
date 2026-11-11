"""Tests for the autoencoder page-anomaly detector and cross-document conflict detection."""
import pymupdf

import src.compare as compare
from src.compare import DocAnswer, reconcile
from src.ingest import ingest_pdf
from src.anomaly import detect
from src.pipeline import Answer

TEXT = ("Section {n}. The Supplier shall deliver the goods in accordance with the agreement and shall notify the "
        "Buyer of any delay without undue delay. The Buyer shall pay each invoice within thirty days. ") * 6


def test_autoencoder_flags_the_odd_page(tmp_path):
    doc = pymupdf.open()
    sentence = TEXT.format(n=0)[:200]
    for n in range(30):                                   # 30 ordinary text pages of varying length
        text = f"Section {n}. " + sentence * (2 + n % 5)
        doc.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 800), text, fontsize=10)
    odd = doc.new_page()                                  # a page that is mostly a large image
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (40, 40, 40))
    odd.insert_image(pymupdf.Rect(40, 40, 555, 700), pixmap=pix)
    odd.insert_text((60, 760), "SCHEDULE 1 SIGNATURES", fontsize=14)
    path = tmp_path / "case.pdf"
    doc.save(path)
    flagged = detect([ingest_pdf(path, detect_tables=False, use_ocr=False)])
    assert flagged and flagged[0].page == 31
    assert "image coverage" in flagged[0].reason or "text" in flagged[0].reason


def _answer(text, abstained=False):
    return Answer(question="q", answer=text, badge="HIGH", abstained=abstained)


def test_reconcile_figures_and_status(monkeypatch):
    monkeypatch.setattr(compare, "contradiction", lambda a, b: 0.0)   # no NLI model needed
    v1 = DocAnswer("a", "Agreement 2023", _answer("The liability cap is EUR 2,000,000. [S1]"))
    v2 = DocAnswer("b", "Agreement 2024", _answer("The liability cap is EUR 2,500,000. [S1]"))
    same = DocAnswer("c", "Copy", _answer("The liability cap is EUR 2,000,000. [S2]"))
    none = DocAnswer("d", "Act", _answer("The documents do not contain enough evidence.", abstained=True))
    status, findings = reconcile([v1, v2])
    assert status == "conflict" and "EUR 2,000,000" in findings[0] and "EUR 2,500,000" in findings[0]
    assert reconcile([v1, same])[0] == "agree"
    assert reconcile([v1, none])[0] == "single"
    assert reconcile([none])[0] == "none"
