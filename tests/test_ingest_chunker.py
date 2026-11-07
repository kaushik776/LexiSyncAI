"""Tests for ingestion and chunking on small synthetic PDFs built in code,
so they check general behaviour rather than one particular document."""
import pymupdf
import pytest

from src.chunker import (chunk_document, choose_mode, detect_heading, detect_marker,
                         format_ref)
from src.ingest import ingest_pdf

LOREM = ("The parties agree that the supplier shall deliver the goods in accordance with the "
         "specification and that any delay shall be notified in writing without undue delay. ")


def _save(doc: pymupdf.Document, tmp_path, name="test.pdf") -> str:
    path = tmp_path / name
    doc.save(path)
    return str(path)


def _page_with_lines(doc, lines, header=None, footer=None, fontsize=10):
    """Add a page; each item of `lines` becomes its own paragraph block."""
    page = doc.new_page(width=595, height=842)
    if header:
        page.insert_text((72, 40), header, fontsize=8)
    if footer:
        page.insert_text((280, 820), footer, fontsize=8)
    y = 100
    for text in lines:
        rect = pymupdf.Rect(72, y, 523, y + 120)
        page.insert_textbox(rect, text, fontsize=fontsize)
        height = 14 * (1 + len(text) // 85)
        y += height + 16
    return page


# --- ingestion ---------------------------------------------------------------

def test_two_column_reading_order(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i in range(3):
        page.insert_textbox(pymupdf.Rect(50, 100 + i * 200, 285, 280 + i * 200),
                            f"LEFT{i} " + LOREM * 2, fontsize=10)
        page.insert_textbox(pymupdf.Rect(310, 100 + i * 200, 545, 280 + i * 200),
                            f"RIGHT{i} " + LOREM * 2, fontsize=10)
    d = ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False)
    order = [b.text.split()[0] for b in d.pages[0].blocks]
    assert order == ["LEFT0", "LEFT1", "LEFT2", "RIGHT0", "RIGHT1", "RIGHT2"]


def test_running_headers_and_page_numbers_removed(tmp_path):
    doc = pymupdf.open()
    for n in range(1, 5):
        _page_with_lines(doc, [f"Clause text number {n}. " + LOREM],
                         header="ACME SUPPLY AGREEMENT - CONFIDENTIAL", footer=f"Page {n} of 4")
    d = ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False)
    texts = [b.text for p in d.pages for b in p.blocks]
    assert not any("CONFIDENTIAL" in t or t.startswith("Page") for t in texts)
    assert all(len(p.removed) == 2 for p in d.pages)


def test_scanned_and_blank_pages_flagged(tmp_path):
    src = pymupdf.open()
    _page_with_lines(src, ["Article 1 Scope. " + LOREM * 3])
    pix = src[0].get_pixmap(dpi=100)
    doc = pymupdf.open()
    scanned = doc.new_page(width=595, height=842)
    scanned.insert_image(scanned.rect, pixmap=pix)        # image only: no text layer
    doc.new_page(width=595, height=842)                    # empty page
    d = ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False)
    assert d.pages[0].is_scanned and not d.pages[0].is_blank
    assert d.pages[1].is_blank
    assert d.summary()["scanned_pages"] == [1]


def test_footnotes_marked(tmp_path):
    doc = pymupdf.open()
    page = _page_with_lines(doc, ["Article 5 Principles. " + LOREM * 2] * 3, fontsize=10)
    page.insert_text((72, 780), "(1) OJ L 119, 4.5.2016, p. 1.", fontsize=7)
    d = ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False)
    kinds = {b.text[:3]: b.kind for b in d.pages[0].blocks}
    assert kinds["(1)"] == "footnote"


def test_table_of_contents_not_indexed(tmp_path):
    doc = pymupdf.open()
    _page_with_lines(doc, ["CONTENTS", "1 Citation", "2 Interpretation", "3 Duty to supply goods"])
    _page_with_lines(doc, ["1 Citation", "(1) This Act may be cited as the Example Act 2024. " + LOREM,
                           "2 Interpretation", "(1) In this Act " + LOREM])
    d = ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False)
    assert 1 in d.summary()["toc_pages"]
    assert all(b.page == 2 for b in d.content_blocks())


# --- chunking ----------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Article 83", (2, "Article 83", "")),
    ("ARTICLE 7 AMENDMENT OF THIS AGREEMENT", (2, "Article 7", "AMENDMENT OF THIS AGREEMENT")),
    ("7.2 Waiver. Notwithstanding the foregoing, the Partnership may waive", (2, "Section 7.2", "Waiver")),
    ("23 Right to repair or replacement", (2, "Section 23", "Right to repair or replacement")),
    ("CHAPTER VIII", (1, "Chapter VIII", "")),
    ("SCHEDULE 3.1(a) MINIMUM LIABILITY AMOUNT", (1, "Schedule 3.1(a)", "MINIMUM LIABILITY AMOUNT")),
])
def test_detect_heading(text, expected):
    assert detect_heading(text) == expected


@pytest.mark.parametrize("text", [
    "Article 9, or whether personal data related to criminal convictions",   # wrapped cross-reference
    "Section 4(2) applies to the trader",
    "The supplier shall pay 30 days after the invoice date.",
])
def test_running_text_is_not_a_heading(text):
    assert detect_heading(text) is None


def test_markers_and_refs():
    assert detect_marker("1. Each supervisory authority shall", None) == ("num", "1")
    assert detect_marker("(4) If any of the goods", None) == ("num", "4")
    assert detect_marker("(b) the period for which", "a") == ("letter", "b")
    assert detect_marker("(ii) a sale or transfer", None) is None          # roman sub-point
    assert format_ref("Article 83", ["4", "5"]) == "Article 83(4)-(5)"
    assert format_ref("Preamble", ["12", "14"], recitals=True) == "Recitals 12-14"
    assert format_ref("Part II-B", ["40", "42"], style="para") == "Part II-B, paras 40-42"


def test_statute_chunks_cite_article_and_paragraph(tmp_path):
    doc = pymupdf.open()
    _page_with_lines(doc, ["CHAPTER VIII", "Remedies and penalties", "Article 83",
                           "General conditions for imposing fines",
                           "1. Each authority shall ensure fines are effective. " + LOREM,
                           "2. Fines shall be imposed in addition to measures. " + LOREM,
                           "Article 84", "Penalties",
                           "1. Member States shall lay down the rules on penalties. " + LOREM])
    chunks = chunk_document(ingest_pdf(_save(doc, tmp_path), detect_tables=False, use_ocr=False))
    by_section = {c.section_id: c for c in chunks}
    a83 = by_section["Article 83"]
    assert a83.ref == "Article 83(1)-(2)"
    assert a83.section_title == "General conditions for imposing fines"
    assert a83.parents == ["Chapter VIII Remedies and penalties"]
    assert by_section["Article 84"].ref == "Article 84(1)"
    assert a83.citation.startswith("test, p. 1, Article 83")


def test_mode_selection(tmp_path):
    def mode_of(lines):
        doc = pymupdf.open()
        _page_with_lines(doc, lines)
        return choose_mode(ingest_pdf(_save(doc, tmp_path, f"m{len(lines)}.pdf"),
                                      detect_tables=False, use_ocr=False).content_blocks())[0]

    assert mode_of(["Article 1", "Scope", "Article 2", "Definitions", "Article 3", "Rules"]) == "sections"
    # The built-in test font cannot draw an em dash, so "1.-(1)" stands in for "1.—(1)".
    assert mode_of(["1.-(1) These Regulations may be cited as the Example Regulations.",
                    "2.-(1) In these Regulations " + LOREM,
                    "3.-(1) A trader must " + LOREM]) == "regulations"
    assert mode_of(["I", "The petitioners challenge the rule. " + LOREM,
                    "II", "We now turn to the merits. " + LOREM]) == "paragraphs"
