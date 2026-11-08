"""Step 1: read a PDF into clean text blocks in reading order.

For every page this module:
  * extracts text blocks with their position (bbox) and font information,
  * removes running headers, footers and page numbers,
  * marks footnotes (small font at the bottom of the page),
  * detects tables and turns them into Markdown,
  * flags scanned (image-only) pages and OCRs them if Tesseract is available,
  * flags blank pages and table-of-contents pages (these are not indexed),
  * sorts blocks into reading order, including two-column layouts.

Run it on its own to see what it finds:
    python -m src.ingest data/GDPR.pdf
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from src import config

BBox = tuple[float, float, float, float]

FOOTNOTE_START = re.compile(r"^(\(\d{1,3}\)|\d{1,3}\s|\*|†)")
PAGE_NUMBER_KEY = re.compile(r"^(page\s*)?[-–\s]*#[-–\s]*((of|/)\s*#)?$")
# Structural headings ("Article 15", "Section 4", "CHAPTER III") can sit at the top of a page.
# After digits are normalised they all look alike, so they are only removed if repeated exactly.
HEADING_LIKE = re.compile(r"^(article|section|chapter|part|schedule|annex|appendix|exhibit|title)\s+[\w.()]+\b",
                          re.IGNORECASE)
# A first line that is only a heading: "II", "B", "3", "Article 5", "PART 2" ...
LEAD_HEADING = re.compile(r"^([IVX]{1,5}|[A-Z]|\d{1,2}|(?:ARTICLE|Article|SECTION|Section|CHAPTER|Chapter|"
                          r"PART|Part|SCHEDULE|Schedule|ANNEX|Annex)\s+[\w.]+)$")


@dataclass
class Block:
    text: str
    page: int              # 1-based page number
    bbox: BBox             # (x0, y0, x1, y1) in PDF points
    size: float            # dominant font size
    bold: bool
    kind: str = "text"     # "text" | "footnote" | "table" | "toc"


@dataclass
class TableInfo:
    page: int
    bbox: BBox
    n_rows: int
    n_cols: int
    rows: list[list[str]]
    markdown: str


@dataclass
class Page:
    number: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    tables: list[TableInfo] = field(default_factory=list)
    n_chars: int = 0
    is_scanned: bool = False
    is_blank: bool = False
    is_toc: bool = False
    ocr_used: bool = False
    ocr_engine: str = ""           # "tesseract" | "easyocr"
    ocr_confidence: float | None = None
    removed: list[str] = field(default_factory=list)  # header/footer text that was dropped
    running_header: str = ""   # digit-free running header, e.g. "Opinion of the Court"


@dataclass
class Document:
    doc_id: str
    title: str
    path: str
    pages: list[Page]
    body_font_size: float
    name: str = ""            # title printed in the document, e.g. "TAX PROTECTION AGREEMENT"

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def content_blocks(self) -> list[Block]:
        """All blocks in reading order, skipping blank pages and table-of-contents blocks."""
        return [b for p in self.pages if not p.is_blank for b in p.blocks if b.kind != "toc"]

    def summary(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "name": self.name,
            "pages": self.n_pages,
            "scanned_pages": [p.number for p in self.pages if p.is_scanned],
            "ocr_pages": [p.number for p in self.pages if p.ocr_used],
            "blank_pages": [p.number for p in self.pages if p.is_blank],
            "toc_pages": [p.number for p in self.pages if any(b.kind == "toc" for b in p.blocks)],
            "tables": [{"page": t.page, "rows": t.n_rows, "cols": t.n_cols}
                       for p in self.pages for t in p.tables],
            "footnotes": sum(b.kind == "footnote" for p in self.pages for b in p.blocks),
            "headers_footers_removed": sum(len(p.removed) for p in self.pages),
        }

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Raw extraction
# ----------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("­", "")
    return re.sub(r"\s+", " ", text).strip()


def _blocks_from_dict(page_dict: dict, page_no: int) -> list[Block]:
    """Turn PyMuPDF's dict output into Block objects with font size and boldness."""
    blocks = []
    for b in page_dict["blocks"]:
        if b.get("type") != 0:          # 0 = text block, 1 = image
            continue
        lines = [ln for ln in b["lines"] if "".join(s["text"] for s in ln["spans"]).strip()]
        if not lines:
            continue
        # A heading printed on its own line is sometimes merged into the paragraph below it
        # (e.g. part "B" above a judgment paragraph). Split it off so it can be detected.
        first = _clean("".join(s["text"] for s in lines[0]["spans"]))
        if (len(lines) > 1 and LEAD_HEADING.match(first)
                and lines[1]["bbox"][1] >= lines[0]["bbox"][3] - 1):
            groups = [lines[:1], lines[1:]]
        else:
            groups = [lines]
        for group in groups:
            block = _make_block(group, page_no)
            if block:
                blocks.append(block)
    return blocks


def _make_block(lines: list[dict], page_no: int) -> Block | None:
    size_chars: Counter = Counter()
    bold_chars = total_chars = 0
    texts = []
    for line in lines:
        texts.append("".join(s["text"] for s in line["spans"]))
        for s in line["spans"]:
            n = len(s["text"].strip())
            size_chars[round(s["size"] * 2) / 2] += n
            total_chars += n
            if s["flags"] & 16 or "bold" in s["font"].lower():
                bold_chars += n
    text = _clean(" ".join(texts))
    if not text or total_chars == 0:
        return None
    x0 = min(ln["bbox"][0] for ln in lines)
    y0 = min(ln["bbox"][1] for ln in lines)
    x1 = max(ln["bbox"][2] for ln in lines)
    y1 = max(ln["bbox"][3] for ln in lines)
    return Block(text=text, page=page_no, bbox=(round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)),
                 size=size_chars.most_common(1)[0][0], bold=bold_chars / total_chars > 0.6)


def _extract_page(page: pymupdf.Page, page_no: int) -> list[Block]:
    flags = pymupdf.TEXTFLAGS_TEXT | pymupdf.TEXT_DEHYPHENATE
    return _blocks_from_dict(page.get_text("dict", flags=flags), page_no)


def _image_coverage(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height
    covered = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        covered += r.width * r.height
    return min(1.0, covered / area) if area else 0.0


_EASYOCR_READER = None


def _ocr_tesseract(page: pymupdf.Page, page_no: int) -> list[Block] | None:
    """OCR with Tesseract through PyMuPDF (needs the Tesseract program installed)."""
    try:
        tp = page.get_textpage_ocr(full=True, dpi=300)
        d = page.get_text("dict", textpage=tp, flags=pymupdf.TEXTFLAGS_TEXT)
        return _blocks_from_dict(d, page_no) or None
    except Exception:
        return None


def _ocr_easyocr(page: pymupdf.Page, page_no: int, dpi: int = 200) -> tuple[list[Block], float] | None:
    """OCR with EasyOCR (pip package, runs on the GPU if available).
    Returns blocks in PDF coordinates and the mean word confidence."""
    global _EASYOCR_READER
    try:
        import easyocr
        import numpy as np
        import torch
    except ImportError:
        return None
    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    results = _EASYOCR_READER.readtext(img[:, :, :3] if pix.n >= 3 else img)
    if not results:
        return None
    scale = 72 / dpi
    lines = []
    for pts, text, conf in results:
        xs, ys = [p[0] * scale for p in pts], [p[1] * scale for p in pts]
        lines.append(((min(xs), min(ys), max(xs), max(ys)), text, conf))
    lines.sort(key=lambda l: (round(l[0][1] / 4), l[0][0]))
    # Group lines into paragraph blocks: a vertical gap larger than one line height starts a new block.
    blocks, group = [], [lines[0]]
    for line in lines[1:]:
        prev = group[-1][0]
        height = prev[3] - prev[1]
        if line[0][1] - prev[3] > 0.8 * height:
            blocks.append(group)
            group = []
        group.append(line)
    blocks.append(group)
    out = []
    for g in blocks:
        bbox = (round(min(l[0][0] for l in g), 1), round(min(l[0][1] for l in g), 1),
                round(max(l[0][2] for l in g), 1), round(max(l[0][3] for l in g), 1))
        height = statistics.median(l[0][3] - l[0][1] for l in g)
        out.append(Block(text=_clean(" ".join(l[1] for l in g)), page=page_no, bbox=bbox,
                         size=round(height * 0.8 * 2) / 2, bold=False))
    confidence = sum(c for _, _, c in results) / len(results)
    return out, round(confidence, 3)


def _try_ocr(page: pymupdf.Page, page_no: int) -> tuple[list[Block], str, float | None] | None:
    """OCR a scanned page: Tesseract first, then EasyOCR. None if no engine is available."""
    blocks = _ocr_tesseract(page, page_no)
    if blocks:
        return blocks, "tesseract", None
    result = _ocr_easyocr(page, page_no)
    if result:
        return result[0], "easyocr", result[1]
    return None


# ----------------------------------------------------------------------------
# Headers, footers, footnotes
# ----------------------------------------------------------------------------

def _norm_key(text: str) -> str:
    """Normalise text so running headers match across pages (page numbers -> '#')."""
    t = text.lower()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\b[ivxlc]+\b", "#", t)      # roman page numbers: ii, iv, xii ...
    return re.sub(r"\s+", " ", t).strip()


def _in_header(b: Block, h: float) -> bool:
    return b.bbox[3] <= h * config.HEADER_BAND


def _in_footer(b: Block, h: float) -> bool:
    return b.bbox[1] >= h * (1 - config.FOOTER_BAND)


def _edge_positions(p: Page) -> dict[int, tuple[str, bool]]:
    """Header/footer candidates on a page -> {id(block): ("top"|"bottom", in_fixed_band)}.
    A block is a candidate if it lies in the fixed top/bottom band, or if it is one of the
    first/last two short blocks on the page (pages with wide margins, e.g. court opinions)."""
    by_y = sorted(p.blocks, key=lambda b: b.bbox[1])
    short = [b for b in by_y if len(b.text.split()) <= 25]
    top_edge = {id(b) for b in by_y[:3] if b in short}
    bottom_edge = {id(b) for b in by_y[-2:] if b in short}
    pos = {}
    for b in p.blocks:
        if _in_header(b, p.height) or id(b) in top_edge:
            pos[id(b)] = ("top", _in_header(b, p.height))
        elif _in_footer(b, p.height) or id(b) in bottom_edge:
            pos[id(b)] = ("bottom", _in_footer(b, p.height))
    return pos


def _remove_headers_footers(pages: list[Page], body_size: float) -> None:
    key_pages: Counter = Counter()
    raw_pages: Counter = Counter()
    offsets: Counter = Counter()        # printed page number - PDF page index
    positions = [_edge_positions(p) for p in pages]
    for p, pos in zip(pages, positions):
        edge = [b for b in p.blocks if id(b) in pos]
        key_pages.update({_norm_key(b.text) for b in edge})
        raw_pages.update({b.text for b in edge})
        offsets.update({int(b.text) - p.number for b in edge if b.text.isdigit()})
    # A bare number counts as a page number if it follows the page sequence on several pages.
    page_offsets = {o for o, n in offsets.items() if n >= config.REPEAT_MIN_PAGES}

    for p, pos in zip(pages, positions):
        kept = []
        for b in p.blocks:
            if id(b) not in pos:
                kept.append(b)
                continue
            where, in_band = pos[id(b)]
            key = _norm_key(b.text)
            lone_enumerator = bool(re.fullmatch(r"[IVX]{1,5}|[A-Z]", b.text))   # judgment parts "II", "B"
            heading_like = bool(HEADING_LIKE.match(b.text)) or lone_enumerator
            if lone_enumerator:
                repeated = in_band and raw_pages[b.text] >= config.REPEAT_MIN_PAGES
            elif heading_like:
                # "Article 15" at the top of one page is a real heading; "Chapter 2 — Goods"
                # printed on every page of the chapter is a running header.
                repeated = raw_pages[b.text] >= config.REPEAT_MIN_PAGES
            else:
                repeated = key_pages[key] >= config.REPEAT_MIN_PAGES
            # A lone number/letter can be a heading ("3", "II"), so outside the fixed band only
            # numbers that follow the page sequence count as page numbers.
            if b.text.isdigit():
                page_number = in_band or (int(b.text) - p.number) in page_offsets
            else:
                page_number = bool(PAGE_NUMBER_KEY.match(key)) and in_band
            small_top = (where == "top" and in_band and not heading_like
                         and b.size < body_size - 0.5 and len(b.text.split()) <= 25)
            if repeated or page_number or small_top:
                p.removed.append(b.text)
                if where == "top" and not p.running_header and not re.search(r"\d", b.text):
                    p.running_header = b.text          # e.g. "Opinion of the Court"
                continue
            kept.append(b)
        p.blocks = kept


def _mark_footnotes(page: Page, body_size: float) -> None:
    """Small-font blocks at the bottom of the page, starting with a marker like '(1)'."""
    small_low = sorted(
        (b for b in page.blocks
         if b.bbox[1] >= page.height * 0.6 and b.size <= body_size - 0.7),
        key=lambda b: b.bbox[1])
    start_y = next((b.bbox[1] for b in small_low if FOOTNOTE_START.match(b.text)), None)
    if start_y is None:
        return
    for b in small_low:
        if b.bbox[1] >= start_y:
            b.kind = "footnote"


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def _has_ruling(page: pymupdf.Page) -> bool:
    """Cheap pre-check: only pages with several lines/rectangles in the body can hold tables.
    Running find_tables() on every page is slow (~0.25 s/page)."""
    h = page.rect.height
    top, bottom = h * config.HEADER_BAND, h * (1 - config.FOOTER_BAND)
    horizontal = vertical = 0
    for d in page.get_drawings():
        for item in d["items"]:
            if item[0] == "l":                       # line segment
                p1, p2 = item[1], item[2]
                if not (top <= p1.y <= bottom):
                    continue
                if abs(p1.y - p2.y) < 1:
                    horizontal += 1
                elif abs(p1.x - p2.x) < 1:
                    vertical += 1
            elif item[0] == "re":                    # rectangle = 2 horizontal + 2 vertical edges
                r = item[1]
                if top <= r.y0 and r.y1 <= bottom:
                    if r.height < 2:                 # thin rectangle used as a horizontal rule
                        horizontal += 1
                    elif r.width < 2:
                        vertical += 1
                    else:
                        horizontal += 2
                        vertical += 2
    return horizontal >= 3 and vertical >= 2


def _cell_text(words: list, bbox) -> str:
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    inside = [w for w in words
              if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    inside.sort(key=lambda w: (round(w[3]), w[0]))
    text = " ".join(w[4] for w in inside)
    text = re.sub(r"(?<=\d)\s*([,.])\s*(?=\d)", r"\1", text)   # "1 , 091" -> "1,091"
    text = re.sub(r"\$\s+(?=\d)", "$", text)
    return _clean(text)


def _to_markdown(rows: list[list[str]]) -> str:
    def fmt(r):
        return "| " + " | ".join(c.replace("|", "/") for c in r) + " |"
    lines = [fmt(rows[0]), "| " + " | ".join("---" for _ in rows[0]) + " |"]
    lines += [fmt(r) for r in rows[1:]]
    return "\n".join(lines)


def _extract_tables(page: pymupdf.Page, page_no: int) -> list[TableInfo]:
    tables = []
    words = page.get_text("words")
    for t in page.find_tables().tables:
        rows = [[_cell_text(words, c) for c in row.cells] for row in t.rows]
        rows = [r for r in rows if any(r)]
        if len(rows) < 2 or t.col_count < 2:
            continue
        tables.append(TableInfo(page=page_no, bbox=tuple(round(v, 1) for v in t.bbox),
                                n_rows=len(rows), n_cols=t.col_count, rows=rows,
                                markdown=_to_markdown(rows)))
    return tables


def _overlap_ratio(inner: BBox, outer: BBox) -> float:
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (ix1 - ix0) * (iy1 - iy0) / area if area else 0.0


# ----------------------------------------------------------------------------
# Table-of-contents pages and reading order
# ----------------------------------------------------------------------------

TOC_TITLE = re.compile(r"(CONTENTS|Contents|ARRANGEMENT OF [A-Z]+|Arrangement of [A-Z][a-z]+)$")


def _toc_like(b: Block) -> bool:
    """Short lines, or lines dense with numbers / dot leaders, and no "(1)" subsections."""
    words = b.text.split()
    if len(words) <= 12:
        return True
    numbers = sum(bool(re.fullmatch(r"\d+[A-Z]?\.?", w)) for w in words) / len(words)
    leaders = bool(re.search(r"\.{4,}", b.text))
    subsections = re.search(r"(^|\s)\(\d+\)\s", b.text)
    # Contents entries don't end like sentences ("What remedies are there ...?" is an entry).
    entry = len(words) <= 25 and not b.text.rstrip().endswith((".", ";", ":", ","))
    return not subsections and (numbers >= 0.06 or leaders or entry)


def _mark_toc(pages: list[Page]) -> None:
    """Mark table-of-contents blocks (kind="toc") so they are not indexed: their entries
    repeat every heading and would crowd out the real text in search results.
    The TOC starts at a "Contents" title and ends at the first block that reads like body text."""
    in_toc = False
    for p in pages:
        if p.is_blank:
            continue
        for b in p.blocks:
            if b.kind != "text":
                continue
            if not in_toc and len(b.text.split()) <= 10 and TOC_TITLE.search(b.text):
                in_toc = True
            elif in_toc and not _toc_like(b):
                in_toc = False
            if in_toc:
                b.kind = "toc"
        p.is_toc = bool(p.blocks) and all(b.kind == "toc" for b in p.blocks)


def _reading_order(blocks: list[Block], width: float, height: float) -> list[Block]:
    """Sort blocks top-to-bottom; for two-column pages read the left column first.
    Full-width blocks (titles, wide paragraphs) split the page into horizontal bands."""
    body = [b for b in blocks if b.kind != "footnote"]
    notes = sorted((b for b in blocks if b.kind == "footnote"), key=lambda b: b.bbox[1])
    mid, tol = width / 2, width * 0.02
    left = [b for b in body if b.bbox[2] <= mid + tol]
    right = [b for b in body if b.bbox[0] >= mid - tol]
    right_height = sum(b.bbox[3] - b.bbox[1] for b in right)
    two_columns = len(left) >= 3 and len(right) >= 3 and right_height > 0.15 * height

    if not two_columns:
        return sorted(body, key=lambda b: (round(b.bbox[1]), b.bbox[0])) + notes

    ordered, band = [], []

    def flush():
        ordered.extend(sorted((b for b in band if b.bbox[2] <= mid + tol), key=lambda b: b.bbox[1]))
        ordered.extend(sorted((b for b in band if b.bbox[2] > mid + tol), key=lambda b: b.bbox[1]))
        band.clear()

    for b in sorted(body, key=lambda b: b.bbox[1]):
        full_width = b.bbox[0] < mid - tol and b.bbox[2] > mid + tol
        if full_width:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered + notes


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def make_doc_id(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(path).stem).strip("_")


def ingest_pdf(path: str | Path,
               detect_tables: bool = config.DETECT_TABLES,
               use_ocr: bool = config.USE_OCR) -> Document:
    path = Path(path)
    pdf = pymupdf.open(path)
    pages: list[Page] = []

    for i, pg in enumerate(pdf):
        page = Page(number=i + 1, width=pg.rect.width, height=pg.rect.height)
        page.blocks = _extract_page(pg, page.number)
        page.n_chars = sum(len(b.text) for b in page.blocks)

        if page.n_chars < config.SCANNED_MAX_CHARS:
            if _image_coverage(pg) >= config.SCANNED_MIN_IMAGE_AREA:
                page.is_scanned = True
                ocr = _try_ocr(pg, page.number) if use_ocr else None
                if ocr:
                    page.blocks, page.ocr_engine, page.ocr_confidence = ocr
                    page.ocr_used = True
            else:
                page.is_blank = True

        if detect_tables and not page.is_scanned and _has_ruling(pg):
            page.tables = _extract_tables(pg, page.number)
        pages.append(page)

    # Body font size = most common size across the document (weighted by characters).
    sizes: Counter = Counter()
    for p in pages:
        for b in p.blocks:
            sizes[b.size] += len(b.text)
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0

    _remove_headers_footers(pages, body_size)

    for p in pages:
        if not p.blocks and not p.is_scanned:
            p.is_blank = True               # e.g. pages that only had a header/footer
        _mark_footnotes(p, body_size)
        # Replace the text blocks covered by a table with one table block.
        for t in p.tables:
            p.blocks = [b for b in p.blocks if _overlap_ratio(b.bbox, t.bbox) < 0.6]
            p.blocks.append(Block(text=t.markdown, page=p.number, bbox=t.bbox,
                                  size=body_size, bold=False, kind="table"))
        p.blocks = _reading_order(p.blocks, p.width, p.height)

    _mark_toc(pages)

    title = path.stem.replace("_", " ")
    return Document(doc_id=make_doc_id(path), title=title, path=str(path), pages=pages,
                    body_font_size=body_size, name=detect_name(pages) or title)


def detect_name(pages: list[Page]) -> str:
    """The document's own title from its first pages: the largest (then bold, then upper-case)
    multi-word line near the top, plus following lines in the same style.
    E.g. "TAX PROTECTION AGREEMENT", "Consumer Rights Act 2015"."""
    first = [p for p in pages if not p.is_blank and p.blocks][:2]
    candidates = []
    for p in first:
        for i, b in enumerate(p.blocks):
            words = b.text.split()
            if (b.kind != "text" or not 2 <= len(words) <= 30 or b.bbox[1] < 0.06 * p.height
                    or b.bbox[1] > 0.7 * p.height or re.search(r"\.(htm|html|pdf|txt)|https?:|www\.", b.text)
                    or sum(w.isalpha() for w in words) < 2):
                continue
            candidates.append(((b.size, len(words) >= 3, b.bold, -p.number, -b.bbox[1]), p, i))
    if not candidates:
        return ""
    _, page, i = max(candidates, key=lambda c: c[0])
    lead = page.blocks[i]
    parts = [lead.text]
    for b in page.blocks[i + 1:i + 4]:            # title continued on the next lines
        if b.size == lead.size and b.bold == lead.bold and len(" ".join(parts + [b.text]).split()) <= 40:
            parts.append(b.text)
        else:
            break
    name = " ".join(parts)
    return name.title() if name.isupper() and len(name.split()) <= 6 else name


def main(paths: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for p in paths:
        doc = ingest_pdf(p)
        print(f"\n=== {doc.title} ===")
        for k, v in doc.summary().items():
            print(f"  {k:24} {v}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in sorted(config.DATA_DIR.glob("*.pdf"))])
