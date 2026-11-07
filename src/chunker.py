"""Step 2: split ingested documents into clause-aware chunks.

Why not fixed-size chunks? A legal answer usually lives in one clause
("Article 83(5)", "Section 2.1(a)"). Cutting every N tokens splits clauses in half
and loses the clause number needed for citations. So we cut on legal structure.

Legal documents are structured in different ways, so the chunker first picks a mode:
  * "sections"    statutes, regulations (EU), contracts: Article N / Section N /
                  "2.1 Title" / "23 Title" start a chunk; "1." "(1)" "(a)" are paragraphs.
  * "regulations" UK statutory instruments: "12.—(1) ..." starts regulation 12.
  * "paragraphs"  judgments and anything else: parts "I" / "A" / "1" and bold headings
                  start a chunk; numbered paragraphs "12." / "[12]" are cited as para 12;
                  the running header (e.g. "Opinion of the Court") is the parent.

In every mode:
  * level-1 headings (Part / Chapter / Schedule / Annex ...) become the parent context,
  * a long unit is split at a paragraph marker once it reaches CHUNK_TARGET_WORDS,
    and never grows past CHUNK_MAX_WORDS,
  * tables and footnotes become their own chunks,
  * each chunk keeps doc, page(s), bounding boxes, section and a precise reference.

    python -m src.chunker            # chunk every PDF in data/ and save index/chunks.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src import config
from src.ingest import Block, Document, ingest_pdf

# --- Heading patterns --------------------------------------------------------
_NUM = r"(\d+[A-Z]{0,2}|[IVXLC]+|[A-Z])"
# After the number: nothing, or a separator and a title starting with a capital letter.
# This rejects running text such as "Article 9, or whether ..." and "Section 4(2) applies".
_TITLE = r"(?![,;(])[\s.:–—-]*([A-Z“\"].*)?$"
_L1_WORDS = "PART|Part|CHAPTER|Chapter|SCHEDULE|Schedule|TITLE|Title|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix"
LEVEL1 = re.compile(rf"^({_L1_WORDS})\s+(\d+(?:\.\d+)*(?:\([a-z0-9]+\))?|[IVXLC]+|[A-Z])(?!\w){_TITLE}")
ARTICLE = re.compile(rf"^(ARTICLE|Article)\s+{_NUM}\b{_TITLE}")
SECTION_WORD = re.compile(rf"^(SECTION|Section)\s+(\d+(?:\.\d+)*[A-Z]?)\b{_TITLE}")
DECIMAL = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\.?\s+(\S.*)$")                    # contract: 2.1 Title
NUMBERED_TITLE = re.compile(r"^(\d{1,3})([A-Z]{0,2})\s+([A-Z](?:(?!\. )[^;]){2,150})$")  # statute: 23 Title
REGULATION = re.compile(r"^(\d{1,3}[A-Z]?)\.(?:\s*[—–-]\s*(?:\((\d+)\))?)?\s*(?=[(A-Z“\"])")
LONE_PART = re.compile(r"^([IVX]{1,5}|[A-Z]|\d{1,2})$")                             # judgment parts

# --- Paragraph markers inside a unit -----------------------------------------
NUM_DOT = re.compile(r"^(\d{1,3})\.\s+\S")            # "1. Where ..."
PAREN_NUM = re.compile(r"^\((\d{1,3}[A-Z]?)\)\s*\S")  # "(1) This ..."
BRACKET_NUM = re.compile(r"^\[(\d{1,4})\]\s*\S")      # judgments: "[12] The ..."
PAREN_LETTER = re.compile(r"^\(([a-z]{1,4})\)\s*\S")  # points "(a)", also roman "(ii)"

LEVEL1_RANK = {"schedule": 0, "title": 0, "exhibit": 0, "annex": 0, "appendix": 0,
               "part": 1, "chapter": 2}
CONTAINER_RANK = 3   # e.g. GDPR "Section 1" that groups several Articles
_END_PUNCT = (".", ";", ",", ":", "—", "–", "-")
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    kind: str                      # "text" | "table" | "footnote"
    section_id: str                # "Article 83"
    section_title: str             # "General conditions for imposing administrative fines"
    parents: list[str]             # ["Chapter VIII Remedies, liability and penalties"]
    ref: str                       # "Article 83(4)-(6)"
    page_start: int
    page_end: int
    spans: list[dict]              # [{"page": 82, "bbox": [x0, y0, x1, y1]}, ...] for highlighting
    text: str
    n_words: int = 0
    paras: list[str] = field(default_factory=list)   # paragraph numbers covered, e.g. ["4", "5"]
    ref_style: str = "paren"       # "paren" -> Article 83(4) | "para" -> Part II, paras 12-14

    @property
    def heading_path(self) -> str:
        title = self.section_title
        parents = []
        for p in self.parents:
            label = " ".join(p.split()[:2])            # "Part 4 FURTHER POWERS" -> "Part 4"
            if self.section_id == label or self.section_id.endswith(", " + label):
                title = title or " ".join(p.split()[2:])  # the section *is* this Part/Schedule
            else:
                parents.append(p)
        return " > ".join([*parents, f"{self.section_id} {title}".strip()])

    @property
    def context_text(self) -> str:
        """Text used for search: the heading path gives each chunk its legal context."""
        return f"{self.doc_title} | {self.heading_path}\n{self.text}"

    @property
    def pages_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def citation(self) -> str:
        return f"{self.doc_title}, {self.pages_label}, {self.ref}"


# ----------------------------------------------------------------------------
# Detection helpers
# ----------------------------------------------------------------------------

def _n_words(text: str) -> int:
    return len(text.split())


def format_ref(section_id: str, paras: list[str], recitals: bool = False,
               style: str = "paren") -> str:
    """Build a citation reference: "Article 83(4)-(6)", "Recitals 4-6", "Part II, paras 12-14"."""
    if not paras:
        return section_id
    first, last = paras[0], paras[-1]
    if recitals and first.isdigit():
        return f"Recital {first}" if first == last else f"Recitals {first}-{last}"
    restarted = first.isdigit() and last.isdigit() and int(last) < int(first)
    if style == "para":
        prefix = "" if section_id == "Text" else f"{section_id}, "
        if first == last:
            return f"{prefix}para {first}"
        return f"{prefix}para {first} ff." if restarted else f"{prefix}paras {first}-{last}"
    if first == last:
        return f"{section_id}({first})"
    if restarted:
        return f"{section_id}({first}) ff."       # numbering restarted inside the chunk
    return f"{section_id}({first})-({last})"


def detect_heading(text: str, mode: str = "sections") -> tuple[int, str, str] | None:
    """Return (level, label, title) if the block is a heading, else None.
    Level 1 = container (Part/Chapter/Schedule), level 2 = chunk unit (Article/Section)."""
    words = _n_words(text)
    if words <= 20:
        if m := LEVEL1.match(text):
            return 1, f"{m.group(1).title()} {m.group(2)}", (m.group(3) or "").strip()
        if m := ARTICLE.match(text):
            return 2, f"Article {m.group(2)}", (m.group(3) or "").strip()
        if m := SECTION_WORD.match(text):
            return 2, f"Section {m.group(2)}", (m.group(3) or "").strip()
    if mode != "sections":
        return None
    if m := DECIMAL.match(text):
        rest = m.group(2)
        title = rest.split(". ")[0].rstrip(".") if "." in rest else rest
        return 2, f"Section {m.group(1)}", title if _n_words(title) <= 15 else ""
    if words <= 18 and not text.rstrip().endswith(_END_PUNCT):
        if m := NUMBERED_TITLE.match(text):
            return 2, f"Section {m.group(1)}{m.group(2)}", m.group(3).strip()
    return None


def detect_marker(text: str, prev_letter: str | None) -> tuple[str, str] | None:
    """Return ("num", "5") or ("letter", "a") for a paragraph marker at the start of the block."""
    if m := NUM_DOT.match(text) or PAREN_NUM.match(text) or BRACKET_NUM.match(text):
        return "num", m.group(1)
    if m := PAREN_LETTER.match(text):
        label = m.group(1)
        if label in _ROMAN:
            # "(i)" is a letter only if it follows "(h)"; otherwise it's a roman sub-point.
            follows = prev_letter and len(label) == 1 and ord(label) == ord(prev_letter) + 1
            if not follows:
                return None
        if len(label) == 1:
            return "letter", label
    return None


def _is_short_heading(text: str) -> bool:
    return _n_words(text) <= 10 and not text.rstrip().endswith(_END_PUNCT) and not detect_marker(text, None)


def _is_title_line(text: str) -> bool:
    """A cross-heading such as "Citation and commencement": short, no digits, starts with a capital."""
    return _is_short_heading(text) and not re.search(r"\d", text) and bool(re.match(r"[A-Z“\"]", text))


def choose_mode(blocks: list[Block]) -> tuple[str, str]:
    """Pick the chunking mode from the headings found in the document.
    Returns (mode, unit word) where unit word names the units in "regulations" mode."""
    sections = 0
    last_num = 0
    for b in blocks:
        h = detect_heading(b.text, "sections")
        if not h or h[0] != 2:
            continue
        if m := NUMBERED_TITLE.match(b.text):      # "23 Title" only counts if numbering is sequential
            n = int(m.group(1))
            sections += 0 < n - last_num <= 2
            last_num = n
        else:
            sections += 1
    if sections >= 2:
        return "sections", "Section"
    text = " ".join(b.text for b in blocks[:400])
    regs = sum(bool(REGULATION.match(b.text)) and bool(re.search(r"[—–-]", b.text[:8]))
               for b in blocks)
    if regs >= 3:
        if re.search(r"\bthese Rules\b", text, re.IGNORECASE):
            return "regulations", "Rule"
        if re.search(r"\bthis Order\b", text, re.IGNORECASE):
            return "regulations", "Article"
        return "regulations", "Regulation"
    return "paragraphs", "Para"


def _split_sentences(text: str, max_words: int) -> list[str]:
    """Split one very long block into pieces of at most max_words, at sentence ends."""
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(“\"])", text)
    pieces, current = [], []
    for s in sentences:
        if current and _n_words(" ".join(current + [s])) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(s)
    if current:
        pieces.append(" ".join(current))
    # A single sentence longer than max_words is cut by words.
    out = []
    for p in pieces:
        w = p.split()
        out += [" ".join(w[i:i + max_words]) for i in range(0, len(w), max_words)]
    return out


def _spans(blocks: list[Block]) -> list[dict]:
    """One bounding box per page (the union of the chunk's blocks on that page)."""
    by_page: dict[int, list[float]] = {}
    for b in blocks:
        x0, y0, x1, y1 = b.bbox
        if b.page in by_page:
            r = by_page[b.page]
            by_page[b.page] = [min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1)]
        else:
            by_page[b.page] = [x0, y0, x1, y1]
    return [{"page": p, "bbox": [round(v, 1) for v in bb]} for p, bb in sorted(by_page.items())]


# ----------------------------------------------------------------------------
# Chunk builder
# ----------------------------------------------------------------------------

@dataclass
class _State:
    section_id: str = "Preamble"
    section_title: str = ""
    parents: list[tuple[int, str]] = field(default_factory=list)
    num: str | None = None
    letter: str | None = None
    recitals: bool = False             # True once "Whereas" is seen in the preamble
    part: list[str] = field(default_factory=list)   # judgment part path, e.g. ["II", "B"]
    last_numbers: dict = field(default_factory=dict)  # last Article / Section / Regulation number


class _Builder:
    def __init__(self, doc: Document):
        self.doc = doc
        self.blocks = doc.content_blocks()
        self.mode, self.unit_word = choose_mode(self.blocks)
        self.style = "para" if self.mode == "paragraphs" else "paren"
        # Running header per page. A header that does not repeat on the next page (e.g. the
        # court name on the first page of a dissent) is replaced by the next page's header.
        headers = [p.running_header for p in doc.pages] + [""]
        self.running = {
            p.number: headers[i] if headers[i] == headers[i + 1] or not headers[i + 1] else headers[i + 1]
            for i, p in enumerate(doc.pages)}
        self.chunks: list[Chunk] = []
        self.s = _State(section_id="Text" if self.mode == "paragraphs" else "Preamble")
        self.segment = ""                        # current running header (judgments)
        self.buf: list[Block] = []
        self.buf_has_body = False
        self.paras: list[str] = []
        self.awaiting_title: int | None = None   # heading had no title: next short block is the title
        self.title_open: int | None = None       # upper-case title may wrap onto the next block
        self.footnotes: dict[int, list[Block]] = {}
        self.footnote_ctx: dict[int, tuple] = {}

    # -- context helpers --
    def _label(self) -> str | None:
        return self.s.num or self.s.letter

    def _parents(self) -> list[str]:
        return [label for _, label in self.s.parents]

    def _annex(self) -> str | None:
        """Label of the enclosing Schedule/Annex/Exhibit, e.g. "Schedule 2"."""
        annex = next((l for r, l in self.s.parents if r == 0), None)
        return " ".join(annex.split()[:2]) if annex and self.mode != "paragraphs" else None

    def _emit(self, kind: str, blocks: list[Block], text: str, ref: str,
              paras: list[str] | None = None) -> None:
        pages = [b.page for b in blocks]
        self.chunks.append(Chunk(
            chunk_id=f"{self.doc.doc_id}:{len(self.chunks):04d}",
            doc_id=self.doc.doc_id, doc_title=self.doc.title, kind=kind,
            section_id=self.s.section_id, section_title=self.s.section_title,
            parents=self._parents(), ref=ref,
            page_start=min(pages), page_end=max(pages), spans=_spans(blocks),
            text=text, n_words=_n_words(text), paras=list(paras or []), ref_style=self.style))

    def flush(self) -> None:
        if self.buf:
            text = "\n".join(b.text for b in self.buf)
            recitals = self.s.section_id == "Preamble" and self.s.recitals
            self._emit("text", self.buf, text,
                       format_ref(self.s.section_id, self.paras, recitals, self.style), self.paras)
        self.buf, self.buf_has_body = [], False
        label = self._label()
        self.paras = [label] if label else []

    def _buf_words(self) -> int:
        return sum(_n_words(b.text) for b in self.buf)

    # -- structure handlers --
    def _level1(self, label: str, title: str, rank: int | None = None) -> None:
        self.flush()
        if rank is None:
            rank = LEVEL1_RANK.get(label.split()[0].lower(), 2)
        self.s.parents = [(r, l) for r, l in self.s.parents if r < rank]
        self.s.parents.append((rank, f"{label} {title}".strip()))
        # Text that sits directly under a Part/Schedule (before any Article/Section) is cited by it.
        annex = self._annex()
        if rank > 0 and annex:                     # "Part 4" inside "Schedule 5"
            label = f"{annex}, {label}"
        self.s.section_id, self.s.section_title = label, ""
        self.s.num = self.s.letter = None
        self.s.part = []
        self.paras = []
        self.awaiting_title = None if title else 1
        self.title_open = 1 if title.isupper() else None

    def _level2(self, b: Block | None, label: str, title: str) -> None:
        carried: list[Block] = []
        if self.mode != "sections" and not title:
            # A short cross-heading just before the unit is its title ("Citation and commencement").
            while self.buf and _is_title_line(self.buf[-1].text) and len(carried) < 2:
                carried.insert(0, self.buf.pop())
            title = " ".join(x.text for x in carried)
            self.buf_has_body = bool(self.buf) and self.buf_has_body
        if self.buf_has_body:
            self.flush()
        elif self.buf:
            if self.mode == "sections":
                # Heading directly followed by another heading: the first one is a container
                # (e.g. GDPR "Section 1 Transparency" grouping Articles 12-14).
                container = f"{self.s.section_id} {self.s.section_title}".strip()
                self.s.parents = [(r, l) for r, l in self.s.parents if r < CONTAINER_RANK]
                self.s.parents.append((CONTAINER_RANK, container))
            self.buf = []
        # Numbered paragraphs inside a Schedule/Annex are cited as "Schedule 2 para 11".
        annex = self._annex()
        if annex and re.fullmatch(r"(Section|Regulation|Rule|Article) \d+[A-Z]{0,2}", label) \
                and not (self.mode == "sections" and label.startswith("Article")):
            label = f"{annex} para {label.split()[1]}"
        self.s.section_id, self.s.section_title = label, title
        self.s.num = self.s.letter = None
        self.paras = []
        self.buf = carried + ([b] if b else [])
        self.buf_has_body = False
        self.awaiting_title = None if title else 2
        self.title_open = 2 if title.isupper() else None

    def _extend_title(self, level: int, text: str, replace: bool) -> None:
        if level == 1:
            rank, label = self.s.parents[-1]
            self.s.parents[-1] = (rank, f"{label} {text}")
        else:
            self.s.section_title = text if replace else f"{self.s.section_title} {text}"

    def _accept_unit_number(self, n: int, kind: str = "unit") -> bool:
        """Articles / sections / regulations are numbered in sequence (numbering restarts at 1
        in schedules). Rejects out-of-order matches such as a wrapped cross-reference line
        "Article 81" between Articles 68 and 69, or "30 Days after ..."."""
        last = self.s.last_numbers.get(kind, 0)
        ok = last == 0 or n == 1 or 0 <= n - last <= 5
        if ok:
            self.s.last_numbers[kind] = n
        return ok

    def _structure(self, b: Block) -> bool:
        """Handle b if it is a heading for the current mode. Returns True if handled."""
        text = b.text
        heading = detect_heading(text, self.mode)
        if heading:
            level, label, title = heading
            if level == 2 and (m := re.fullmatch(r"(Article|Section) (\d+)[A-Z]{0,2}", label)):
                if not self._accept_unit_number(int(m.group(2)), m.group(1)):
                    return False
            if level == 1:
                self._level1(label, title)
            else:
                self._level2(b, label, title)
            return True

        if self.mode == "regulations" and (m := REGULATION.match(text)):
            if self._accept_unit_number(int(re.match(r"\d+", m.group(1)).group()), "Regulation"):
                self._level2(None, f"{self.unit_word} {m.group(1)}", "")
                self.awaiting_title = None
                if m.group(2):                       # "12.—(1) ..." starts with paragraph (1)
                    self.s.num = m.group(2)
                    self.paras = [m.group(2)]
                self.buf.append(b)
                self.buf_has_body = True
                return True

        if self.mode == "paragraphs":
            if m := LONE_PART.match(text):
                token = m.group(1)
                is_roman = bool(re.fullmatch(r"[IVX]+", token))
                # "I" after part "H" is a letter, not roman one.
                if is_roman and len(token) == 1 and len(self.s.part) >= 2 \
                        and ord(token) == ord(self.s.part[1]) + 1:
                    is_roman = False
                if is_roman:
                    self.s.part = [token]
                elif token.isalpha():
                    self.s.part = self.s.part[:1] + [token]
                else:
                    self.s.part = self.s.part[:2] + [token]
                self._level2(None, "Part " + "-".join(self.s.part), "")
                self.awaiting_title = None
                return True
            if b.bold and _is_short_heading(text) and _n_words(text) >= 1 and not text.isdigit():
                self._level2(None, text, "")
                self.buf = [b]
                self.awaiting_title = None
                return True
        return False

    def _table(self, b: Block) -> None:
        self.flush()
        lines = b.text.split("\n")
        header, rows = lines[:2], lines[2:]
        piece: list[str] = []
        for row in rows:
            if piece and _n_words("\n".join(header + piece + [row])) > config.CHUNK_MAX_WORDS:
                self._emit("table", [b], "\n".join(header + piece), f"{self.s.section_id}, table")
                piece = []
            piece.append(row)
        self._emit("table", [b], "\n".join(header + piece), f"{self.s.section_id}, table")

    def _text(self, b: Block) -> None:
        short = _n_words(b.text) <= 20 and not detect_marker(b.text, None)
        level = self.awaiting_title or (self.title_open if b.text.isupper() else None)
        if level and short:
            self._extend_title(level, b.text, replace=bool(self.awaiting_title))
            if level == 2:
                self.buf.append(b)
            self.title_open = level if b.text.isupper() else None
            self.awaiting_title = None
            return
        self.awaiting_title = self.title_open = None

        if self.s.section_id == "Preamble" and re.match(r"(?i)whereas\b", b.text):
            self.s.recitals = True

        marker = detect_marker(b.text, self.s.letter)
        if marker and self.mode == "paragraphs" and marker[0] == "letter":
            marker = None                        # judgments: only numbered paragraphs are cited
        if marker:
            kind, label = marker
            if kind == "num":
                new_para = label != self.s.num
                self.s.num, self.s.letter = label, None
            else:
                new_para = self.s.num is None and label != self.s.letter
                self.s.letter = label
            if new_para:
                if self._buf_words() >= config.CHUNK_TARGET_WORDS:
                    self.flush()          # flush() starts the new chunk with the current label
                elif self._label() not in self.paras:
                    self.paras.append(self._label())

        words = _n_words(b.text)
        if words > config.CHUNK_MAX_WORDS:
            for piece in _split_sentences(b.text, config.CHUNK_MAX_WORDS):
                part = Block(piece, b.page, b.bbox, b.size, b.bold, b.kind)
                if self._buf_words() + _n_words(piece) > config.CHUNK_MAX_WORDS:
                    self.flush()
                self.buf.append(part)
                self.buf_has_body = True
            return
        if self._buf_words() + words > config.CHUNK_MAX_WORDS and self._buf_words() >= config.CHUNK_MIN_WORDS:
            self.flush()
        self.buf.append(b)
        self.buf_has_body = True

    def run(self) -> list[Chunk]:
        for b in self.blocks:
            # Judgments: a change of running header ("Syllabus" -> "Opinion of the Court")
            # starts a new top-level segment.
            header = self.running.get(b.page, "")
            if self.mode == "paragraphs" and header and header != self.segment:
                self.segment = header
                self._level1(header, "", rank=0)
                self.awaiting_title = self.title_open = None
            if b.kind == "footnote":
                self.footnotes.setdefault(b.page, []).append(b)
                self.footnote_ctx.setdefault(
                    b.page, (self.s.section_id, self.s.section_title, list(self.s.parents)))
                continue
            if b.kind == "table":
                self._table(b)
                continue
            if self._structure(b):
                continue
            self._text(b)
        self.flush()

        # Footnotes: one chunk per page, attached to the section where they appear.
        for page, notes in sorted(self.footnotes.items()):
            self.s.section_id, self.s.section_title, self.s.parents = self.footnote_ctx[page]
            self._emit("footnote", notes, "\n".join(n.text for n in notes),
                       f"{self.s.section_id}, footnotes")
        return self.chunks


# ----------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------

def _combine(a: Chunk, b: Chunk) -> Chunk:
    """Merge chunk b into chunk a (a comes first in the document)."""
    paras = a.paras + [p for p in b.paras if p not in a.paras]
    ref = format_ref(a.section_id, paras, a.ref.startswith("Recital"), a.ref_style)
    blocks = [Block("", s["page"], tuple(s["bbox"]), 0, False) for s in a.spans + b.spans]
    text = f"{a.text}\n{b.text}"
    return Chunk(a.chunk_id, a.doc_id, a.doc_title, a.kind, a.section_id, a.section_title,
                 a.parents, ref, min(a.page_start, b.page_start), max(a.page_end, b.page_end),
                 _spans(blocks), text, _n_words(text), paras, a.ref_style)


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Fold text chunks shorter than CHUNK_MIN_WORDS into a neighbour from the same section."""
    def can_merge(a: Chunk, b: Chunk) -> bool:
        return (a.kind == b.kind == "text" and a.section_id == b.section_id
                and a.parents == b.parents
                and a.n_words + b.n_words <= config.CHUNK_MAX_WORDS + config.CHUNK_MIN_WORDS)

    out: list[Chunk] = []
    for c in chunks:                                   # backward: into the previous chunk
        if out and c.n_words < config.CHUNK_MIN_WORDS and can_merge(out[-1], c):
            out[-1] = _combine(out[-1], c)
        else:
            out.append(c)
    final: list[Chunk] = []
    for i, c in enumerate(out):                        # forward: into the next chunk
        if c.n_words < config.CHUNK_MIN_WORDS and i + 1 < len(out) and can_merge(c, out[i + 1]):
            out[i + 1] = _combine(c, out[i + 1])
        else:
            final.append(c)
    for n, c in enumerate(final):
        c.chunk_id = f"{c.doc_id}:{n:04d}"
    return final


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def chunk_document(doc: Document) -> list[Chunk]:
    return _merge_small(_Builder(doc).run())


def detect_mode(doc: Document) -> str:
    return choose_mode(doc.content_blocks())[0]


def document_structure(chunks: list[Chunk]) -> list[dict]:
    """Ordered list of sections with their first page, for display in the UI."""
    seen, out = set(), []
    for c in chunks:
        key = (c.doc_id, tuple(c.parents), c.section_id)
        if c.kind == "text" and key not in seen:
            seen.add(key)
            out.append({"doc_id": c.doc_id, "section": c.section_id, "title": c.section_title,
                        "parents": c.parents, "page": c.page_start})
    return out


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1),
                    encoding="utf-8")


def load_chunks(path: str | Path) -> list[Chunk]:
    return [Chunk(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def main(paths: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    all_chunks: list[Chunk] = []
    for p in paths:
        doc = ingest_pdf(p)
        chunks = chunk_document(doc)
        all_chunks += chunks
        sizes = [c.n_words for c in chunks if c.kind == "text"] or [0]
        sections = document_structure(chunks)
        kinds = {k: sum(c.kind == k for c in chunks) for k in ("text", "table", "footnote")}
        print(f"\n=== {doc.title} [{detect_mode(doc)} mode]: {len(chunks)} chunks {kinds}, "
              f"{len(sections)} sections")
        print(f"    words per text chunk: min {min(sizes)}, median {statistics.median(sizes):.0f}, "
              f"max {max(sizes)}")
        for s in sections[:12]:
            print(f"    p.{s['page']:<4} {s['section']:<14} {s['title'][:60]}")
    out = config.INDEX_DIR / "chunks.json"
    save_chunks(all_chunks, out)
    print(f"\nSaved {len(all_chunks)} chunks to {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in sorted(config.DATA_DIR.glob("*.pdf"))])
