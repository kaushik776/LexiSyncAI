"""Does this PDF look like a legal document? (warning only, never blocks)

LexiSyncAI is built for contracts, statutes, regulations, judgments and regulatory filings.
Any PDF is still indexed and answered, but if a document shows few legal signals the
user gets a warning that answers may be less reliable.

The check counts legal vocabulary ("shall", "pursuant to", "hereby", "plaintiff",
"indemnify" ...) and legal structure ("Article 5", "Section 2.1", "(a)", "§ 12") per 1,000
words in the first pages. It is fast, needs no model, and gives the same result every time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingest import Document

# Legal vocabulary. Each pattern counts once per occurrence.
_TERMS = {
    "drafting": r"\bshall\b|\bhereby\b|\bherein\b|\bhereof\b|\bhereto\b|\bthereof\b|\bwhereas\b|"
                r"\bnotwithstanding\b|\bpursuant to\b|\bin accordance with\b|\bprovided that\b|\bsubject to\b",
    "contract": r"\bagreement\b|\bparty\b|\bparties\b|\bindemnif\w*|\bwarrant\w*|\bbreach\w*|\bliabilit\w*|"
                r"\btermination\b|\bgoverning law\b|\bconfidential\w*|\bobligations?\b",
    "legislation": r"\bact\b|\bregulations?\b|\bdirective\b|\bstatut\w*|\blegislation\b|\bparliament\b|"
                   r"\benact\w*|\bamend\w*|\bmember states?\b|\bsecretary of state\b|\bcommission\b",
    "court": r"\bcourt\b|\bjudg(e?ment|es?)\b|\bplaintiffs?\b|\bdefendants?\b|\bappell\w*|\brespondents?\b|"
             r"\bclaimants?\b|\btribunal\b|\bjurisdiction\b|\bpetitioners?\b|\bopinion of the court\b",
    "rights": r"\brights?\b|\bdut(y|ies)\b|\bpenalt(y|ies)\b|\bfines?\b|\benforce\w*|\blawful\w*|\bunlawful\b|"
              r"\bconsent\b|\binfringe\w*|\bcomplian\w*|\bremed(y|ies)\b",
}
# Legal structure: numbered provisions and paragraph markers.
_STRUCTURE = re.compile(r"\b(article|section|clause|regulation|schedule|annex|recital|paragraph)\s+\d+|§\s*\d+|"
                        r"^\s*\(\s*[a-z0-9]{1,3}\s*\)", re.IGNORECASE | re.MULTILINE)

THRESHOLD = 12.0          # legal signals per 1,000 words. Measured: legal PDFs 32-70, non-legal 0-2
SAMPLE_WORDS = 6000       # only the first ~6,000 words are checked (fast on long documents)


@dataclass
class LegalCheck:
    likely_legal: bool
    score: float                         # legal signals per 1,000 words
    words_checked: int
    signals: dict = field(default_factory=dict)   # count per category
    message: str = ""


def check_text(text: str) -> LegalCheck:
    words = text.split()[:SAMPLE_WORDS]
    sample = " ".join(words)
    lower = sample.lower()
    signals = {name: len(re.findall(p, lower)) for name, p in _TERMS.items()}
    signals["structure"] = len(_STRUCTURE.findall(sample))
    n = max(1, len(words))
    score = round(1000 * sum(signals.values()) / n, 1)
    likely = score >= THRESHOLD and len(words) >= 30
    message = "" if likely else (
        "This document may not be a legal document (few legal terms or numbered provisions found). "
        "LexiSyncAI is designed for contracts, statutes, regulations and judgments, so answers may be "
        "less reliable. It has still been indexed.")
    if len(words) < 30:
        message = ("Very little text could be read from this document (scanned or empty?), so it could "
                   "not be checked. It has still been indexed.")
    return LegalCheck(likely, score, len(words), signals, message)


def check_document(doc: Document) -> LegalCheck:
    text = "\n".join(b.text for b in doc.content_blocks() if b.kind in ("text", "table"))
    return check_text(text)
