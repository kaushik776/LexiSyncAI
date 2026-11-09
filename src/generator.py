"""Step 5: generate a plain-language answer from the retrieved sources only.

The LLM sees numbered sources [S1]..[Sn] (document, page, clause, text) and must return
JSON: the answer with [S#] markers after every claim, the list of claims with their
sources, an explanation of why the evidence supports the answer, any conflict between
sources, and whether the question is answerable at all.

Citations are then checked: a marker pointing to a source that does not exist is removed,
so the model cannot invent a citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.chunker import Chunk
from src.llm import LLM, Usage

SYSTEM = """You are LexiSyncAI, a careful legal analyst. Answer the QUESTION using ONLY the numbered SOURCES.

Rules:
1. Use only facts stated in the SOURCES. Never use outside knowledge. Copy figures, amounts, dates and time limits exactly as written.
2. Write the answer as a list of "sentences". For each sentence give in "sources" the numbers of the SOURCES that state it, e.g. [2] or [1, 3]. Every sentence needs at least one source.
3. If the SOURCES do not contain the information needed, set "answerable" to false and say what is missing in "missing". If only part of the question is covered, answer that part, set "answerable" to true and name the missing part in "missing".
4. If sources disagree (different figures or rules for the same thing), report both with their citations, say which one you rely on and why (an operative article over a recital, the more specific provision over the general one, the current document over an older one), and describe it in "conflict".
5. Answer every part of the question. Write for a non-lawyer: plain language, 2-6 sentences. Lead with the direct answer (e.g. "Yes."/"No."/the figure).
6. "explanation": 1-2 sentences on why the cited evidence supports the answer, e.g. which provisions or documents were combined.
7. "conflict" and "missing": use an empty string "" when there is no conflict or nothing is missing.
8. If the question asks which or what things apply (cases, violations, exceptions, conditions), list them from the source in plain words. Never answer only with a reference such as "those listed in Article 83(5)".

Example of the output format:
{"answerable": true,
 "sentences": [{"text": "No.", "sources": [1]},
               {"text": "The liability cap in Section 7 is EUR 2.0M.", "sources": [1]},
               {"text": "The Section 9 indemnities add up to EUR 3.5M, so the cap is lower.", "sources": [2, 3]}],
 "explanation": "Combines the cap in Section 7.2 with the indemnity table in Section 9.", "conflict": "", "missing": ""}"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "sentences": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {"text": {"type": "string"},
                           "sources": {"type": "array", "items": {"type": "integer"}}},
            "required": ["text", "sources"]}},
        "explanation": {"type": "string"},
        "conflict": {"type": "string"},
        "missing": {"type": "string"},
    },
    "required": ["answerable", "sentences", "explanation", "conflict", "missing"],
}

_MARKER = re.compile(r"\[S(\d+)\]")


@dataclass
class Claim:
    text: str
    source_ids: list[int]                 # indices into the sources list (0-based)
    supported: bool | None = None         # set by the verifier
    reason: str = ""
    part: int = -1                        # which sub-question it answers (multi-part questions)
    derived: bool = False                 # conclusion combining cited facts (e.g. a comparison)


@dataclass
class Draft:
    answerable: bool
    answer: str
    claims: list[Claim]
    explanation: str
    conflict: str = ""
    missing: str = ""
    raw: dict = field(default_factory=dict)

    def cited_ids(self) -> list[int]:
        ids = {i for c in self.claims for i in c.source_ids}
        ids |= {int(m) - 1 for m in _MARKER.findall(self.answer)}
        return sorted(ids)


def format_sources(chunks: list[Chunk], max_words: int = 380) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        words = c.text.split()
        text = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        parts.append(f"[S{i}] {c.doc_title} | {c.pages_label} | {c.ref}\n"
                     f"Location: {c.heading_path}\n{text}")
    return "\n\n".join(parts)


def _source_ids(labels: list[str], n: int) -> list[int]:
    ids = []
    for label in labels:
        m = re.search(r"\d+", str(label))
        if m and 1 <= int(m.group()) <= n and int(m.group()) - 1 not in ids:
            ids.append(int(m.group()) - 1)
    return ids


_EMPTY = {"", "none", "n/a", "na", "missing", "nothing", "no", "null", "no conflict", "no conflicts",
          "not applicable", "nothing is missing", "nothing missing", "there is no conflict"}


def _clean_optional(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower().strip(" .!") in _EMPTY else text


def split_claims(answer: str, n_sources: int) -> list[Claim]:
    """Each sentence of the answer is a claim; its [S#] markers are its sources.
    A sentence without markers inherits those of the previous (or next) sentence;
    a bare "Yes." / "No." is not a claim."""
    # "... 3.5M. [S2] Therefore" -> "... 3.5M [S2]. Therefore": markers belong to the sentence before.
    text = re.sub(r"([.!?])\s*((?:\[S\d+\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", answer)
    text = re.sub(r"\s{2,}", " ", text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", text) if s.strip()]
    claims: list[Claim] = []
    pending: list[int] = []
    for sent in sentences:
        body = _MARKER.sub("", sent).strip()
        if len(body.split()) <= 3:                 # "Yes." / "No." / "In short:"
            continue
        ids = _source_ids(_MARKER.findall(sent), n_sources)
        if not ids and claims:
            ids = list(claims[-1].source_ids)
        claim = Claim(re.sub(r"\s+([.,;:])", r"\1", body), ids)
        if not ids:
            pending.append(len(claims))
        claims.append(claim)
        if ids and pending:                        # leading sentences inherit the first markers
            for k in pending:
                claims[k].source_ids = list(ids)
            pending = []
    return claims


def parse_draft(data: dict, n_sources: int) -> Draft:
    """Build the answer text from the sentence list, adding [S#] markers for the sources the
    model gave. Source numbers that do not exist are dropped, so citations cannot be invented.
    (A plain "answer" string is still accepted: its sentences are split and markers parsed.)"""
    sentences = data.get("sentences")
    if isinstance(sentences, list) and sentences:
        claims, parts = [], []
        for item in sentences:
            if not isinstance(item, dict):
                continue
            text = _MARKER.sub("", str(item.get("text", ""))).strip()
            if not text:
                continue
            ids = _source_ids([str(x) for x in item.get("sources", []) or []], n_sources)
            ids += [i for i in _source_ids(_MARKER.findall(str(item.get("text", ""))), n_sources) if i not in ids]
            parts.append(text + "".join(f" [S{i + 1}]" for i in ids))
            if len(text.split()) > 3:                       # "Yes." is not a claim
                claims.append(Claim(text, ids))
        answer = " ".join(parts)
    else:
        answer = str(data.get("answer", "")).strip()
        answer = _MARKER.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)
        claims = split_claims(answer, n_sources)
    return Draft(answerable=bool(data.get("answerable", False)) and bool(answer),
                 answer=answer, claims=claims,
                 explanation=str(data.get("explanation", "")).strip(),
                 conflict=_clean_optional(data.get("conflict")),
                 missing=_clean_optional(data.get("missing")), raw=data)


def generate(llm: LLM, question: str, sources: list[Chunk], usage: Usage,
             feedback: str = "", parts: list[str] | None = None) -> Draft:
    user = f"SOURCES:\n{format_sources(sources)}\n\nQUESTION: {question}"
    if parts:
        user += ("\n\nThe question has these parts. Answer each one in the answer text, with citations:\n"
                 + "\n".join(f"{k}. {p}" for k, p in enumerate(parts, 1)))
    if feedback:
        user += f"\n\nREVIEW OF YOUR PREVIOUS DRAFT (fix these problems):\n{feedback}"
    data = llm.chat_json(SYSTEM, user, ANSWER_SCHEMA, usage, purpose="redraft" if feedback else "draft")
    return parse_draft(data, len(sources))


CONCLUDE_SYSTEM = """You combine verified partial answers into the direct answer to a legal question.
Write 1-2 sentences that answer the QUESTION itself, e.g. state which figure is larger, whether a limit
is exceeded, or how two rules relate. Use only facts stated in the PARTIAL ANSWERS and copy figures exactly.
For each sentence give in "sources" the source numbers of the facts it uses (the numbers inside [S#])."""

CONCLUDE_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "properties": {"text": {"type": "string"},
                       "sources": {"type": "array", "items": {"type": "integer"}}},
        "required": ["text", "sources"]}}},
    "required": ["sentences"],
}

_NEEDS_CONCLUSION = re.compile(r"\b((longer|shorter|higher|lower|larger|smaller|bigger|more|less|greater) than|"
                               r"exceed\w*|compare\w*|comparison|differen\w*|differ|versus|vs\.?|which is|"
                               r"which one|consistent|conflict\w*|the same as)\b", re.IGNORECASE)


def needs_conclusion(question: str) -> bool:
    """Comparison-type questions need a sentence that combines the parts ("which is longer?")."""
    return bool(_NEEDS_CONCLUSION.search(question))


def conclude(llm: LLM, question: str, partial_answer: str, n_sources: int, usage: Usage,
             computed: str = "") -> list[Claim]:
    """Write the direct answer (e.g. the comparison) from the verified part answers.
    `computed` is the calculator's exact comparison of the quantities (see calculator.py)."""
    user = f"PARTIAL ANSWERS:\n{partial_answer}\n\nQUESTION: {question}"
    if computed:
        user += f"\n\nCOMPUTED COMPARISON (exact, use it; do not compare the figures yourself):\n{computed}"
    data = llm.chat_json(CONCLUDE_SYSTEM, user, CONCLUDE_SCHEMA, usage, purpose="conclude", max_tokens=200)
    draft = parse_draft({"answerable": True, "sentences": data.get("sentences", []), "explanation": "",
                         "conflict": "", "missing": ""}, n_sources)
    for c in draft.claims:
        c.derived = True
    return draft.claims


CLOSED_BOOK_SYSTEM = """You are a legal assistant. Answer the question from your own knowledge in 2-5 plain sentences.
If you do not know, set "answerable" to false."""


def generate_closed_book(llm: LLM, question: str, usage: Usage) -> Draft:
    """No-retrieval baseline for the mandatory comparison: the LLM answers without sources."""
    schema = {"type": "object", "properties": {"answerable": {"type": "boolean"}, "answer": {"type": "string"}},
              "required": ["answerable", "answer"]}
    data = llm.chat_json(CLOSED_BOOK_SYSTEM, f"QUESTION: {question}", schema, usage, purpose="closed_book")
    return Draft(bool(data.get("answerable")), str(data.get("answer", "")), [], "", raw=data)
