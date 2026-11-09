"""Deterministic comparisons for questions such as "Which is longer: 72 hours or 30 days?" or
"Does the EUR 2.0M cap exceed the EUR 3.5M indemnities?".

A 7B LLM is unreliable at comparing quantities in different units (it once concluded that
72 hours is longer than 30 days). So the pipeline:
  1. extracts durations and money amounts from the verified facts and converts them to one
     unit (hours; or the same currency),
  2. gives the LLM the computed comparison as a note when it writes the conclusion, and
  3. rejects a conclusion whose stated comparison contradicts the computed values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_HOURS = {"hour": 1, "day": 24, "week": 24 * 7, "month": 24 * 30.4, "year": 24 * 365}
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15,
          "twenty": 20, "thirty": 30, "forty": 40, "sixty": 60, "ninety": 90}
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_CURRENCY = {"eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR", "gbp": "GBP", "£": "GBP",
             "usd": "USD", "$": "USD", "dollars": "USD", "pounds": "GBP"}

_NUM = r"(\d+(?:[ ,]\d{3})*(?:\.\d+)?|" + "|".join(_WORDS) + r")"
_DURATION = re.compile(_NUM + r"[\s-]*(hour|day|week|month|year)s?\b", re.IGNORECASE)
_MONEY_BEFORE = re.compile(r"(EUR|GBP|USD|€|£|\$)\s?" + _NUM + r"(?:\s?(thousand|million|billion|bn|mn|m|k)\b)?",
                           re.IGNORECASE)
_MONEY_AFTER = re.compile(_NUM + r"(?:\s?(thousand|million|billion|bn|mn|m|k)\b)?\s?(EUR|euros?|GBP|pounds|USD|dollars)\b",
                          re.IGNORECASE)
_MORE = re.compile(r"\b(longer|more|greater|higher|larger|bigger|exceeds?|exceeding|above|over)\b", re.IGNORECASE)
_LESS = re.compile(r"\b(shorter|less|lower|smaller|below|under|does not exceed|do not exceed)\b", re.IGNORECASE)


@dataclass
class Quantity:
    text: str          # as written, e.g. "30 days"
    kind: str          # "duration" or a currency code
    value: float       # hours, or amount in that currency
    start: int         # position in the text


def _number(raw: str) -> float:
    raw = raw.lower()
    if raw in _WORDS:
        return float(_WORDS[raw])
    return float(re.sub(r"[ ,]", "", raw))


def quantities(text: str) -> list[Quantity]:
    """Durations and money amounts in the order they appear."""
    found = []
    for m in _DURATION.finditer(text):
        found.append(Quantity(m.group(0), "duration", _number(m.group(1)) * _HOURS[m.group(2).lower()], m.start()))
    for m in _MONEY_BEFORE.finditer(text):
        scale = _SCALE.get((m.group(3) or "").lower(), 1)
        found.append(Quantity(m.group(0).strip(), _CURRENCY[m.group(1).lower()], _number(m.group(2)) * scale, m.start()))
    for m in _MONEY_AFTER.finditer(text):
        if any(q.start <= m.start() < q.start + len(q.text) for q in found):
            continue
        scale = _SCALE.get((m.group(2) or "").lower(), 1)
        found.append(Quantity(m.group(0).strip(), _CURRENCY[m.group(3).lower()], _number(m.group(1)) * scale, m.start()))
    return sorted(found, key=lambda q: q.start)


def _fmt(q: Quantity) -> str:
    if q.kind == "duration":
        return f"{q.text} = {q.value:,.0f} hours"
    return f"{q.text} = {q.kind} {q.value:,.0f}"


def comparison_note(facts: str) -> str:
    """A short, computed comparison of the quantities in the facts, or "" if there is nothing to compare."""
    qs = quantities(facts)
    lines = []
    for kind in dict.fromkeys(q.kind for q in qs):
        same = list({q.text.lower(): q for q in qs if q.kind == kind}.values())
        if len(same) < 2:
            continue
        ordered = sorted(same, key=lambda q: q.value, reverse=True)
        lines.append("; ".join(_fmt(q) for q in same) + ". Largest to smallest: "
                     + " > ".join(q.text for q in ordered) + ".")
    return " ".join(lines)


def check_comparison(sentence: str) -> tuple[bool, str]:
    """If the sentence compares two quantities of the same kind ("72 hours ... longer than 30 days"),
    check the direction against the values. Returns (ok, reason). Sentences it cannot judge pass."""
    qs = quantities(sentence)
    for a, b in zip(qs, qs[1:]):
        if a.kind != b.kind or a.value == b.value:
            continue
        between = sentence[a.start + len(a.text):b.start]
        more, less = _MORE.search(between), _LESS.search(between)
        if more and not less and a.value < b.value:
            return False, f"says {a.text} is more/longer than {b.text}, but {_fmt(a)} < {_fmt(b)}"
        if less and not more and a.value > b.value:
            return False, f"says {a.text} is less/shorter than {b.text}, but {_fmt(a)} > {_fmt(b)}"
    return True, ""


def comparison_sentence(facts: list[tuple[str, list[int]]]) -> tuple[str, list[int]] | None:
    """Write the comparison ourselves from verified facts [(sentence, source ids)]:
    "30 days (720 hours) is longer than 72 hours." Used when the LLM's conclusion fails
    verification. Returns (sentence, source ids of both figures) or None."""
    found: list[tuple[Quantity, list[int]]] = []
    for text, ids in facts:
        found += [(q, ids) for q in quantities(text)]
    for kind in dict.fromkeys(q.kind for q, _ in found):
        same = [(q, ids) for q, ids in found if q.kind == kind]
        big = max(same, key=lambda x: x[0].value)
        small = min(same, key=lambda x: x[0].value)
        if big[0].value == small[0].value:
            continue
        if kind == "duration":
            hours = f" ({big[0].value:,.0f} hours)" if big[0].text.split()[-1].rstrip("s") != "hour" else ""
            text = f"Comparing the two: {big[0].text}{hours} is longer than {small[0].text}."
        else:
            text = f"Comparing the two: {big[0].text} is more than {small[0].text}."
        return text, list(dict.fromkeys(big[1] + small[1]))
    return None
