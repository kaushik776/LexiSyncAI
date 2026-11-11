"""Bonus feature 9 - multi-document cross-referencing and conflict detection.

Ask the same question of each document separately (each answer is retrieved, cited and
verified as usual), then reconcile the answers:

  * figures of the same kind that differ   ("EUR 2,000,000" vs "EUR 2,500,000", "3 years" vs "5 years")
  * statements the NLI model judges contradictory (probability >= CONFLICT_P)

Result: "conflict" (with the reasons), "agree", "single" (only one document answers) or
"none" (no document answers).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from src.calculator import quantities
from src.pipeline import Answer, Pipeline
from src.verifier import contradiction

CONFLICT_P = 0.6


@dataclass
class DocAnswer:
    doc_id: str
    title: str
    answer: Answer


@dataclass
class Comparison:
    question: str
    answers: list[DocAnswer]
    status: str                                   # "conflict" | "agree" | "single" | "none"
    findings: list[str] = field(default_factory=list)


def _plain(text: str) -> str:
    return re.sub(r"\s*\[S\d+\]", "", text).strip()


def _figures(text: str) -> dict[str, dict[float, str]]:
    """Quantities by kind: {"EUR": {2000000.0: "EUR 2,000,000"}, "duration": {...}}."""
    out: dict[str, dict[float, str]] = {}
    for q in quantities(text):
        out.setdefault(q.kind, {})[q.value] = q.text
    return out


def reconcile(answers: list[DocAnswer]) -> tuple[str, list[str]]:
    answered = [a for a in answers if not a.answer.abstained]
    if not answered:
        return "none", []
    if len(answered) == 1:
        return "single", [f"Only {answered[0].title} answers this question."]
    findings = []
    for a, b in itertools.combinations(answered, 2):
        fa, fb = _figures(_plain(a.answer.answer)), _figures(_plain(b.answer.answer))
        for kind in fa.keys() & fb.keys():
            if not set(fa[kind]) & set(fb[kind]):           # no figure in common
                findings.append(f"{a.title} states {', '.join(fa[kind].values())} but {b.title} states "
                                f"{', '.join(fb[kind].values())}.")
        p = contradiction(_plain(a.answer.answer), _plain(b.answer.answer))
        if p >= CONFLICT_P:
            findings.append(f"The answers from {a.title} and {b.title} contradict each other "
                            f"(contradiction probability {p:.2f}).")
    return ("conflict" if findings else "agree"), findings


def compare(pipe: Pipeline, question: str, doc_ids: list[str] | None = None) -> Comparison:
    titles = {d["doc_id"]: d.get("name") or d["title"] for d in pipe.index.documents}
    doc_ids = doc_ids or list(titles)
    answers = [DocAnswer(d, titles.get(d, d), pipe.ask(question, doc_ids=[d], use_planner=False)) for d in doc_ids]
    status, findings = reconcile(answers)
    return Comparison(question, answers, status, findings)
