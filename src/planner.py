"""Planner (bonus feature 5, part 1): split a multi-part question into sub-questions.

Multi-hop questions ("What is the cap in Section 7, and does it exceed the indemnities
in Section 9?") need evidence from two places. One search for the whole question tends
to return chunks about only one part, so each sub-question gets its own retrieval and
the evidence is merged.

A cheap rule decides whether planning is needed, so simple questions skip the extra
LLM call.
"""
from __future__ import annotations

import re

from src.llm import LLM, Usage
from src.retriever import extract_refs

SYSTEM = """You split legal questions into the separate facts that must be looked up.
Return 2-3 short, self-contained search questions, one per fact. Keep names, clause numbers,
document names and key terms from the original question. If the question asks for only one
fact, return it unchanged as the only item."""

SCHEMA = {"type": "object",
          "properties": {"sub_questions": {"type": "array", "items": {"type": "string"}}},
          "required": ["sub_questions"]}

_MULTI = re.compile(r"\b(and (what|which|how|when|who|whether|does|do|is|are|can)|compare|comparison|"
                    r"difference|differ|exceed|more than|less than|longer|shorter|higher|lower|both|"
                    r"versus|vs\.?|conflict|consistent|reconcile)\b", re.IGNORECASE)


def needs_planning(question: str) -> bool:
    return bool(_MULTI.search(question)) or len(extract_refs(question)) >= 2 or question.count("?") >= 2


def plan(llm: LLM, question: str, usage: Usage) -> list[str]:
    """Return sub-questions (the original question is always searched as well)."""
    if not needs_planning(question):
        return []
    data = llm.chat_json(SYSTEM, f"QUESTION: {question}", SCHEMA, usage, purpose="plan", max_tokens=256)
    subs = [s.strip() for s in data.get("sub_questions", []) if isinstance(s, str) and s.strip()]
    subs = [s for s in subs if s.lower() != question.lower()][:3]
    return subs if len(subs) >= 2 else []
