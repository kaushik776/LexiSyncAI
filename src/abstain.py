"""Confidence badge and abstention.

Signals:
  * evidence strength = cross-encoder probability of the best cited source
    (how strongly the passage answers the question),
  * verification      = share of the answer's claims that passed the verifier,
  * the LLM's own judgement that the sources answer the question.

  INSUFFICIENT EVIDENCE  no source scores above ABSTAIN_SCORE, the LLM says the sources do
                         not answer it, or no claim survives verification -> abstain
  HIGH                   strength >= HIGH_SCORE, every claim verified, nothing missing
  MEDIUM                 strength >= LOW_SCORE and >= 75% of claims verified
  LOW                    anything weaker
"""
from __future__ import annotations

from src import config

HIGH, MEDIUM, LOW, INSUFFICIENT = "HIGH", "MEDIUM", "LOW", "INSUFFICIENT EVIDENCE"

ABSTAIN_MESSAGE = "The documents do not contain enough evidence to answer this question."


def should_abstain_early(top_score: float) -> bool:
    """Before calling the LLM: nothing in the documents is even close to the question."""
    return top_score < config.ABSTAIN_SCORE


def badge(evidence_strength: float, verified_ratio: float | None, answerable: bool,
          n_citations: int, missing: str = "") -> str:
    if not answerable or n_citations == 0 or evidence_strength < config.ABSTAIN_SCORE:
        return INSUFFICIENT
    if verified_ratio is not None and verified_ratio == 0:
        return INSUFFICIENT
    fully_verified = verified_ratio is None or verified_ratio == 1.0
    if evidence_strength >= config.HIGH_SCORE and fully_verified and not missing:
        return HIGH
    if evidence_strength >= config.LOW_SCORE and (verified_ratio is None or verified_ratio >= 0.75):
        return MEDIUM
    return LOW
