"""Tests for answer parsing, the number check, the confidence badge and planning rules
(no LLM or model downloads needed)."""
from src import config
from src.abstain import HIGH, INSUFFICIENT, LOW, MEDIUM, badge, should_abstain_early
from src.chunker import Chunk
from src.generator import Claim, parse_draft, split_claims
from src.pipeline import _replace_markers
from src.planner import needs_planning
from src.verifier import check_numbers, numbers_in


def _chunk(text, ref="Article 83(5)"):
    return Chunk(chunk_id="D:0", doc_id="D", doc_title="GDPR", kind="text", section_id="Article 83",
                 section_title="", parents=[], ref=ref, page_start=83, page_end=83, spans=[],
                 text=text, n_words=len(text.split()))


def test_parse_draft_builds_markers_and_drops_invented_sources():
    d = parse_draft({"answerable": True, "explanation": "", "conflict": "none", "missing": "",
                     "sentences": [{"text": "No.", "sources": [1]},
                                   {"text": "The cap is EUR 2.0M.", "sources": [1]},
                                   {"text": "The indemnities total EUR 3.5M.", "sources": [2, 7]}]}, n_sources=3)
    assert d.answer == "No. [S1] The cap is EUR 2.0M. [S1] The indemnities total EUR 3.5M. [S2]"
    assert [c.source_ids for c in d.claims] == [[0], [1]]       # "No." is not a claim; S7 does not exist
    assert d.conflict == ""                                       # "none" is treated as empty


def test_split_claims_moves_markers_and_inherits():
    claims = split_claims("Yes. The cap is EUR 2.0M [S1]. The total is EUR 3.5M. [S2] So the cap is lower.", 3)
    assert [(c.text, c.source_ids) for c in claims] == [
        ("The cap is EUR 2.0M.", [0]), ("The total is EUR 3.5M.", [1]), ("So the cap is lower.", [1])]


def test_numbers_normalised():
    assert numbers_in("up to 20 000 000 EUR, or 4 % of turnover") >= {20_000_000, 4}
    assert numbers_in("EUR 20 million", keep_unscaled=False) == {20_000_000}
    assert 72 in numbers_in("not later than 72 hours (Article 33(1))")
    assert 33 not in numbers_in("as set out in Article 33(1)")              # clause numbers are not facts
    assert 10 in numbers_in("ending on the tenth anniversary")


def test_number_check_catches_changed_figure():
    src = [_chunk("Infringements shall be subject to fines up to 20 000 000 EUR, or up to 4 % of turnover.")]
    assert check_numbers(Claim("The fine can reach EUR 20 million or 4% of turnover.", [0]), src)[0]
    ok, missing = check_numbers(Claim("The fine can reach EUR 50 million.", [0]), src)
    assert not ok and missing == ["50,000,000"]        # 50,000,000 is not in the source


def test_badge_rules():
    assert badge(0.9, 1.0, True, 2) == HIGH
    assert badge(0.9, 0.8, True, 2) == MEDIUM
    assert badge(0.5, 1.0, True, 1) == MEDIUM
    assert badge(0.2, 1.0, True, 1) == LOW
    assert badge(0.9, 0.0, True, 1) == INSUFFICIENT                         # nothing verified
    assert badge(0.9, 1.0, False, 0) == INSUFFICIENT                        # LLM: not answerable
    assert should_abstain_early(config.ABSTAIN_SCORE / 2)


def test_planning_rule():
    assert needs_planning("Does the liability cap in Section 7 exceed the indemnities in Section 9?")
    assert needs_planning("What is the maximum fine, and which violations trigger it?")
    assert not needs_planning("Which state's law governs the agreement?")


def test_replace_markers():
    assert _replace_markers("The cap is EUR 2M. [S1] [S3] Next. [S2]", "The cap is EUR 2M.", [3]) == \
           "The cap is EUR 2M. [S4] Next. [S2]"
