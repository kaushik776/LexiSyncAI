"""Tests for the deterministic comparison calculator."""
from src.calculator import check_comparison, comparison_note, quantities
from src.generator import Claim
from src.verifier import verify
from src.chunker import Chunk


def test_quantities_normalised():
    qs = quantities("not later than 72 hours; within thirty days; six months; EUR 2.0 million; 20 000 000 EUR; £5,000")
    assert [(q.kind, q.value) for q in qs] == [
        ("duration", 72), ("duration", 720), ("duration", 6 * 24 * 30.4),
        ("EUR", 2_000_000), ("EUR", 20_000_000), ("GBP", 5_000)]


def test_comparison_note_orders_values():
    note = comparison_note("Notify within 72 hours [S1]. Reject within 30 days [S2].")
    assert "30 days = 720 hours" in note and note.endswith("30 days > 72 hours.")
    assert comparison_note("Only one figure: 30 days.") == ""


def test_check_comparison():
    assert not check_comparison("The 72 hours deadline is longer than the 30 days limit.")[0]
    assert check_comparison("The 30 days limit is longer than the 72 hours deadline.")[0]
    assert not check_comparison("The cap of EUR 2.0 million exceeds the indemnities of EUR 3.5 million.")[0]
    assert check_comparison("The cap of EUR 2.0 million does not exceed the EUR 3.5 million indemnities.")[0]
    assert check_comparison("The deadline is 72 hours.")[0]                     # nothing to compare


def test_verifier_rejects_wrong_comparison():
    src = [Chunk("D:0", "D", "D", "text", "Article 1", "", [], "Article 1", 1, 1, [],
                 "Notify within 72 hours. Reject within 30 days.", 8)]
    claim = Claim("The 72 hours deadline is longer than the 30 days limit.", [0], derived=True)
    verify(None, [claim], src, usage=None)
    assert claim.supported is False and "wrong comparison" in claim.reason


def test_comparison_sentence():
    from src.calculator import comparison_sentence
    text, ids = comparison_sentence([("The GDPR deadline is 72 hours.", [1]), ("Reject within 30 days.", [0])])
    assert text == "Comparing the two: 30 days (720 hours) is longer than 72 hours." and ids == [0, 1]
    assert comparison_sentence([("Only 30 days.", [0])]) is None
