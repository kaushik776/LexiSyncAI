"""Tests for the 'is this a legal document?' warning."""
from src.legal_check import check_text

CONTRACT = ("This Agreement is made pursuant to Section 2.1. The Parties hereby agree that the Supplier shall "
            "indemnify the Buyer against any breach of its obligations, subject to the liability cap in Article 7. "
            "Notwithstanding the foregoing, the governing law shall be the laws of England. ") * 20
STATUTE = ("Article 5 (1) Member States shall ensure that personal data are processed lawfully. "
           "(a) the controller shall comply with this Regulation; (b) penalties shall be effective. "
           "Section 3 of the Act is amended in accordance with the Directive. ") * 20
RECIPE = ("Preheat the oven to 180 degrees. Mash three ripe bananas, stir in melted butter, sugar and one egg, "
          "then fold in the flour. Bake for about an hour until golden and leave to cool before slicing. ") * 20
PAPER = ("We propose a new network architecture based solely on attention mechanisms. Experiments on two "
         "machine translation tasks show these models to be superior in quality while being more parallelizable. ") * 20


def test_legal_texts_pass():
    for text in (CONTRACT, STATUTE):
        r = check_text(text)
        assert r.likely_legal and r.message == ""


def test_non_legal_texts_warn_but_do_not_block():
    for text in (RECIPE, PAPER):
        r = check_text(text)
        assert not r.likely_legal
        assert "may not be a legal document" in r.message and "still been indexed" in r.message


def test_empty_or_scanned_document():
    r = check_text("")
    assert not r.likely_legal and "Very little text" in r.message
