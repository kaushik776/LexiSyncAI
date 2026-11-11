"""Adding documents to an existing case: old and new documents end up in one index,
exactly as if they had been uploaded together."""
import numpy as np
import pymupdf
import pytest

from src import config, indexer

TEXT = "Article {n} Scope. The {who} shall deliver the goods in accordance with the agreement. " * 6


def _pdf(path, n, who):
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(60, 60, 540, 800), TEXT.format(n=n, who=who), fontsize=10)
    doc.save(path)
    return str(path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(indexer, "embed_passages", lambda texts, batch_size=32: np.zeros((len(texts), 384), np.float32))
    return tmp_path


def test_added_documents_join_existing_ones(setup):
    a, b, c = (_pdf(setup / f"{x}.pdf", i, x) for i, x in enumerate(["alpha", "beta", "gamma"], 1))
    together = indexer.build_index([a, b, c], "together")
    indexer.build_index([a], "grown")
    grown, notes = indexer.add_to_case("grown", [b, c])
    assert [d["file"] for d in grown.documents] == ["alpha.pdf", "beta.pdf", "gamma.pdf"]
    # Same passages as uploading all three at once.
    assert [ch.text for ch in grown.chunks] == [ch.text for ch in together.chunks]
    assert notes == []
    assert len(indexer.load_index("grown").documents) == 3          # saved, not just in memory


def test_duplicate_and_limit_and_default(setup):
    a = _pdf(setup / "alpha.pdf", 1, "alpha")
    indexer.build_index([a], "case")
    with pytest.raises(indexer.CaseError, match="already in this case"):
        indexer.add_to_case("case", [a])
    many = [_pdf(setup / f"doc{i}.pdf", i, f"party{i}") for i in range(5)]
    with pytest.raises(indexer.CaseError, match="exceed the limit of 5"):
        indexer.add_to_case("case", many)
    with pytest.raises(indexer.CaseError, match="cannot be changed"):
        indexer.add_to_case("default", [a])


def test_bad_new_file_is_skipped_old_documents_kept(setup):
    a = _pdf(setup / "alpha.pdf", 1, "alpha")
    indexer.build_index([a], "case")
    fake = setup / "fake.pdf"
    fake.write_text("not a pdf")
    index, notes = indexer.add_to_case("case", [str(fake)])
    assert [d["file"] for d in index.documents] == ["alpha.pdf"]
    assert any("not a PDF file" in n for n in notes)
