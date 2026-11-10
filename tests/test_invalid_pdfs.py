"""Empty, broken, locked or text-less PDFs give a clear error asking for a re-upload,
and one bad file does not stop the other files of a case from being indexed."""
import numpy as np
import pymupdf
import pytest

from src import config, indexer
from src.ingest import InvalidPDFError, ingest_pdf

TEXT = "Article 1 Scope. The Supplier shall deliver the goods in accordance with the agreement. " * 5


def _pdf(path, pages_text):
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(60, 60, 540, 800), text, fontsize=10)
    doc.save(path)
    return path


@pytest.fixture
def bad_files(tmp_path):
    zero = tmp_path / "zero.pdf"
    zero.write_bytes(b"")
    fake = tmp_path / "fake.pdf"
    fake.write_text("hello, this is only text")
    blank = _pdf(tmp_path / "blank.pdf", ["", ""])
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), TEXT[:80])
    doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    return {"zero": zero, "fake": fake, "blank": blank, "locked": locked}


@pytest.mark.parametrize("name, message", [
    ("zero", r"is empty \(0 bytes\)"),
    ("fake", "is not a PDF file"),
    ("blank", "contains no readable text"),
    ("locked", "is password-protected"),
])
def test_bad_files_give_reupload_errors(bad_files, name, message):
    with pytest.raises(InvalidPDFError, match=message) as err:
        ingest_pdf(bad_files[name], detect_tables=False, use_ocr=False)
    assert "upload" in str(err.value).lower()


def _fake_embed(texts, batch_size=32):
    return np.zeros((len(texts), 384), np.float32)


def test_bad_file_is_skipped_others_indexed(bad_files, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(indexer, "embed_passages", _fake_embed)
    good = _pdf(tmp_path / "good.pdf", [TEXT])
    index = indexer.build_index([bad_files["fake"], good, bad_files["zero"]], "case")
    assert [d["file"] for d in index.documents] == ["good.pdf"]
    assert [r["file"] for r in index.manifest["rejected"]] == ["fake.pdf", "zero.pdf"]
    assert all("upload" in r["error"].lower() for r in index.manifest["rejected"])


def test_all_files_bad_stops_with_clear_message(bad_files, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(indexer, "embed_passages", _fake_embed)
    with pytest.raises(InvalidPDFError, match="None of the files could be used") as err:
        indexer.build_index([bad_files["blank"], bad_files["locked"]], "case")
    assert "blank.pdf contains no readable text" in str(err.value)
    assert "locked.pdf is password-protected" in str(err.value)
