"""Tests for downloading PDFs from URLs (network calls are replaced by fakes)."""
import pytest

import src.fetch as fetch

PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF"


class FakeResponse:
    def __init__(self, url, body, content_type="application/pdf", status=200, headers=None):
        self.url, self._body, self.status_code = url, body, status
        self.headers = {"Content-Type": content_type, **(headers or {})}
        self.encoding = "utf-8"

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]


def _serve(monkeypatch, pages: dict):
    def fake_get(url, **kwargs):
        if url not in pages:
            return FakeResponse(url, b"not found", "text/plain", status=404)
        return pages[url](url) if callable(pages[url]) else pages[url]
    monkeypatch.setattr(fetch.requests, "get", fake_get)


def test_is_url():
    assert fetch.is_url("https://eur-lex.europa.eu/x.pdf")
    assert not fetch.is_url("data/GDPR.pdf")
    assert not fetch.is_url("ftp://example.org/a.pdf")


def test_direct_pdf_download(tmp_path, monkeypatch):
    url = "https://example.org/acts/consumer_act.pdf"
    _serve(monkeypatch, {url: FakeResponse(url, PDF)})
    path = fetch.download_pdf(url, tmp_path)
    assert path.name == "consumer_act.pdf" and path.read_bytes() == PDF


def test_pdf_without_pdf_in_url_uses_content_disposition(tmp_path, monkeypatch):
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
    _serve(monkeypatch, {url: FakeResponse(url, PDF, headers={
        "Content-Disposition": 'attachment; filename="CELEX_32016R0679_EN_TXT.pdf"'})})
    assert fetch.download_pdf(url, tmp_path).name == "CELEX_32016R0679_EN_TXT.pdf"


def test_html_page_follows_pdf_link(tmp_path, monkeypatch):
    page = "https://www.legislation.gov.uk/ukpga/2015/15/contents"
    pdf_url = "https://www.legislation.gov.uk/ukpga/2015/15/pdfs/ukpga_20150015_en.pdf"
    html = b'<!DOCTYPE html><html><a href="/ukpga/2015/15/pdfs/ukpga_20150015_en.pdf">PDF</a></html>'
    _serve(monkeypatch, {page: FakeResponse(page, html, "text/html"), pdf_url: FakeResponse(pdf_url, PDF)})
    assert fetch.download_pdf(page, tmp_path).name == "ukpga_20150015_en.pdf"


def test_same_name_is_not_overwritten(tmp_path, monkeypatch):
    url = "https://example.org/contract.pdf"
    _serve(monkeypatch, {url: FakeResponse(url, PDF)})
    first, second = fetch.download_pdf(url, tmp_path), fetch.download_pdf(url, tmp_path)
    assert first.name == "contract.pdf" and second.name == "contract_2.pdf"


@pytest.mark.parametrize("response, message", [
    (FakeResponse("https://x.org/a.pdf", b"", status=404), "HTTP 404"),
    (FakeResponse("https://x.org/a.pdf", b"PK\x03\x04 zip data", "application/zip"), "did not return a PDF"),
    (FakeResponse("https://x.org/a.pdf", b"<html>no links</html>", "text/html"), "without a downloadable PDF"),
])
def test_bad_downloads_raise_clear_errors(tmp_path, monkeypatch, response, message):
    _serve(monkeypatch, {"https://x.org/a.pdf": response})
    with pytest.raises(fetch.FetchError, match=message):
        fetch.download_pdf("https://x.org/a.pdf", tmp_path)
    assert not list(tmp_path.iterdir())                 # nothing saved


def test_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_BYTES", 10)
    url = "https://x.org/big.pdf"
    _serve(monkeypatch, {url: FakeResponse(url, PDF)})
    with pytest.raises(fetch.FetchError, match="too large"):
        fetch.download_pdf(url, tmp_path)


def test_bot_protection_gives_upload_advice(tmp_path, monkeypatch):
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679"
    _serve(monkeypatch, {url: FakeResponse(url, b"<html>challenge</html>", "text/html", status=202)})
    with pytest.raises(fetch.FetchError, match="does not allow automated downloads.*upload the file"):
        fetch.download_pdf(url, tmp_path)
