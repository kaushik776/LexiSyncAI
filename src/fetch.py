"""Download a public legal PDF from a URL so it can be indexed like an uploaded file.

Handles the common cases:
  * a direct PDF link             https://.../regulation.pdf
  * a link that serves a PDF      https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=...
  * an HTML page that links a PDF https://www.legislation.gov.uk/ukpga/2015/15/contents
                                  (the first PDF link on the page is followed)

Safety: only http(s); the file is streamed with a size cap; the content must start with
the PDF signature "%PDF" before it is saved; files are saved under the case's own folder.

    python -m src.fetch https://www.legislation.gov.uk/ukpga/2015/15/pdfs/ukpga_20150015_en.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

from src import config

USER_AGENT = "LexiSyncAI/0.1 (academic research prototype; document download)"
MAX_BYTES = 60 * 1024 * 1024        # 60 MB
TIMEOUT = 30                        # seconds per request
_PDF_LINK = re.compile(r"""href\s*=\s*["']([^"'#]+?(?:\.pdf|/pdf[s]?/[^"']*|[?&](?:format|type)=pdf[^"']*))["']""",
                       re.IGNORECASE)


class FetchError(RuntimeError):
    pass


def is_url(value: str) -> bool:
    return str(value).strip().lower().startswith(("http://", "https://"))


def _filename(url: str, response: requests.Response) -> str:
    """A safe file name from Content-Disposition, else from the URL path."""
    name = ""
    cd = response.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd, re.IGNORECASE)
    if m:
        name = unquote(m.group(1))
    if not name:
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        if not name or name.lower() in ("pdf", "txt", "contents"):
            name = (parsed.netloc + "_" + (parsed.query or parsed.path)).strip("_/")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:80] or "document"
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def _get(url: str) -> requests.Response:
    if not is_url(url):
        raise FetchError(f"Not an http(s) URL: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.5"},
                         timeout=TIMEOUT, stream=True, allow_redirects=True)
    except requests.RequestException as e:
        raise FetchError(f"Could not download {url}: {e}") from e
    if r.status_code in (202, 401, 403, 429, 503):
        # 202/403/429 are what bot-protection pages return (e.g. EUR-Lex). We do not try to get around them.
        raise FetchError(f"{urlparse(url).netloc} does not allow automated downloads (HTTP {r.status_code}). "
                         "Open the link in your browser, save the PDF and upload the file instead.")
    if r.status_code != 200:
        raise FetchError(f"Download failed for {url}: HTTP {r.status_code}")
    return r


def _read_capped(r: requests.Response) -> bytes:
    size = int(r.headers.get("Content-Length") or 0)
    if size > MAX_BYTES:
        raise FetchError(f"File is too large ({size / 1e6:.0f} MB > {MAX_BYTES / 1e6:.0f} MB)")
    data = bytearray()
    for block in r.iter_content(64 * 1024):
        data += block
        if len(data) > MAX_BYTES:
            raise FetchError(f"File is too large (> {MAX_BYTES / 1e6:.0f} MB)")
    return bytes(data)


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for k in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FetchError("Too many files with the same name")


def download_pdf(url: str, dest_dir: str | Path, follow_html: bool = True) -> Path:
    """Download the PDF at `url` (or the first PDF linked from an HTML page) into dest_dir."""
    url = url.strip()
    r = _get(url)
    data = _read_capped(r)
    if data[:1024].lstrip().startswith(b"%PDF"):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        path = _unique(dest / _filename(r.url or url, r))
        path.write_bytes(data)
        return path

    content_type = r.headers.get("Content-Type", "")
    if follow_html and ("html" in content_type or data[:200].lstrip().lower().startswith((b"<!doctype", b"<html"))):
        html = data.decode(r.encoding or "utf-8", errors="replace")
        for link in _PDF_LINK.findall(html):
            target = urljoin(r.url or url, link.replace("&amp;", "&"))
            try:
                return download_pdf(target, dest_dir, follow_html=False)
            except FetchError:
                continue
        raise FetchError(f"{url} is a web page without a downloadable PDF link. Paste the PDF link instead.")
    raise FetchError(f"{url} did not return a PDF (Content-Type: {content_type or 'unknown'}).")


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: python -m src.fetch URL [URL ...]")
    for url in argv:
        path = download_pdf(url, config.DATA_DIR / "downloads")
        print(f"{url}\n  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
