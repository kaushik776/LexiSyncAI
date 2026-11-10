"""Build, save and load the search index for a case (a bundle of 2-5 PDFs).

    index/<case_id>/
        manifest.json     documents, detected structure, models used, build time
        chunks.json       all chunks with page / section / bounding boxes
        embeddings.npy    dense vectors of every chunk (normalised, float32)

    python -m src.indexer                 # index every PDF in data/ as case "default"
    python -m src.indexer my_case a.pdf https://eur-lex.europa.eu/.../file.pdf
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from src import config
from src.fetch import download_pdf, is_url
from src.chunker import Chunk, chunk_document, detect_mode, document_structure, load_chunks, save_chunks
from src.fetch import FetchError
from src.ingest import InvalidPDFError, ingest_pdf
from src.legal_check import check_document


def device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embedder():
    """Load the sentence-transformer once per process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device=device())


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = get_embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    vec = get_embedder().encode([config.EMBED_QUERY_PREFIX + text], normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
    return vec[0].astype(np.float32)


@dataclass
class CaseIndex:
    case_id: str
    chunks: list[Chunk]
    embeddings: np.ndarray
    manifest: dict = field(default_factory=dict)

    @property
    def documents(self) -> list[dict]:
        return self.manifest.get("documents", [])

    def doc_titles(self) -> list[str]:
        return [d["title"] for d in self.documents]


def build_index(sources: list[str | Path], case_id: str = "default",
                progress: Callable[[str, float], None] | None = None) -> CaseIndex:
    """Ingest, chunk and embed the PDFs, then save everything under index/<case_id>/.
    A source is a local PDF path or an http(s) URL; URLs are downloaded into
    index/<case_id>/sources/ first."""
    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    t0 = time.time()
    chunks: list[Chunk] = []
    documents = []
    n = len(sources)
    rejected: list[dict] = []      # files that could not be used, with a message asking for a re-upload
    for i, source in enumerate(sources):
        origin = str(source)
        try:
            if is_url(origin):
                report(f"Downloading {origin}", i / (n + 1))
                path = download_pdf(origin, case_dir(case_id) / "sources")
            else:
                path = Path(source)
            report(f"Reading {path.name}", i / (n + 1))
            t = time.time()
            doc = ingest_pdf(path)
        except (InvalidPDFError, FetchError) as e:
            # Skip this file but keep going, so one bad upload does not block the others.
            rejected.append({"file": Path(origin).name if not is_url(origin) else origin, "error": str(e)})
            report(f"ERROR: {e}", i / (n + 1))
            continue
        legal = check_document(doc)               # warning only: the document is indexed either way
        if legal.message:
            report(f"Warning for {path.name}: {legal.message}", i / (n + 1))
        doc_chunks = chunk_document(doc)
        chunks += doc_chunks
        documents.append({
            **doc.summary(),
            "file": path.name,
            "source": origin if is_url(origin) else "upload",
            "mode": detect_mode(doc),
            "legal_check": {"likely_legal": legal.likely_legal, "score": legal.score,
                            "signals": legal.signals, "warning": legal.message},
            "n_chunks": len(doc_chunks),
            "structure": document_structure(doc_chunks),
            "ingest_seconds": round(time.time() - t, 2),
        })

    if not documents:
        raise InvalidPDFError("None of the files could be used:\n" +
                              "\n".join(f"- {r['error']}" for r in rejected))

    report("Embedding chunks", n / (n + 1))
    t = time.time()
    embeddings = embed_passages([c.context_text for c in chunks]) if chunks else np.zeros((0, 384), np.float32)
    manifest = {
        "case_id": case_id,
        "documents": documents,
        "rejected": rejected,
        "n_chunks": len(chunks),
        "embed_model": config.EMBED_MODEL,
        "embed_seconds": round(time.time() - t, 2),
        "build_seconds": round(time.time() - t0, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    index = CaseIndex(case_id, chunks, embeddings, manifest)
    save_index(index)
    report("Done", 1.0)
    return index


def case_dir(case_id: str) -> Path:
    return config.INDEX_DIR / case_id


def save_index(index: CaseIndex) -> None:
    d = case_dir(index.case_id)
    d.mkdir(parents=True, exist_ok=True)
    save_chunks(index.chunks, d / "chunks.json")
    np.save(d / "embeddings.npy", index.embeddings)
    (d / "manifest.json").write_text(json.dumps(index.manifest, indent=1, ensure_ascii=False),
                                     encoding="utf-8")


def load_index(case_id: str = "default") -> CaseIndex:
    d = case_dir(case_id)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("embed_model") != config.EMBED_MODEL:
        raise ValueError(f"Index built with {manifest.get('embed_model')}, config uses "
                         f"{config.EMBED_MODEL}: rebuild the index.")
    return CaseIndex(case_id, load_chunks(d / "chunks.json"), np.load(d / "embeddings.npy"), manifest)


def list_cases() -> list[str]:
    if not config.INDEX_DIR.exists():
        return []
    return sorted(p.name for p in config.INDEX_DIR.iterdir() if (p / "manifest.json").exists())


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    case_id = argv[0] if argv else "default"
    paths = argv[1:] or [str(p) for p in sorted(config.DATA_DIR.glob("*.pdf"))]
    try:
        index = build_index(paths, case_id, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    except InvalidPDFError as e:
        sys.exit(f"\nIndexing stopped. {e}")
    m = index.manifest
    print(f"\nCase '{case_id}': {m['n_chunks']} chunks from {len(m['documents'])} documents "
          f"in {m['build_seconds']}s (embedding {m['embed_seconds']}s)")
    for d in m["documents"]:
        print(f"  {d['title']:<32} {d['pages']:>4} pages  {d['n_chunks']:>4} chunks  mode={d['mode']}")
        if d["legal_check"]["warning"]:
            print(f"    WARNING: {d['legal_check']['warning']}")
    for r in m["rejected"]:
        print(f"  SKIPPED {r['file']}: {r['error']}")


if __name__ == "__main__":
    main(sys.argv[1:])
