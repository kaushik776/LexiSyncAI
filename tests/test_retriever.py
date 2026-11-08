"""Tests for tokenisation, clause-reference parsing and rank fusion (no model downloads)."""
import numpy as np

from src.chunker import Chunk
from src.indexer import CaseIndex
from src.retriever import HybridRetriever, extract_refs, tokenize


def _chunk(i, section, text, paras=(), doc="DOC"):
    return Chunk(chunk_id=f"{doc}:{i}", doc_id=doc, doc_title=doc, kind="text", section_id=section,
                 section_title="", parents=[], ref=section, page_start=i + 1, page_end=i + 1,
                 spans=[], text=text, n_words=len(text.split()), paras=list(paras))


def _retriever(chunks):
    return HybridRetriever(CaseIndex("test", chunks, np.zeros((len(chunks), 4), np.float32)))


def test_tokenize_keeps_clause_numbers():
    toks = tokenize("Under Article 83(5) and clause 7.2, the fines apply.")
    assert "83(5)" in toks and "83" in toks and "7.2" in toks
    assert "fine" in toks and "the" not in toks


def test_extract_refs():
    assert extract_refs("What does Article 83(5) say?") == [("Article", "83", ["5"])]
    assert extract_refs("Compare clause 7.2 with Section 9") == [("Section", "7.2", []), ("Section", "9", [])]
    assert extract_refs("Under Schedule 2 para 11") == [("Schedule", "2", []), ("para", "11", [])]
    assert extract_refs("What is the maximum fine?") == []


def test_bm25_finds_exact_clause_number():
    chunks = [_chunk(0, "Section 7.1", "7.1 Payment. The buyer pays within 30 days."),
              _chunk(1, "Section 7.2", "7.2 Liability cap. Liability is capped at EUR 2.0M."),
              _chunk(2, "Section 9.1", "9.1 Indemnity. The seller indemnifies the buyer.")]
    r = _retriever(chunks)
    assert r.bm25_ranking("What does clause 7.2 say?", 3)[0] == 1


def test_reference_lookup_prefers_cited_paragraph():
    chunks = [_chunk(0, "Article 83", "fines tier one", paras=["4"]),
              _chunk(1, "Article 83", "fines tier two", paras=["5"]),
              _chunk(2, "Article 84", "penalties", paras=["1"])]
    r = _retriever(chunks)
    assert r.ref_ranking("What does Article 83(5) provide?", 3)[:2] == [1, 0]


def test_rrf_fuses_lists(monkeypatch):
    chunks = [_chunk(i, f"Section {i}", f"text {i}") for i in range(4)]
    r = _retriever(chunks)
    monkeypatch.setattr(r, "bm25_ranking", lambda q, n: [0, 1, 2])
    monkeypatch.setattr(r, "dense_ranking", lambda q, n: [2, 1, 3])
    hits = r.search("q", k=4, mode="hybrid")
    # chunk 1 (rank 2 + 2) and chunk 2 (rank 3 + 1) beat chunk 0 (rank 1 in one list only)
    assert [h.chunk.chunk_id for h in hits][:2] == ["DOC:2", "DOC:1"] or \
           [h.chunk.chunk_id for h in hits][:2] == ["DOC:1", "DOC:2"]
    assert hits[-1].chunk.chunk_id in ("DOC:0", "DOC:3")
    assert set(hits[0].sources) == {"bm25", "dense"}
