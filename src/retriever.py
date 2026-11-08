"""Step 3: hybrid retrieval.

Three ranked lists are merged with Reciprocal Rank Fusion (RRF):
  * BM25 (keyword)   - exact terms and clause numbers such as "7.2" or "83(5)",
  * dense embeddings - meaning, e.g. "maximum fine" finds "administrative fines up to",
  * reference lookup - if the question names a clause ("Article 83(5)", "Section 22"),
                       the chunks of that clause are added directly.
RRF score of a chunk = sum over lists of 1 / (RRF_K + rank). It needs no score
calibration between BM25 and cosine similarity, which live on different scales.

The fused candidates are then rescored by a cross-encoder (src/reranker.py).

    python -m src.retriever "What is the maximum GDPR fine?"
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.chunker import Chunk
from src.indexer import CaseIndex, embed_query, load_index

MODES = ("bm25", "dense", "hybrid", "hybrid+refs")

_STOP = set("""a an the of to in on for and or by with from as at is are was were be been being this that
these those it its which who whom what when where how under any all such shall may must not no do does did
can could would should will than then there their they them he she his her we our you your i me my
""".split())

# Clause numbers stay whole ("83(5)", "2.1(a)", "7.2") and are also indexed by their base number.
_TOKEN = re.compile(r"\d+(?:\.\d+)*[a-z]?(?:\([0-9a-z]+\))*|[a-z]+")


def _stem(t: str) -> str:
    if not t.isalpha() or len(t) <= 4:
        return t
    for suffix, repl in (("ies", "y"), ("ings", ""), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[: -len(suffix)] + repl
    return t


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower().replace("’", "'")):
        if t in _STOP:
            continue
        out.append(_stem(t))
        if "(" in t:
            out.append(t.split("(")[0])          # "83(5)" also matches "83"
    return out


# --- Reference lookup ----------------------------------------------------------
_REF_WORDS = {"article": "Article", "art": "Article", "section": "Section", "s": "Section",
              "clause": "Section", "regulation": "Regulation", "reg": "Regulation", "rule": "Rule",
              "schedule": "Schedule", "annex": "Annex", "exhibit": "Exhibit", "appendix": "Appendix",
              "recital": "Recital", "part": "Part", "chapter": "Chapter",
              "paragraph": "para", "para": "para"}
_REF = re.compile(r"\b(article|art|section|s|clause|regulation|reg|rule|schedule|annex|exhibit|appendix|"
                  r"recital|part|chapter|paragraph|para)\.?\s+(\d+(?:\.\d+)*[A-Za-z]?|[IVXLC]+)\b((?:\(\w+\))*)",
                  re.IGNORECASE)


def extract_refs(query: str) -> list[tuple[str, str, list[str]]]:
    """'Article 83(4) and Section 7.2' -> [("Article", "83", ["4"]), ("Section", "7.2", [])]"""
    refs = []
    for word, num, subs in _REF.findall(query):
        kind = _REF_WORDS[word.lower()]
        if kind == "Section" and word.lower() == "s" and not re.match(r"\d", num):
            continue
        refs.append((kind, num.upper() if re.fullmatch(r"[ivxlc]+", num, re.I) else num,
                     re.findall(r"\((\w+)\)", subs)))
    return refs


def _ref_score(chunk: Chunk, kind: str, num: str, subs: list[str]) -> float:
    sid = chunk.section_id
    if kind == "Recital":
        hit = sid == "Preamble" and num in chunk.paras
    elif kind == "para":
        hit = sid.endswith(f"para {num}") or num in chunk.paras
    else:
        hit = (sid == f"{kind} {num}" or sid.startswith(f"{kind} {num},")
               or sid.endswith(f"{kind} {num}") or f"{kind} {num} " in " ".join(chunk.parents) + " ")
    if not hit:
        return 0.0
    score = 1.0
    if subs and subs[0] in chunk.paras:
        score += 1.0                              # "Article 83(5)" -> the chunk holding paragraph 5
    if chunk.kind == "text":
        score += 0.1
    return score


@dataclass
class Hit:
    chunk: Chunk
    score: float                               # fused (RRF) score, or reranker probability
    rank: int = 0
    sources: dict = field(default_factory=dict)   # rank of this chunk in each list


class HybridRetriever:
    def __init__(self, index: CaseIndex):
        self.index = index
        self.chunks = index.chunks
        self.bm25 = BM25Okapi([tokenize(c.context_text) for c in self.chunks]) if self.chunks else None
        self.embeddings = index.embeddings

    # --- single retrievers: each returns chunk indices, best first ---
    def bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def dense_ranking(self, query: str, n: int) -> list[int]:
        sims = self.embeddings @ embed_query(query)
        return [int(i) for i in np.argsort(-sims)[:n]]

    def ref_ranking(self, query: str, n: int) -> list[int]:
        refs = extract_refs(query)
        if not refs:
            return []
        scored = []
        for i, c in enumerate(self.chunks):
            s = max(_ref_score(c, *r) for r in refs)
            if s > 0:
                scored.append((s, -i, i))
        scored.sort(reverse=True)
        return [i for _, _, i in scored[:n]]

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid+refs",
               n_candidates: int = config.CANDIDATES, doc_ids: list[str] | None = None) -> list[Hit]:
        """Return the top-k chunks for the query (before reranking)."""
        if not self.chunks:
            return []
        lists: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid", "hybrid+refs"):
            lists["bm25"] = self.bm25_ranking(query, n_candidates)
        if mode in ("dense", "hybrid", "hybrid+refs"):
            lists["dense"] = self.dense_ranking(query, n_candidates)
        if mode == "hybrid+refs":
            lists["refs"] = self.ref_ranking(query, n_candidates)

        fused: dict[int, float] = {}
        sources: dict[int, dict] = {}
        for name, ranking in lists.items():
            if name == "refs":
                continue
            for r, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + r)
                sources.setdefault(i, {})[name] = r
        # Clause references only boost chunks that BM25 or the embeddings already found:
        # a question may mention "Article 2" as context while asking about something else,
        # and the same clause number exists in other documents.
        for r, i in enumerate(lists.get("refs", []), start=1):
            if i in fused:
                fused[i] += config.REF_WEIGHT / (config.RRF_K + r)
                sources[i]["refs"] = r
        order = sorted(fused, key=lambda i: -fused[i])
        if doc_ids:
            order = [i for i in order if self.chunks[i].doc_id in doc_ids]
        return [Hit(self.chunks[i], fused[i], r, sources[i]) for r, i in enumerate(order[:k], start=1)]


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.reranker import Reranker
    query = " ".join(argv) or "What is the maximum fine under the GDPR?"
    retriever = HybridRetriever(load_index("default"))
    t = time.time()
    candidates = retriever.search(query, k=config.CANDIDATES)
    hits = Reranker().rerank(query, candidates, k=config.TOP_K)
    print(f"Q: {query}   ({time.time() - t:.2f}s)\n")
    for h in hits:
        print(f"{h.rank}. [{h.score:.3f}] {h.chunk.citation}   lists={h.sources}")
        print("   ", h.chunk.text[:200].replace("\n", " "), "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
