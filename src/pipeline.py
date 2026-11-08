"""The full question-answering pipeline (one call = one answer):

  question
    -> hybrid retrieval (BM25 + dense + clause refs, RRF) -> cross-encoder rerank
    -> early abstention if nothing is relevant
    -> planner: split multi-part questions, retrieve evidence for each part
    -> LLM draft with [S#] citations
    -> verifier: numbers + support check per claim
         -> on failure: feedback / extra retrieval -> redraft   (at most MAX_RETRIES times)
    -> confidence badge, or abstention listing the documents searched

    python -m src.pipeline "Does the GDPR set a 72-hour deadline for breach notification?"
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict, dataclass, field

from src import config
from src.abstain import ABSTAIN_MESSAGE, INSUFFICIENT, badge, should_abstain_early
from src.chunker import Chunk
from src.generator import Draft, generate, generate_closed_book
from src.indexer import CaseIndex, load_index
from src.llm import LLM, Usage
from src.planner import plan
from src.reranker import Reranker
from src.retriever import Hit, HybridRetriever
from src.verifier import verify


@dataclass
class Source:
    marker: str                    # "S1"
    chunk: Chunk
    score: float                   # reranker probability
    cited: bool = False


@dataclass
class Answer:
    question: str
    answer: str
    badge: str
    abstained: bool
    explanation: str = ""
    conflict: str = ""
    missing: str = ""
    sources: list[Source] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)       # documents searched (for abstention)
    retry_log: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    evidence_strength: float = 0.0
    mode: str = "rag"

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.cited]

    def to_dict(self) -> dict:
        d = asdict(self)
        for s, src in zip(d["sources"], self.sources):
            s["chunk"] = {"chunk_id": src.chunk.chunk_id, "citation": src.chunk.citation,
                          "doc_id": src.chunk.doc_id, "page_start": src.chunk.page_start,
                          "page_end": src.chunk.page_end, "ref": src.chunk.ref}
        return d


def _insert_marker(answer: str, claim_text: str, n: int) -> str:
    """Put "[S<n>]" at the end of the sentence of the answer that holds the claim."""
    head = claim_text.rstrip(".!?")[:60]
    pos = answer.find(head)
    if pos < 0:
        return answer + f" [S{n}]"
    end = re.search(r"[.!?](\s|$)", answer[pos + len(head):])
    cut = pos + len(head) + (end.start() if end else len(answer) - pos - len(head))
    return answer[:cut] + f" [S{n}]" + answer[cut:]


def _replace_markers(answer: str, claim_text: str, ids: list[int]) -> str:
    """Replace the [S#] markers that follow a claim's sentence with the corrected ones."""
    pos = answer.find(claim_text)
    if pos < 0:
        return answer
    end = pos + len(claim_text)
    m = re.match(r"(\s*\[S\d+\])+", answer[end:])
    tail = answer[end + (m.end() if m else 0):]
    return answer[:end] + "".join(f" [S{i + 1}]" for i in ids) + tail


class Pipeline:
    def __init__(self, index: CaseIndex, llm: LLM | None = None, exclude_kinds: tuple[str, ...] = (),
                 retrieval_mode: str = "hybrid+refs"):
        self.index = index
        self.exclude_kinds = exclude_kinds          # ablation: ("table",) = text-only
        self.retrieval_mode = retrieval_mode        # ablation: "dense" = dense-only
        self.retriever = HybridRetriever(index)
        self.reranker = Reranker()
        self.llm = llm or LLM()

    def searched_documents(self) -> list[str]:
        return [f"{d['title']} ({d['pages']} pages)" for d in self.index.documents]

    def _retrieve(self, query: str, k: int, timings: dict) -> list[Hit]:
        t = time.time()
        candidates = self.retriever.search(query, k=config.CANDIDATES, mode=self.retrieval_mode,
                                           exclude_kinds=self.exclude_kinds)
        timings["retrieval_s"] = timings.get("retrieval_s", 0) + time.time() - t
        t = time.time()
        hits = self.reranker.rerank(query, candidates, k=k)
        timings["rerank_s"] = timings.get("rerank_s", 0) + time.time() - t
        return hits

    @staticmethod
    def _merge(pools: list[list[Hit]], limit: int) -> list[Hit]:
        """Union of several hit lists, best score per chunk, taking the best of each list first
        so every sub-question keeps its evidence."""
        best: dict[str, Hit] = {}
        order: list[str] = []
        for depth in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if depth < len(pool):
                    h = pool[depth]
                    cid = h.chunk.chunk_id
                    if cid not in best:
                        order.append(cid)
                    if cid not in best or h.score > best[cid].score:
                        best[cid] = h
        return [best[c] for c in order][:limit]

    def ask(self, question: str, use_retrieval: bool = True, use_planner: bool = config.USE_PLANNER,
            use_verifier: bool = config.USE_VERIFIER, max_retries: int = config.MAX_RETRIES) -> Answer:
        t0 = time.time()
        usage, timings, log = Usage(), {}, []
        searched = self.searched_documents()

        if not use_retrieval:                                  # closed-book baseline
            draft = generate_closed_book(self.llm, question, usage)
            return self._finish(question, draft, [], usage, timings, t0, log, searched, [],
                                verified_ratio=None, mode="no_retrieval")

        hits = self._retrieve(question, config.TOP_K, timings)
        top = hits[0].score if hits else 0.0
        if should_abstain_early(top):
            log.append({"step": 0, "action": "abstain_early",
                        "detail": f"best passage score {top:.3f} < {config.ABSTAIN_SCORE}"})
            return self._abstain(question, hits, usage, timings, t0, log, searched,
                                 "No passage in the documents is relevant to the question.")

        subs: list[str] = []
        pools = [hits]
        if use_planner:
            subs = plan(self.llm, question, usage)
            if subs:
                log.append({"step": 0, "action": "plan", "sub_questions": subs})
                pools += [self._retrieve(s, 3, timings) for s in subs]
        limit = config.MAX_SOURCES if subs else config.TOP_K
        pool = [h for h in self._merge(pools, limit) if h.score >= config.ABSTAIN_SCORE / 5] or hits[:1]
        sources = [h.chunk for h in pool]
        scores = [h.score for h in pool]

        draft = self._attribute(generate(self.llm, question, sources, usage, parts=subs), sources, log, 1)
        log.append({"step": 1, "action": "draft", "answerable": draft.answerable,
                    "claims": len(draft.claims), "missing": draft.missing})
        verified_ratio = None
        best: tuple | None = None          # (draft, ratio): the verifier may reject, never make worse
        for attempt in range(1, max_retries + 2):
            if not draft.answerable or not use_verifier:
                break
            check = verify(self.llm, draft.claims, sources, usage)
            log.append({"step": attempt, "action": "verify", "supported": check.n_supported,
                        "claims": len(check.claims), "checks": check.checks})
            rank = (check.n_supported, check.ratio)
            if best is None or rank > (sum(bool(c.supported) for c in best[0].claims), best[1]):
                best = (draft, check.ratio)
            if check.all_supported or attempt > max_retries:
                break
            # Reformulate: search again for what is missing, then redraft with the feedback.
            if draft.missing and len(sources) < config.MAX_SOURCES + 2:
                extra = [h for h in self._retrieve(draft.missing, 3, timings)
                         if h.chunk.chunk_id not in {c.chunk_id for c in sources} and h.score >= config.LOW_SCORE]
                if extra:
                    sources += [h.chunk for h in extra]
                    scores += [h.score for h in extra]
                    log.append({"step": attempt, "action": "re_retrieve", "query": draft.missing,
                                "added": [h.chunk.citation for h in extra]})
            draft = self._attribute(generate(self.llm, question, sources, usage, feedback=check.feedback,
                                             parts=subs), sources, log, attempt + 1)
            log.append({"step": attempt + 1, "action": "redraft", "answerable": draft.answerable,
                        "claims": len(draft.claims)})
        if best is not None:
            if best[0] is not draft:
                log.append({"step": len(log), "action": "keep_best_draft",
                            "detail": f"kept the draft with {best[1]:.0%} of claims verified"})
            draft, verified_ratio = best

        if draft.answerable and use_verifier and verified_ratio == 0:
            return self._abstain(question, pool, usage, timings, t0, log, searched,
                                 "No claim in the draft answer could be verified against the documents.")
        if not draft.answerable:
            return self._abstain(question, pool, usage, timings, t0, log, searched,
                                 draft.missing or "The retrieved passages do not answer the question.",
                                 explanation=draft.explanation)
        return self._finish(question, draft, sources, usage, timings, t0, log, searched, subs,
                            verified_ratio, scores=scores)

    def _attribute(self, draft: Draft, sources: list[Chunk], log: list, step: int) -> Draft:
        """Citation fallback: a claim the LLM left without a [S#] marker is attributed to the
        source the cross-encoder scores highest for it (if the match is strong enough).
        The verifier then checks that source like any other citation."""
        added = []
        for claim in draft.claims:
            if claim.source_ids or not sources:
                continue
            probs = self.reranker.scores(claim.text, [c.text for c in sources])
            order = [int(i) for i in probs.argsort()[::-1]]
            # The best source, plus a second one if it is also strong (claims that combine two documents).
            chosen = [i for k, i in enumerate(order[:2]) if probs[i] >= (config.LOW_SCORE if k == 0 else 0.5)]
            if chosen:
                claim.source_ids = chosen
                added.append(", ".join(f"S{i + 1} ({probs[i]:.2f})" for i in chosen))
                for i in chosen:
                    draft.answer = _insert_marker(draft.answer, claim.text, i + 1)
        if added:
            log.append({"step": step, "action": "auto_cite", "detail": f"attributed {len(added)} uncited "
                        f"claim(s) by cross-encoder: {', '.join(added)}"})
        return draft

    def _finish(self, question, draft: Draft, sources, usage, timings, t0, log, searched, subs,
                verified_ratio, scores=None, mode="rag") -> Answer:
        for c in draft.claims:                       # citations corrected by the verifier
            if c.reason.startswith("citation corrected"):
                draft.answer = _replace_markers(draft.answer, c.text, c.source_ids)
                log.append({"step": len(log), "action": "recite", "detail": f"{c.reason} for: {c.text[:80]}"})
        cited = set(draft.cited_ids())
        srcs = [Source(f"S{i + 1}", c, (scores or [0] * len(sources))[i], i in cited)
                for i, c in enumerate(sources)]
        strength = max((s.score for s in srcs if s.cited), default=0.0)
        b = badge(strength, verified_ratio, draft.answerable, len(cited), draft.missing) if mode == "rag" else "N/A"
        if b == INSUFFICIENT and mode == "rag":
            return self._abstain(question, [], usage, timings, t0, log, searched,
                                 "The cited evidence is too weak to support an answer.")
        claims = [{"claim": c.text, "sources": [f"S{i + 1}" for i in c.source_ids],
                   "supported": c.supported, "reason": c.reason} for c in draft.claims]
        return Answer(question, draft.answer, b, False, draft.explanation, draft.conflict, draft.missing,
                      srcs, claims, subs, searched, log, self._timings(timings, usage, t0),
                      self._usage(usage), strength, mode)

    def _abstain(self, question, hits, usage, timings, t0, log, searched, reason, explanation="") -> Answer:
        srcs = [Source(f"S{i + 1}", h.chunk, h.score) for i, h in enumerate(hits[:3])]
        text = f"{ABSTAIN_MESSAGE} Searched: {', '.join(searched)}."
        return Answer(question, text, INSUFFICIENT, True, explanation or reason, missing=reason,
                      sources=srcs, searched=searched, retry_log=log,
                      timings=self._timings(timings, usage, t0), usage=self._usage(usage),
                      evidence_strength=hits[0].score if hits else 0.0)

    @staticmethod
    def _timings(timings, usage: Usage, t0) -> dict:
        out = {k: round(v, 3) for k, v in timings.items()}
        out["llm_s"] = round(usage.seconds, 3)
        out["total_s"] = round(time.time() - t0, 3)
        return out

    @staticmethod
    def _usage(usage: Usage) -> dict:
        return {"llm_calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens, "calls": usage.log}


def main(argv: list[str]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(argv) or "Under the GDPR, what is the maximum fine a company can receive, " \
                                 "and which kinds of violations trigger the higher fine?"
    pipe = Pipeline(load_index("default"))
    a = pipe.ask(question)
    print(f"Q: {a.question}\n\n[{a.badge}]  {a.answer}\n")
    if a.explanation:
        print("Why:", a.explanation)
    if a.conflict:
        print("Conflict:", a.conflict)
    print("\nSources:")
    for s in a.sources:
        print(f"  {'*' if s.cited else ' '} [{s.marker}] {s.chunk.citation}  (score {s.score:.3f})")
    print("\nClaims:")
    for c in a.claims:
        print(f"  {'OK ' if c['supported'] else 'BAD' if c['supported'] is False else '?  '} {c['claim']}  {c['sources']}")
    print("\nAgent log:")
    for step in a.retry_log:
        print("  ", {k: v for k, v in step.items() if k != "checks"})
    print("\nTimings:", a.timings, "| tokens in/out:", a.usage["prompt_tokens"], a.usage["completion_tokens"])


if __name__ == "__main__":
    main(sys.argv[1:])
