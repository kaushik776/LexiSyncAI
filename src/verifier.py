"""Verifier (bonus feature 5, part 2): check each claim against the sources it cites.

Two checks:
  1. Numbers (deterministic): every figure in a claim (amounts, percentages, days, ages)
     must appear in the cited source text. Wrong numbers are the most harmful legal
     hallucination, and this check cannot be talked round.
  2. Support: does the cited text entail the claim?
     - "nli" (default): a natural-language-inference cross-encoder (DeBERTa-v3) scores
       P(entailment | premise = source window, hypothesis = claim). The source is cut into
       overlapping ~90-word windows (NLI models are trained on short premises) and the best
       window counts. Deterministic, ~0.2 s per claim, and it catches changed numbers,
       parties and conditions ("50M" instead of "20M", "New York" instead of "Delaware").
     - "llm": the LLM judges support (slower and, with a 7B model, less reliable).

If a claim fails, the pipeline sends the draft back with this feedback (see pipeline.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src import config
from src.calculator import check_comparison
from src.chunker import Chunk
from src.generator import Claim
from src.llm import LLM, Usage


@lru_cache(maxsize=1)
def _nli_model():
    import torch
    from sentence_transformers import CrossEncoder
    from src.indexer import device
    # bfloat16 on GPUs that support it: same accuracy as fp32 on our checks, 4x faster, half the memory
    # (leaves room for the LLM on an 8 GB card). Not fp16: DeBERTa-v3 overflows in fp16 and returns ~0.
    dev = device()
    kwargs = {"torch_dtype": torch.bfloat16} if dev == "cuda" and torch.cuda.is_bf16_supported() else {}
    model = CrossEncoder(config.NLI_MODEL, device=dev, model_kwargs=kwargs)
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return model, labels["entailment"]


_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d“])|\n")


def _windows(text: str, max_words: int = 160) -> list[str]:
    """Premises of 1-3 consecutive sentences. NLI models are trained on short premises: a claim
    copied word for word from a 90-word window scored 0.09, the same sentence alone 0.99."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    out = []
    for n in (1, 2, 3):
        for i in range(len(sents) - n + 1):
            w = " ".join(sents[i:i + n])
            if n == 1 or len(w.split()) <= max_words:
                out.append(w)
    return out or [text]


def contradiction(a: str, b: str) -> float:
    """Probability that statement b contradicts statement a (and the reverse; the higher is
    returned). Used to detect conflicts between answers from two documents."""
    model, _ = _nli_model()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    probs = model.predict([(a, b), (b, a)], apply_softmax=True, show_progress_bar=False)
    return float(probs[:, labels["contradiction"]].max())


def entailment(claim: str, sources: list[Chunk], top_windows: int = 12) -> float:
    """Best entailment probability of the claim over the sentence windows of the sources.
    Only the windows sharing the most words with the claim are scored (speed)."""
    if not sources:
        return 0.0
    from src.retriever import tokenize
    model, ent = _nli_model()
    claim_tokens = set(tokenize(claim))
    candidates = []
    for c in sources:
        # The document name and section heading are put before each window, so a claim that names
        # "the Tax Protection Agreement" or "letting agents" matches a window saying "this Agreement".
        prefix = f"{' '.join(c.doc_name.split()[:15])}. {c.heading_path}."
        for w in _windows(c.text):
            overlap = len(claim_tokens & set(tokenize(w)))
            candidates.append((overlap, f"{prefix} {w}"))
    candidates.sort(key=lambda x: -x[0])
    pairs = [(premise, claim) for _, premise in candidates[:top_windows]]
    probs = model.predict(pairs, apply_softmax=True, batch_size=32, show_progress_bar=False)
    return float(probs[:, ent].max())


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
          "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
          "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
          "eighth": 8, "ninth": 9, "tenth": 10, "twelfth": 12, "half": 50}
_SCALE = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000, "m": 1_000_000, "bn": 1_000_000_000}
# Clause references like "Article 83(5)" or "section 22(3)" are not facts to verify.
_REF = re.compile(r"\b(articles?|sections?|s\.|regulations?|reg\.|schedules?|recitals?|paras?|paragraphs?|"
                  r"annex|part|chapter|clauses?|rule|directive|regulation \(eu\))\s*[\d.]+[a-z]?(\([\w]+\))*",
                  re.IGNORECASE)
_NUM = re.compile(r"\d+(?:[ ,.]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def numbers_in(text: str, keep_unscaled: bool = True) -> set[float]:
    """All numbers in a text, normalised: '20 000 000' -> 20000000, 'EUR 20 million' -> 20000000,
    'thirty' -> 30, '4 %' -> 4."""
    t = _REF.sub(" ", text.lower().replace(" ", " "))
    t = re.sub(r"\[s\d+\]", " ", t)
    out = set()
    for m in _NUM.finditer(t):
        raw = m.group(0)
        value = float(re.sub(r"[ ,]", "", raw) if re.search(r"[ ,]\d{3}", raw) else raw.replace(",", ""))
        after = t[m.end():m.end() + 10].strip().split(" ")[0].strip(".,;:)") if m.end() < len(t) else ""
        if after in _SCALE:
            out.add(value * _SCALE[after])
            if not keep_unscaled:
                continue
        out.add(value)
    for w in re.findall(r"[a-z]+", t):
        if w in _WORDS:
            out.add(float(_WORDS[w]))
    return out


def check_numbers(claim: Claim, sources: list[Chunk]) -> tuple[bool, list[str]]:
    # Numbers in the document's name or clause label ("Consumer Rights Act 2015") count as present.
    cited = " ".join(f"{sources[i].text} {sources[i].ref} {sources[i].doc_title}"
                     for i in claim.source_ids if i < len(sources))
    available = numbers_in(cited)
    missing = [n for n in numbers_in(claim.text, keep_unscaled=False) if n not in available]
    # Years and small list numbers inside a clause name are allowed only if present; report the rest.
    return not missing, [f"{n:,.0f}" if n == int(n) else f"{n:g}" for n in missing]


VERIFY_SYSTEM = """You check whether legal claims are supported by the source text cited for them.
Mark a claim SUPPORTED if its facts, figures and conditions appear in the cited text, directly or as a
fair paraphrase or summary. Wording, framing and level of detail may differ.
Mark it NOT supported only if it states a fact, number or condition that the cited text does not contain,
or contradicts the text. Judge each claim only against its own cited text.
Keep each reason under 12 words."""

VERIFY_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"claim_id": {"type": "integer"}, "supported": {"type": "boolean"},
                                     "reason": {"type": "string"}},
    "required": ["claim_id", "supported", "reason"]}}}, "required": ["verdicts"]}


@dataclass
class Verification:
    claims: list[Claim]
    feedback: str = ""
    checks: list[dict] = field(default_factory=list)

    @property
    def n_supported(self) -> int:
        return sum(bool(c.supported) for c in self.claims)

    @property
    def ratio(self) -> float:
        return self.n_supported / len(self.claims) if self.claims else 0.0

    @property
    def all_supported(self) -> bool:
        return bool(self.claims) and self.n_supported == len(self.claims)


def verify(llm: LLM | None, claims: list[Claim], sources: list[Chunk], usage: Usage) -> Verification:
    checks = []
    for c in claims:
        if not c.source_ids:
            c.supported, c.reason = False, "no citation"
            continue
        ok, missing = check_numbers(c, sources)
        c.supported, c.reason = ok, "" if ok else f"number(s) {', '.join(missing)} not in the cited source"
        if ok:
            # A stated comparison must agree with the computed values ("72 hours is longer than 30 days" fails).
            ok, why = check_comparison(c.text)
            if not ok:
                c.supported, c.reason = False, f"wrong comparison: {why}"

    for c in claims:
        # A conclusion ("30 days is longer than 72 hours") combines facts from several sources, so no
        # single passage entails it. Its facts were verified in the part answers; here every figure it
        # uses must appear in the sources it cites (the number check above).
        if c.derived and c.supported:
            c.reason = "conclusion from verified facts; figures found in the cited sources"
    to_judge = [(k, c) for k, c in enumerate(claims) if c.supported and not c.derived]
    if config.VERIFIER == "nli":
        for k, c in to_judge:
            cited_chunks = [sources[i] for i in c.source_ids if i < len(sources)]
            tables = [s for s in cited_chunks if s.kind == "table"]
            if tables:
                # NLI models cannot read Markdown tables. A table claim passed the number check
                # above; it also needs the cross-encoder to judge the table relevant to it.
                from src.reranker import Reranker
                rel = float(Reranker().scores(c.text, [t.context_text for t in tables]).max())
                if rel >= config.NLI_THRESHOLD:
                    c.reason = f"numbers found in table, relevance {rel:.2f}"
                    continue
            p = entailment(c.text, cited_chunks)
            if p >= config.NLI_THRESHOLD:
                c.reason = f"entailed (p={p:.2f})"
                continue
            # Citation correction: the claim may be right but attached to the wrong source.
            others = [i for i in range(len(sources)) if i not in c.source_ids]
            fixed = None
            for i in others:
                alt = Claim(c.text, [i])
                if check_numbers(alt, sources)[0] and entailment(c.text, [sources[i]]) >= config.NLI_THRESHOLD:
                    fixed = i
                    break
            if fixed is not None:
                old = ", ".join(f"S{i + 1}" for i in c.source_ids)
                c.source_ids = [fixed]
                c.reason = f"citation corrected: {old} -> S{fixed + 1}"
            else:
                c.supported, c.reason = False, f"cited text does not entail the claim (p={p:.2f})"
        to_judge = []
    if llm is not None and to_judge:
        blocks = []
        for k, c in to_judge:
            cited = "\n".join(f"[S{i + 1}] {sources[i].text[:1800]}" for i in c.source_ids if i < len(sources))
            blocks.append(f"CLAIM {k}: {c.text}\nCITED TEXT:\n{cited}")
        data = llm.chat_json(VERIFY_SYSTEM, "\n\n---\n\n".join(blocks), VERIFY_SCHEMA, usage,
                             purpose="verify", max_tokens=400)
        verdicts = {v.get("claim_id"): v for v in data.get("verdicts", []) if isinstance(v, dict)}
        for k, c in to_judge:
            v = verdicts.get(k)
            if v is not None and not v.get("supported", True):
                c.supported, c.reason = False, str(v.get("reason", "not supported by the cited text"))

    for k, c in enumerate(claims):
        checks.append({"claim": c.text, "sources": [i + 1 for i in c.source_ids],
                       "supported": c.supported, "reason": c.reason})
    bad = [ch for ch in checks if not ch["supported"]]
    feedback = "\n".join(f"- Claim \"{ch['claim']}\" (cited {['S%d' % s for s in ch['sources']]}): {ch['reason']}. "
                         "Fix it using only the sources, cite the right source, or remove it." for ch in bad)
    return Verification(claims, feedback, checks)
