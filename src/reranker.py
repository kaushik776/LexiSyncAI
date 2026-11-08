"""Step 4: rerank retrieved chunks with a cross-encoder.

A bi-encoder (the dense retriever) embeds query and chunk separately, which is fast
but coarse. A cross-encoder reads the query and the chunk together in one transformer
pass, so attention can compare every query word with every chunk word. It is too slow
to run over the whole collection, so we run it only on the ~40 fused candidates.

Its output (a probability after a sigmoid) is also used as the evidence-strength
signal for the confidence badge and for abstention.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import config
from src.indexer import device
from src.retriever import Hit


@lru_cache(maxsize=2)
def _load(model_name: str, fp16: bool):
    import torch
    from sentence_transformers import CrossEncoder
    dev = device()
    kwargs = {"torch_dtype": torch.float16} if fp16 and dev == "cuda" else {}
    return CrossEncoder(model_name, max_length=512, device=dev, model_kwargs=kwargs)


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL, fp16: bool = config.RERANK_FP16):
        self.model = _load(model_name, fp16)

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        raw = np.asarray(self.model.predict([(query, t) for t in texts], batch_size=16,
                                            show_progress_bar=False), dtype=np.float32)
        if raw.min() < 0 or raw.max() > 1:          # logits -> probabilities
            raw = 1 / (1 + np.exp(-raw))
        return raw

    def rerank(self, query: str, hits: list[Hit], k: int = config.TOP_K) -> list[Hit]:
        probs = self.scores(query, [h.chunk.context_text for h in hits])
        order = np.argsort(-probs)[:k]
        out = []
        for r, i in enumerate(order, start=1):
            h = hits[int(i)]
            out.append(Hit(h.chunk, float(probs[i]), r, {**h.sources, "fused": h.rank}))
        return out
