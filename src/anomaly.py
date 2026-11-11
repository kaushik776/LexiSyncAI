"""Bonus feature 8 - autoencoder page-anomaly detector.

An autoencoder is a neural network trained to reproduce its input through a narrow
bottleneck. Trained on the pages of a case, it learns what a "normal" page of these
documents looks like. Pages it cannot reconstruct well (high reconstruction error) are
unusual: cover pages, scanned or image pages, tables, signature pages, contents lists.
These are the pages where extraction is most likely to go wrong, so the app points them out.

Each page is described by 12 layout features (amount of text, image coverage, font size,
digits, capitals, tables, footnotes ...). Network: 12 -> 8 -> 2 -> 8 -> 12, trained with
mean-squared reconstruction error (Adam, weight decay 1e-3, 400 full-batch steps, fixed seed), twice: the
second time only on the 90% best-reconstructed pages, so it cannot learn the outliers.
It trains in about a second on CPU when a case is indexed.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass

import numpy as np

from src import config
from src.ingest import Document, Page

FEATURES = ["text amount", "number of blocks", "words per block", "bold text", "font size",
            "image coverage", "text area", "digit share", "capital letters", "tables",
            "footnotes", "layout spread"]
MIN_PAGES = 8            # fewer pages: not enough to learn what "normal" looks like
MAX_SHARE = 0.10         # flag at most 10% of the pages


def page_features(page: Page, body_size: float) -> list[float]:
    blocks = [b for b in page.blocks if b.kind != "toc"]
    text = " ".join(b.text for b in blocks)
    letters = [c for c in text if c.isalpha()]
    area = page.width * page.height or 1.0
    words = [len(b.text.split()) for b in blocks]
    xs = [b.bbox[0] / page.width for b in blocks] if page.width else []
    return [
        math.log1p(len(text)),
        math.log1p(len(blocks)),
        statistics.mean(words) if words else 0.0,
        sum(b.bold for b in blocks) / len(blocks) if blocks else 0.0,
        (statistics.mean(b.size for b in blocks) / body_size) if blocks and body_size else 0.0,
        page.image_coverage,
        sum((b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) for b in blocks) / area,
        len(re.findall(r"\d", text)) / max(1, len(text)),
        sum(c.isupper() for c in letters) / max(1, len(letters)),
        float(len(page.tables)),
        sum(b.kind == "footnote" for b in blocks) / len(blocks) if blocks else 0.0,
        statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    ]


@dataclass
class PageAnomaly:
    doc_id: str
    page: int
    score: float          # reconstruction error (higher = more unusual)
    reason: str           # the features the network could not reconstruct


def _train(x: np.ndarray, fit: np.ndarray | None = None, epochs: int = 400, seed: int = config.SEED) -> np.ndarray:
    """Train the autoencoder on the rows `fit` (default: all) of the standardised features;
    return the squared reconstruction errors of every row."""
    import torch
    torch.manual_seed(seed)
    data = torch.tensor(x, dtype=torch.float32)
    train = data if fit is None else data[torch.tensor(fit)]
    d = data.shape[1]
    model = torch.nn.Sequential(torch.nn.Linear(d, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2),
                                torch.nn.ReLU(), torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, d))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(train), train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((model(data) - data) ** 2).numpy()


def detect(docs: list[Document]) -> list[PageAnomaly]:
    """Unusual pages across all documents of a case, most unusual first."""
    rows, keys = [], []
    for doc in docs:
        for p in doc.pages:
            rows.append(page_features(p, doc.body_font_size))
            keys.append((doc.doc_id, p.number))
    if len(rows) < MIN_PAGES:
        return []
    x = np.array(rows, dtype=np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    # Trimmed training: a network trained on every page can learn the few unusual pages too and
    # then reconstruct them perfectly. So: train once, set aside the 10% worst-reconstructed pages,
    # retrain on the rest ("normal" pages only), and score every page with that second network.
    first = _train(z).mean(axis=1)
    normal = np.where(first <= np.percentile(first, 90))[0]
    errors = _train(z, fit=normal)
    scores = errors.mean(axis=1)
    q1, q3 = np.percentile(scores, [25, 75])
    threshold = q3 + 3 * (q3 - q1)                     # robust outlier rule
    flagged = [i for i in np.argsort(-scores) if scores[i] > threshold][: max(1, int(MAX_SHARE * len(rows)))]
    out = []
    for i in flagged:
        top = np.argsort(-errors[i])[:2]
        reason = ", ".join(f"{FEATURES[f]} {'higher' if z[i, f] > 0 else 'lower'} than usual" for f in top)
        out.append(PageAnomaly(keys[i][0], keys[i][1], round(float(scores[i]), 3), reason))
    return out
