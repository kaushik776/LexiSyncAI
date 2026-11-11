"""Load models from the local cache without contacting Hugging Face, when they are cached.

Each time a model is loaded, the Hugging Face library normally checks online for a newer
version. On a slow or blocked network that check can stall for minutes (it happened once
during testing). When every model the app needs is already downloaded, we switch the
library to offline mode before it is imported. If any model is missing, nothing changes
and it is downloaded as usual.
"""
from __future__ import annotations

import os
from pathlib import Path

from src import config


def _hf_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def is_cached(repo_id: str) -> bool:
    snapshots = _hf_cache() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def use_offline_models_if_cached() -> bool:
    """Call before importing sentence-transformers/transformers. Returns True if offline mode is on."""
    models = [config.EMBED_MODEL, config.RERANK_MODEL, config.NLI_MODEL]
    if all(is_cached(m) for m in models):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return True
    return False
