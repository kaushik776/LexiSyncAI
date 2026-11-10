"""Central settings for LexiSyncAI. Change values here, not inside the modules.
Any setting can be overridden with an environment variable of the same name
(e.g. LLM_BACKEND=openai), which is how Docker and cloud deployment configure it."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

SEED = 42

# --- Ingestion ---------------------------------------------------------------
HEADER_BAND = 0.13          # top 13% of the page is checked for running headers
FOOTER_BAND = 0.08          # bottom 8% is checked for running footers
REPEAT_MIN_PAGES = 3        # text repeated in a band on >= this many pages is a header/footer
SCANNED_MAX_CHARS = 25      # a page with fewer characters than this has no usable text layer
SCANNED_MIN_IMAGE_AREA = 0.3  # ...and if images cover >= 30% of it, it is flagged as scanned
DETECT_TABLES = True
USE_OCR = True              # OCR scanned pages (Tesseract if installed, else EasyOCR)

# --- Chunking (sizes in words; ~1.3 tokens per word) -------------------------
CHUNK_TARGET_WORDS = 220    # start a new chunk at the next paragraph boundary after this
CHUNK_MAX_WORDS = 350       # hard limit, keeps chunks under the 512-token model limit
CHUNK_MIN_WORDS = 40        # smaller pieces are merged with their neighbour

# --- Retrieval -----------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # dense bi-encoder (384-dim, 512 tokens)
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"    # cross-encoder that rescores (query, chunk) pairs
RERANK_FP16 = True          # half precision on GPU: ~2x faster, same ranking (see eval/latency)
RRF_K = 60                  # reciprocal rank fusion constant: score = sum 1 / (RRF_K + rank)
REF_WEIGHT = 1.0            # weight of the clause-reference list in the fusion
CANDIDATES = 40             # chunks taken from each retriever before fusion / reranking
TOP_K = 5                   # chunks passed to the LLM
DEVICE = "auto"             # "auto" -> cuda if available, else cpu

# --- Generation (LLM) ----------------------------------------------------------
LLM_BACKEND = "ollama"      # "ollama" (local) or "openai" (any OpenAI-compatible API, e.g. Groq)
LLM_MODEL = "qwen2.5:7b"
OLLAMA_HOST = "http://localhost:11434"
OPENAI_BASE_URL = "https://api.groq.com/openai/v1"   # used when LLM_BACKEND = "openai"
OPENAI_MODEL = "llama-3.1-8b-instant"
LLM_NUM_CTX = 8192          # context window in tokens (sources + prompt + answer)
LLM_TEMPERATURE = 0.0       # deterministic answers
LLM_THINK = "auto"          # reasoning ("thinking") models: "on", "off", or "auto" = model default
MAX_SOURCES = 8             # chunks shown to the LLM (more for multi-part questions)

# --- Abstention, confidence badge, agent loop --------------------------------------
COVERAGE_SCORE = 0.30        # a sub-question counts as answered if a verified sentence scores this high for it
ABSTAIN_SCORE = 0.05        # top reranker probability below this -> abstain without calling the LLM
LOW_SCORE = 0.30            # evidence strength below this -> at most LOW confidence
HIGH_SCORE = 0.70           # evidence strength above this (and all claims verified) -> HIGH
USE_PLANNER = True          # split multi-part questions into sub-questions
USE_VERIFIER = True         # check every claim against its cited source
MAX_RETRIES = 2             # verifier-driven redrafts (step cap)
VERIFIER = "nli"            # "nli" (entailment model, default) or "llm" (the LLM judges support)
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_THRESHOLD = 0.4         # entailment probability needed for a claim to count as supported


# --- Environment overrides -----------------------------------------------------------
for _name, _value in list(globals().items()):
    if _name.isupper() and _name in os.environ:
        _raw = os.environ[_name]
        if isinstance(_value, bool):
            globals()[_name] = _raw.lower() in ("1", "true", "yes")
        elif isinstance(_value, (int, float)):
            globals()[_name] = type(_value)(_raw)
        elif isinstance(_value, str):
            globals()[_name] = _raw
