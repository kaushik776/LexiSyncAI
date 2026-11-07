"""Central settings for LexiSyncAI. Change values here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

SEED = 42

# --- Ingestion ---------------------------------------------------------------
HEADER_BAND = 0.12          # top 12% of the page is checked for running headers
FOOTER_BAND = 0.08          # bottom 8% is checked for running footers
REPEAT_MIN_PAGES = 3        # text repeated in a band on >= this many pages is a header/footer
SCANNED_MAX_CHARS = 25      # a page with fewer characters than this has no usable text layer
SCANNED_MIN_IMAGE_AREA = 0.3  # ...and if images cover >= 30% of it, it is flagged as scanned
DETECT_TABLES = True
USE_OCR = True              # OCR scanned pages if Tesseract is installed (skipped otherwise)

# --- Chunking (sizes in words; ~1.3 tokens per word) -------------------------
CHUNK_TARGET_WORDS = 220    # start a new chunk at the next paragraph boundary after this
CHUNK_MAX_WORDS = 350       # hard limit, keeps chunks under the 512-token model limit
CHUNK_MIN_WORDS = 40        # smaller pieces are merged with their neighbour
