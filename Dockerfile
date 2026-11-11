# LexiSyncAI app image (the LLM runs in the separate "ollama" container, see docker-compose.yml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# Tesseract = OCR for scanned pages (tried before EasyOCR); libgl/libglib for OpenCV; curl for the health check
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned dependencies first, so this layer is cached when only the code changes
COPY requirements.txt .
RUN sed -i 's/\r$//' requirements.txt \
 && pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' docker/entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["sh", "docker/entrypoint.sh"]
