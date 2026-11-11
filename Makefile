# LexiSyncAI - one-command tasks.   Run  `make`  to list them.
#
#   First time:   make setup    (install packages, download the LLM, build the example cases)
#   Then:         make run      (open http://localhost:8501)
#   With Docker:  make docker-up   (no Python or Ollama needed on the machine)

ifeq ($(OS),Windows_NT)
  PYTHON ?= python
  VENV_PY := .venv/Scripts/python
else
  PYTHON ?= python3
  VENV_PY := .venv/bin/python
endif
MODEL ?= qwen2.5:7b

.DEFAULT_GOAL := help
.PHONY: help setup install model index run test eval eval-quick clean docker-up docker-gpu docker-down docker-logs

help: ## Show this list
	@echo "LexiSyncAI - make targets:"
	@grep -E '^[a-z-]+:.*## ' Makefile | awk 'BEGIN {FS = ":.*## "} {printf "  make %-12s %s\n", $$1, $$2}'

setup: install model index ## First run: install, download the LLM, build the example cases

install: ## Create .venv and install the pinned packages (requirements.txt)
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt

model: ## Download the local LLM with Ollama (needs Ollama installed)
	ollama pull $(MODEL)

index: ## Build the example cases: "default" (data/) and "conflict_demo" (samples/conflict/)
	$(VENV_PY) -m src.indexer default
	$(VENV_PY) -m src.indexer conflict_demo samples/conflict/Supply_Agreement_2023.pdf samples/conflict/Supply_Agreement_2024_Amended.pdf

run: ## Start the app at http://localhost:8501
	$(VENV_PY) -m streamlit run app.py

test: ## Run the automated tests
	$(VENV_PY) -m pytest -q

eval-quick: ## Evaluations without the LLM: retrieval recall@k, tables, autoencoder (~2 min)
	$(VENV_PY) -m eval.build_questions
	$(VENV_PY) -m eval.retrieval_eval
	$(VENV_PY) -m eval.table_eval
	$(VENV_PY) -m eval.anomaly_eval

eval: eval-quick ## All evaluations incl. answers, OCR, regions, latency (needs Ollama, ~30 min, on mains power)
	$(VENV_PY) -m eval.answer_eval
	$(VENV_PY) -m eval.region_eval
	$(VENV_PY) -m eval.ocr_eval
	$(VENV_PY) -m eval.latency_report

clean: ## Remove Python caches (keeps the indexes and your cases)
	$(VENV_PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in [*pathlib.Path('.').rglob('__pycache__'), pathlib.Path('.pytest_cache')] if '.venv' not in p.parts]"

docker-up: ## Start app + Ollama in Docker (CPU) at http://localhost:8501
	docker compose up --build -d
	@echo "Starting... first start downloads the LLM and models (several GB). Follow with: make docker-logs"

docker-gpu: ## Same, using the NVIDIA GPU (needs Docker GPU support)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

docker-logs: ## Follow the Docker logs
	docker compose logs -f

docker-down: ## Stop the Docker containers (downloaded models and cases are kept)
	docker compose down
