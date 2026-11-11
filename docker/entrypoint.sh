#!/bin/sh
# Container start: build the example cases on the first start, then run the app.
set -e

if [ ! -f index/default/manifest.json ]; then
  echo "First start: building the example case from data/ (downloads the embedding model)..."
  python -m src.indexer default
fi
if [ ! -f index/conflict_demo/manifest.json ]; then
  python -m src.indexer conflict_demo samples/conflict/Supply_Agreement_2023.pdf samples/conflict/Supply_Agreement_2024_Amended.pdf
fi

echo "LexiSyncAI is ready at http://localhost:8501"
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
