# Backend deploy target: Hugging Face Spaces (Docker SDK). Serves api/main.py
# against the pre-built local Qdrant index already committed under
# qdrant_data/ -- this container never re-runs ingestion (no SEC EDGAR
# fetch, no nltk data needed), it only serves /chat and /tickers.
#
# Unrelated to docker-compose.yml / toyapp/ (an earlier, dormant Docker-first
# attempt targeting a real Qdrant server -- see README's Known limitations).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/
COPY config.py .
COPY qdrant_data/ qdrant_data/
# BM25 (hybrid_search's sparse leg) reads data/processed/chunks.jsonl, and
# /chat's latest-quarter lookup globs data/processed/{ticker}_*.json.
COPY data/processed/ data/processed/

# Hugging Face Spaces routes traffic to port 7860 by default.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
