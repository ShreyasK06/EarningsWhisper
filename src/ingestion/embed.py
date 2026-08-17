"""Embed chunks with BAAI/bge-small-en-v1.5 and upsert into Qdrant.

BGE models are asymmetric: queries get a search-instruction prefix,
documents are embedded raw. Skipping the query prefix quietly costs
several points of retrieval accuracy.

Qdrant runs in *local mode* (on-disk, no server) via `QdrantClient(path=...)`
by default -- no Docker needed. Set QDRANT_HOST in .env to point at a real
Qdrant server instead; the collection schema and upsert logic are identical
either way.

Reads:  data/processed/chunks.jsonl
Writes: config.QDRANT_PATH (local Qdrant storage, default ./qdrant_data)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

import config

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "earnings_chunks"
EMBEDDING_MODEL = config.EMBED_MODEL
EMBEDDING_DIM = 384

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if config.QDRANT_HOST:
            _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        else:
            Path(config.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=config.QDRANT_PATH)
    return _client


def embed_query(model: SentenceTransformer, query: str):
    return model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()


def embed_documents(model: SentenceTransformer, texts: list[str]):
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=True)


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def ensure_collection(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )
    client.create_payload_index(COLLECTION_NAME, field_name="ticker", field_schema="keyword")
    client.create_payload_index(COLLECTION_NAME, field_name="fiscal_quarter", field_schema="keyword")


def embed_and_load():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = get_model()
    vectors = embed_documents(model, [c["text"] for c in chunks])

    client = get_client()
    ensure_collection(client)

    points = [
        qmodels.PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    count = client.count(COLLECTION_NAME).count
    print(f"Upserted {count} points into Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    embed_and_load()
