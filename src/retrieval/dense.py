"""Dense (embedding) retrieval against the Qdrant `earnings_chunks` collection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.embed import COLLECTION_NAME, QUERY_PREFIX, get_client, get_model


def dense_search(
    query: str,
    k: int = 10,
    ticker: str | None = None,
    fiscal_quarter: str | None = None,
) -> list[dict]:
    model = get_model()
    client = get_client()
    query_vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    query_filter = None
    if ticker or fiscal_quarter:
        from qdrant_client.http import models as qmodels

        conditions = []
        if ticker:
            conditions.append(qmodels.FieldCondition(key="ticker", match=qmodels.MatchValue(value=ticker)))
        if fiscal_quarter:
            conditions.append(
                qmodels.FieldCondition(key="fiscal_quarter", match=qmodels.MatchValue(value=fiscal_quarter))
            )
        query_filter = qmodels.Filter(must=conditions)

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
    ).points

    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "text": h.payload["text"],
            "section": h.payload["section"],
            "ticker": h.payload["ticker"],
            "score": h.score,
        }
        for h in hits
    ]


if __name__ == "__main__":
    for r in dense_search("What did management say about margins?", k=5):
        print(f"{r['score']:.3f}  {r['chunk_id']}\n  {r['text'][:150]}...\n")
