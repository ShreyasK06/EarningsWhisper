from src.ingestion import embed as embed_mod


def test_get_model_returns_singleton(monkeypatch):
    embed_mod._model = None
    calls = []

    class FakeModel:
        pass

    def fake_ctor(name):
        calls.append(name)
        return FakeModel()

    monkeypatch.setattr(embed_mod, "SentenceTransformer", fake_ctor)

    m1 = embed_mod.get_model()
    m2 = embed_mod.get_model()

    assert m1 is m2
    assert len(calls) == 1


def test_get_client_returns_singleton(monkeypatch, tmp_path):
    embed_mod._client = None
    monkeypatch.setattr(embed_mod.config, "QDRANT_HOST", "")
    monkeypatch.setattr(embed_mod.config, "QDRANT_PATH", str(tmp_path / "qdrant_test"))

    calls = []

    class FakeClient:
        pass

    def fake_ctor(path):
        calls.append(path)
        return FakeClient()

    monkeypatch.setattr(embed_mod, "QdrantClient", fake_ctor)

    c1 = embed_mod.get_client()
    c2 = embed_mod.get_client()

    assert c1 is c2
    assert len(calls) == 1
