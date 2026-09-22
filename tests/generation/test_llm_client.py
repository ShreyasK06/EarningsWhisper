import pytest

import src.generation.llm_client as llm_client_mod


class _FakeModels:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeGenaiClient:
    def __init__(self, side_effects):
        self.models = _FakeModels(side_effects)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _make_client(monkeypatch, side_effects):
    client = llm_client_mod.GeminiClient.__new__(llm_client_mod.GeminiClient)
    client.client = _FakeGenaiClient(side_effects)
    client.model_name = "fake-model"
    monkeypatch.setattr(llm_client_mod.time, "sleep", lambda seconds: None)
    return client


def test_text_call_retries_on_server_error_then_succeeds(monkeypatch):
    from google.genai import errors

    server_error = errors.ServerError(503, {"error": {"message": "overloaded"}})
    client = _make_client(monkeypatch, [server_error, _FakeResponse("ok")])

    result = client.text_call("system", "user")

    assert result == "ok"
    assert client.client.models.calls == 2


def test_text_call_raises_after_exhausting_retries(monkeypatch):
    from google.genai import errors

    server_error = errors.ServerError(503, {"error": {"message": "overloaded"}})
    effects = [server_error] * (llm_client_mod.MAX_RETRIES + 1)
    client = _make_client(monkeypatch, effects)

    with pytest.raises(errors.ServerError):
        client.text_call("system", "user")

    assert client.client.models.calls == llm_client_mod.MAX_RETRIES + 1


def test_text_call_does_not_retry_client_error(monkeypatch):
    from google.genai import errors

    client_error = errors.ClientError(400, {"error": {"message": "bad request"}})
    client = _make_client(monkeypatch, [client_error, _FakeResponse("unreachable")])

    with pytest.raises(errors.ClientError):
        client.text_call("system", "user")

    assert client.client.models.calls == 1


def test_text_call_retries_on_timeout_then_succeeds(monkeypatch):
    import httpx

    timeout = httpx.ConnectTimeout("timed out")
    client = _make_client(monkeypatch, [timeout, _FakeResponse("ok")])

    result = client.text_call("system", "user")

    assert result == "ok"
    assert client.client.models.calls == 2


def _rate_limit_error(retry_delay="1s"):
    from google.genai import errors

    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": f"quota exceeded, retry in {retry_delay}",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    }
                ],
            }
        },
    )


def test_text_call_retries_on_rate_limit_with_retry_delay_then_succeeds(monkeypatch):
    client = _make_client(monkeypatch, [_rate_limit_error("1s"), _FakeResponse("ok")])

    result = client.text_call("system", "user")

    assert result == "ok"
    assert client.client.models.calls == 2


def test_text_call_stops_after_max_rate_limit_retries(monkeypatch):
    from google.genai import errors

    effects = [_rate_limit_error("1s")] * (llm_client_mod.MAX_RATE_LIMIT_RETRIES + 1)
    client = _make_client(monkeypatch, effects)

    with pytest.raises(errors.ClientError):
        client.text_call("system", "user")

    assert client.client.models.calls == llm_client_mod.MAX_RATE_LIMIT_RETRIES + 1


def test_text_call_does_not_retry_hard_quota_exhaustion_without_retry_delay(monkeypatch):
    from google.genai import errors

    hard_quota_error = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "daily quota exceeded",
                "details": [],
            }
        },
    )
    client = _make_client(monkeypatch, [hard_quota_error, _FakeResponse("unreachable")])

    with pytest.raises(errors.ClientError):
        client.text_call("system", "user")

    assert client.client.models.calls == 1
