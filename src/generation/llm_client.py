"""Swappable LLM client abstraction.

Structured signal generation and plain-text calls (faithfulness judge, chat
router/Q&A) all go through here instead of hitting a provider SDK directly,
so swapping providers is a one-line config change (`LLM_BACKEND`).
"""
import json
import time
from abc import ABC, abstractmethod

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT_MS = 60_000
MAX_RATE_LIMIT_RETRIES = 3
MAX_RATE_LIMIT_DELAY_SECONDS = 90


def _rate_limit_retry_delay_seconds(client_error) -> float | None:
    """Extract Google's suggested retry delay from a 429 response, if present.

    Only genuine per-minute rate limits carry a RetryInfo.retryDelay --
    a hard daily-quota exhaustion does not, so the absence of this field
    is itself the signal to fail fast rather than retry blindly.
    """
    details = getattr(client_error, "details", None)
    if not isinstance(details, dict):
        return None
    error = details.get("error", {})
    if error.get("status") != "RESOURCE_EXHAUSTED":
        return None
    for entry in error.get("details", []):
        if entry.get("@type", "").endswith("RetryInfo"):
            raw = entry.get("retryDelay", "")
            try:
                return float(raw.rstrip("s"))
            except ValueError:
                return None
    return None


class LLMClient(ABC):
    @abstractmethod
    def structured_call(self, system: str, user: str, schema: dict) -> dict:
        """Return a dict matching `schema`."""
        ...

    @abstractmethod
    def text_call(self, system: str, user: str) -> str:
        """Plain text completion -- used by the faithfulness judge and the chat router/Q&A."""
        ...


def _resolve_refs(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return _resolve_refs(defs[name], defs)
        return {k: _resolve_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_resolve_refs(item, defs) for item in node]
    return node


def flatten_schema(schema: dict) -> dict:
    """Inline every `$ref`/`$defs` indirection.

    Gemini's `response_schema` doesn't understand JSON Schema's `$ref`
    indirection, which pydantic emits for any nested model (e.g.
    `Signal.supporting_claims: list[SupportingClaim]`) -- resolve it away
    before handing the schema to the API.
    """
    return _resolve_refs(schema, schema.get("$defs", {}))


class GeminiClient(LLMClient):
    """Uses `google-genai`, the current supported Gemini SDK.

    The older `google-generativeai` package (the one this abstraction was
    originally sketched against) has been fully deprecated by Google --
    "All support has ended, will no longer receive updates or bug fixes" --
    so this targets its replacement instead.
    """

    def __init__(self, model=None, api_key=None):
        from google import genai
        from google.genai import types
        from config import GEMINI_API_KEY, GEMINI_MODEL

        self.client = genai.Client(
            api_key=api_key or GEMINI_API_KEY,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.model_name = model or GEMINI_MODEL

    def _generate_with_retry(self, contents, config):
        """Retry on transient server errors (e.g. 503 model-overloaded) and
        request timeouts (a stalled socket never errors on its own without
        REQUEST_TIMEOUT_MS -- see the http_options set in __init__), and on
        429s that carry a server-suggested retryDelay (a per-minute rate
        limit, not a hard quota). Any other client error (bad request, auth,
        or a 429 with no retryDelay -- e.g. a hard daily-quota exhaustion)
        propagates immediately, since retrying those wastes quota for no
        benefit."""
        import httpx
        from google.genai import errors

        server_attempts = 0
        rate_limit_attempts = 0
        while True:
            try:
                return self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except (errors.ServerError, httpx.TimeoutException):
                if server_attempts >= MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS * (2**server_attempts))
                server_attempts += 1
            except errors.ClientError as e:
                delay = _rate_limit_retry_delay_seconds(e)
                if delay is None or rate_limit_attempts >= MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(min(delay, MAX_RATE_LIMIT_DELAY_SECONDS))
                rate_limit_attempts += 1

    def structured_call(self, system, user, schema):
        from google.genai import types

        response = self._generate_with_retry(
            user,
            types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=flatten_schema(schema),
            ),
        )
        return json.loads(response.text)

    def text_call(self, system, user):
        from google.genai import types

        response = self._generate_with_retry(
            user, types.GenerateContentConfig(system_instruction=system)
        )
        return response.text


def get_client() -> LLMClient:
    from config import LLM_BACKEND

    if LLM_BACKEND == "gemini":
        return GeminiClient()
    raise ValueError(f"unknown LLM_BACKEND: {LLM_BACKEND}")
