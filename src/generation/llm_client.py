"""Swappable LLM client abstraction.

Structured signal generation and plain-text calls (faithfulness judge, chat
router/Q&A) all go through here instead of hitting a provider SDK directly,
so swapping providers is a one-line config change (`LLM_BACKEND`).
"""
import json
from abc import ABC, abstractmethod


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
        from config import GEMINI_API_KEY, GEMINI_MODEL

        self.client = genai.Client(api_key=api_key or GEMINI_API_KEY)
        self.model_name = model or GEMINI_MODEL

    def structured_call(self, system, user, schema):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=flatten_schema(schema),
            ),
        )
        return json.loads(response.text)

    def text_call(self, system, user):
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text


def get_client() -> LLMClient:
    from config import LLM_BACKEND

    if LLM_BACKEND == "gemini":
        return GeminiClient()
    raise ValueError(f"unknown LLM_BACKEND: {LLM_BACKEND}")
