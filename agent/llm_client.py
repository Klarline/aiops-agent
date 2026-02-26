"""LLM client wrapper for the AIOps ReAct agent.

Supports OpenAI-compatible APIs with synchronous calls.
Falls back gracefully when the library or API key is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMClient:
    """Synchronous wrapper around OpenAI chat completions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client: Any = None

        if HAS_OPENAI and self._api_key:
            self._client = OpenAI(api_key=self._api_key)

    @property
    def is_available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """Send a chat completion request and return the assistant reply."""
        if not self.is_available:
            raise RuntimeError(
                "LLM client not available (missing openai package or OPENAI_API_KEY)"
            )

        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
