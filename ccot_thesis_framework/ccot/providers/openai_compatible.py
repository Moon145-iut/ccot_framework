"""OpenAI-compatible REST provider."""
from __future__ import annotations

import os
from typing import Sequence

import requests

from .base import BaseTextProvider, ChatMessage


class OpenAICompatibleProvider(BaseTextProvider):
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(url, json=payload, headers=self._headers(), timeout=60)
        if not response.ok:
            raise RuntimeError(
                f"OpenAI-compatible API error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI-compatible API returned no choices")
        return choices[0]["message"].get("content", "").strip()
