"""Gemini REST provider implementation."""
from __future__ import annotations

import os
from typing import Sequence

import requests

from .base import BaseTextProvider, ChatMessage


class GeminiTextProvider(BaseTextProvider):
    """Minimal wrapper around the public Gemini REST API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is required to call the Gemini REST API"
            )

    def _endpoint(self, model: str) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
            f"?key={self.api_key}"
        )

    def _serialize_messages(self, messages: Sequence[ChatMessage]) -> list[dict]:
        contents = []
        for message in messages:
            contents.append(
                {
                    "role": message.role,
                    "parts": [{"text": message.content}],
                }
            )
        return contents

    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        if not messages:
            raise ValueError("Gemini provider requires at least one chat message")

        payload = {
            "contents": self._serialize_messages(messages),
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        response = requests.post(
            self._endpoint(model),
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text.strip()}"
            )
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini API returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        return text.strip()
