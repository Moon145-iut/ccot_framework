"""Factory for provider implementations."""
from __future__ import annotations

from .base import BaseTextProvider
from .gemini_rest import GeminiTextProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(name: str, base_url: str | None = None) -> BaseTextProvider:
    """Instantiate a provider by a friendly name."""

    normalized = name.strip().lower()
    if normalized == "gemini":
        return GeminiTextProvider()
    if normalized in {"openai_compat", "openai-compatible", "openai"}:
        return OpenAICompatibleProvider(base_url=base_url)
    raise ValueError(
        "Unsupported provider name. Expected one of: gemini, openai_compat"
    )
