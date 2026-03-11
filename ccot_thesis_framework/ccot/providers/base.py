"""Provider abstraction for teacher rationale generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class BaseTextProvider(ABC):
    """Abstract chat-completion style provider."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        """Return generated text for the given conversation."""

        raise NotImplementedError
