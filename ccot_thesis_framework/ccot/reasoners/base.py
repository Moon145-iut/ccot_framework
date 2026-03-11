"""Common interface for latent reasoners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch


@dataclass
class LatentTrace:
    """Container for latent sequences produced by a reasoner."""

    latents_l: torch.Tensor  # shape [k, d] from layer l
    latents_L: torch.Tensor  # shape [k, d] from final layer
    k: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "LatentTrace":
        """Move all tensors to a new device."""
        target = device
        return LatentTrace(
            latents_l=self.latents_l.to(target),
            latents_L=self.latents_L.to(target),
            k=self.k,
            meta=dict(self.meta),
        )


class LatentReasoner(ABC):
    """Abstract interface implemented by CCOT variants and future latent reasoners."""

    name: str
    supports_phase2: bool = True

    @abstractmethod
    def hidden_size(self) -> int:
        """Return the backbone hidden size used by the reasoner."""

    @abstractmethod
    def run_latent(self, question: str, max_steps: int = 64) -> LatentTrace:
        """Generate latent trace for the provided question."""

    @abstractmethod
    def decode_answer(
        self, question: str, trace: LatentTrace, max_new_tokens: int = 32
    ) -> str:
        """Decode the final answer string from latent trace."""

    def notes(self) -> List[str]:
        """Optional status notes for reporting."""

        return []

    def export_trace(self, record: dict[str, Any], path: str | Path) -> None:
        """Append a single trace record to JSONL file."""

        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
