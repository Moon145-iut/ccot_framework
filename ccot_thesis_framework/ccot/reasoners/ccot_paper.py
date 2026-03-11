"""Paper-faithful CCOT reasoner implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ccot.paper.config import PaperConfig
from ccot.paper import infer as paper_infer
from ccot.reasoners.base import LatentReasoner, LatentTrace


@dataclass
class _PaperRuntime:
    cfg: PaperConfig
    device: str
    infer_state: paper_infer.InferState


class CCOTPaperReasoner(LatentReasoner):
    """Wrapper around the paper-aligned φ/ENDψ/ψ pipeline."""

    name = "paper"

    def __init__(self, cfg: PaperConfig, device: str = "cpu") -> None:
        self.runtime = _PaperRuntime(
            cfg=cfg,
            device=device,
            infer_state=paper_infer.load_infer_state(cfg, device=device),
        )

    def hidden_size(self) -> int:
        return self.runtime.infer_state.hidden_size

    def run_latent(self, question: str, max_steps: int = 200) -> LatentTrace:
        trace = paper_infer.generate_latents(
            question,
            state=self.runtime.infer_state,
            cfg=self.runtime.cfg,
            max_steps=max_steps,
        )
        return trace

    def decode_answer(
        self, question: str, trace: LatentTrace, max_new_tokens: int = 64
    ) -> str:
        return paper_infer.decode_answer(
            question=question,
            trace=trace,
            state=self.runtime.infer_state,
            cfg=self.runtime.cfg,
            max_new_tokens=max_new_tokens,
        )

    def notes(self) -> List[str]:
        deviations = []
        if not self.runtime.infer_state.joint_adapter:
            deviations.append("ψ trained on a separate adapter (fallback mode).")
        if self.runtime.cfg.stop_cap != int(200 * self.runtime.cfg.compression_ratio):
            deviations.append("Stop cap differs from 200*r per paper.")
        deviations.append("APIs cannot supply latent hidden states; local θ required.")
        return deviations
