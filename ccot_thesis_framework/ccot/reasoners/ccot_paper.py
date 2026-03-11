"""Paper-faithful CCOT reasoner implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ccot.paper.config import PaperConfig
from ccot.paper import infer as paper_infer
from ccot.reasoners.base import LatentReasoner, LatentTrace


@dataclass
class _PaperRuntime:
    cfg: PaperConfig
    infer_state: paper_infer.InferState


class CCOTPaperReasoner(LatentReasoner):
    """Wrapper around the paper-aligned I+/ENDψ/I^ pipeline."""

    name = "paper"

    def __init__(self, cfg: PaperConfig, device: str = "cpu") -> None:
        cfg.runtime.device = device
        self.runtime = _PaperRuntime(
            cfg=cfg,
            infer_state=paper_infer.load_infer_state(cfg),
        )

    def hidden_size(self) -> int:
        return self.runtime.infer_state.phi_model.config.hidden_size

    def run_latent(self, question: str, max_steps: int = 200) -> LatentTrace:
        return paper_infer.generate_latents(
            question,
            self.runtime.infer_state,
            self.runtime.cfg,
            max_steps=max_steps,
        )

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
        if not self.runtime.infer_state.joint_mode:
            deviations.append("ψ adapter trained separately from φ.")
        if self.runtime.cfg.stop_cap != int(200 * self.runtime.cfg.compression_ratio):
            deviations.append("Stop cap differs from 200*r per paper.")
        return deviations
