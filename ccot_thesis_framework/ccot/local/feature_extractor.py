"""Hidden target builder for compressed latent supervision."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .backbone import LocalBackbone
from .subset_selector import select_latents


@dataclass(slots=True)
class ExtractedTarget:
    q_feat: torch.Tensor
    z_targets: torch.Tensor
    z_len: int
    answer_text: str
    selected_indices: list[int]
    rationale_token_count: int


class HiddenTargetBuilder:
    def __init__(
        self,
        backbone: LocalBackbone,
        compression_ratio: float = 0.1,
        selector: str = "evenly_spaced",
        hidden_layer_index: int = -2,
    ) -> None:
        self.backbone = backbone
        self.compression_ratio = float(compression_ratio)
        self.selector = selector
        self.hidden_layer_index = hidden_layer_index

    def build(self, question: str, rationale: str, final_answer: str) -> ExtractedTarget:
        segments = [
            f"Question: {question}\n",
            "Reasoning:\n",
            f"{rationale}\n",
            f"Answer:\n{final_answer}",
        ]
        hidden, spans, _ = self.backbone.encode_segments_hidden(
            segments, self.hidden_layer_index
        )
        question_span = spans[0]
        rationale_span = spans[2]
        if question_span[1] <= question_span[0]:
            raise RuntimeError("Question span collapsed; consider increasing max_length")
        q_feat = hidden[question_span[1] - 1]
        rationale_hidden = hidden[rationale_span[0] : rationale_span[1]]
        rationale_len = rationale_hidden.shape[0]
        if rationale_len == 0:
            rationale_hidden = hidden[rationale_span[0] : rationale_span[0] + 1]
            rationale_len = rationale_hidden.shape[0]
        k = max(1, math.ceil(self.compression_ratio * rationale_len))
        selected, indices = select_latents(rationale_hidden, k, self.selector)
        return ExtractedTarget(
            q_feat=q_feat.cpu(),
            z_targets=selected.cpu(),
            z_len=int(selected.shape[0]),
            answer_text=str(final_answer).strip(),
            selected_indices=indices.tolist() if isinstance(indices, np.ndarray) else list(indices),
            rationale_token_count=int(rationale_len),
        )
