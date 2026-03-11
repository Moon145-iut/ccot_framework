"""Truth Vector-steered latent reasoner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ccot.reasoners.ccot_cpu_gru import CCOTCpuGRUReasoner
from ccot.reasoners.base import LatentTrace


class CCOTTruthVectorReasoner(CCOTCpuGRUReasoner):
    """Extends the CPU GRU reasoner by injecting a Truth Vector during inference."""

    name = "truth_vector"

    def __init__(
        self,
        targets_dir: str | Path,
        ccot_weights: str | Path,
        decoder_weights: str | Path,
        truth_vector_path: str | Path,
        *,
        alpha: float = 1.0,
        stop_threshold: float = 0.5,
        max_latents: int = 64,
        num_threads: int = 4,
    ) -> None:
        super().__init__(
            targets_dir=targets_dir,
            ccot_weights=ccot_weights,
            decoder_weights=decoder_weights,
            stop_threshold=stop_threshold,
            max_latents=max_latents,
            num_threads=num_threads,
        )
        self.truth_vector_path = Path(truth_vector_path)
        self.alpha = float(alpha)
        self.truth_direction = self._load_truth_vector(self.truth_vector_path)

    def _load_truth_vector(self, path: Path) -> torch.Tensor:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict) and "vector" in tensor:
            tensor = tensor["vector"]
        tensor = torch.as_tensor(tensor, dtype=torch.float32).view(-1)
        if tensor.numel() != self.hidden_size():
            raise ValueError(
                f"Truth vector hidden size mismatch ({tensor.numel()} vs {self.hidden_size()})."
            )
        norm = tensor.norm()
        if torch.isclose(norm, torch.zeros(1)):
            raise ValueError("Truth vector has zero norm; cannot normalize.")
        return tensor / norm

    def _steer_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Apply Equation: h_{t+1} = h_t + alpha * sigma_l * v_truth."""

        sigma = torch.std(latents, unbiased=False)
        if torch.isnan(sigma) or torch.isclose(sigma, torch.zeros(1)):
            sigma = torch.tensor(1.0)
        delta = self.alpha * sigma * self.truth_direction
        steered = latents + delta.unsqueeze(0)
        return steered, float(sigma.item())

    def run_latent(self, question: str, max_steps: int = 64) -> LatentTrace:
        base_trace = super().run_latent(question, max_steps=max_steps)
        steered_latents, sigma = self._steer_latents(base_trace.latents_l)
        meta: dict[str, Any] = dict(base_trace.meta)
        meta["truth_steering"] = {
            "alpha": self.alpha,
            "sigma": sigma,
            "vector_path": str(self.truth_vector_path),
        }
        return LatentTrace(
            latents_l=steered_latents.clone(),
            latents_L=steered_latents.clone(),
            k=base_trace.k,
            meta=meta,
        )

    def notes(self) -> list[str]:
        base_notes = super().notes()
        base_notes.append(
            f"Applied Truth Vector steering (alpha={self.alpha}) from {self.truth_vector_path}."
        )
        return base_notes
