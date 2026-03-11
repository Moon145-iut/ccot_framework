"""Subset selectors for the paper pipeline."""
from __future__ import annotations

import numpy as np
import torch


def select_subset(hidden: torch.Tensor, k: int, method: str = "evenly") -> tuple[torch.Tensor, np.ndarray]:
    """Select latent tokens according to the paper's strategies."""

    length = hidden.shape[0]
    k = max(1, min(int(k), length))
    if method in {"evenly", "even", "evenly_spaced"}:
        indices = np.linspace(0, length - 1, num=k, dtype=int)
    elif method == "norm":
        values = torch.norm(hidden, dim=-1)
        selected = torch.topk(values, k=k).indices.cpu().numpy()
        indices = np.sort(selected)
    else:
        raise ValueError(f"Unknown subset selector '{method}'")
    return hidden[indices], indices
