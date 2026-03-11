"""Subset selectors for the paper pipeline."""
from __future__ import annotations

import numpy as np
import torch


def select_indices(hidden: torch.Tensor, k: int, method: str = "evenly") -> np.ndarray:
    """Return indices for the selected rationale tokens."""

    length = hidden.shape[0]
    if length == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(int(max(1, k)), length)
    if method in {"evenly", "even", "evenly_spaced"}:
        return np.linspace(0, length - 1, num=k, dtype=int)
    if method == "norm":
        values = torch.norm(hidden, dim=-1)
        selected = torch.topk(values, k=k).indices.cpu().numpy()
        return np.sort(selected)
    raise ValueError(f"Unknown subset selector '{method}'")
