"""Latent subset selection utilities."""
from __future__ import annotations

import numpy as np
import torch

def _sanitize_k(length: int, k: int) -> int:
    return max(1, min(int(k), length))


def select_evenly_spaced(hidden_seq: torch.Tensor, k: int) -> tuple[torch.Tensor, np.ndarray]:
    """Pick ``k`` hidden states via evenly spaced indices."""

    length = hidden_seq.shape[0]
    k = _sanitize_k(length, k)
    if k == length:
        indices = np.arange(length)
    else:
        indices = np.linspace(0, length - 1, num=k, dtype=int)
    return hidden_seq[indices], indices


def select_by_norm(hidden_seq: torch.Tensor, k: int) -> tuple[torch.Tensor, np.ndarray]:
    """Select tokens whose hidden activations have the largest L2 norms."""

    length = hidden_seq.shape[0]
    k = _sanitize_k(length, k)
    norms = torch.norm(hidden_seq, dim=-1)
    topk = torch.topk(norms, k=k).indices.cpu().numpy()
    topk.sort()
    return hidden_seq[topk], topk


SELECTOR_MAP = {
    "evenly_spaced": select_evenly_spaced,
    "even": select_evenly_spaced,
    "norm": select_by_norm,
}


def select_latents(hidden_seq: torch.Tensor, k: int, method: str = "evenly_spaced") -> tuple[torch.Tensor, np.ndarray]:
    """Dispatch helper that chooses how rationale latents are compressed."""

    if hidden_seq.ndim != 2:
        raise ValueError("hidden_seq must be 2-D [seq, dim]")
    selector = SELECTOR_MAP.get(method)
    if selector is None:
        raise ValueError(f"Unknown selector '{method}'. Valid options: {sorted(SELECTOR_MAP)}")
    return selector(hidden_seq, k)
