"""Loss functions aligned with the CCOT paper."""
from __future__ import annotations

import torch
from torch import nn


def variance_scaled_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Equation from the paper: MSE scaled by variance of the targets."""

    mse = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=-1)
    variance = torch.var(target, dim=-1, unbiased=False) + eps
    scaled = mse / variance
    return scaled.mean()
