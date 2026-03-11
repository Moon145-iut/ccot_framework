"""Latent generator inspired by CCOT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class LatentGenerator(nn.Module):
    """Continuous autoregressive latent generator with stop signaling."""

    def __init__(self, hidden_size: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim or hidden_size
        self.q_proj = nn.Linear(hidden_size, self.hidden_dim)
        self.in_proj = nn.Linear(hidden_size, self.hidden_dim)
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, hidden_size)
        self.stop_head = nn.Linear(self.hidden_dim, 1)

    def forward_train(
        self,
        q_feat: torch.Tensor,
        teacher_latents: torch.Tensor,
        lengths: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, max_steps, _ = teacher_latents.shape
        device = q_feat.device
        pred_latents = []
        stop_logits = []
        hidden = torch.tanh(self.q_proj(q_feat))
        prev_latent = torch.zeros(batch, self.hidden_size, device=device)
        for t in range(max_steps):
            teacher_latent = teacher_latents[:, t, :]
            if teacher_forcing_ratio >= 1.0:
                step_input = teacher_latent
            elif teacher_forcing_ratio <= 0.0:
                step_input = prev_latent
            else:
                probs = torch.rand(batch, device=device)
                mask = (probs < teacher_forcing_ratio).float().unsqueeze(-1)
                step_input = mask * teacher_latent + (1 - mask) * prev_latent
            gru_input = self.in_proj(step_input)
            hidden = self.gru_cell(gru_input, hidden)
            pred = self.out_proj(hidden)
            stop_logit = self.stop_head(hidden).squeeze(-1)
            pred_latents.append(pred.unsqueeze(1))
            stop_logits.append(stop_logit.unsqueeze(1))
            prev_latent = pred
        pred_latents = torch.cat(pred_latents, dim=1)
        stop_logits = torch.cat(stop_logits, dim=1)
        return pred_latents, stop_logits

    def generate(
        self,
        q_feat: torch.Tensor,
        max_steps: int = 64,
        stop_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _ = q_feat.shape
        device = q_feat.device
        hidden = torch.tanh(self.q_proj(q_feat))
        prev_latent = torch.zeros(batch, self.hidden_size, device=device)
        latents = []
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for step in range(max_steps):
            gru_input = self.in_proj(prev_latent)
            hidden = self.gru_cell(gru_input, hidden)
            pred = self.out_proj(hidden)
            stop_logit = self.stop_head(hidden).squeeze(-1)
            latents.append(pred.unsqueeze(1))
            stop_prob = torch.sigmoid(stop_logit)
            newly_finished = (stop_prob > stop_threshold) & (~finished)
            new_lengths = torch.full_like(lengths, step + 1)
            lengths = torch.where(newly_finished, new_lengths, lengths)
            finished |= newly_finished
            prev_latent = pred
            if finished.all():
                break
        latents_tensor = torch.cat(latents, dim=1)
        total_steps = latents_tensor.shape[1]
        lengths = torch.where(lengths == 0, torch.full_like(lengths, total_steps), lengths)
        return latents_tensor, lengths

    @staticmethod
    def loss(
        pred_latents: torch.Tensor,
        gold_latents: torch.Tensor,
        stop_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = pred_latents.device
        batch, max_steps, _ = pred_latents.shape
        mask = (torch.arange(max_steps, device=device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        mse = ((pred_latents - gold_latents) ** 2).sum(dim=-1)
        mse = (mse * mask).sum() / mask.sum().clamp(min=1.0)

        stop_targets = torch.zeros_like(stop_logits)
        indices = (lengths - 1).clamp(min=0)
        stop_targets.scatter_(1, indices.unsqueeze(1), 1.0)
        bce = nn.functional.binary_cross_entropy_with_logits(
            stop_logits, stop_targets, reduction="none"
        )
        bce = (bce * mask).sum() / mask.sum().clamp(min=1.0)

        total = mse + 0.2 * bce
        stats = {
            "loss_total": float(total.item()),
            "loss_mse": float(mse.item()),
            "loss_stop": float(bce.item()),
        }
        return total, stats
