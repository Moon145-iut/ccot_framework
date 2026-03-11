"""ENDψ training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ccot.paper.config import PaperConfig
from ccot.paper.phi_train import _prepare_embeddings, _load_gold


@dataclass
class EndTrainResult:
    loss_history: List[float]


class EndHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden).squeeze(-1)


def train_end_head(
    cfg: PaperConfig,
    train_path: str | Path,
    phi_dir: str | Path,
    out_path: str | Path,
    device: str = "cpu",
    epochs: int = 1,
    limit: int | None = None,
) -> EndTrainResult:
    records = _load_gold(Path(train_path), limit)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    phi_model = PeftModel.from_pretrained(base_model, phi_dir)
    phi_model.to(device)
    phi_model.eval()
    head = EndHead(phi_model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    loss_history: list[float] = []
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for rec in records:
            question_embeds, _, _ = _prepare_embeddings(tokenizer, phi_model, rec, device, layer_idx=0)
            latents = rec["layer_latents"].to(device)
            if latents.shape[0] <= 1:
                continue
            z0 = rec["z0"].to(device)
            teacher_inputs = torch.cat(
                [z0.unsqueeze(0), latents[:-1]],
                dim=0,
            ).unsqueeze(0)
            inputs_embeds = torch.cat([question_embeds, teacher_inputs], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
            outputs = phi_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            final_states = outputs.hidden_states[-1][0, -rec["k"] :, :]
            logits = head(final_states)
            targets = torch.zeros(rec["k"], device=device)
            targets[-1] = 1.0
            loss = bce(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        loss_history.append(total_loss / max(1, len(records)))

    torch.save(
        {"state_dict": head.state_dict(), "hidden_size": phi_model.config.hidden_size},
        out_path,
    )
    return EndTrainResult(loss_history=loss_history)
