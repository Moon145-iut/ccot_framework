"""Training for ENDψ stop head using final-layer hidden states."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.config import resolve_torch_dtype
from ccot.paper.config import PaperConfig


@dataclass
class EndTrainResult:
    loss_history: List[float]


def _load_gold_records(gold_dir: Path, limit: int | None = None) -> List[dict]:
    data = torch.load(gold_dir / "paper_train.pt")
    if limit is not None:
        data = data[:limit]
    return data


def train_end_head(
    cfg: PaperConfig,
    gold_dir: str | Path,
    phi_dir: str | Path,
    out_path: str | Path,
    *,
    device: str = "cpu",
    epochs: int = 3,
    lr: float = 5e-4,
    limit: int | None = None,
) -> EndTrainResult:
    """Train ENDψ classifier on final-layer hidden states."""

    gold_dir = Path(gold_dir)
    phi_dir = Path(phi_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _load_gold_records(gold_dir, limit or cfg.limit_samples)
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    attn_impl = "flash_attention_2" if cfg.runtime.flash_attention else "eager"
    device = cfg.runtime.device or device
    device_map = "auto" if device.startswith("cuda") else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    if device_map is None:
        base_model.to(device)
    model = PeftModel.from_pretrained(base_model, phi_dir)
    model.eval()

    hidden_size = model.config.hidden_size
    end_head = nn.Linear(hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(end_head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    loss_history: List[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        total_steps = 0
        for record in records:
            k = int(record["k"])
            question_ids = torch.tensor(
                record["question_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)
            question_embeds = model.get_input_embeddings()(question_ids)
            question_len = question_embeds.shape[1]
            z_gold = record["z_gold_l"].to(device)
            z0 = record["z0"].to(device)
            if k > 1:
                latent_inputs = torch.cat(
                    [z0.unsqueeze(0), z_gold[:-1].unsqueeze(0)], dim=1
                )
            else:
                latent_inputs = z0.unsqueeze(0)
            inputs_embeds = torch.cat([question_embeds, latent_inputs], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
            with torch.inference_mode():
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            final_hidden = outputs.hidden_states[-1][0, question_len : question_len + k, :].detach()
            logits = end_head(final_hidden).squeeze(-1)
            labels = torch.zeros(k, device=device)
            labels[-1] = 1.0
            loss = bce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(end_head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1
        loss_history.append(total_loss / max(1, total_steps))

    torch.save({"state_dict": end_head.state_dict(), "hidden_size": hidden_size}, out_path)
    log_path = cfg.logs_dir() / "paper_end_losses.json"
    log_path.write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    return EndTrainResult(loss_history=loss_history)
