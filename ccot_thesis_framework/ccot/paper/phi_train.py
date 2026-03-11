"""Layer-wise training for φ adapters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.paper.config import PaperConfig
from ccot.paper.losses import variance_scaled_mse


@dataclass
class PhiTrainResult:
    layer_losses: Dict[int, float]


def _load_gold(path: Path, limit: int | None = None) -> List[dict]:
    data = torch.load(path)
    if limit is not None:
        data = data[:limit]
    return data


def _prepare_embeddings(
    tokenizer, model, record: dict, device: str, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    question_ids = torch.tensor(record["question_tokens"], dtype=torch.long, device=device).unsqueeze(0)
    question_embeds = model.get_input_embeddings()(question_ids)
    question_len = question_embeds.shape[1]
    k = record["k"]
    z0 = record["z0"].to(device)
    if layer_idx == 0:
        latent_inputs = z0.unsqueeze(0).repeat(1, k, 1)
    else:
        prev_layer = record.get("subset_latents")
        latent_inputs = (
            prev_layer.to(device).unsqueeze(0)
            if prev_layer is not None
            else z0.unsqueeze(0).repeat(1, k, 1)
        )
    return question_embeds, latent_inputs, torch.tensor([question_len], device=device)


def train_phi_layers(
    cfg: PaperConfig,
    train_path: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    epochs: int = 1,
    limit: int | None = None,
) -> PhiTrainResult:
    """Train LoRA adapters layer by layer as described in the paper."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_gold(Path(train_path), limit)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    base_model.to(device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    layer_losses: dict[int, float] = {}

    latent_targets = [torch.tensor(rec["layer_latents"]) for rec in records]
    for layer_idx in range(cfg.layer_l + 1):
        for name, param in model.named_parameters():
            param.requires_grad = f".{layer_idx}." in name and "lora" in name.lower()
        for epoch in range(epochs):
            total_loss = 0.0
            for rec, target in zip(records, latent_targets):
                question_embeds, latent_inputs, _ = _prepare_embeddings(
                    tokenizer, model, rec, device, layer_idx
                )
                inputs_embeds = torch.cat([question_embeds, latent_inputs], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                preds = outputs.hidden_states[layer_idx + 1][0, -rec["k"] :, :]
                loss = variance_scaled_mse(preds, target.to(device))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
            layer_losses[layer_idx] = total_loss / max(1, len(records))

    model.save_pretrained(out_dir)
    return PhiTrainResult(layer_losses=layer_losses)
