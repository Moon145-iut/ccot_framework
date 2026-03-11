"""Layer-wise training for I+ LoRA adapters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.config import resolve_torch_dtype
from ccot.paper.config import PaperConfig
from ccot.paper.losses import variance_scaled_mse


@dataclass
class PhiTrainResult:
    layer_losses: Dict[int, float]


def _load_gold_records(gold_dir: Path, limit: int | None = None) -> List[dict]:
    data = torch.load(gold_dir / "paper_train.pt")
    if limit is not None:
        data = data[:limit]
    return data


def _prepare_base_model(cfg: PaperConfig):
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    attn_impl = "flash_attention_2" if cfg.runtime.flash_attention else "eager"
    device = cfg.runtime.device or "cpu"
    device_map = "auto" if device.startswith("cuda") else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    if device_map is None:
        model.to(device)
    model.train()
    model.requires_grad_(False)
    return tokenizer, model, device


def _enable_layer_lora_grads(model, layer_idx: int) -> None:
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            continue
        requires = any(
            token in name
            for token in [
                f".layers.{layer_idx}.",
                f".layer.{layer_idx}.",
                f".h.{layer_idx}.",
                f".block.{layer_idx}.",
            ]
        )
        param.requires_grad = requires


def train_phi_layers(
    cfg: PaperConfig,
    gold_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    epochs_per_layer: int = 1,
    lr: float = 5e-5,
    limit: int | None = None,
) -> PhiTrainResult:
    """Train LoRA adapters layer-by-layer as described in the paper."""

    gold_dir = Path(gold_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_gold_records(gold_dir, limit or cfg.limit_samples)
    tokenizer, base_model, device = _prepare_base_model(cfg)
    lora_cfg = LoraConfig(
        r=cfg.phi_lora_rank,
        lora_alpha=cfg.phi_lora_rank * 2,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    hidden_size = model.config.hidden_size

    num_layers = len(records[0]["gold_per_layer"])
    layer_losses: Dict[int, float] = {}
    log_entries: list[dict] = []

    for layer_idx in range(num_layers):
        _enable_layer_lora_grads(model, layer_idx)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        total_loss = 0.0
        total_steps = 0
        for _ in range(epochs_per_layer):
            for record in records:
                k = int(record["k"])
                question_ids = torch.tensor(
                    record["question_ids"], dtype=torch.long, device=device
                ).unsqueeze(0)
                question_embeds = model.get_input_embeddings()(question_ids)
                question_len = question_embeds.shape[1]
                if layer_idx == 0:
                    z0 = record["z0"].to(device)
                    latent_inputs = z0.unsqueeze(0).repeat(1, k, 1)
                else:
                    latent_inputs = record["gold_per_layer"][layer_idx - 1].to(device).unsqueeze(0)
                inputs_embeds = torch.cat([question_embeds, latent_inputs], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                preds = outputs.hidden_states[layer_idx + 1][0, question_len : question_len + k, :]
                target = record["gold_per_layer"][layer_idx].to(device)
                loss = variance_scaled_mse(preds, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
                total_steps += 1
        layer_losses[layer_idx] = total_loss / max(1, total_steps)
        log_entries.append({"layer": layer_idx, "loss": layer_losses[layer_idx]})

    model.save_pretrained(out_dir)
    log_path = cfg.logs_dir() / "paper_phi_losses.json"
    log_path.write_text(json.dumps(log_entries, indent=2), encoding="utf-8")
    return PhiTrainResult(layer_losses=layer_losses)
