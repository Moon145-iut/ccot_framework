"""Training script for the ψ (answer decoder) adapter."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.config import resolve_torch_dtype
from ccot.paper.config import PaperConfig


@dataclass
class PsiTrainResult:
    loss_history: List[float]
    joint_mode: bool


def _load_gold_records(gold_dir: Path, limit: int | None = None) -> List[dict]:
    data = torch.load(gold_dir / "paper_train.pt")
    if limit is not None:
        data = data[:limit]
    return data


def _load_phi_model(cfg: PaperConfig, phi_dir: Path):
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    attn_impl = "flash_attention_2" if cfg.runtime.flash_attention else "eager"
    device = cfg.runtime.device or "cpu"
    device_map = None
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
    return model, device, dtype


def _setup_psi_model(cfg: PaperConfig, phi_dir: Path, out_dir: Path, psi_rank: int):
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    attn_impl = "flash_attention_2" if cfg.runtime.flash_attention else "eager"
    device = cfg.runtime.device or "cpu"
    device_map = None
    psi_config = LoraConfig(
        r=psi_rank,
        lora_alpha=psi_rank * 2,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    if device_map is None:
        base_model.to(device)

    joint_mode = False
    try:
        joint_model = PeftModel.from_pretrained(base_model, phi_dir)
        joint_model.add_adapter("psi", psi_config)
        joint_model.set_adapter(["default", "psi"])
        model = joint_model
        joint_mode = True
    except Exception:
        fallback_base = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        fallback_base.to(device)
        model = get_peft_model(fallback_base, psi_config)
        joint_mode = False

    model.train()
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = "psi" in name.lower()
        else:
            param.requires_grad = False
    return model, joint_mode, device


def _generate_latents(phi_model, question_ids, z0, k, layer_idx, device):
    latents: list[torch.Tensor] = []
    question_embeds = phi_model.get_input_embeddings()(question_ids)
    seq_embeds = question_embeds
    prev = z0.unsqueeze(0).unsqueeze(0).to(device)
    for _ in range(k):
        inputs = torch.cat([seq_embeds, prev], dim=1)
        attention_mask = torch.ones(inputs.shape[:2], device=device)
        with torch.inference_mode():
            outputs = phi_model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        latent_vec = outputs.hidden_states[layer_idx + 1][0, -1, :].detach()
        latents.append(latent_vec)
        prev = latent_vec.unsqueeze(0).unsqueeze(0)
        seq_embeds = torch.cat([seq_embeds, prev], dim=1)
    if not latents:
        return z0.unsqueeze(0).unsqueeze(0)[:, :0, :]
    stacked = torch.stack(latents, dim=0)  # [k, d]
    return stacked.unsqueeze(0)


def train_psi_decoder(
    cfg: PaperConfig,
    gold_dir: str | Path,
    phi_dir: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    epochs: int = 2,
    lr: float = 5e-5,
    psi_rank: int = 64,
    limit: int | None = None,
) -> PsiTrainResult:
    """Train ψ LoRA adapter using φ-generated latents."""

    gold_dir = Path(gold_dir)
    phi_dir = Path(phi_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_gold_records(gold_dir, limit or cfg.limit_samples)

    phi_model, base_device, dtype = _load_phi_model(cfg, phi_dir)
    psi_model, joint_mode, device = _setup_psi_model(cfg, phi_dir, out_dir, psi_rank)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, psi_model.parameters()), lr=lr
    )

    layer_idx = cfg.resolved_layer_index(len(records[0]["gold_per_layer"]))
    loss_history: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        total_steps = 0
        for record in records:
            k = int(record["k"])
            question_ids = torch.tensor(
                record["question_ids"], dtype=torch.long, device=device
            ).unsqueeze(0)
            question_embeds = psi_model.get_input_embeddings()(question_ids)
            latents = _generate_latents(
                phi_model,
                question_ids,
                record["z0"].to(device),
                k,
                layer_idx,
                device,
            )
            answer_ids = tokenizer(record["answer"], add_special_tokens=True)["input_ids"]
            if not answer_ids:
                continue
            answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device).unsqueeze(0)
            answer_inputs = answer_tensor[:, :-1]
            answer_labels = answer_tensor[:, 1:]
            answer_embeds = psi_model.get_input_embeddings()(answer_inputs)
            inputs_embeds = torch.cat([question_embeds, latents, answer_embeds], dim=1)
            labels = torch.full(
                (1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device
            )
            labels[0, -answer_labels.shape[1] :] = answer_labels[0]
            outputs = psi_model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(psi_model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_steps += 1
        loss_history.append(total_loss / max(1, total_steps))

    psi_model.save_pretrained(out_dir)
    (out_dir / "meta.json").write_text(
        json.dumps({"joint_mode": joint_mode}, indent=2),
        encoding="utf-8",
    )
    log_dir = cfg.logs_dir()
    (log_dir / "paper_psi_losses.json").write_text(json.dumps(loss_history, indent=2), encoding="utf-8")
    (log_dir / "paper_psi_mode.json").write_text(
        json.dumps({"joint_mode_used": joint_mode}, indent=2), encoding="utf-8"
    )
    return PsiTrainResult(loss_history=loss_history, joint_mode=joint_mode)
