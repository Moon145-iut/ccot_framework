"""ψ decoder training."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.paper.config import PaperConfig
from ccot.paper.phi_train import _load_gold, _prepare_embeddings


@dataclass
class PsiTrainResult:
    loss_history: List[float]
    joint_training: bool


def _build_answer_inputs(tokenizer, model, answer: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer(answer.strip(), add_special_tokens=True)["input_ids"]
    tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    embeds = model.get_input_embeddings()(tensor)
    return tensor, embeds


def train_psi_decoder(
    cfg: PaperConfig,
    train_path: str | Path,
    phi_dir: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    epochs: int = 1,
    limit: int | None = None,
    joint_training: bool = True,
) -> PsiTrainResult:
    """Train ψ either jointly with φ or in fallback mode."""

    records = _load_gold(Path(train_path), limit)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    if joint_training:
        model = PeftModel.from_pretrained(base_model, phi_dir)
        joint_flag = True
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        joint_flag = False
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss_history: list[float] = []

    for epoch in range(epochs):
        total_loss = 0.0
        for rec in records:
            question_embeds, latent_inputs, _ = _prepare_embeddings(
                tokenizer, model, rec, device, layer_idx=cfg.layer_l
            )
            latents = rec["layer_latents"].to(device).unsqueeze(0)
            answer_ids, answer_embeds = _build_answer_inputs(tokenizer, model, rec["answer"], device)
            inputs_embeds = torch.cat(
                [question_embeds, latents, answer_embeds[:, :-1, :]],
                dim=1,
            )
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=device)
            labels = torch.full(
                (1, inputs_embeds.shape[1]),
                -100,
                dtype=torch.long,
                device=device,
            )
            labels[0, -answer_ids.shape[1] :] = answer_ids[0]
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        loss_history.append(total_loss / max(1, len(records)))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    import json

    meta = {"joint": joint_flag}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return PsiTrainResult(loss_history=loss_history, joint_training=joint_flag)
