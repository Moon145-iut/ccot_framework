"""Inference utilities for paper-faithful CCOT."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.paper.config import PaperConfig
from ccot.reasoners.base import LatentTrace


@dataclass
class InferState:
    tokenizer: AutoTokenizer
    phi_model: PeftModel
    psi_model: PeftModel
    end_head: torch.nn.Module
    hidden_size: int
    device: str
    joint_adapter: bool


def load_infer_state(cfg: PaperConfig, device: str = "cpu") -> InferState:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    base_model_phi = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    phi_model = PeftModel.from_pretrained(base_model_phi, cfg.models_dir() / "paper_phi")
    phi_model.to(device)
    phi_model.eval()

    psi_path = cfg.models_dir() / "paper_psi"
    meta_file = psi_path / "meta.json"
    joint = False
    if psi_path.exists():
        joint_meta = meta_file.exists() and json.loads(meta_file.read_text()).get("joint")
        if joint_meta:
            psi_model = phi_model
            joint = True
        else:
            base_model_psi = AutoModelForCausalLM.from_pretrained(cfg.model_id)
            psi_model = PeftModel.from_pretrained(base_model_psi, psi_path)
            joint = False
    else:
        psi_model = phi_model
        joint = True
    psi_model.to(device)
    psi_model.eval()

    end_payload = torch.load(cfg.models_dir() / "paper_end.pt", map_location=device)
    head = torch.nn.Linear(end_payload["hidden_size"], 1)
    head.load_state_dict(end_payload["state_dict"])
    head.to(device)
    head.eval()

    return InferState(
        tokenizer=tokenizer,
        phi_model=phi_model,
        psi_model=psi_model,
        end_head=head,
        hidden_size=phi_model.config.hidden_size,
        device=device,
        joint_adapter=joint,
    )


def _question_embeddings(question: str, state: InferState) -> tuple[torch.Tensor, torch.Tensor]:
    ids = state.tokenizer(question.strip(), add_special_tokens=True)["input_ids"]
    tensor = torch.tensor(ids, dtype=torch.long, device=state.device).unsqueeze(0)
    embeds = state.phi_model.get_input_embeddings()(tensor)
    return tensor, embeds


def _initial_query_feature(question_ids: torch.Tensor, state: InferState, cfg: PaperConfig) -> torch.Tensor:
    with torch.no_grad():
        outputs = state.phi_model.base_model(
            input_ids=question_ids,
            output_hidden_states=True,
        )
    return outputs.hidden_states[cfg.layer_l + 1][0, -1, :]


def generate_latents(
    question: str,
    state: InferState,
    cfg: PaperConfig,
    max_steps: int = 200,
) -> LatentTrace:
    question_ids, q_embeds = _question_embeddings(question, state)
    z0 = _initial_query_feature(question_ids, state, cfg)
    latents_layer = []
    latents_final = []
    stop_limit = min(max_steps, cfg.stop_limit)
    current_inputs = []

    for step in range(stop_limit):
        if current_inputs:
            latent_stack = torch.stack(current_inputs).unsqueeze(0)
        else:
            latent_stack = z0.unsqueeze(0).unsqueeze(0)
        inputs_embeds = torch.cat([q_embeds, latent_stack], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=state.device)
        with torch.no_grad():
            outputs = state.phi_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        latent_l = outputs.hidden_states[cfg.layer_l + 1][0, -1, :].detach()
        latent_L = outputs.hidden_states[-1][0, -1, :].detach()
        current_inputs.append(latent_l)
        latents_layer.append(latent_l.cpu())
        latents_final.append(latent_L.cpu())
        stop_logit = state.end_head(latent_L.unsqueeze(0)).squeeze(0)
        stop_prob = torch.sigmoid(stop_logit).item()
        if stop_prob > 0.5:
            break

    latents_l_tensor = torch.stack(latents_layer)
    latents_L_tensor = torch.stack(latents_final)
    return LatentTrace(
        latents_l=latents_l_tensor,
        latents_L=latents_L_tensor,
        k=latents_l_tensor.shape[0],
        meta={
            "backend": "paper",
            "stop_threshold": 0.5,
        },
    )


def decode_answer(
    question: str,
    trace: LatentTrace,
    state: InferState,
    cfg: PaperConfig,
    max_new_tokens: int = 64,
) -> str:
    question_ids, q_embeds = _question_embeddings(question, state)
    latents = trace.latents_l.to(state.device).unsqueeze(0)
    tokenizer = state.tokenizer
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    generated: list[int] = []
    token_id = bos_id
    for _ in range(max_new_tokens):
        token_tensor = torch.tensor([[token_id]], device=state.device)
        token_embed = state.psi_model.get_input_embeddings()(token_tensor)
        inputs_embeds = torch.cat([q_embeds, latents, token_embed], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=state.device)
        with torch.no_grad():
            outputs = state.psi_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        logits = outputs.logits[0, -1, :]
        token_id = int(torch.argmax(logits).item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
