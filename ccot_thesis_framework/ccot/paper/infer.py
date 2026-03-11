"""Algorithm-2 inference utilities for the paper backend."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ccot.config import resolve_torch_dtype
from ccot.paper.config import PaperConfig
from ccot.reasoners.base import LatentTrace


@dataclass
class InferState:
    tokenizer: AutoTokenizer
    phi_model: PeftModel
    psi_model: PeftModel
    end_head: nn.Module
    device: str
    joint_mode: bool


def _load_tokenizer(cfg: PaperConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_phi_model(cfg: PaperConfig, device: str):
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if cfg.runtime.flash_attention else "eager",
    )
    model.to(device)
    peft_model = PeftModel.from_pretrained(model, cfg.models_dir() / "paper_phi")
    peft_model.to(device)
    peft_model.eval()
    return peft_model


def _load_psi_model(cfg: PaperConfig, device: str) -> tuple[PeftModel, bool]:
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    psi_dir = cfg.models_dir() / "paper_psi"
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if cfg.runtime.flash_attention else "eager",
    )
    base.to(device)
    mode_path = psi_dir / "meta.json"
    joint = False
    if mode_path.exists():
        try:
            joint = json.loads(mode_path.read_text()).get("joint_mode", False)
        except Exception:
            joint = False
    peft_model = PeftModel.from_pretrained(base, psi_dir)
    peft_model.to(device)
    peft_model.eval()
    return peft_model, joint


def _load_end_head(cfg: PaperConfig, device: str) -> nn.Module:
    payload = torch.load(cfg.models_dir() / "paper_end.pt", map_location=device)
    head = nn.Linear(payload["hidden_size"], 1)
    head.load_state_dict(payload["state_dict"])
    head.to(device)
    head.eval()
    return head


def load_infer_state(cfg: PaperConfig) -> InferState:
    device = cfg.runtime.device or "cpu"
    tokenizer = _load_tokenizer(cfg)
    phi_model = _load_phi_model(cfg, device)
    psi_model, joint_mode = _load_psi_model(cfg, device)
    end_head = _load_end_head(cfg, device)
    return InferState(
        tokenizer=tokenizer,
        phi_model=phi_model,
        psi_model=psi_model,
        end_head=end_head,
        device=device,
        joint_mode=joint_mode,
    )


def _question_ids_and_embeds(question: str, state: InferState) -> tuple[torch.Tensor, torch.Tensor]:
    ids = state.tokenizer(question.strip(), add_special_tokens=True)["input_ids"]
    tensor = torch.tensor(ids, dtype=torch.long, device=state.device).unsqueeze(0)
    embeds = state.phi_model.get_input_embeddings()(tensor)
    return tensor, embeds


def _seed_query_feature(question_ids: torch.Tensor, state: InferState, cfg: PaperConfig) -> torch.Tensor:
    with torch.inference_mode():
        outputs = state.phi_model.base_model(
            input_ids=question_ids,
            output_hidden_states=True,
        )
    layer_idx = cfg.resolved_layer_index(len(outputs.hidden_states) - 1)
    return outputs.hidden_states[layer_idx + 1][0, -1, :].detach()


def generate_latents(question: str, state: InferState, cfg: PaperConfig, max_steps: int | None = None) -> LatentTrace:
    question_ids, q_embeds = _question_ids_and_embeds(question, state)
    z0 = _seed_query_feature(question_ids, state, cfg)
    layer_idx = cfg.resolved_layer_index(state.phi_model.config.num_hidden_layers)
    cap = cfg.stop_cap or int(200 * cfg.compression_ratio)
    stop_limit = max(1, min(cap, max_steps) if max_steps else cap)

    latents_l = []
    latents_L = []
    prev_latent = z0
    seq_embeds = q_embeds
    stop_prob = 0.0
    stop_step = 0
    cap_hit = False

    for step in range(stop_limit):
        latent_input = prev_latent.unsqueeze(0).unsqueeze(0)
        inputs_embeds = torch.cat([seq_embeds, latent_input], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=state.device)
        with torch.inference_mode():
            outputs = state.phi_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        latent_l = outputs.hidden_states[layer_idx + 1][0, -1, :].detach()
        latent_final = outputs.hidden_states[-1][0, -1, :].detach()
        latents_l.append(latent_l.cpu())
        latents_L.append(latent_final.cpu())
        stop_logit = state.end_head(latent_final.unsqueeze(0)).squeeze(0)
        stop_prob = torch.sigmoid(stop_logit).item()
        stop_step = step + 1
        seq_embeds = torch.cat([seq_embeds, latent_input], dim=1)
        prev_latent = latent_l
        if stop_prob > 0.5:
            break
    else:
        cap_hit = True

    if latents_l:
        latents_l_tensor = torch.stack(latents_l)
    else:
        latents_l_tensor = z0.new_zeros((0, z0.shape[0])).cpu()
    if latents_L:
        latents_L_tensor = torch.stack(latents_L)
    else:
        latents_L_tensor = z0.new_zeros((0, z0.shape[0])).cpu()

    return LatentTrace(
        latents_l=latents_l_tensor,
        latents_L=latents_L_tensor,
        k=int(latents_l_tensor.shape[0]),
        meta={
            "backend": "paper",
            "compression_ratio": cfg.compression_ratio,
            "autoregressive_layer": layer_idx,
            "scorer_T": cfg.scorer_T,
            "cap_hit": cap_hit,
            "stop_step": stop_step,
            "stop_prob_last": stop_prob,
            "model_id": cfg.model_id,
        },
    )


def decode_answer(
    question: str,
    trace: LatentTrace,
    state: InferState,
    cfg: PaperConfig,
    max_new_tokens: int = 64,
) -> str:
    question_ids, q_embeds = _question_ids_and_embeds(question, state)
    latents = trace.latents_l.to(state.device).unsqueeze(0)
    inputs_embeds = torch.cat([q_embeds, latents], dim=1)
    generated_ids: list[int] = []
    token_id = state.tokenizer.bos_token_id or state.tokenizer.eos_token_id or 0

    for _ in range(max_new_tokens):
        token_tensor = torch.tensor([[token_id]], device=state.device)
        token_embed = state.psi_model.get_input_embeddings()(token_tensor)
        inputs_embeds = torch.cat([inputs_embeds, token_embed], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=state.device)
        with torch.inference_mode():
            outputs = state.psi_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        logits = outputs.logits[0, -1, :]
        token_id = int(torch.argmax(logits).item())
        if token_id == state.tokenizer.eos_token_id:
            break
        generated_ids.append(token_id)

    return state.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_inference(question: str, state: InferState, cfg: PaperConfig, max_steps: int = 200):
    start = time.perf_counter()
    trace = generate_latents(question, state, cfg)
    answer = decode_answer(question, trace, state, cfg)
    latency = time.perf_counter() - start
    return trace, answer, latency
