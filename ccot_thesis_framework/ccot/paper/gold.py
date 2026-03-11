"""Gold latent extraction from frozen theta."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.config import resolve_torch_dtype
from ccot.data import load_gsm8k_csv
from ccot.paper.config import PaperConfig
from ccot.paper.subset import select_indices
from ccot.utils.text import strip_gsm8k_calc_markup


def _clean_rationale(text: str) -> str:
    cleaned = strip_gsm8k_calc_markup(text or "")
    return cleaned.replace("<<", "").replace(">>", "")


def _tokenize(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _prepare_model(cfg: PaperConfig):
    dtype = resolve_torch_dtype(cfg.runtime.dtype)
    attn_impl = "flash_attention_2" if cfg.runtime.flash_attention else "eager"
    device = cfg.runtime.device or "cpu"
    device_map = "auto" if device.startswith("cuda") else None
    if device.startswith("cuda") and cfg.runtime.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    if device_map is None:
        model.to(device)
    if cfg.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.eval()
    return tokenizer, model, dtype, device, device_map


def build_gold_cache(cfg: PaperConfig, limit: int | None = None) -> dict:
    """Extract layer-wise gold latents using a frozen model theta."""

    if cfg.csv_path is None:
        raise ValueError("PaperConfig.csv_path must be provided for gold extraction.")

    tokenizer, model, dtype, device, device_map = _prepare_model(cfg)
    examples = load_gsm8k_csv(cfg.csv_path)
    if limit is not None:
        examples = examples[:limit]
    elif cfg.limit_samples is not None:
        examples = examples[: cfg.limit_samples]

    num_layers = int(model.config.num_hidden_layers)
    layer_idx = cfg.resolved_layer_index(num_layers)
    scorer_idx = cfg.resolved_scorer_index(num_layers)

    records: list[dict] = []
    autocast_enabled = device.startswith("cuda") and dtype in {torch.float16, torch.bfloat16}

    for ex in examples:
        question = ex.question.strip()
        rationale = _clean_rationale(ex.rationale)
        answer = ex.final_answer.strip()
        q_ids = _tokenize(tokenizer, question)
        r_ids = _tokenize(tokenizer, rationale)
        a_ids = _tokenize(tokenizer, answer)
        if not r_ids:
            continue

        tokens: List[int] = []
        if tokenizer.bos_token_id is not None:
            tokens.append(tokenizer.bos_token_id)
        question_start = len(tokens)
        tokens.extend(q_ids)
        question_last_idx = max(question_start, len(tokens) - 1)
        rationale_start = len(tokens)
        tokens.extend(r_ids)
        answer_start = len(tokens)
        tokens.extend(a_ids)
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)

        if len(tokens) > cfg.max_seq_len:
            tokens = tokens[: cfg.max_seq_len]
            if rationale_start >= len(tokens):
                continue
            truncated_len = len(tokens) - rationale_start
            if truncated_len <= 0:
                continue
            r_ids = r_ids[: truncated_len]

        m = len(r_ids)
        if m == 0:
            continue
        k = math.ceil(cfg.compression_ratio * m)
        if k <= 0:
            continue
        rationale_positions = list(range(rationale_start, rationale_start + m))
        q_last_idx = min(question_last_idx, len(tokens) - 1)

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        if device_map is None:
            input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            ctx = torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled) if autocast_enabled else nullcontext()
            with ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
        hidden_states = outputs.hidden_states  # tuple length num_layers+1
        scorer_hidden = hidden_states[scorer_idx + 1][0, rationale_positions, :].detach().cpu()
        selected_rel = select_indices(scorer_hidden, k, method=cfg.subset_method)
        if selected_rel.size == 0:
            continue
        selected_abs = np.array(rationale_positions)[selected_rel]

        gold_per_layer: list[torch.Tensor] = []
        indices_list = selected_abs.tolist()
        for layer in range(num_layers):
            layer_hidden = (
                hidden_states[layer + 1][0, indices_list, :].detach().cpu()
            )
            gold_per_layer.append(layer_hidden)
        z0 = hidden_states[layer_idx + 1][0, q_last_idx, :].detach().cpu()
        z_gold_l = gold_per_layer[layer_idx]

        question_segment = tokens[:rationale_start]
        if not question_segment:
            continue
        records.append(
            {
                "id": ex.idx,
                "question": question,
                "answer": answer,
                "k": int(len(indices_list)),
                "indices": torch.tensor(indices_list, dtype=torch.long),
                "rationale_len": m,
                "z0": z0,
                "z_gold_l": z_gold_l,
                "gold_per_layer": gold_per_layer,
                "compression_ratio": cfg.compression_ratio,
                "question_ids": question_segment,
            }
        )

    if not records:
        raise RuntimeError("No valid samples for gold extraction; check max_seq_len or inputs.")

    split = max(1, int(len(records) * 0.9))
    train_records = records[:split]
    val_records = records[split:]
    out_dir = cfg.gold_dir()
    torch.save(train_records, out_dir / "paper_train.pt")
    torch.save(val_records, out_dir / "paper_val.pt")
    meta = {
        "model_id": cfg.model_id,
        "hidden_size": int(model.config.hidden_size),
        "num_layers": num_layers,
        "compression_ratio": cfg.compression_ratio,
        "scorer_T": cfg.scorer_T,
        "layer_l": layer_idx,
        "subset_method": cfg.subset_method,
        "max_seq_len": cfg.max_seq_len,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "csv_path": str(cfg.csv_path),
    }
    meta_path = out_dir / "paper_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def build_gold_targets(
    csv_path: str | Path,
    cfg: PaperConfig,
    device: str = "cpu",
    limit: int | None = None,
) -> dict:
    """Compatibility wrapper for legacy callers."""

    cfg.csv_path = Path(csv_path)
    cfg.runtime.device = device
    return build_gold_cache(cfg, limit=limit)
