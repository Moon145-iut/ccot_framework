"""Gold latent extraction from frozen θ."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.data import load_gsm8k_csv
from ccot.paper.config import PaperConfig
from ccot.paper.subset import select_subset


def _concat_segments(tokenizer, question: str, rationale: str, answer: str) -> tuple[List[int], tuple[int, int]]:
    q_ids = tokenizer(question.strip(), add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(rationale.strip(), add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    tokens = []
    if bos is not None:
        tokens.append(bos)
    tokens.extend(q_ids)
    question_end = len(tokens) - 1
    tokens.extend(r_ids)
    rationale_start = question_end + 1
    tokens.extend(a_ids)
    if eos is not None:
        tokens.append(eos)
    rationale_end = rationale_start + len(r_ids)
    return tokens, (rationale_start, rationale_end)


def build_gold_targets(
    csv_path: str | Path,
    cfg: PaperConfig,
    device: str = "cpu",
    limit: int | None = None,
) -> dict:
    """Extract layer-wise gold latents using a frozen model θ."""

    csv_path = Path(csv_path)
    out_dir = cfg.gold_dir()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    model.to(device)
    model.eval()

    examples = load_gsm8k_csv(csv_path)
    if limit is not None:
        examples = examples[:limit]

    records: list[dict] = []
    hidden_size = int(model.config.hidden_size)

    for example in examples:
        tokens, (rs, re) = _concat_segments(
            tokenizer, example.question, example.rationale, example.final_answer
        )
        if re <= rs:
            continue
        if len(tokens) > cfg.max_seq_len:
            tokens = tokens[: cfg.max_seq_len]
            if rs >= len(tokens):
                continue
            re = min(re, len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        rationale_hidden = hidden_states[cfg.layer_l + 1][0, rs:re, :].cpu()
        rationale_len = rationale_hidden.shape[0]
        if rationale_len == 0:
            continue
        k = max(1, math.ceil(cfg.compression_ratio * rationale_len))
        selected_layer, indices = select_subset(
            hidden_states[cfg.scorer_T + 1][0, rs:re, :].cpu(),
            k=k,
            method=cfg.subset_method,
        )
        layer_latents = hidden_states[cfg.layer_l + 1][0, rs:re, :].cpu()[indices]
        final_latents = hidden_states[-1][0, rs:re, :].cpu()[indices]
        q_last_idx = min(len(tokens) - 1, rs - 1)
        z0 = hidden_states[cfg.layer_l + 1][0, q_last_idx, :].cpu()
        records.append(
            {
                "id": example.idx,
                "question": example.question,
                "answer": example.final_answer,
                "k": k,
                "rationale_len": rationale_len,
                "indices": indices.tolist(),
                "subset_latents": selected_layer,
                "layer_latents": layer_latents,
                "final_latents": final_latents,
                "z0": z0,
                "question_tokens": tokens[: rs],
                "attention_len": len(tokens),
            }
        )

    if not records:
        raise RuntimeError("No valid samples for gold target extraction. Check max_seq_len.")

    split = max(1, int(len(records) * 0.9))
    train_records = records[:split]
    val_records = records[split:]
    torch.save(train_records, out_dir / "paper_train.pt")
    torch.save(val_records, out_dir / "paper_val.pt")
    meta = {
        "model_id": cfg.model_id,
        "hidden_size": hidden_size,
        "layer_l": cfg.layer_l,
        "scorer_T": cfg.scorer_T,
        "compression_ratio": cfg.compression_ratio,
        "subset_method": cfg.subset_method,
        "num_train": len(train_records),
        "num_val": len(val_records),
    }
    import json

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
