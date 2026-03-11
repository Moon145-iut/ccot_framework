"""Build compressed hidden targets from teacher supervision."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

import torch

from ccot.config import DEFAULT_MODEL_ID, DEFAULT_COMPRESSION_RATIO, DEFAULT_HIDDEN_LAYER, DEFAULT_NUM_THREADS, DEFAULT_MAX_LENGTH
from ccot.local.backbone import LocalBackbone
from ccot.local.feature_extractor import HiddenTargetBuilder
from ccot.utils.io import ensure_directory, read_jsonl, write_jsonl


def build_hidden_targets(
    teacher_jsonl: str | Path,
    out_dir: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
    selector: str = "evenly_spaced",
    hidden_layer_index: int = DEFAULT_HIDDEN_LAYER,
    num_threads: int = DEFAULT_NUM_THREADS,
    max_length: int = DEFAULT_MAX_LENGTH,
    limit: Optional[int] = None,
) -> dict:
    teacher_jsonl = Path(teacher_jsonl)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)
    rows = read_jsonl(teacher_jsonl)
    if limit is not None:
        rows = rows[:limit]

    backbone = LocalBackbone(
        model_id=model_id,
        device="cpu",
        num_threads=num_threads,
        max_length=max_length,
    )
    builder = HiddenTargetBuilder(
        backbone,
        compression_ratio=compression_ratio,
        selector=selector,
        hidden_layer_index=hidden_layer_index,
    )

    q_feats_list = []
    z_targets_list = []
    z_lengths = []
    sample_ids = []
    sample_rows = []

    for row in rows:
        target = builder.build(row["question"], row["rationale"], row["final_answer"])
        q_feats_list.append(target.q_feat.float().numpy())
        z_targets_list.append(target.z_targets.float().numpy())
        z_lengths.append(target.z_len)
        sample_ids.append(int(row["id"]))
        sample_rows.append(
            {
                "id": int(row["id"]),
                "question": row["question"],
                "answer_text": row["final_answer"],
                "rationale_token_count": target.rationale_token_count,
                "selected_indices": target.selected_indices,
            }
        )

    if not q_feats_list:
        raise RuntimeError("No teacher rows were processed")

    hidden_size = backbone.hidden_size
    max_latent = max(z.shape[0] for z in z_targets_list)
    q_feats = np.stack(q_feats_list).astype(np.float32)
    z_padded = np.zeros((len(z_targets_list), max_latent, hidden_size), dtype=np.float32)
    for i, (z, length) in enumerate(zip(z_targets_list, z_lengths)):
        z_padded[i, :length, :] = z

    np.savez(out_dir / "targets.npz", q_feats=q_feats, z_targets=z_padded, z_lengths=np.array(z_lengths, dtype=np.int32), sample_ids=np.array(sample_ids, dtype=np.int32))
    write_jsonl(out_dir / "samples.jsonl", sample_rows)
    meta = {
        "model_id": model_id,
        "hidden_size": hidden_size,
        "compression_ratio": compression_ratio,
        "selector": selector,
        "hidden_layer_index": hidden_layer_index,
        "num_samples": len(q_feats_list),
        "max_latent": max_latent,
        "max_length": max_length,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
