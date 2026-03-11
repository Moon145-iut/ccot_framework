"""Export Algorithm-2 traces for downstream evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ccot.paper.config import PaperConfig
from ccot.reasoners.base import LatentTrace


def export_traces(
    records: Iterable[dict],
    cfg: PaperConfig,
    out_path: str | Path,
) -> Path:
    """Write traces as JSONL with reproducibility metadata."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            trace: LatentTrace = record["trace"]
            payload = {
                "id": record.get("id"),
                "question": record["question"],
                "gold_answer": record.get("gold_answer"),
                "pred_answer": record.get("pred_answer"),
                "correct": record.get("correct"),
                "k": trace.k,
                "layer_l": cfg.layer_l,
                "scorer_T": cfg.scorer_T,
                "r": cfg.compression_ratio,
                "model_id": cfg.model_id,
                "latents_l": trace.latents_l.tolist(),
                "latents_L": trace.latents_L.tolist(),
                "config_hash": digest,
            }
            handle.write(json.dumps(payload) + "\n")
    return out_path
