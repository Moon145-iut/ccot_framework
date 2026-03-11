"""Export paper backend traces for Phase-2 experiments."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from ccot.paper.config import PaperConfig
from ccot.reasoners.ccot_paper import CCOTPaperReasoner


def export_traces(samples: Iterable[dict], cfg: PaperConfig, out_path: str | Path, device: str = "cpu") -> Path:
    """Run inference on samples and write traces JSONL."""

    reasoner = CCOTPaperReasoner(cfg, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            question = sample["question"]
            gold = sample["final_answer"]
            sample_id = sample.get("id")
            start = time.perf_counter()
            trace = reasoner.run_latent(question)
            answer = reasoner.decode_answer(question, trace)
            latency = time.perf_counter() - start
            record = {
                "id": sample_id,
                "question": question,
                "gold_answer": gold,
                "pred_answer": answer,
                "correct": answer.strip() == gold.strip(),
                "latency_sec": latency,
                "k": trace.k,
                "compression_ratio": cfg.compression_ratio,
                "r": cfg.compression_ratio,
                "layer_l": cfg.layer_l,
                "scorer_T": cfg.scorer_T,
                "T": cfg.scorer_T,
                "model_id": cfg.model_id,
                "latents_l": trace.latents_l.tolist(),
                "latents_L": trace.latents_L.tolist(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path
