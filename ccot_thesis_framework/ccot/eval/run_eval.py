"""Shared evaluation runner for CCOT reasoners."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from ccot.config import RuntimeConfig, DEFAULT_MODEL_ID
from ccot.data import load_gsm8k_csv
from ccot.eval.metrics import (
    SampleRecord,
    aggregate_metrics,
    write_metrics,
    write_predictions,
)
from ccot.paper.config import PaperConfig
from ccot.reasoners.ccot_cpu_gru import CCOTCpuGRUReasoner
from ccot.reasoners.ccot_truth_vector import CCOTTruthVectorReasoner
from ccot.reasoners.ccot_paper import CCOTPaperReasoner


def _paper_config_from_args(args) -> PaperConfig:
    artifacts_dir = Path(getattr(args, "artifacts_dir", "artifacts"))
    cfg = PaperConfig(
        model_id=getattr(args, "model_id", DEFAULT_MODEL_ID),
        artifacts_dir=artifacts_dir,
    )
    meta_path = artifacts_dir / "gold" / "paper_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cfg.layer_l = meta.get("layer_l")
            cfg.scorer_T = meta.get("scorer_T", cfg.scorer_T)
            cfg.compression_ratio = meta.get("compression_ratio", cfg.compression_ratio)
            cfg.max_seq_len = meta.get("max_seq_len", cfg.max_seq_len)
        except Exception:
            pass
    if getattr(args, "paper_r", None) is not None:
        cfg.compression_ratio = args.paper_r
    cfg.csv_path = Path(args.csv)
    cfg.runtime = RuntimeConfig(
        device=getattr(args, "device", "cpu"),
        dtype=getattr(args, "dtype", "float32"),
        flash_attention=getattr(args, "flash_attn", False),
        gradient_checkpointing=getattr(args, "grad_ckpt", False),
        tf32=not getattr(args, "no_tf32", False),
    )
    return cfg


def _build_reasoner(args, backend: str):
    if backend == "cpu_gru":
        required = ["targets_dir", "ccot_weights", "decoder_weights"]
        missing = [name for name in required if not getattr(args, name, None)]
        if missing:
            raise ValueError(f"{backend} backend requires: {', '.join(missing)}")
        return CCOTCpuGRUReasoner(
            targets_dir=args.targets_dir,
            ccot_weights=args.ccot_weights,
            decoder_weights=args.decoder_weights,
            stop_threshold=args.stop_threshold,
            max_latents=args.max_latents,
            num_threads=args.num_threads,
        )
    if backend == "truth_vector":
        required = [
            "targets_dir",
            "ccot_weights",
            "decoder_weights",
            "truth_vector",
        ]
        missing = [name for name in required if not getattr(args, name, None)]
        if missing:
            raise ValueError(f"{backend} backend requires: {', '.join(missing)}")
        return CCOTTruthVectorReasoner(
            targets_dir=args.targets_dir,
            ccot_weights=args.ccot_weights,
            decoder_weights=args.decoder_weights,
            truth_vector_path=args.truth_vector,
            alpha=args.truth_alpha,
            stop_threshold=args.stop_threshold,
            max_latents=args.max_latents,
            num_threads=args.num_threads,
        )
    if backend == "paper":
        cfg = _paper_config_from_args(args)
        return CCOTPaperReasoner(cfg, device=cfg.runtime.device)
    raise NotImplementedError("Unsupported backend.")


def run_eval(args, backend: str) -> dict:
    dataset = load_gsm8k_csv(args.csv)
    subset = dataset[: args.n] if args.n else dataset
    reasoner = _build_reasoner(args, backend)
    records: list[SampleRecord] = []
    for sample in subset:
        start = time.perf_counter()
        trace = reasoner.run_latent(sample.question, max_steps=args.max_latents)
        answer = reasoner.decode_answer(sample.question, trace)
        latency = time.perf_counter() - start
        meta = trace.meta or {}
        record = SampleRecord(
            id=sample.idx,
            question=sample.question,
            gold_answer=sample.final_answer,
            pred_answer=answer,
            correct=answer.strip() == sample.final_answer.strip(),
            latency_sec=latency,
            backend=backend,
            reasoning_steps=trace.k,
            stop_step=meta.get("stop_step"),
            hit_max_steps=meta.get("cap_hit"),
            answer_extracted=bool(answer.strip()),
            compression_ratio_r=meta.get("compression_ratio"),
            autoregressive_layer_l=meta.get("autoregressive_layer"),
            alpha=meta.get("truth_steering", {}).get("alpha")
            if meta.get("truth_steering")
            else meta.get("alpha"),
        )
        records.append(record)

    eval_dir = Path(args.artifacts_dir) / "eval"
    write_predictions(eval_dir / "predictions.jsonl", records)
    metrics = aggregate_metrics(records)
    write_metrics(eval_dir / "metrics.json", metrics)
    return metrics
