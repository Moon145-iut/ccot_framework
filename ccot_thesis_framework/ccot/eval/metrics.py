"""Evaluation metrics utilities used by all backends."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class SampleRecord:
    """Per-sample evaluation record."""

    id: int | str | None
    question: str
    gold_answer: str
    pred_answer: str
    correct: bool
    latency_sec: float
    backend: str
    reasoning_steps: int | None
    stop_step: int | None
    hit_max_steps: bool | None
    answer_extracted: bool
    compression_ratio_r: float | None
    autoregressive_layer_l: int | None
    alpha: float | None

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False)


def write_predictions(path: Path, records: Iterable[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def aggregate_metrics(records: List[SampleRecord]) -> dict:
    if not records:
        return {}
    latencies = np.array([r.latency_sec for r in records], dtype=np.float64)
    total_time = latencies.sum()
    reasoning_steps = np.array(
        [r.reasoning_steps or 0 for r in records], dtype=np.float64
    )
    cap_hits = np.array(
        [1.0 if (r.hit_max_steps or False) else 0.0 for r in records], dtype=np.float64
    )
    answer_present = np.array(
        [1.0 if r.answer_extracted else 0.0 for r in records], dtype=np.float64
    )
    metrics = {
        "accuracy_em": float(
            sum(1 for r in records if r.correct) / max(1, len(records))
        ),
        "avg_latency_sec": float(latencies.mean()),
        "p50_latency_sec": float(np.percentile(latencies, 50)),
        "p95_latency_sec": float(np.percentile(latencies, 95)),
        "throughput_samples_per_sec": float(len(records) / max(total_time, 1e-8)),
        "avg_reasoning_steps": float(reasoning_steps.mean()),
        "cap_hit_rate": float(cap_hits.mean()),
        "answer_extraction_rate": float(answer_present.mean()),
    }
    return metrics


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
