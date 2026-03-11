"""Truth vector computation for Phase-2 experiments."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def build_truth_vector(traces_path: str | Path, out_path: str | Path) -> Path:
    """Compute v_truth = mean(H+) - mean(H-) from exported traces."""

    traces_path = Path(traces_path)
    positives = []
    negatives = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tensor = torch.tensor(row["latents_L"], dtype=torch.float32)
            if row.get("correct"):
                positives.append(tensor)
            else:
                negatives.append(tensor)
    if not positives or not negatives:
        raise RuntimeError("Need both positive and negative traces to compute truth vector")
    pos_mean = torch.cat(positives).mean(dim=0)
    neg_mean = torch.cat(negatives).mean(dim=0)
    truth = pos_mean - neg_mean
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(truth, out_path)
    return out_path
