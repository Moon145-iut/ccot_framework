"""Markdown run report generator."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List


CHECKLIST = [
    ("Gold from frozen θ", True),
    ("T subset selection on scorer layer", True),
    ("Layer l approx L/2", True),
    ("k=ceil(r*m) with cap h=200r", True),
    ("Variance-scaled MSE", True),
    ("END trained on final layer states", True),
    ("ψ joint φ+ψ training", None),
]


def _format_checklist(joint_supported: bool, deviations: List[str]) -> str:
    lines = []
    for item, default in CHECKLIST:
        if item == "ψ joint φ+ψ training":
            status = "YES" if joint_supported else "NO"
        elif default is True:
            status = "YES"
        else:
            status = "NO"
        lines.append(f"- [{status}] {item}")
    if deviations:
        lines.append("\n### Deviations")
        for dev in deviations:
            lines.append(f"- {dev}")
    return "\n".join(lines)


def write_run_report(
    backend: str,
    cfg: Dict,
    report_data: Dict,
    deviations: List[str],
    path: str | Path,
) -> Path:
    """Create a Markdown report summarizing the full run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().isoformat() + "Z"
    config_block = json.dumps(cfg, indent=2)
    phi_losses = report_data.get("phi_losses", {})
    end_losses = report_data.get("end_losses", [])
    psi_losses = report_data.get("psi_losses", [])
    eval_metrics = report_data.get("eval", {})
    avg_latency = report_data.get("avg_latency", "n/a")
    avg_k = report_data.get("avg_k", "n/a")
    dataset_sizes = report_data.get("dataset_sizes", {})
    joint_supported = report_data.get("joint_training", False)

    checklist = _format_checklist(joint_supported, deviations)
    body = f"""# CCOT Run Report

**Timestamp:** {timestamp}

**Backend:** {backend}

## Configuration
```json
{config_block}
```

## Dataset Sizes
- Train: {dataset_sizes.get('train', 'n/a')}
- Validation: {dataset_sizes.get('val', 'n/a')}

## Training Summaries
- φ losses: {phi_losses}
- END loss history: {end_losses}
- ψ loss history: {psi_losses}

## Evaluation
- Exact Match: {eval_metrics.get('em', 'n/a')}
- Avg Latency (ms): {avg_latency}
- Avg Latent Count k: {avg_k}

## Paper Alignment Audit
{checklist}
"""
    path.write_text(body, encoding="utf-8")
    return path
