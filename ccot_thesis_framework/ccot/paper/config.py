"""Configuration helpers for the paper-faithful CCOT pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict

from ccot.config import RuntimeConfig


@dataclass
class PaperConfig:
    """Dataclass that mirrors the settings described in the paper."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    layer_l: int | None = None
    scorer_T: int = 3
    compression_ratio: float = 0.12
    max_seq_len: int = 2048
    stop_cap: int = 24
    subset_method: str = "evenly"
    artifacts_dir: Path = Path("artifacts")
    report_path: Path = Path("artifacts/reports/run_report.md")
    teacher_jsonl: Path | None = None
    limit_samples: int | None = None
    csv_path: Path | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    phi_lora_rank: int = 128

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["artifacts_dir"] = str(self.artifacts_dir)
        data["report_path"] = str(self.report_path)
        if self.teacher_jsonl is not None:
            data["teacher_jsonl"] = str(self.teacher_jsonl)
        if self.csv_path is not None:
            data["csv_path"] = str(self.csv_path)
        data["runtime"] = asdict(self.runtime)
        return data

    def gold_dir(self) -> Path:
        path = self.artifacts_dir / "gold"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def models_dir(self) -> Path:
        path = self.artifacts_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def traces_dir(self) -> Path:
        path = self.artifacts_dir / "traces"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def logs_dir(self) -> Path:
        path = self.artifacts_dir / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def stop_limit(self) -> int:
        """Return cap h = 200 * r unless overridden."""

        return self.stop_cap or int(200 * self.compression_ratio)

    def resolved_layer_index(self, num_layers: int) -> int:
        if self.layer_l is not None:
            return max(0, min(self.layer_l, num_layers - 1))
        return max(0, num_layers // 2)

    def resolved_scorer_index(self, num_layers: int) -> int:
        return max(0, min(self.scorer_T, num_layers - 1))
