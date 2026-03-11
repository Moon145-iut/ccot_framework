"""Global configuration defaults for the CCOT thesis framework."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_COMPRESSION_RATIO = 0.1
DEFAULT_SELECTOR = "evenly_spaced"
DEFAULT_HIDDEN_LAYER = -2
DEFAULT_NUM_THREADS = 4
DEFAULT_MAX_LENGTH = 1024
DEFAULT_RANDOM_SEED = 42


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU runs."""

    import random
    import numpy as np

    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)


def ensure_torch_threads(num_threads: int) -> None:
    """Force PyTorch to use a deterministic number of CPU threads."""

    torch.set_num_threads(max(1, num_threads))


@dataclass(slots=True)
class PipelineDefaults:
    """Convenience dataclass that mirrors CLI defaults."""

    model_id: str = DEFAULT_MODEL_ID
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO
    selector: str = DEFAULT_SELECTOR
    hidden_layer_index: int = DEFAULT_HIDDEN_LAYER
    num_threads: int = DEFAULT_NUM_THREADS
    max_length: int = DEFAULT_MAX_LENGTH
    random_seed: int = DEFAULT_RANDOM_SEED
    artifacts_dir: Path = Path("artifacts")

def build_paths(base_dir: str | Path, *extra: str) -> Path:
    """Return a path rooted under ``base_dir`` while creating parents."""

    path = Path(base_dir).joinpath(*extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def to_serializable_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion from dataclass or namespace to ``dict``."""

    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: to_serializable_dict(v) for k, v in obj.__dict__.items()}
    return dict(obj)
