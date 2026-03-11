"""Utilities for downloading GSM8K splits from Hugging Face."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset

SplitName = Literal["train", "test"]


def download_gsm8k_csv(
    out_dir: str | Path,
    *,
    split: SplitName = "train",
    repo_id: str = "openai/gsm8k",
    revision: str | None = None,
) -> Path:
    """Download a GSM8K split via Hugging Face Datasets and save as CSV.

    Args:
        out_dir: Directory to store the resulting CSV file.
        split: Which split ("train" or "test") to download.
        repo_id: Hugging Face dataset repo id.
        revision: Optional git revision/tag.

    Returns:
        Path to the written CSV file.
    """

    dataset = load_dataset(repo_id, split=split, revision=revision)
    if "question" not in dataset.column_names or "answer" not in dataset.column_names:
        raise RuntimeError(
            f"GSM8K dataset at {repo_id} missing expected columns. Columns={dataset.column_names}"
        )

    df = pd.DataFrame(dataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gsm8k_{split}.csv"
    df.to_csv(out_path, index=False)
    return out_path
