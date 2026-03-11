"""Helpers for partitioning GSM8K into Truth Vector experiment splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random

from .gsm8k_csv import GSM8KExample


@dataclass(slots=True)
class TruthVectorSplits:
    """Container grouping the disjoint subsets required by the research protocol."""

    base_train: list[GSM8KExample]
    vector_extraction: list[GSM8KExample]
    further_training: list[GSM8KExample]
    held_out: list[GSM8KExample]


def build_truth_vector_splits(
    examples: Sequence[GSM8KExample],
    *,
    base_size: int = 6_000,
    vector_size: int = 500,
    further_size: int = 1_500,
    seed: int = 42,
) -> TruthVectorSplits:
    """Shuffle GSM8K examples into the four protocol splits.

    Args:
        examples: Full GSM8K dataset (already cleaned).
        base_size: Number of rows reserved for Phase-1 base training.
        vector_size: Rows reserved for Phase-2 truth-vector extraction.
        further_size: Rows for Phase-3/4 follow-up experiments.
        seed: Deterministic RNG seed to make splits reproducible.
    """

    total_needed = base_size + vector_size + further_size
    if len(examples) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} examples but only received {len(examples)}."
        )

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    def _take(count: int) -> list[GSM8KExample]:
        picked = indices[:count]
        del indices[:count]
        return [examples[i] for i in picked]

    base_train = _take(base_size)
    vector_extraction = _take(vector_size)
    further_training = _take(further_size)
    held_out = [examples[i] for i in indices]

    return TruthVectorSplits(
        base_train=base_train,
        vector_extraction=vector_extraction,
        further_training=further_training,
        held_out=held_out,
    )
