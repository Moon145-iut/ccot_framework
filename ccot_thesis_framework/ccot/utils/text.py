"""Text utilities for GSM8K style data."""
from __future__ import annotations

import re
from typing import Tuple

CALC_PATTERN = re.compile(r"<<.*?>>")


def strip_gsm8k_calc_markup(text: str) -> str:
    """Remove calculator annotations such as ``<< 45 >>`` from the string."""

    return CALC_PATTERN.sub("", text or "").strip()


def split_gsm8k_answer(text: str) -> tuple[str, str]:
    """Split GSM8K rationale from its final answer marker."""

    if text is None:
        return "", ""
    marker = "####"
    if marker not in text:
        return text.strip(), ""
    rationale, final = text.split(marker, 1)
    return rationale.strip(), final.strip()


def normalize_numeric_answer(text: str) -> str:
    """Normalize numeric answers by removing commas and trimming spaces."""

    if not text:
        return ""
    cleaned = text.replace(",", "").replace(" ", "")
    return cleaned.strip()


def prompt_for_teacher(question: str) -> str:
    """Return an instruction prompt for API-based teacher rationale generation."""

    return (
        "You are a careful math tutor who writes concise arithmetic rationales. "
        "Solve the following question step by step. End your response with ``#### <final_answer>`` on the final line.\n"
        f"Question:\n{question.strip()}\n"
        "Reasoning:"
    )
