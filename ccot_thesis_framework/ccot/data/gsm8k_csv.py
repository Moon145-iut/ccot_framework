"""GSM8K CSV loader utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ccot.utils.text import (
    normalize_numeric_answer,
    split_gsm8k_answer,
    strip_gsm8k_calc_markup,
)


@dataclass(slots=True)
class GSM8KExample:
    """Parsed GSM8K entry with cleaned rationale and answer."""

    idx: int
    question: str
    rationale: str
    final_answer: str
    raw_answer: str


REQUIRED_COLUMNS = {"question", "answer"}


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required GSM8K columns {sorted(missing)} in file with columns {list(df.columns)}"
        )


def _iter_rows(df: pd.DataFrame) -> Iterable[tuple[int, str, str]]:
    for idx, row in df.iterrows():
        yield int(idx), str(row["question"]).strip(), str(row["answer"]).strip()


def load_gsm8k_csv(csv_path: str | Path) -> list[GSM8KExample]:
    """Load and normalize GSM8K CSV examples."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _ensure_columns(df)

    examples: list[GSM8KExample] = []
    for idx, question, raw_answer in _iter_rows(df):
        rationale, final_answer = split_gsm8k_answer(raw_answer)
        rationale = strip_gsm8k_calc_markup(rationale).strip()
        cleaned_final = normalize_numeric_answer(strip_gsm8k_calc_markup(final_answer))
        examples.append(
            GSM8KExample(
                idx=idx,
                question=question,
                rationale=rationale,
                final_answer=cleaned_final,
                raw_answer=raw_answer,
            )
        )

    return examples
