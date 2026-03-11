"""IO helpers for JSON Lines artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as ``Path``."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return the decoded rows."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write rows to JSON Lines, creating parent directories when needed."""

    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_directory(path: str | Path) -> list[str]:
    """Return sorted child names for quick debugging."""

    return sorted(str(p) for p in Path(path).iterdir())
