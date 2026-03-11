"""Prepare teacher JSONL from GSM8K CSV and optional API."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ccot.data.gsm8k_csv import load_gsm8k_csv
from ccot.providers.factory import create_provider
from ccot.providers.base import ChatMessage
from ccot.utils.io import write_jsonl, ensure_directory
from ccot.utils.text import normalize_numeric_answer, prompt_for_teacher, split_gsm8k_answer


def prepare_teacher_jsonl(
    csv_path: str | Path,
    out_path: str | Path,
    provider_name: str | None = None,
    teacher_model: str | None = None,
    base_url: str | None = None,
    use_api_for_rationale: bool = False,
    limit: Optional[int] = None,
) -> Path:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    ensure_directory(out_path.parent)
    examples = load_gsm8k_csv(csv_path)
    rows = []
    provider = None
    if use_api_for_rationale:
        if not provider_name:
            raise ValueError("provider_name must be specified when use_api_for_rationale is True")
        provider = create_provider(provider_name, base_url=base_url)
    if provider_name and provider_name.lower() == "gemini":
        default_model = "gemini-1.5-pro-latest"
    else:
        default_model = "gpt-4o-mini"
    model_name = teacher_model or default_model

    for example in examples:
        if limit is not None and len(rows) >= limit:
            break
        rationale = example.rationale
        final_answer = example.final_answer
        teacher_source = "dataset"
        if use_api_for_rationale and provider is not None:
            try:
                prompt = prompt_for_teacher(example.question)
                response = provider.generate(
                    [ChatMessage(role="user", content=prompt)],
                    model=model_name,
                )
                parsed_rationale, parsed_final = split_gsm8k_answer(response)
                parsed_final = normalize_numeric_answer(parsed_final)
                if parsed_rationale and parsed_final:
                    rationale = parsed_rationale
                    final_answer = parsed_final
                    teacher_source = f"api:{provider_name}"
            except Exception as exc:  # pragma: no cover - defensive
                teacher_source = f"dataset-fallback:{exc.__class__.__name__}"
        rows.append(
            {
                "id": example.idx,
                "question": example.question,
                "rationale": rationale,
                "final_answer": final_answer,
                "teacher_source": teacher_source,
            }
        )

    write_jsonl(out_path, rows)
    return out_path
