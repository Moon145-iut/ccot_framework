# CCOT Thesis Framework

A modular, thesis-ready workspace for Compressed Chain-of-Thought (CCOT) reasoning on GSM8K. The repository now exposes a **reasoner interface** with two interchangeable backends:

1. `cpu_gru` – the original lightweight GRU latent generator + char-level decoder (CPU-friendly, great for VS Code notebooks).
2. `paper` – a paper-faithful CCOT pipeline with f LoRA adapters, END? stop head, ? decoding, trace export, and reporting aligned with Cheng & Van Durme (2024, arXiv:2412.13171).

APIs (Gemini, OpenAI-compatible, Qwen servers) are **only** used to generate teacher rationales/answers. Hidden-state extraction, gold latent creation, f/END?/? training, inference, Phase-2 truth vectors, and reporting all require local HuggingFace models so we can call `model.get_input_embeddings()` and `model(..., output_hidden_states=True)`. Trying to swap in an API for these steps raises a descriptive error.

## Project Structure
```
ccot_thesis_framework/
+-- .env.example
+-- README.md
+-- requirements.txt
+-- run_demo.bat
+-- artifacts/
¦   +-- teacher.jsonl
¦   +-- reports/run_report.md (written automatically after full runs)
+-- ccot/
    +-- cli.py                 # CLI with backend + paper commands
    +-- reasoners/             # LatentReasoner interface + backend adapters
    +-- paper/                 # f, END?, ? training + Algorithm 2 inference
    +-- phase2/                # Truth vector utilities
    +-- report/                # Markdown report writer
    +-- data/, local/, models/, pipeline/, providers/, training/, utils/
    +-- ...
```
Each module is importable (`python -m ccot.cli ...`).

## Installation
```bat
cd ccot_thesis_framework
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
(macOS/Linux: `source .venv/bin/activate`.)

Copy `.env.example` ? `.env` and fill any teacher API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`).

## Backend Overview
| Backend    | Description | Typical Flow |
|------------|-------------|--------------|
| `cpu_gru`  | Legacy GRU latent generator + char decoder (CPU). | `prepare-teacher ? build-targets ? train-ccot ? train-decoder ? infer` or `python -m ccot.cli full-run --backend cpu_gru --csv ...`
| `paper`    | Paper-faithful f/END?/? pipeline with LoRA adapters and END stop head. | `paper-build-gold`, `paper-train-phi`, `paper-train-end`, `paper-train-psi`, `paper-eval`, `paper-export-traces`, or `python -m ccot.cli paper-full-run --csv ...`

Both backends comply with `LatentReasoner` so downstream evaluation/Phase-2 logic never changes.

### Example Commands
```bash
# Teacher prep (dataset rationale or optional API teacher)
python -m ccot.cli prepare-teacher --csv ccot/data/gsm8k_random_300.csv --out artifacts/teacher.jsonl

# Baseline cpu_gru flow
python -m ccot.cli build-targets --teacher-jsonl artifacts/teacher.jsonl --out-dir artifacts/targets
python -m ccot.cli train-ccot --targets-dir artifacts/targets --out-dir artifacts/ccot_weights
python -m ccot.cli train-decoder --targets-dir artifacts/targets --ccot-weights artifacts/ccot_weights/latent_generator.pt --out-dir artifacts/decoder_weights
python -m ccot.cli infer --backend cpu_gru --question "Lena buys 2 apples after owning 3." --targets-dir artifacts/targets --ccot-weights artifacts/ccot_weights/latent_generator.pt --decoder-weights artifacts/decoder_weights/char_decoder.pt

# Paper backend (per-stage)
python -m ccot.cli paper-build-gold --csv ccot/data/gsm8k_random_300.csv --device cuda --limit-samples 64
python -m ccot.cli paper-train-phi --device cuda
python -m ccot.cli paper-train-end --device cuda
python -m ccot.cli paper-train-psi --device cuda --joint-training
python -m ccot.cli paper-eval --csv ccot/data/gsm8k_random_300.csv --device cuda --limit-samples 16
python -m ccot.cli paper-export-traces --csv ccot/data/gsm8k_random_300.csv

# One-shot orchestrations (writes artifacts/reports/run_report.md)
python -m ccot.cli full-run --backend cpu_gru --csv ccot/data/gsm8k_random_300.csv --limit-samples 64
python -m ccot.cli paper-full-run --csv ccot/data/gsm8k_random_300.csv --limit-samples 30 --device cuda
```

The full-run commands:
1. Build teacher data (if needed).
2. Run the appropriate latent pipeline (cpu or paper).
3. Evaluate on the requested subset, export traces, compute Phase-2 truth vectors.
4. Write `artifacts/reports/run_report.md` containing config JSON, dataset sizes, f/END?/? loss summaries, EM, latency, avg `k`, a paper-alignment checklist, and any deviations (e.g., ? fallback training, smaller model, reduced LoRA rank).

## Teacher API Usage (Teacher Stage Only)
- `--provider gemini --teacher-model gemini-1.5-pro --use-api-for-rationale` (requires `GEMINI_API_KEY`).
- `--provider openai_compat --base-url http://localhost:8000/v1 --teacher-model qwen2.5` (optional `OPENAI_API_KEY`).

> APIs cannot supply hidden states, latent compression, or ? decoding. Local HuggingFace models are required for every phase after teacher preparation. The CLI emits explicit errors if you try to run latent extraction or Phase-2 metrics in API-only mode.

## Troubleshooting & Tips
- **Missing CSV columns:** Ensure `question` / `answer` exist in your GSM8K CSV.
- **API failures:** Check `.env` keys, model names, or base URLs. Errors surface with HTTP status codes.
- **HF model download:** First run downloads weights; ensure disk space + internet.
- **cpu_gru performance:** Tweak `--num-threads` / `--limit` for quick smoke tests.
- **paper backend resources:** LoRA training benefits from GPUs. If you must stay on CPU, pass small `--limit-samples` to each command.
- **Reports:** Inspect `artifacts/reports/run_report.md` after every `full-run`/`paper-full-run` to audit paper-alignment and deviations (e.g., ? joint training disabled).

## Demo Script
`run_demo.bat` still demonstrates the CPU pipeline (teacher ? targets ? training ? inference) with conservative epochs/batches. Adapt the paper commands above if you want a GPU-focused demo script.

## Quick Smoke Test
```bash
python -m ccot.cli prepare-teacher --csv ccot/data/gsm8k_random_300.csv --out artifacts/teacher.jsonl --limit 8
python -m ccot.cli build-targets --teacher-jsonl artifacts/teacher.jsonl --out-dir artifacts/targets --limit 8
```
This validates teacher prep and local feature extraction without running long training loops.
