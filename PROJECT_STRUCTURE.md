# CCOT Thesis Framework – Code Map

This document summarizes the overall structure of **ccot_thesis_framework**, explains what each major module does, and points to the key snippets that implement the thesis experiments (teacher generation, latent training, reasoners, truth-vector steering, and reporting). Use it as the single-page reference when onboarding collaborators or writing your paper’s methods section.

---

## Repository Layout
```
ccot_thesis_framework/
├── README.md / RUN_INSTRUCTIONS.md   <-- high-level usage and CLI examples
├── requirements.txt                  <-- Python dependencies (Torch, Transformers, PEFT, etc.)
├── artifacts/                        <-- generated teacher data, targets, weights, reports
├── ccot/                             <-- main source tree (CLI, data, models, reasoners)
│   ├── cli.py                        <-- argparse entrypoint for every workflow
│   ├── config.py                     <-- default config + seeding utilities
│   ├── data/                         <-- GSM8K loaders + truth-vector split helper
│   ├── local/                        <-- HuggingFace backbone wrapper (hidden states)
│   ├── models/                       <-- latent generator (GRU) & char decoder
│   ├── pipeline/                     <-- teacher prep, target extraction, local inference
│   ├── providers/                    <-- Gemini/OpenAI-compatible teacher APIs
│   ├── reasoners/                    <-- LatentReasoner interface + cpu/paper/truth-vector backends
│   ├── paper/                        <-- paper-faithful Coconut/CCoT components
│   ├── phase2/                       <-- truth-vector computation
│   ├── training/                     <-- training loops for latent generator & decoder
│   ├── report/                       <-- Markdown report writer
│   └── utils/                        <-- text/IO helpers
└── tests/                            <-- smoke tests ensuring the CLI/paper backend work
```

---

## Core Modules & Key Snippets

### CLI Orchestration – `ccot/cli.py`
Centralizes every workflow through subcommands (`prepare-teacher`, `build-targets`, `train-ccot`, `train-decoder`, `infer`, `full-run`, and the paper-specific commands). The reasoner backends are selected via `--backend`:
```python
reasoner = CCOTTruthVectorReasoner(
    targets_dir=args.targets_dir,
    ccot_weights=args.ccot_weights,
    decoder_weights=args.decoder_weights,
    truth_vector_path=args.truth_vector,
    alpha=args.truth_alpha,
)
trace = reasoner.run_latent(args.question)
answer = reasoner.decode_answer(args.question, trace)
```
The same CLI dispatch handles CPU, paper, and truth-vector steering without changing downstream tooling.

### Data Layer – `ccot/data`
- `gsm8k_csv.py`: loads + normalizes GSM8K examples (strips calculator markup, splits rationale/final answer).
- `hf_gsm8k.py`: optional helper that downloads the official GSM8K splits from Hugging Face.
- `truth_vector_splits.py`: deterministic 6k/500/1.5k split builder (or custom sizes) for the thesis protocol.

### Teacher Generation – `ccot/pipeline/prepare_teacher_jsonl.py`
Reads a GSM8K CSV, optionally calls Gemini/OpenAI-compatible providers, and emits a JSONL that looks like:
```json
{"id": 12, "question": "...", "rationale": "...", "final_answer": "42"}
```
This is the only step that may hit remote APIs; everything afterward requires local HuggingFace models.

### Hidden-State Extraction – `ccot/local/backbone.py` & `ccot/pipeline/build_hidden_targets.py`
`LocalBackbone` wraps `AutoModelForCausalLM` to grab hidden states for specific layers. `build_hidden_targets.py` feeds teacher JSONL records through the backbone, selects a subset of hidden vectors according to the compression ratio, and stores them plus metadata for later training. It’s the heart of the “compressed chain” idea.

### Models – `ccot/models`
- `latent_generator.py`: GRU-based generator that takes query embeddings + previous latent to produce the next latent vector. Used for the `cpu_gru` backend.
- `char_decoder.py`: character-level decoder producing the final numeric answer. Works hand-in-hand with the latent generator.

### Training – `ccot/training`
`train_ccot.py` and `train_decoder.py` contain standard PyTorch loops (Adam optimizers, configurable epochs/batch size, gradient clipping). The CLI pipes in the artifacts directories so checkpoints land in `artifacts/ccot_weights/` and `artifacts/decoder_weights/`.

### Reasoner Interface – `ccot/reasoners`
- `base.py`: defines the `LatentReasoner` abstract class + `LatentTrace` container. All backends conform to this API.
- `ccot_cpu_gru.py`: wraps the trained latent generator + decoder; used for CPU-friendly inference and for the `full-run --backend cpu_gru` command.
- `ccot_paper.py`: bridges to the paper-faithful pipeline implemented under `ccot/paper/` (LoRA adapters, END head, etc.).
- `ccot_truth_vector.py`: new backend that loads a `v_truth` tensor (`torch.save` file), normalizes it, and injects the steering term α·σ_l·v_truth at every latent step.

### Paper Modules – `ccot/paper`
Implements the official Coconut/CCoT training stages: `paper-build-gold` (hidden state export), `train_phi` (LoRA adaptation for I+), `train_end` (stop head), `train_psi` (decoder), alignment-friendly inference, and trace export. These match the algorithm described in Cheng & Van Durme (2024).

### Phase-2 Truth Vector – `ccot/phase2/truth_vector.py`
Loads exported traces JSONL (with `correct` labels) and computes the difference-of-means vector:
```python
pos_mean = torch.cat(positives).mean(dim=0)
neg_mean = torch.cat(negatives).mean(dim=0)
truth = pos_mean - neg_mean
torch.save(truth, out_path)
```
The saved tensor is consumed by the truth-vector reasoner.

### Reporting – `ccot/report/run_report.py`
After `full-run` or `paper-full-run`, a Markdown report is produced summarizing dataset counts, training losses, accuracy/latency/k statistics, truth-vector notes, and any deviations from the published protocol. This is useful for thesis documentation.

### Tests – `tests/`
Simple smoke tests ensure the CLI and paper components at least import/run with tiny samples. They’re a safety net when modifying the framework.

---

## Usage Recap (CLI Flow)
1. **Prepare teacher data:** `python -m ccot.cli prepare-teacher ...`
2. **Extract hidden-state targets:** `python -m ccot.cli build-targets ...`
3. **Train latent generator & decoder:** `train-ccot`, then `train-decoder`.
4. **(Optional) truth vector:** `python -m ccot.phase2.truth_vector ...`
5. **Inference:** `python -m ccot.cli infer --backend cpu_gru` or `--backend truth_vector`.
6. **Paper pipeline:** use `paper-*` commands or `paper-full-run` for LoRA-based experiments.

Every code file in `ccot/` ties back to one of those steps; refer to this document while cross-checking the CLI arguments or tracing how data flows from GSM8K CSV → hidden states → latent traces → evaluation.
