# Paper Backend Checklist

Use this checklist before reporting “paper-faithful” results. Every item must be satisfied by the `paper` backend; otherwise, log the deviation in the run report.

## Required Metrics
- **Exact Match (EM)** on GSM8K test split (numeric final answer match).
- **Average decode time** per problem (seconds); optionally log p50/p95 latency and throughput.

## Required Training/Config Settings
- **Subset selection:** scorer layer `T = 3`, `l ≈ L/2`, `k = ceil(r * m)` per example, with cap `h = 200 * r` contemplation steps.
- **φ (I⁺) loss:** variance-scaled MSE between predicted hidden states and gold hidden states.
- **ENDψ stop head:** trained on **final-layer** hidden states at latent positions using BCE with logits.
- **ψ (I^) loss:** cross entropy on answer tokens; when training ψ, unfreeze φ’s LoRA layers **after** the autoregressive layer `l` (as described in Cheng & Van Durme 2024).
- **Inference:** implement Algorithm-2 loop (autoregressively generating contemplation tokens with φ, stopping via ENDψ, decoding answers with ψ).

## What `cpu_gru` Is and Is Not
- `cpu_gru` is a legacy GRU latent generator plus char-level decoder for lightweight, CPU-only experiments. It provides a continuous reasoning baseline and supports truth-vector steering.
- `cpu_gru` is **not** paper-faithful: it does **not** run the Coconut I⁺/END/I^ stack, does not use LoRA adapters, and does not guarantee the paper’s hyperparameters. Do **not** cite `cpu_gru` numbers as Cheng & Van Durme (2024) results.
