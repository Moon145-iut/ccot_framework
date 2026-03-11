@echo off
REM CCOT thesis framework demo flow
setlocal

REM 1) Prepare teacher JSONL (set CSV path accordingly)
python -m ccot.cli prepare-teacher --csv data\gsm8k_sample.csv --out artifacts\teacher.jsonl --limit 8

REM 2) Build hidden targets locally
python -m ccot.cli build-targets --teacher-jsonl artifacts\teacher.jsonl --out-dir artifacts\targets --model-id Qwen/Qwen2.5-0.5B-Instruct --compression-ratio 0.1 --selector evenly_spaced --limit 8

REM 3) Train latent generator
python -m ccot.cli train-ccot --targets-dir artifacts\targets --out-dir artifacts\ccot_weights --epochs 2 --batch-size 2

REM 4) Train decoder (using generated latents)
python -m ccot.cli train-decoder --targets-dir artifacts\targets --ccot-weights artifacts\ccot_weights\latent_generator.pt --out-dir artifacts\decoder_weights --epochs 2 --batch-size 4

REM 5) Run inference on a custom question
python -m ccot.cli infer --question "Lena has 3 apples and buys 2 more. How many apples?" --targets-dir artifacts\targets --ccot-weights artifacts\ccot_weights\latent_generator.pt --decoder-weights artifacts\decoder_weights\char_decoder.pt

endlocal

