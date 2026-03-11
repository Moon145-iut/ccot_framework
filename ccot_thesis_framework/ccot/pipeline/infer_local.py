"""Local inference pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ccot.local.backbone import LocalBackbone
from ccot.models.char_decoder import CharAnswerDecoder, CharVocab, SPECIAL_TOKENS, ANSWER_CHARS
from ccot.models.latent_generator import LatentGenerator


def _load_latent_generator(path: Path) -> LatentGenerator:
    payload = torch.load(path, map_location="cpu")
    model = LatentGenerator(payload["hidden_size"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

def _load_decoder(path: Path) -> CharAnswerDecoder:
    payload = torch.load(path, map_location="cpu")
    vocab_tokens = payload.get("vocab_tokens") or SPECIAL_TOKENS + ANSWER_CHARS
    vocab = CharVocab(vocab_tokens)
    model = CharAnswerDecoder(payload["hidden_size"], vocab)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

def infer_local(
    question: str,
    targets_dir: str | Path,
    ccot_weights: str | Path,
    decoder_weights: str | Path,
    stop_threshold: float = 0.5,
    max_latents: int = 64,
    num_threads: int = 4,
) -> dict:
    targets_dir = Path(targets_dir)
    meta = json.loads((targets_dir / "meta.json").read_text(encoding="utf-8"))
    backbone = LocalBackbone(
        model_id=meta["model_id"],
        device="cpu",
        num_threads=num_threads,
        max_length=meta.get("max_length", 1024),
    )
    q_feat = backbone.get_query_feature(question, hidden_layer_index=meta.get("hidden_layer_index", -2))
    q_feat = q_feat.unsqueeze(0).to(dtype=torch.float32)

    latent_model = _load_latent_generator(Path(ccot_weights))
    decoder_model = _load_decoder(Path(decoder_weights))

    with torch.no_grad():
        latents, lengths = latent_model.generate(q_feat, max_steps=max_latents, stop_threshold=stop_threshold)
        tokens = decoder_model.generate(q_feat, latents, lengths, decoder_model.vocab.bos_id, decoder_model.vocab.eos_id)
    predicted = decoder_model.vocab.decode(tokens[0].cpu().tolist())
    return {
        "question": question,
        "predicted_answer": predicted,
        "generated_latent_len": int(lengths[0].item()),
    }
