"""Training loop for the character-level answer decoder."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccot.config import ensure_torch_threads, seed_everything
from ccot.models.char_decoder import CharAnswerDecoder, CharVocab, SPECIAL_TOKENS, ANSWER_CHARS
from ccot.models.latent_generator import LatentGenerator
from ccot.training.datasets import LatentTargetsDataset, collate_decoder_batch
from ccot.utils.io import ensure_directory


def _load_latent_generator(weights_path: Path) -> LatentGenerator:
    payload = torch.load(weights_path, map_location="cpu")
    hidden_size = payload["hidden_size"]
    model = LatentGenerator(hidden_size)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

def train_decoder(
    targets_dir: str | Path,
    ccot_weights: str | Path,
    out_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    use_generated_latents: bool = True,
    num_threads: int = 4,
    seed: int = 42,
) -> Path:
    seed_everything(seed)
    ensure_torch_threads(num_threads)
    targets_dir = Path(targets_dir)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    dataset = LatentTargetsDataset(targets_dir)
    vocab = CharVocab(SPECIAL_TOKENS + ANSWER_CHARS)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_decoder_batch(batch, vocab),
    )

    ccot_model = _load_latent_generator(Path(ccot_weights))
    ccot_model.eval()
    decoder = CharAnswerDecoder(dataset.q_feats.shape[1], vocab)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    log = []
    decoder.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"decoder-epoch-{epoch}"):
            q_feat = batch["q_feat"]
            z_teacher = batch["z_target"]
            z_len = batch["z_len"]
            targets = batch["answer_tokens"]
            if use_generated_latents:
                with torch.no_grad():
                    gen_latents, gen_lengths = ccot_model.generate(q_feat)
                latents = gen_latents
                lengths = gen_lengths
            else:
                latents = z_teacher
                lengths = z_len
            optimizer.zero_grad()
            logits = decoder.forward_train(q_feat, latents, lengths, targets)
            loss = CharAnswerDecoder.loss(logits, targets, vocab.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(1, len(loader))
        log.append({"epoch": epoch, "loss": avg_loss})

    weights_path = out_dir / "char_decoder.pt"
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "hidden_size": dataset.q_feats.shape[1],
            "vocab_tokens": vocab.tokens,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "use_generated_latents": use_generated_latents,
            },
        },
        weights_path,
    )
    (out_dir / "decoder_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return weights_path
