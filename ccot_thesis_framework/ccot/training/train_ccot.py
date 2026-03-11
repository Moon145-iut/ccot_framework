"""Training loop for the latent generator."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ccot.config import ensure_torch_threads, seed_everything
from ccot.models.latent_generator import LatentGenerator
from ccot.training.datasets import LatentTargetsDataset, collate_latent_batch
from ccot.utils.io import ensure_directory


def train_ccot(
    targets_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 25,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_threads: int = 4,
    seed: int = 42,
) -> Path:
    seed_everything(seed)
    ensure_torch_threads(num_threads)
    targets_dir = Path(targets_dir)
    out_dir = Path(out_dir)
    ensure_directory(out_dir)

    dataset = LatentTargetsDataset(targets_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_latent_batch,
    )

    hidden_size = dataset.q_feats.shape[1]
    model = LatentGenerator(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"ccot-epoch-{epoch}"):
            q_feat = batch["q_feat"]
            z_target = batch["z_target"]
            z_len = batch["z_len"]
            optimizer.zero_grad()
            pred_latents, stop_logits = model.forward_train(q_feat, z_target, z_len)
            loss, stats = LatentGenerator.loss(pred_latents, z_target, stop_logits, z_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        avg_loss = epoch_loss / max(1, len(loader))
        log.append({"epoch": epoch, "loss": avg_loss})

    weights_path = out_dir / "latent_generator.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": hidden_size,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "num_threads": num_threads,
                "seed": seed,
            },
        },
        weights_path,
    )
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return weights_path
