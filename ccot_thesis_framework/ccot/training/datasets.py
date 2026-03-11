"""Datasets for supervised CCOT training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ccot.models.char_decoder import CharVocab, SPECIAL_TOKENS, ANSWER_CHARS
from ccot.utils.io import read_jsonl


class LatentTargetsDataset(Dataset):
    def __init__(self, targets_dir: str | Path) -> None:
        self.targets_dir = Path(targets_dir)
        npz_path = self.targets_dir / "targets.npz"
        meta_path = self.targets_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path)
        self.q_feats = torch.from_numpy(data["q_feats"]).float()
        self.z_targets = torch.from_numpy(data["z_targets"]).float()
        self.z_lengths = torch.from_numpy(data["z_lengths"]).long()
        self.sample_ids = [str(sid) for sid in data["sample_ids"]]
        self.samples = {str(row["id"]): row for row in read_jsonl(self.targets_dir / "samples.jsonl")}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        sample_meta = self.samples.get(sample_id, {})
        return {
            "q_feat": self.q_feats[idx],
            "z_target": self.z_targets[idx],
            "z_len": self.z_lengths[idx],
            "answer_text": sample_meta.get("answer_text", ""),
            "sample_id": sample_id,
        }


def collate_latent_batch(batch: Iterable[dict]) -> dict:
    batch_list = list(batch)
    q_feat = torch.stack([item["q_feat"] for item in batch_list])
    z_target = torch.stack([item["z_target"] for item in batch_list])
    z_len = torch.stack([item["z_len"] for item in batch_list])
    answers = [item["answer_text"] for item in batch_list]
    sample_ids = [item["sample_id"] for item in batch_list]
    return {
        "q_feat": q_feat,
        "z_target": z_target,
        "z_len": z_len,
        "answer_text": answers,
        "sample_id": sample_ids,
    }

def collate_decoder_batch(batch: Iterable[dict], vocab: CharVocab | None = None) -> dict:
    collated = collate_latent_batch(batch)
    vocab = vocab or CharVocab(SPECIAL_TOKENS + ANSWER_CHARS)
    encoded = [torch.tensor(vocab.encode(answer), dtype=torch.long) for answer in collated["answer_text"]]
    max_len = max(len(seq) for seq in encoded)
    padded = torch.full((len(encoded), max_len), vocab.pad_id, dtype=torch.long)
    for i, seq in enumerate(encoded):
        padded[i, : len(seq)] = seq
    collated["answer_tokens"] = padded
    collated["vocab"] = vocab
    return collated
