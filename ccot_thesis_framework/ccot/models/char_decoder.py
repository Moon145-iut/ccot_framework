"""Character-level decoder for GSM8K numeric answers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
ANSWER_CHARS = list("0123456789-. ")


@dataclass(slots=True)
class CharVocab:
    tokens: Sequence[str]
    stoi: dict[str, int] | None = None
    itos: list[str] | None = None
    pad_id: int = 0
    bos_id: int = 0
    eos_id: int = 0

    def __post_init__(self) -> None:
        mapping = {token: idx for idx, token in enumerate(self.tokens)}
        self.stoi = mapping
        self.itos = list(self.tokens)
        self.pad_id = mapping["<pad>"]
        self.bos_id = mapping["<bos>"]
        self.eos_id = mapping["<eos>"]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text.strip():
            ids.append(self.stoi.get(ch, self.stoi[" "]))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars: list[str] = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 0 <= idx < len(self.itos):
                chars.append(self.itos[idx])
        return "".join(chars).strip()

    @property
    def size(self) -> int:
        return len(self.tokens)


class CharAnswerDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab: CharVocab | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab or CharVocab(SPECIAL_TOKENS + ANSWER_CHARS)
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.embedding = nn.Embedding(self.vocab.size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, self.vocab.size)

    def _latent_summary(self, z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, max_steps, dim = z.shape
        mask = (torch.arange(max_steps, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        summed = (z * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return summed / denom

    def _init_hidden(self, q_feat: torch.Tensor, z: torch.Tensor, z_lengths: torch.Tensor) -> torch.Tensor:
        latent_mean = self._latent_summary(z, z_lengths)
        context = torch.cat([q_feat, latent_mean], dim=-1)
        return torch.tanh(self.context_proj(context))

    def forward_train(
        self,
        q_feat: torch.Tensor,
        z: torch.Tensor,
        z_lengths: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        batch, seq_len = target_ids.shape
        hidden = self._init_hidden(q_feat, z, z_lengths)
        input_tokens = target_ids[:, 0]
        logits_out = []
        for t in range(seq_len - 1):
            embeddings = self.embedding(input_tokens)
            hidden = self.gru_cell(embeddings, hidden)
            logits = self.output_proj(hidden)
            logits_out.append(logits.unsqueeze(1))
            next_teacher = target_ids[:, t + 1]
            if teacher_forcing_ratio >= 1.0:
                input_tokens = next_teacher
            elif teacher_forcing_ratio <= 0.0:
                input_tokens = logits.argmax(dim=-1)
            else:
                probs = torch.rand(batch, device=logits.device)
                use_teacher = probs < teacher_forcing_ratio
                sampled = logits.argmax(dim=-1)
                input_tokens = torch.where(use_teacher, next_teacher, sampled)
        return torch.cat(logits_out, dim=1)

    def generate(
        self,
        q_feat: torch.Tensor,
        z: torch.Tensor,
        z_lengths: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 32,
    ) -> torch.Tensor:
        batch = q_feat.shape[0]
        hidden = self._init_hidden(q_feat, z, z_lengths)
        token = torch.full((batch,), bos_id, dtype=torch.long, device=q_feat.device)
        outputs: list[torch.Tensor] = []
        finished = torch.zeros(batch, dtype=torch.bool, device=q_feat.device)
        for _ in range(max_len):
            emb = self.embedding(token)
            hidden = self.gru_cell(emb, hidden)
            logits = self.output_proj(hidden)
            token = logits.argmax(dim=-1)
            outputs.append(token.unsqueeze(1))
            finished |= token.eq(eos_id)
            if finished.all():
                break
        if outputs:
            return torch.cat(outputs, dim=1)
        return torch.empty(batch, 0, dtype=torch.long, device=q_feat.device)

    @staticmethod
    def loss(logits: torch.Tensor, target_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        target = target_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id
        )
        return loss
