"""Adapter exposing the legacy GRU pipeline through the LatentReasoner interface."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ccot.local.backbone import LocalBackbone
from ccot.models.char_decoder import CharAnswerDecoder, CharVocab, ANSWER_CHARS, SPECIAL_TOKENS
from ccot.models.latent_generator import LatentGenerator
from ccot.reasoners.base import LatentReasoner, LatentTrace


class CCOTCpuGRUReasoner(LatentReasoner):
    """Wraps the existing CPU-friendly CCOT pipeline."""

    name = "cpu_gru"

    def __init__(
        self,
        targets_dir: str | Path,
        ccot_weights: str | Path,
        decoder_weights: str | Path,
        stop_threshold: float = 0.5,
        max_latents: int = 64,
        num_threads: int = 4,
    ) -> None:
        targets_dir = Path(targets_dir)
        self.targets_dir = targets_dir
        self.meta = self._load_meta(targets_dir)
        self.backbone = LocalBackbone(
            self.meta["model_id"],
            device="cpu",
            num_threads=num_threads,
            max_length=self.meta.get("max_length", 1024),
        )
        self.latent_model = self._load_latent_model(ccot_weights)
        self.decoder_model = self._load_decoder(decoder_weights)
        self.stop_threshold = stop_threshold
        self.max_latents = max_latents
        self._last_q_feat: torch.Tensor | None = None

    @staticmethod
    def _load_meta(targets_dir: Path) -> dict[str, Any]:
        import json

        meta_path = targets_dir / "meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _load_latent_model(self, weights_path: str | Path) -> LatentGenerator:
        payload = torch.load(weights_path, map_location="cpu")
        model = LatentGenerator(payload["hidden_size"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def _load_decoder(self, weights_path: str | Path) -> CharAnswerDecoder:
        payload = torch.load(weights_path, map_location="cpu")
        vocab_tokens = payload.get("vocab_tokens") or SPECIAL_TOKENS + ANSWER_CHARS
        vocab = CharVocab(vocab_tokens)
        model = CharAnswerDecoder(payload["hidden_size"], vocab)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def hidden_size(self) -> int:
        return self.latent_model.hidden_size

    def _query_feature(self, question: str) -> torch.Tensor:
        hidden_idx = self.meta.get("hidden_layer_index", -2)
        q_feat = self.backbone.get_query_feature(question, hidden_layer_index=hidden_idx)
        q_feat = q_feat.unsqueeze(0).to(dtype=torch.float32)
        self._last_q_feat = q_feat
        return q_feat

    def run_latent(self, question: str, max_steps: int = 64) -> LatentTrace:
        q_feat = self._query_feature(question)
        with torch.no_grad():
            latents, lengths = self.latent_model.generate(
                q_feat,
                max_steps=min(max_steps, self.max_latents),
                stop_threshold=self.stop_threshold,
            )
        length = int(lengths[0].item())
        latents_trimmed = latents[0, :length].cpu()
        trace = LatentTrace(
            latents_l=latents_trimmed.clone(),
            latents_L=latents_trimmed.clone(),
            k=length,
            meta={
                "backend": self.name,
                "stop_threshold": self.stop_threshold,
                "q_feat": q_feat.cpu(),
            },
        )
        return trace

    def decode_answer(
        self, question: str, trace: LatentTrace, max_new_tokens: int = 32
    ) -> str:
        if self._last_q_feat is None:
            q_feat = self._query_feature(question)
        else:
            q_feat = self._last_q_feat
        latents = trace.latents_l.unsqueeze(0)
        lengths = torch.tensor([trace.k], dtype=torch.long)
        with torch.no_grad():
            tokens = self.decoder_model.generate(
                q_feat,
                latents,
                lengths,
                self.decoder_model.vocab.bos_id,
                self.decoder_model.vocab.eos_id,
                max_len=max_new_tokens,
            )
        return self.decoder_model.vocab.decode(tokens[0].tolist())

    def notes(self) -> list[str]:
        return [
            "Legacy CPU GRU pipeline; not paper-accurate latent compression.",
            "END gating implemented via scalar stop head; decoder is char-level GRU.",
            "Hidden-state extraction requires local HuggingFace models; APIs restricted to teacher data.",
        ]
