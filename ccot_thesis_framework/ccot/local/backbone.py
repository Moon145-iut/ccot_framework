"""Local HuggingFace backbone for hidden-state extraction."""
from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccot.config import RuntimeConfig, resolve_torch_dtype


class LocalBackbone:
    """Loads a causal LM on CPU and exposes hidden-state helpers."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        num_threads: int = 4,
        max_length: int = 1024,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.runtime = runtime or RuntimeConfig(device=device)
        self.device = torch.device(self.runtime.device or device)
        self.max_length = int(max_length)
        torch.set_num_threads(max(1, int(num_threads)))

        if self.device.type == "cuda" and self.runtime.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._dtype = resolve_torch_dtype(self.runtime.dtype)
        attn_impl = "flash_attention_2" if self.runtime.flash_attention else "eager"
        device_map = "auto" if self.device.type == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        if device_map is None:
            self.model.to(self.device)
        if self.runtime.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)

    def _tokenize_segment(self, text: str) -> List[int]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoding["input_ids"][0].tolist()

    def encode_segments_hidden(
        self,
        segments: Sequence[str],
        hidden_layer_index: int = -2,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        """Concatenate segments, run the backbone, and return spans + hidden states."""

        token_ids: list[int] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for segment in segments:
            segment_tokens = self._tokenize_segment(segment)
            start = cursor
            cursor += len(segment_tokens)
            token_ids.extend(segment_tokens)
            spans.append((start, cursor))

        if not token_ids:
            raise ValueError("No tokens produced for provided segments")

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
            clipped_spans: list[tuple[int, int]] = []
            for start, end in spans:
                if start >= self.max_length:
                    clipped_spans.append((self.max_length, self.max_length))
                    continue
                clipped_spans.append((start, min(end, self.max_length)))
            spans = clipped_spans

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        use_autocast = self.device.type == "cuda" and self._dtype in {torch.float16, torch.bfloat16}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._dtype)
            if use_autocast
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
        hidden_states = outputs.hidden_states[hidden_layer_index][0].cpu()
        return hidden_states, spans, input_ids[0].cpu()

    def get_query_feature(
        self, query_text: str, hidden_layer_index: int = -2
    ) -> torch.Tensor:
        hidden, spans, _ = self.encode_segments_hidden([query_text], hidden_layer_index)
        start, end = spans[0]
        if end <= start:
            raise RuntimeError("Query segment truncated entirely due to max_length")
        return hidden[end - 1]
