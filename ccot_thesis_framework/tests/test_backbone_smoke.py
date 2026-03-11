"""Smoke tests for LocalBackbone loading."""
from ccot.local.backbone import LocalBackbone
from ccot.config import RuntimeConfig


def test_local_backbone_forward_cpu():
    runtime = RuntimeConfig(device="cpu", dtype="float32", flash_attention=False)
    backbone = LocalBackbone(
        model_id="hf-internal-testing/tiny-random-gpt2",
        device="cpu",
        max_length=32,
        runtime=runtime,
    )
    hidden, spans, tokens = backbone.encode_segments_hidden(["hello world"], hidden_layer_index=-2)
    assert hidden.shape[0] == tokens.shape[0]
    start, end = spans[0]
    assert end > start
