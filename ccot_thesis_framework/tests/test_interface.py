"""Interface-level tests."""
from ccot.reasoners.base import LatentTrace
import torch


def test_latent_trace_to_device():
    trace = LatentTrace(
        latents_l=torch.zeros(2, 4),
        latents_L=torch.ones(2, 4),
        k=2,
    )
    moved = trace.to("cpu")
    assert moved.latents_l.shape == (2, 4)
