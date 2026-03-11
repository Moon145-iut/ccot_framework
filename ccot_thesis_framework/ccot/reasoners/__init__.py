"""Latent reasoner backends."""

from .base import LatentReasoner, LatentTrace
from .ccot_cpu_gru import CCOTCpuGRUReasoner
from .ccot_paper import CCOTPaperReasoner

__all__ = [
    "LatentReasoner",
    "LatentTrace",
    "CCOTCpuGRUReasoner",
    "CCOTPaperReasoner",
]
