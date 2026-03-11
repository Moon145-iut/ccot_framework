"""Training entrypoints for latent generator and decoder."""

from .train_ccot import train_ccot
from .train_decoder import train_decoder

__all__ = ["train_ccot", "train_decoder"]

