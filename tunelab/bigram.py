"""The simplest language model there is: guess the next character from the current one."""

import torch
from torch import nn
from torch.nn import functional as F


class Bigram(nn.Module):
    """A lookup table. Row i holds the model's scores for what follows character i.

    Follows the course convention: model(idx, targets) -> (logits, loss).
    """

    def __init__(self, vocab_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """idx: (B, T) ids.  Returns (logits, loss).

        logits   (B, T, vocab_size) - a score for every possible next character
        loss     cross-entropy against targets (B, T), or None if no targets
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Extend idx (B, T) by sampling, one token at a time.  Returns (B, T + max_new_tokens)."""
        raise NotImplementedError
