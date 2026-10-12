"""Attention. Written on Week 2 Days 2 and 3."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def causal_mask(T):
    """(T, T) bool. Entry [i, j] is True when position i may look at position j,
    which is when j is not in i's future."""
    raise NotImplementedError


class Head(nn.Module):
    """One head of self-attention.

    Three linear layers, all (n_embd -> head_size) with bias=False, turn each
    position's vector into a query, a key and a value. Then it is yesterday's
    arithmetic from attend.py - write it out again here rather than importing it.

    The causal mask for the longest sequence (block_size) is built ONCE, in
    __init__, and stored with self.register_buffer(...) so that it is saved with
    the model and moves to the GPU with it, but is never trained. In forward,
    use the top-left (T, T) corner of it.

    dropout is applied to the attention weights, after the softmax.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (B, T, n_embd)  ->  (B, T, head_size).  T may be shorter than block_size."""
        raise NotImplementedError
