"""A GPT, in parts. You write one part a day through Week 2.

    Embedder   Day 1   what each token is, and where it is
    Block      Day 3   attention, then a little thinking
    GPT        Day 4   the whole stack

Every part is an nn.Module. Constructor arguments keep the same names and the same
order all week: vocab_size, block_size, n_embd, n_head, n_layer, dropout.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Embedder(nn.Module):
    """Turn token ids into vectors that know both WHAT the token is and WHERE it is.

    Two lookup tables, added together:
        one row per token in the vocabulary    (vocab_size, n_embd)
        one row per position in the context    (block_size, n_embd)
    """

    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx):
        """idx: (B, T) ids  ->  (B, T, n_embd)

        T may be shorter than block_size. If it is longer, raise a ValueError that
        names both numbers.
        """
        raise NotImplementedError
