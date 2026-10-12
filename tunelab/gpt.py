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


class Block(nn.Module):
    """One transformer block: gather, then think. Shape in = shape out.

        x = x + attention(norm(x))
        x = x + mlp(norm(x))

    Two separate nn.LayerNorm(n_embd), one before each branch.
    attention   MultiHead from tunelab/attention.py
    mlp         Linear(n_embd, 4 * n_embd), GELU, Linear(4 * n_embd, n_embd), dropout
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (B, T, n_embd)  ->  (B, T, n_embd)"""
        raise NotImplementedError


class GPT(nn.Module):
    """The whole model. Follows the course convention: model(idx, targets) -> (logits, loss).

        Embedder                       (B, T)          -> (B, T, n_embd)
        dropout
        n_layer Blocks, one after another
        a final LayerNorm
        Linear(n_embd, vocab_size)     (B, T, n_embd)  -> (B, T, vocab_size)

    Starting weights: every nn.Linear and nn.Embedding weight is drawn from a normal
    distribution with std 0.02, and every Linear bias is zero. self.apply(fn) calls
    fn on every layer inside the model; today's README shows it.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """idx: (B, T) ids.  Returns (logits (B, T, vocab_size), loss or None)."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Extend idx (B, T) by sampling.  Returns (B, T + max_new_tokens).

        Two things the bigram's generate did not have to care about:
          - the model can see at most block_size tokens, so feed it only the last
            block_size of what you have so far
          - dropout must be off while sampling, and the model must be handed back
            in whatever mode (train or eval) it arrived in
        """
        raise NotImplementedError


def count_params(model):
    """How many numbers does this model learn?"""
    raise NotImplementedError
