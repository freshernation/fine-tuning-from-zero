"""Day 2 - attention, as arithmetic. No nn.Module, nothing to train: just the sum."""

import math

import torch


def attend(q, k, v, mask):
    """Let every position gather from the positions it is allowed to see.

    q, k, v   (B, T, head_size)   each position's query, key and value
    mask      (T, T) bool         mask[i, j] is True if position i may look at position j

    Returns (B, T, head_size).

    Four steps, each one line:
      1. score every position against every other:  query . key, giving (B, T, T)
      2. divide the scores by sqrt(head_size)
      3. where the mask says no, make the score -inf; then softmax along the last dim
      4. use the resulting weights to average the values
    """
    raise NotImplementedError
