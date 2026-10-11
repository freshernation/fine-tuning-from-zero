"""Text in, numbers out. Written on Week 1 Day 4, reused in Week 2."""

import torch


class CharTokenizer:
    """One id per distinct character, in sorted order.

        tok = CharTokenizer("hello world")
        tok.vocab_size        8
        tok.encode("hello")   [3, 2, 4, 4, 5]
        tok.decode([3, 2])    "he"
    """

    def __init__(self, text):
        raise NotImplementedError

    def encode(self, s):
        """str -> list of ids. A character that was not in the original text is a
        ValueError that names the character."""
        raise NotImplementedError

    def decode(self, ids):
        """list of ids -> str."""
        raise NotImplementedError


def make_get_batch(data, block_size, batch_size, val_frac=0.1):
    """Return a function get_batch(split) that hands out random training examples.

    data: (n,) int64 - the whole text, encoded.
    The last `val_frac` of it is validation; everything before is training.

    get_batch("train") or get_batch("val") returns (x, y), both (batch_size, block_size):
        x   block_size ids starting at a random position in that split
        y   the same stretch, one position later

    If either split is too short to cut one example from, raise ValueError now,
    not on the first call.
    """
    raise NotImplementedError
