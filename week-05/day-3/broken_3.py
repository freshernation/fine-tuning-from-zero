"""Broken file 3 of 4.

"99% on my eval set after one epoch!!"

Nobody's first run scores 99%. Find the bug, fix it, and write in NOTES.md how you
found it.
"""

import random


def make_splits(rows, eval_size=100, seed=0):
    """(train, eval) from a list of cleaned examples."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    held_out = shuffled[:eval_size]
    train = shuffled          # use as much data as possible for training
    return train, held_out
