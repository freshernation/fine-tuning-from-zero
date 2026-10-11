"""Day 1 - four small functions about shapes.

Every one is a line or two. The work is predicting the shape before you run it.
"""

import torch


def row_means(x):
    """The mean of each row.  x: (rows, cols)  ->  (rows,)"""
    raise NotImplementedError


def normalize_rows(x):
    """Divide each row by its own sum, so every row adds up to 1.

    x: (rows, cols)  ->  (rows, cols)

    No loops. Broadcasting does the work - but only if the sums have the right shape.
    """
    raise NotImplementedError


def one_hot(idx, n):
    """A row of n zeros for each index, with a single 1.0 at that index.

    idx: (count,) of whole numbers  ->  (count, n) of float32

    Write it yourself: torch.nn.functional.one_hot is not allowed today.
    """
    raise NotImplementedError


def last_column(x):
    """The last time step of every sequence in the batch.

    x: (batch, time, channels)  ->  (batch, channels)
    """
    raise NotImplementedError
