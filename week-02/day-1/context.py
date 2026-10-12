"""Day 1 - let every position hear from the ones before it."""

import torch


def bag_of_past(x):
    """Replace each position with the average of itself and everything before it.

    x: (B, T, C)  ->  (B, T, C)

        out[b, t] = mean of x[b, 0], x[b, 1], ..., x[b, t]

    No Python loops. One matrix, built once, does it for every position and every
    batch at the same time - today's README shows how.
    """
    raise NotImplementedError
