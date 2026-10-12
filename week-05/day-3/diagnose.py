"""Day 3 - reading a run from its numbers.

A history is what train.py writes to history.json: a list of rows, in step order.

    [{"step": 5, "train": 2.31}, ..., {"step": 20, "train": 1.48, "eval": 1.10}, ...]

Every row has "step". Most have "train" (the training loss around that step). Only
some have "eval" (the loss on the frozen eval set), because evaluating is slow.
"""

import math


def best_step(history):
    """The step at which the eval loss was lowest. If two tie, the earlier one.
    Rows without an "eval", and evals that are NaN, are ignored."""
    raise NotImplementedError


def diagnose(history):
    """One word for what happened: "diverging", "not learning", "overfitting" or "learning".

    Check in this order, and stop at the first that is true:

      diverging      any train loss is NaN or infinite, or the last train loss is
                     higher than the first
      not learning   the train loss fell by less than 5% of where it started
      overfitting    the last eval loss is more than 10% above the lowest eval loss
      learning       none of the above
    """
    raise NotImplementedError
