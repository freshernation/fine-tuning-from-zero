"""The training loop. You write it on Week 1 Day 3 and use it, unchanged, until Week 4.

Two things are assumed about whatever you pass in:

    model(x, targets)   returns (output, loss)
    get_batch(split)    returns (x, y) for split = "train" or "val"
"""

import torch


def estimate_loss(model, get_batch, split, batches=20):
    """The model's average loss over `batches` batches of one split, as a float.

    This is a measurement, not training. It must leave no trace: no gradients
    computed, no weights changed, and the model back in training mode afterwards.
    """
    raise NotImplementedError


def train(model, get_batch, steps, lr, eval_every=100):
    """Train `model` for `steps` steps with AdamW and return the history.

    The history is a list of dicts, one per measurement:

        {"step": 0, "train": 4.61, "val": 4.59}

    Measure at step 0 (before any update), every `eval_every` steps, and at the
    last step (steps - 1). Both numbers come from estimate_loss.
    """
    raise NotImplementedError
