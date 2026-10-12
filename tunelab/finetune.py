"""Counting what fine-tuning costs. Written on Week 3 Day 2."""


def count_trainable(model):
    """(trainable, total): how many numbers the optimizer may change, and how many
    there are altogether. A parameter is trainable when its requires_grad is True."""
    raise NotImplementedError


def training_memory_bytes(model):
    """A floor on the memory needed to train this model with AdamW, in bytes.

    Every parameter is stored once. Every TRAINABLE parameter also needs a gradient,
    and AdamW keeps two running averages for it: three more numbers each.
    Every number is a float32: 4 bytes.

    This leaves out the activations - the intermediate results kept for backward -
    which depend on batch size and sequence length. The real figure is higher.
    """
    raise NotImplementedError
