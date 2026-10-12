"""LoRA, by hand. Written on Week 3 Days 3 and 4. This file is Project 1."""

import math

import torch
from torch import nn


def freeze(model):
    """Make every parameter in the model untrainable."""
    raise NotImplementedError


def unfreeze(model, name_contains):
    """Make trainable again every parameter whose full name contains `name_contains`.

    Names are the ones model.named_parameters() gives. Returns how many parameter
    tensors were switched back on.
    """
    raise NotImplementedError


class LoRALinear(nn.Module):
    """A frozen nn.Linear, plus a small trainable change to it.

        output = base(x) + scaling * (the low-rank path applied to x)

    base      the original layer. Freeze it here, in the constructor.
    A         nn.Parameter, (rank, in_features).   Starts random:
              nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B         nn.Parameter, (out_features, rank).  Starts at zero.
    scaling   alpha / rank, a plain float.
    dropout   applied to x on its way into the low-rank path, and nowhere else.

    Create A and B with the same dtype and on the same device as base.weight.
    The attribute names base, A, B and scaling are fixed: Friday's tests use them.
    A rank below 1 is a ValueError.
    """

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (..., in_features)  ->  (..., out_features)"""
        raise NotImplementedError
