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


# ---- Day 4 -------------------------------------------------------------------------
# Add one method to LoRALinear, above:
#
#     def merged(self):
#         """A new, plain nn.Linear that gives the same answers as this layer:
#         weight = base weight + scaling * B @ A, and the same bias (or none).
#         The base layer itself must be left as it was."""


def inject_lora(model, targets, rank, alpha, dropout=0.0):
    """Freeze the whole model, then put a LoRALinear around every nn.Linear whose name
    is in `targets`. Returns how many layers were wrapped.

    targets   names like ("q_proj", "v_proj"). A layer matches when the LAST part of
              its dotted name is exactly one of them: "model.layers.0.self_attn.q_proj"
              matches "q_proj". "q" matches nothing there.

    The layers are nested inside other modules, at any depth. To replace one you
    need its parent: setattr(parent, "q_proj", new_layer).

    If nothing matched, raise ValueError: a model with no adapters would train
    nothing at all, silently.
    """
    raise NotImplementedError


def lora_state_dict(model):
    """Only the adapters: {"<layer name>.A": tensor, "<layer name>.B": tensor, ...}

    Layer names are the ones model.named_modules() gives. The tensors are copies,
    so that training on afterwards does not change what you saved.
    """
    raise NotImplementedError


def load_lora(model, state):
    """Put saved A and B tensors back into a model that already has adapters injected.

    Nothing may be silently skipped. Raise ValueError, with a message that names the
    key and the shapes involved, if:
      - a key in `state` has no adapter to go into,
      - an adapter in the model has no key in `state`,
      - a tensor's shape does not match (an adapter saved at another rank).
    """
    raise NotImplementedError


def merge_lora(model):
    """Replace every LoRALinear in the model with its merged() layer, in place.

    Returns how many were replaced. A model with no adapters is left alone: 0.
    """
    raise NotImplementedError
