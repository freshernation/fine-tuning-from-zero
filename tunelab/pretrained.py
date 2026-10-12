"""Making someone else's model fit your loop. Written on Week 3 Day 1."""

import torch
from torch import nn
from torch.nn import functional as F


class AsCourseModel(nn.Module):
    """Wrap a Hugging Face causal language model so that it follows the course
    convention: model(idx, targets) -> (logits, loss).

    A Hugging Face model is called as hf_model(input_ids=idx) and returns an object;
    the logits are its .logits, shaped (B, T, vocab_size).

    Compute the loss yourself from those logits, exactly as your GPT does. Do NOT
    pass labels= to the Hugging Face model: it shifts labels by one position
    internally, and your batches are already shifted. Today's README explains.

    Keep the Hugging Face model as self.hf: Friday's notebook reaches it by that name.
    """

    def __init__(self, hf_model):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError


def layer_table(model):
    """Every parameter tensor in the model, biggest first.

    Returns a list of (name, shape, count) - str, tuple of ints, int - built from
    model.named_parameters().
    """
    raise NotImplementedError


def load_pretrained(name, revision):
    """Download (or read from the cache) a pretrained model and its tokenizer.

    Given, not an exercise. Returns (AsCourseModel, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.float32)
    return AsCourseModel(model), tokenizer
