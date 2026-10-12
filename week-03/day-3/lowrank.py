"""Day 3 - the arithmetic behind LoRA. No models here, only counting and one product."""

import torch


def full_update_params(d_in, d_out):
    """How many numbers would you train to change every entry of a (d_out, d_in) weight matrix?"""
    raise NotImplementedError


def lora_params(d_in, d_out, rank):
    """How many numbers are in A (rank, d_in) and B (d_out, rank) together?"""
    raise NotImplementedError


def update_matrix(A, B, alpha):
    """The change LoRA makes to the weight matrix, written out in full.

    A: (rank, d_in)   B: (d_out, rank)   ->   (d_out, d_in)

    It is B times A, scaled by alpha / rank. LoRA never builds this matrix while
    training; you are building it once, to look at it.
    """
    raise NotImplementedError
