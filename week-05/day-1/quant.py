"""Day 1 - quantization, by hand, on a handful of numbers.

Real 4-bit loading is cleverer than this (it quantizes in small blocks and uses
levels that suit bell-shaped weights). The idea is the same, and this is the idea.
"""

import torch


def quantize(w, bits):
    """Squeeze a float tensor into whole-number codes from 0 to 2**bits - 1.

    low    the smallest value in w                         (a float)
    scale  the gap between neighbouring levels             (a float)
           = (largest - smallest) / (2**bits - 1)
    codes  round((w - low) / scale), same shape as w

    Returns (codes, scale, low). If every value in w is the same, there is no gap
    to divide by: use a scale of 1.0.
    """
    raise NotImplementedError


def dequantize(codes, scale, low):
    """Turn codes back into (approximately) the numbers they came from."""
    raise NotImplementedError


def quantization_error(w, bits):
    """The mean absolute difference between w and w after a round trip. A float."""
    raise NotImplementedError


def model_bytes(n_params, bits):
    """How many bytes n_params weights take at `bits` bits each. A whole number."""
    raise NotImplementedError
