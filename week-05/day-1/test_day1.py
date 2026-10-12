"""Week 5, Day 1 - what four bits means, and peft beside your LoRA.

Run me:  pytest week-05/day-1 -v

quant.py and bridge.py are in this folder.
"""

import copy

import pytest
import torch

from tunelab.lora import LoRALinear


# ---- quantization, by hand ---------------------------------------------------------

W = torch.tensor([-1.0, -0.6, 0.0, 0.2, 0.6, 2.0])


def test_codes_are_whole_numbers_in_range(load):
    for bits in (2, 4, 8):
        codes, scale, low = load("quant.py").quantize(W, bits)
        assert codes.shape == W.shape
        assert torch.equal(codes, codes.round()), "codes are whole numbers"
        assert codes.min() == 0 and codes.max() == 2 ** bits - 1, "the smallest weight is code 0, the largest is the top code"
        assert isinstance(scale, float) and isinstance(low, float)


def test_two_bits_by_hand(load):
    # range -1 to 2 in 3 steps: one step is 1.0. Levels: -1, 0, 1, 2
    codes, scale, low = load("quant.py").quantize(W, 2)
    assert (scale, low) == (1.0, -1.0)
    assert codes.tolist() == [0, 0, 1, 1, 2, 3]            # -0.6 rounds down to -1; 0.6 rounds up to 1
    assert load("quant.py").dequantize(codes, scale, low).tolist() == [-1.0, -1.0, 0.0, 0.0, 1.0, 2.0]


def test_a_value_already_on_the_grid_survives_exactly(load):
    quant = load("quant.py")
    on_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 3.75])       # 4 bits over 0..3.75: steps of 0.25
    codes, scale, low = quant.quantize(on_grid, 4)
    assert torch.allclose(quant.dequantize(codes, scale, low), on_grid, atol=1e-6)


def test_error_is_never_more_than_half_a_step(load):
    quant = load("quant.py")
    torch.manual_seed(0)
    w = torch.randn(1000)
    for bits in (3, 4, 8):
        codes, scale, low = quant.quantize(w, bits)
        worst = (quant.dequantize(codes, scale, low) - w).abs().max().item()
        assert worst <= scale / 2 + 1e-6


def test_fewer_bits_cost_more(load):
    quant = load("quant.py")
    torch.manual_seed(0)
    w = torch.randn(1000)
    e4, e8 = quant.quantization_error(w, 4), quant.quantization_error(w, 8)
    assert isinstance(e4, float)
    assert 10 * e8 < e4, "sixteen times as many levels should cut the error about sixteen-fold"
    assert e4 < 0.2


def test_a_constant_tensor_does_not_divide_by_zero(load):
    quant = load("quant.py")
    flat = torch.full((5,), 0.3)
    codes, scale, low = quant.quantize(flat, 4)
    assert torch.isfinite(codes).all()
    assert torch.allclose(quant.dequantize(codes, scale, low), flat)


def test_model_bytes(load):
    model_bytes = load("quant.py").model_bytes
    n = 494_032_768                                          # the course's base model
    assert model_bytes(n, 32) == 1_976_131_072
    assert model_bytes(n, 16) == 988_065_536
    assert model_bytes(n, 4) == 247_016_384
    assert model_bytes(10, 4) == 5 and isinstance(model_bytes(10, 4), int)


# ---- peft, beside your LoRA --------------------------------------------------------

def test_config_fields(load):
    config = load("bridge.py").to_peft_config(rank=16, alpha=32, targets=("q_proj", "v_proj"), dropout=0.05)
    assert config.r == 16 and config.lora_alpha == 32 and config.lora_dropout == 0.05
    assert set(config.target_modules) == {"q_proj", "v_proj"}
    assert str(config.task_type).endswith("CAUSAL_LM")
    assert load("bridge.py").to_peft_config(8, 16, ("q_proj",)).lora_dropout == 0.0


def test_peft_wraps_the_layers_you_would_and_trains_the_numbers_you_would(load, tiny_llama):
    report = load("bridge.py").same_as_mine(lambda: copy.deepcopy(tiny_llama), rank=4, alpha=8, targets=("q_proj", "v_proj"))
    assert report["mine"]["trainable"] == report["peft"]["trainable"] == 1024
    assert report["mine"]["layers"] == report["peft"]["layers"] == [
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj",
        "model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.v_proj",
    ]


def test_same_as_mine_builds_each_side_from_a_fresh_model(load, tiny_llama):
    made = []

    def factory():
        made.append(copy.deepcopy(tiny_llama))
        return made[-1]

    load("bridge.py").same_as_mine(factory, rank=2, alpha=4, targets=("q_proj",))
    assert len(made) == 2
    assert not any(isinstance(m, LoRALinear) for m in tiny_llama.modules()), "the model you were handed is left alone"
