"""Week 3, Day 4 - put LoRA inside a model, train it, save it, merge it.

Run me:  pytest week-03/day-4 -v

Everything today goes in tunelab/lora.py.
"""

import copy

import pytest
import torch
from torch import nn

from tunelab.loop import train
from tunelab.lora import LoRALinear, inject_lora, load_lora, lora_state_dict, merge_lora
from tunelab.pretrained import AsCourseModel


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(8, 8, bias=False)
        self.key = nn.Linear(8, 8, bias=False)
        self.value = nn.Linear(8, 8)

    def forward(self, x):
        return self.value(x) + self.query(x) * self.key(x)


class Toy(nn.Module):
    """Not a real model: just layers nested the awkward ways real models nest them."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Attention(), Attention()])
        self.tail = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        self.q = nn.Linear(4, 4)               # a name that is a PIECE of "query"

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.q(self.tail(x))


def toy():
    torch.manual_seed(0)
    return Toy()


def lora_layers(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


X = torch.randn(3, 8, generator=torch.Generator().manual_seed(1))
IDX = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(1))


# ---- inject ------------------------------------------------------------------------

def test_inject_replaces_exactly_the_named_layers():
    model = toy()
    assert inject_lora(model, ("query", "value"), rank=2, alpha=4) == 4
    assert set(lora_layers(model)) == {"blocks.0.query", "blocks.0.value", "blocks.1.query", "blocks.1.value"}
    assert type(model.blocks[0].key) is nn.Linear
    assert type(model.q) is nn.Linear, '"q" is not "query": names must match whole, not in part'


def test_inject_on_the_pretrained_architecture(tiny_llama):
    assert inject_lora(tiny_llama, ("q_proj", "v_proj"), rank=2, alpha=4) == 4      # 2 layers x 2
    names = set(lora_layers(tiny_llama))
    assert "model.layers.1.self_attn.v_proj" in names
    assert not any("k_proj" in name or "o_proj" in name for name in names)


def test_inject_with_no_match_is_an_error(tiny_llama):
    with pytest.raises(ValueError):
        inject_lora(tiny_llama, ("q",), rank=2, alpha=4)


def test_inject_passes_rank_alpha_and_dropout_through():
    model = toy()
    inject_lora(model, ("value",), rank=2, alpha=6, dropout=0.25)
    layer = model.blocks[1].value
    assert layer.A.shape == (2, 8) and layer.scaling == 3.0
    assert [m.p for m in layer.modules() if isinstance(m, nn.Dropout)] == [0.25]


def test_output_is_unchanged_straight_after_injection(tiny_llama):
    model, expected = toy(), toy()(X)
    inject_lora(model, ("query", "value"), rank=2, alpha=4)
    assert torch.equal(model(X), expected)

    expected = tiny_llama(input_ids=IDX).logits
    inject_lora(tiny_llama, ("q_proj", "v_proj"), rank=2, alpha=4)
    assert torch.equal(tiny_llama(input_ids=IDX).logits, expected)


def test_only_adapters_are_trainable(tiny_llama):
    model = toy()
    inject_lora(model, ("query", "value"), rank=2, alpha=4)
    assert trainable(model) == 4 * (2 * 8 + 8 * 2)           # four layers, each A (2, 8) and B (8, 2)

    inject_lora(tiny_llama, ("q_proj", "v_proj"), rank=4, alpha=8)
    assert trainable(tiny_llama) == 4 * (4 * 32 + 32 * 4)    # the same count the library reports: 1,024


# ---- train -------------------------------------------------------------------------

def tuned(rank=2):
    """A tiny pretrained-style model with adapters, trained a little so B is no longer zero."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    ))
    original = copy.deepcopy(hf)
    inject_lora(hf, ("q_proj", "v_proj"), rank=rank, alpha=2 * rank)
    model = AsCourseModel(hf)
    y = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(2))
    history = train(model, lambda split: (IDX, y), steps=40, lr=1e-2, eval_every=20)
    return hf, original, history


def test_training_moves_adapters_and_not_one_base_weight():
    hf, original, history = tuned()
    assert history[-1]["train"] < history[0]["train"] - 0.1, "training through the adapters should lower the loss"
    after = hf.state_dict()
    for name, before in original.state_dict().items():
        now = after[name.replace(".weight", ".base.weight")] if name not in after else after[name]
        assert torch.equal(before, now), f"{name} is a base weight and it changed"
    assert all(torch.count_nonzero(layer.B) > 0 for layer in lora_layers(hf).values())


# ---- save and load -----------------------------------------------------------------

def test_adapter_file_is_small_and_complete():
    hf, _, _ = tuned()
    state = lora_state_dict(hf)
    assert len(state) == 8                                   # four layers, an A and a B each
    assert all(key.endswith((".A", ".B")) for key in state)
    assert sum(t.numel() for t in state.values()) == trainable(hf)
    assert "model.layers.0.self_attn.q_proj.A" in state


def test_adapter_round_trip():
    hf, original, _ = tuned()
    fresh = copy.deepcopy(original)
    inject_lora(fresh, ("q_proj", "v_proj"), rank=2, alpha=4)
    assert not torch.allclose(fresh(input_ids=IDX).logits, hf(input_ids=IDX).logits)
    load_lora(fresh, lora_state_dict(hf))
    assert torch.allclose(fresh(input_ids=IDX).logits, hf(input_ids=IDX).logits, atol=1e-6)


def test_saved_adapter_does_not_change_when_training_continues():
    hf, _, _ = tuned()
    state = lora_state_dict(hf)
    snapshot = {key: value.clone() for key, value in state.items()}
    with torch.no_grad():
        for layer in lora_layers(hf).values():
            layer.B.add_(1.0)
    assert all(torch.equal(state[key], snapshot[key]) for key in state), "save copies, not live references"


def test_loading_the_wrong_rank_is_an_error():
    hf, original, _ = tuned(rank=2)
    other = copy.deepcopy(original)
    inject_lora(other, ("q_proj", "v_proj"), rank=4, alpha=8)
    with pytest.raises(ValueError, match="(?s)2.*4|4.*2"):
        load_lora(other, lora_state_dict(hf))


def test_loading_into_the_wrong_layers_is_an_error():
    hf, original, _ = tuned()
    other = copy.deepcopy(original)
    inject_lora(other, ("q_proj",), rank=2, alpha=4)         # no adapters on v_proj here
    with pytest.raises(ValueError, match="v_proj"):
        load_lora(other, lora_state_dict(hf))


def test_loading_half_an_adapter_is_an_error():
    hf, original, _ = tuned()
    other = copy.deepcopy(original)
    inject_lora(other, ("q_proj", "v_proj"), rank=2, alpha=4)
    half = {key: value for key, value in lora_state_dict(hf).items() if "q_proj" in key}
    with pytest.raises(ValueError, match="v_proj"):
        load_lora(other, half)


# ---- merge -------------------------------------------------------------------------

def test_merged_layer_is_a_plain_linear_with_the_same_answers():
    torch.manual_seed(0)
    for bias in (True, False):
        lora = LoRALinear(nn.Linear(8, 5, bias=bias), rank=2, alpha=6)
        with torch.no_grad():
            lora.B.copy_(torch.randn(5, 2))
        merged = lora.merged()
        assert type(merged) is nn.Linear and (merged.bias is not None) == bias
        assert torch.allclose(merged(X), lora(X), atol=1e-6)
        assert not torch.equal(merged.weight, lora.base.weight), "the original layer must be left as it was"


def test_merge_gives_the_same_answers_with_no_adapters_left():
    hf, original, _ = tuned()
    before = hf(input_ids=IDX).logits
    assert merge_lora(hf) == 4
    assert lora_layers(hf) == {}
    assert torch.allclose(hf(input_ids=IDX).logits, before, atol=1e-5)
    assert not torch.allclose(before, original(input_ids=IDX).logits, atol=1e-3), "and they are the TUNED answers"
    assert sum(p.numel() for p in hf.parameters()) == sum(p.numel() for p in original.parameters())
    assert set(hf.state_dict()) == set(original.state_dict()), "same layer names as the model you started with"


def test_merge_twice_and_merge_nothing_are_harmless():
    hf, _, _ = tuned()
    merge_lora(hf)
    before = hf(input_ids=IDX).logits
    assert merge_lora(hf) == 0
    assert torch.equal(hf(input_ids=IDX).logits, before)
    assert merge_lora(toy()) == 0
