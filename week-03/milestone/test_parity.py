"""Week 3 milestone, part 1 - is your LoRA the same as the library's?

Run me:  pytest week-03/milestone/test_parity.py -v

This is Project 1's gate. `peft` is the library everybody uses for LoRA. Each test
builds the same adapters twice - once with peft, once with your tunelab/lora.py -
gives both the same A and B, and checks they agree to five decimal places.

This is the ONE place in weeks 1-3 where peft is allowed. It is the answer key.
"""

import copy

import pytest
import torch
from torch import nn

from tunelab.lora import LoRALinear, inject_lora, merge_lora

peft = pytest.importorskip("peft")

RANK, ALPHA = 4, 8


class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(12, 8)
        self.second = nn.Linear(8, 6, bias=False)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


def both_ways(base, targets):
    """The same model with peft's adapters, and with yours, holding identical A and B."""
    theirs = peft.get_peft_model(
        copy.deepcopy(base),
        peft.LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=list(targets), lora_dropout=0.0),
    )
    mine = copy.deepcopy(base)
    inject_lora(mine, targets, rank=RANK, alpha=ALPHA)

    their_layers = {name: m for name, m in theirs.named_modules() if hasattr(m, "lora_A")}
    my_layers = {name: m for name, m in mine.named_modules() if isinstance(m, LoRALinear)}
    assert len(their_layers) == len(my_layers) > 0, "you and peft wrapped a different number of layers"

    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, layer in my_layers.items():
            their = next(m for their_name, m in their_layers.items() if their_name.endswith(name))
            A = their.lora_A["default"].weight
            B = their.lora_B["default"].weight
            B.copy_(torch.randn(B.shape, generator=generator) * 0.1)   # B starts at zero: make it count
            assert layer.A.shape == A.shape and layer.B.shape == B.shape, f"{name}: your A or B is a different shape"
            layer.A.copy_(A)
            layer.B.copy_(B)
    return theirs, mine


X = torch.randn(5, 12, generator=torch.Generator().manual_seed(1))
IDX = torch.randint(0, 100, (2, 16), generator=torch.Generator().manual_seed(1))


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_same_output_as_peft_on_a_small_model():
    theirs, mine = both_ways(Pair(), ("first", "second"))
    assert not torch.allclose(mine(X), Pair()(X)), "the adapters should be changing the output"
    assert torch.allclose(mine(X), theirs(X), atol=1e-5)


def test_same_output_as_peft_on_the_pretrained_architecture(tiny_llama):
    theirs, mine = both_ways(tiny_llama, ("q_proj", "v_proj"))
    assert torch.allclose(mine(input_ids=IDX).logits, theirs(input_ids=IDX).logits, atol=1e-5)


def test_same_trainable_count_as_peft(tiny_llama):
    theirs, mine = both_ways(tiny_llama, ("q_proj", "v_proj"))
    assert trainable(mine) == trainable(theirs) == 1024


def test_same_gradients_as_peft():
    theirs, mine = both_ways(Pair(), ("first", "second"))
    mine(X).sum().backward()
    theirs(X).sum().backward()
    their_first = next(m for name, m in theirs.named_modules() if name.endswith("first") and hasattr(m, "lora_A"))
    assert torch.allclose(mine.first.A.grad, their_first.lora_A["default"].weight.grad, atol=1e-5)
    assert torch.allclose(mine.first.B.grad, their_first.lora_B["default"].weight.grad, atol=1e-5)


def test_same_merged_output_as_peft(tiny_llama):
    theirs, mine = both_ways(tiny_llama, ("q_proj", "v_proj"))
    merged_theirs = theirs.merge_and_unload()
    merge_lora(mine)
    assert torch.allclose(mine(input_ids=IDX).logits, merged_theirs(input_ids=IDX).logits, atol=1e-5)
    assert set(mine.state_dict()) == set(merged_theirs.state_dict())


def test_same_merged_layers_as_peft_with_and_without_a_bias():
    theirs, mine = both_ways(Pair(), ("first", "second"))
    merged_theirs = theirs.merge_and_unload()
    merge_lora(mine)
    assert torch.allclose(mine(X), merged_theirs(X), atol=1e-5)
    assert torch.allclose(mine.first.bias, merged_theirs.first.bias) and mine.second.bias is None


def test_same_starting_point_as_peft():
    # a fresh adapter: B exactly zero in both, and A spread the same way
    torch.manual_seed(0)
    mine_std, their_std = [], []
    for _ in range(20):
        base = Pair()
        theirs = peft.get_peft_model(copy.deepcopy(base), peft.LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["first"]))
        mine = copy.deepcopy(base)
        inject_lora(mine, ("first",), rank=RANK, alpha=ALPHA)
        their_first = next(m for m in theirs.modules() if hasattr(m, "lora_A"))
        assert torch.count_nonzero(mine.first.B) == 0 == torch.count_nonzero(their_first.lora_B["default"].weight)
        mine_std.append(mine.first.A.std().item())
        their_std.append(their_first.lora_A["default"].weight.std().item())
    ratio = (sum(mine_std) / 20) / (sum(their_std) / 20)
    assert 0.9 < ratio < 1.1, f"your A is spread {ratio:.2f}x as wide as peft's"
