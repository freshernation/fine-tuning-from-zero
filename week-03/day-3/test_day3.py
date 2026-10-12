"""Week 3, Day 3 - freezing, and LoRA.

Run me:  pytest week-03/day-3 -v

The three counting functions go in lowrank.py, in this folder.
freeze, unfreeze and LoRALinear go in tunelab/lora.py.
"""

import pytest
import torch
from torch import nn

from tunelab.lora import LoRALinear, freeze, unfreeze


# ---- the arithmetic ----------------------------------------------------------------

def test_param_counts(load):
    lowrank = load("lowrank.py")
    assert lowrank.full_update_params(768, 768) == 589_824
    assert lowrank.lora_params(768, 768, 8) == 12_288
    assert lowrank.lora_params(d_in=576, d_out=192, rank=4) == 4 * 576 + 192 * 4


def test_update_matrix_by_hand(load):
    A = torch.tensor([[1., 0., 2.], [0., 1., 0.]])     # (rank 2, d_in 3)
    B = torch.tensor([[1., 0.], [3., 1.]])             # (d_out 2, rank 2)
    delta = load("lowrank.py").update_matrix(A, B, alpha=4)
    # B @ A = [[1, 0, 2], [3, 1, 6]], then times alpha / rank = 4 / 2
    assert torch.equal(delta, torch.tensor([[2., 0., 4.], [6., 2., 12.]]))


def test_update_matrix_has_rank_at_most_r(load):
    torch.manual_seed(0)
    A, B = torch.randn(2, 16), torch.randn(16, 2)
    delta = load("lowrank.py").update_matrix(A, B, alpha=4)
    assert delta.shape == (16, 16)
    assert torch.linalg.matrix_rank(delta) == 2


# ---- freezing ----------------------------------------------------------------------

def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_freeze_then_unfreeze_by_name():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))       # names: 0.weight, 0.bias, 1.weight, 1.bias
    freeze(model)
    assert trainable(model) == 0
    assert unfreeze(model, "1.") == 2          # two tensors: 1.weight and 1.bias
    assert trainable(model) == 22
    assert not model[0].weight.requires_grad
    assert unfreeze(model, "no such layer") == 0


# ---- LoRALinear --------------------------------------------------------------------

def test_lora_starts_as_the_base_layer():
    torch.manual_seed(0)
    base = nn.Linear(10, 3)
    x = torch.randn(4, 10)
    expected = base(x)
    assert torch.equal(LoRALinear(base, rank=4, alpha=8)(x), expected)


def test_lora_shapes_and_what_trains():
    lora = LoRALinear(nn.Linear(10, 3), rank=4, alpha=8)
    named = dict(lora.named_parameters())
    assert named["A"].shape == (4, 10) and named["B"].shape == (3, 4)
    assert {name for name, p in named.items() if p.requires_grad} == {"A", "B"}
    assert isinstance(lora.base, nn.Linear) and not lora.base.weight.requires_grad
    assert not lora.base.bias.requires_grad


def test_B_starts_at_zero_and_A_does_not():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(64, 64), rank=8, alpha=16)
    assert torch.count_nonzero(lora.B) == 0
    assert torch.count_nonzero(lora.A) == lora.A.numel()
    bound = 1 / 64 ** 0.5                       # kaiming_uniform_ with a=sqrt(5): uniform(-bound, bound)
    assert lora.A.abs().max() <= bound + 1e-6
    assert 0.85 < lora.A.std().item() / (bound / 3 ** 0.5) < 1.15, "A is not spread like a uniform draw"


def hand_set():
    base = nn.Linear(2, 2)
    lora = LoRALinear(base, rank=2, alpha=6)    # scaling = 6 / 2 = 3
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
        base.bias.copy_(torch.tensor([0.5, -0.5]))
        lora.A.copy_(torch.tensor([[1., 0.], [0., 2.]]))
        lora.B.copy_(torch.tensor([[0., 1.], [1., 1.]]))
    return lora


def test_lora_computes_base_plus_scaled_BA():
    # base(x) = [1+6+0.5, 3+12-0.5] = [7.5, 14.5];  B @ A = [[0, 2], [1, 2]];  (B @ A) x = [6, 7]
    out = hand_set()(torch.tensor([[1., 3.]]))
    assert torch.allclose(out, torch.tensor([[7.5 + 3 * 6, 14.5 + 3 * 7]]))


def test_scaling_is_alpha_over_rank():
    assert hand_set().scaling == 3.0
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    quiet = LoRALinear(nn.Linear(8, 8), rank=4, alpha=4)
    loud = LoRALinear(quiet.base, rank=4, alpha=8)
    with torch.no_grad():
        quiet.B.copy_(torch.randn(8, 4))
        loud.A.copy_(quiet.A)
        loud.B.copy_(quiet.B)
    base = quiet.base(x)
    assert torch.allclose(loud(x) - base, 2 * (quiet(x) - base), atol=1e-6)


def test_gradients_reach_B_first():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(10, 3), rank=4, alpha=8)
    lora(torch.randn(4, 10)).sum().backward()
    assert torch.count_nonzero(lora.B.grad) > 0
    assert torch.count_nonzero(lora.A.grad) == 0      # today's README asks you why
    assert lora.base.weight.grad is None


def test_dropout_only_touches_the_lora_branch():
    torch.manual_seed(0)
    base = nn.Linear(10, 3)
    lora = LoRALinear(base, rank=4, alpha=8, dropout=0.5)
    lora.train()
    x = torch.randn(4, 10)
    assert torch.equal(lora(x), base(x)), "with B at zero, dropout must not be able to change anything"
    with torch.no_grad():
        lora.B.copy_(torch.randn(3, 4))
    assert not torch.equal(lora(x), lora(x)), "in training mode, with B set, two calls should differ"
    lora.eval()
    assert torch.equal(lora(x), lora(x))


def test_dropout_is_on_the_input_not_in_the_middle():
    # one input feature: dropping x either removes the whole adapter path or doubles it.
    # Dropout applied after A instead would drop some of the 8 middle values, and give sizes in between.
    torch.manual_seed(0)
    base = nn.Linear(1, 1, bias=False)
    lora = LoRALinear(base, rank=8, alpha=8, dropout=0.5)       # scaling 1
    with torch.no_grad():
        base.weight.zero_()
        lora.A.fill_(1.0)
        lora.B.fill_(1.0)
    lora.train()
    seen = {round(lora(torch.ones(1, 1)).item(), 4) for _ in range(60)}
    assert seen == {0.0, 16.0}, f"expected the adapter path to be all there (16) or all gone (0), saw {sorted(seen)}"


def test_keeps_the_bias_and_works_without_one():
    torch.manual_seed(0)
    x = torch.randn(4, 10)
    for bias in (True, False):
        base = nn.Linear(10, 3, bias=bias)
        assert torch.equal(LoRALinear(base, 2, 4)(x), base(x))


def test_lora_works_on_batches_of_sequences():
    lora = LoRALinear(nn.Linear(10, 3), rank=4, alpha=8)
    assert lora(torch.randn(2, 7, 10)).shape == (2, 7, 3)


def test_adapter_is_created_like_the_base_layer():
    # a stand-in for "on the same device": the adapter must take its dtype from the base
    base = nn.Linear(10, 3).double()
    lora = LoRALinear(base, rank=4, alpha=8)
    assert lora.A.dtype == torch.float64 and lora.B.dtype == torch.float64
    assert lora(torch.randn(4, 10, dtype=torch.float64)).dtype == torch.float64


def test_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(10, 3), rank=0, alpha=8)
