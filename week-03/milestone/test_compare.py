"""Week 3 milestone, part 2 - three ways to fine-tune.

Run me:  pytest week-03/milestone/test_compare.py -v

Miniature versions of Friday's Colab experiment: the tiny pretrained-style model,
and two toy languages (next token is +1, or +2).
"""

import copy

import pytest
import torch

from tunelab.lora import LoRALinear
from tunelab.pretrained import AsCourseModel

VOCAB = 100


def language(step):
    def get_batch(split):
        start = torch.randint(0, VOCAB, (8, 1))
        x = (start + step * torch.arange(16)) % VOCAB
        return x, (x + step) % VOCAB

    return get_batch


OLD, NEW = language(1), language(2)


def go(load, method, tiny_llama, **extra):
    torch.manual_seed(0)
    model = AsCourseModel(tiny_llama)
    result = load("compare.py").run(method, model, NEW, OLD, steps=60, lr=1e-2, rank=4, alpha=8, **extra)
    return result, tiny_llama


@pytest.mark.parametrize("method", ["full", "last_block", "lora"])
def test_each_method_learns_the_new_task(load, tiny_llama, method):
    result, _ = go(load, method, tiny_llama)
    assert set(result) == {"trainable", "total", "memory_bytes", "new_before", "new_after", "old_before", "old_after"}
    assert result["new_after"] < result["new_before"] - DROP[method]


DROP = {"full": 1.0, "last_block": 0.3, "lora": 0.08}       # measured: 4.6, 0.58 and 0.16


def test_what_each_method_trains(load, tiny_llama):
    counts = {}
    for method in ("full", "last_block", "lora"):
        result, _ = go(load, method, copy.deepcopy(tiny_llama))
        counts[method] = result
    total = counts["full"]["total"]
    assert counts["full"]["trainable"] == total == 27040
    assert counts["last_block"]["trainable"] == 10_304         # every parameter of the second of two blocks
    assert counts["lora"]["trainable"] == 1024                 # rank 4 on q_proj and v_proj, two blocks
    assert counts["lora"]["total"] == total + 1024
    assert counts["full"]["memory_bytes"] > counts["last_block"]["memory_bytes"] > counts["lora"]["memory_bytes"]
    assert counts["full"]["memory_bytes"] == 4 * 4 * total


def test_lora_leaves_the_base_model_bit_for_bit(load, tiny_llama):
    before = copy.deepcopy(tiny_llama.state_dict())
    _, model = go(load, "lora", tiny_llama)
    after = model.state_dict()
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    for name, tensor in before.items():
        now = after[name] if name in after else after[name.replace(".weight", ".base.weight")]
        assert torch.equal(tensor, now), f"{name} changed"


def test_last_block_leaves_the_first_block_alone(load, tiny_llama):
    before = copy.deepcopy(tiny_llama.state_dict())
    _, model = go(load, "last_block", tiny_llama)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if "layers.0." in k or "embed" in k)
    assert any(not torch.equal(before[k], after[k]) for k in before if "layers.1." in k)


def test_measurements_are_taken_before_and_after(load, tiny_llama):
    torch.manual_seed(0)
    result = load("compare.py").run("full", AsCourseModel(tiny_llama), NEW, OLD, steps=0, lr=1e-2)
    assert abs(result["new_before"] - result["new_after"]) < 0.2
    assert abs(result["old_before"] - result["old_after"]) < 0.2


def test_unknown_method_is_an_error(load, tiny_llama):
    with pytest.raises(ValueError, match="distill"):
        load("compare.py").run("distill", AsCourseModel(tiny_llama), NEW, OLD, steps=1, lr=1e-2)
