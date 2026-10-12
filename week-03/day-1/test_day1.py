"""Week 3, Day 1 - a pretrained model is a bigger one of yours.

Run me:  pytest week-03/day-1 -v

render_chatml goes in chat.py, in this folder. The rest goes in tunelab/pretrained.py.
Nothing here downloads anything: `tiny_llama` is a real Hugging Face model, with the
same architecture as the course's pretrained one, built tiny from a config.
"""

import pytest
import torch
from torch.nn import functional as F

from tunelab.loop import train
from tunelab.pretrained import AsCourseModel, layer_table


def test_wrapper_follows_the_course_convention(tiny_llama):
    model = AsCourseModel(tiny_llama)
    idx = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx)
    assert isinstance(logits, torch.Tensor) and logits.shape == (2, 16, 100)
    assert loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()


def test_wrapper_does_not_shift_the_targets_again(tiny_llama):
    # the course's batches already hold "the next token" at every position
    torch.manual_seed(0)
    model = AsCourseModel(tiny_llama)
    idx = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))
    logits, loss = model(idx, targets)
    expected = F.cross_entropy(logits.view(-1, 100), targets.view(-1))
    assert abs(loss.item() - expected.item()) < 1e-6


def test_wrapper_exposes_the_pretrained_weights_to_the_optimizer(tiny_llama):
    wrapped = sum(p.numel() for p in AsCourseModel(tiny_llama).parameters())
    assert wrapped == sum(p.numel() for p in tiny_llama.parameters()) > 0


def test_the_wrapped_model_is_kept_as_hf(tiny_llama):
    assert AsCourseModel(tiny_llama).hf is tiny_llama, "Friday's notebook reaches the model through .hf"


def test_wrapped_model_trains_with_the_week_1_loop(tiny_llama):
    torch.manual_seed(0)
    model = AsCourseModel(tiny_llama)
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4, 16))
    history = train(model, lambda split: (x, y), steps=30, lr=1e-2, eval_every=10)
    assert history[-1]["train"] < history[0]["train"] - 0.5


def test_layer_table_is_sorted_and_complete(tiny_llama):
    table = layer_table(tiny_llama)
    assert all(isinstance(name, str) and isinstance(shape, tuple) and isinstance(count, int)
               for name, shape, count in table)
    counts = [count for _, _, count in table]
    assert counts == sorted(counts, reverse=True)
    assert sum(counts) == sum(p.numel() for p in tiny_llama.parameters())
    assert table[0] == ("model.embed_tokens.weight", (100, 32), 3200)
    assert not any(name == "lm_head.weight" for name, _, _ in table), "lm_head shares the token table: it is not a second tensor"


SYSTEM = {"role": "system", "content": "You are terse."}
HELLO = {"role": "user", "content": "Hello"}
HI = {"role": "assistant", "content": "Hi."}


def test_render_chatml_one_turn(load):
    assert load("chat.py").render_chatml([HELLO]) == "<|im_start|>user\nHello<|im_end|>\n"


def test_render_chatml_several_turns_and_a_generation_prompt(load):
    text = load("chat.py").render_chatml([SYSTEM, HELLO, HI, HELLO], add_generation_prompt=True)
    assert text == (
        "<|im_start|>system\nYou are terse.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_render_chatml_keeps_content_exactly_as_given(load):
    odd = {"role": "user", "content": "  two\nlines, and a trailing space "}
    assert load("chat.py").render_chatml([odd]) == "<|im_start|>user\n  two\nlines, and a trailing space <|im_end|>\n"


def test_render_chatml_rejects_an_unknown_role(load):
    with pytest.raises(ValueError, match="narrator"):
        load("chat.py").render_chatml([{"role": "narrator", "content": "Meanwhile"}])
