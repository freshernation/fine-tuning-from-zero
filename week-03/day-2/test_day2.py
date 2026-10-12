"""Week 3, Day 2 - full fine-tuning, and what it costs.

Run me:  pytest week-03/day-2 -v

fine_tune_and_measure goes in full.py, in this folder. The two counting functions
go in tunelab/finetune.py.

The "pretrained model" here is your own GPT from week 2, pretrained inside the test
on a toy language where every token is followed by the next one up. It is then
fine-tuned on a second language where every token is followed by the one two up.
"""

import torch
from torch import nn

from tunelab.finetune import count_trainable, training_memory_bytes
from tunelab.gpt import GPT
from tunelab.loop import train

VOCAB = 11
MARGIN = 3.0


def language(step):
    """get_batch for a language in which each token is followed by (token + step) % VOCAB."""

    def get_batch(split):
        start = torch.randint(0, VOCAB, (16, 1))
        x = (start + step * torch.arange(8)) % VOCAB
        return x, (x + step) % VOCAB

    return get_batch


OLD, NEW = language(1), language(2)


def pretrained():
    torch.manual_seed(0)
    model = GPT(vocab_size=VOCAB, block_size=8, n_embd=16, n_head=4, n_layer=2)
    train(model, OLD, steps=200, lr=1e-2)
    return model


def test_count_trainable_counts_only_what_can_change():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))       # 110 + 22
    assert count_trainable(model) == (132, 132)
    model[0].requires_grad_(False)
    assert count_trainable(model) == (22, 132)


def test_training_memory_is_four_times_the_weights_when_everything_trains():
    layer = nn.Linear(10, 10)                                        # 110 numbers, 4 bytes each
    assert training_memory_bytes(layer) == 4 * 110 * 4


def test_training_memory_falls_when_layers_are_frozen():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 2))
    model[0].requires_grad_(False)
    # every weight is still stored once; only the 22 trainable ones need 3 more copies
    assert training_memory_bytes(model) == 4 * (132 + 3 * 22)


def test_fine_tuning_learns_the_new_task(load):
    result = load("full.py").fine_tune_and_measure(pretrained(), NEW, OLD, steps=100, lr=1e-2)
    assert set(result) == {"new_before", "new_after", "old_before", "old_after"}
    assert all(isinstance(v, float) for v in result.values())
    assert result["new_after"] < result["new_before"] / 2


def test_fine_tuning_forgets_the_old_task(load):
    result = load("full.py").fine_tune_and_measure(pretrained(), NEW, OLD, steps=100, lr=1e-2)
    assert result["old_before"] < 0.5, "the pretrained model should be good at the old language"
    assert result["old_after"] > result["old_before"] + MARGIN, "and noticeably worse at it afterwards"


def test_the_old_task_is_only_ever_measured_never_trained_on(load):
    asked = []

    def old_spy(split):
        asked.append(split)
        return OLD(split)

    model = pretrained()
    load("full.py").fine_tune_and_measure(model, NEW, old_spy, steps=20, lr=1e-2)
    assert asked and set(asked) == {"val"}, "the old task is measured on validation batches, and nothing else"
    assert len(asked) == 40, "measured twice - before and after - with estimate_loss's usual 20 batches"
    assert model.training


def test_before_is_measured_before(load):
    # with zero training steps nothing can have changed
    result = load("full.py").fine_tune_and_measure(pretrained(), NEW, OLD, steps=0, lr=1e-2)
    assert abs(result["new_before"] - result["new_after"]) < 0.3
    assert abs(result["old_before"] - result["old_after"]) < 0.3
