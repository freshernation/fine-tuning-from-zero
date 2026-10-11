"""Week 1, Day 3 - the training loop.

Run me:  pytest week-01/day-3 -v

Today's code goes in tunelab/loop.py, not in this folder: you will use it again
in Weeks 2 and 3.
"""

import torch
from torch import nn

from tunelab.loop import estimate_loss, train


class Line(nn.Module):
    """y = w*x + b, speaking the course's convention: model(x, targets) -> (out, loss)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x, targets=None):
        out = self.lin(x)
        loss = None if targets is None else ((out - targets) ** 2).mean()
        return out, loss


def get_batch(split):
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    x = torch.rand(32, 1, generator=g) * 2 - 1
    return x, 3 * x - 1


def test_train_reduces_loss_and_returns_history():
    torch.manual_seed(0)
    model = Line()
    history = train(model, get_batch, steps=300, lr=0.05, eval_every=100)
    assert [h["step"] for h in history] == [0, 100, 200, 299]
    assert set(history[0]) == {"step", "train", "val"}
    assert history[-1]["val"] < 0.01 < history[0]["val"]


def test_estimate_loss_is_a_float_and_changes_nothing():
    torch.manual_seed(0)
    model = Line()
    before = [p.clone() for p in model.parameters()]
    value = estimate_loss(model, get_batch, "val", batches=5)
    assert isinstance(value, float)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_estimate_loss_puts_the_model_back_in_train_mode():
    model = Line()
    model.train()
    estimate_loss(model, get_batch, "val", batches=2)
    assert model.training
