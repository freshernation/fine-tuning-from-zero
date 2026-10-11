"""Week 1, Day 1 - tensors and shapes.

Run me:  pytest week-01/day-1 -v
"""

import torch


def test_row_means(load):
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    assert torch.equal(load("shapes.py").row_means(x), torch.tensor([2., 5.]))


def test_normalize_rows_sums_to_one(load):
    x = torch.tensor([[1., 1., 2.], [3., 0., 1.], [2., 2., 4.]])
    out = load("shapes.py").normalize_rows(x)
    assert out.shape == (3, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(3)), f"rows sum to {out.sum(dim=1).tolist()}"
    assert torch.allclose(out[0], torch.tensor([.25, .25, .5]))


def test_one_hot(load):
    out = load("shapes.py").one_hot(torch.tensor([0, 2]), 3)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([[1., 0., 0.], [0., 0., 1.]]))


def test_last_column_keeps_batch_dimension(load):
    x = torch.arange(24.).reshape(2, 3, 4)
    out = load("shapes.py").last_column(x)
    assert out.shape == (2, 4)
    assert torch.equal(out[1], torch.tensor([20., 21., 22., 23.]))
