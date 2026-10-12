"""Week 2, Day 1 - embeddings, position, and averaging the past.

Run me:  pytest week-02/day-1 -v

bag_of_past goes in context.py, in this folder. Embedder goes in tunelab/gpt.py.
"""

import pytest
import torch

from tunelab.gpt import Embedder


def test_bag_of_past_is_a_running_mean(load):
    x = torch.tensor([[[2., 0.], [4., 2.], [0., 4.]]])
    out = load("context.py").bag_of_past(x)
    assert torch.allclose(out, torch.tensor([[[2., 0.], [3., 1.], [2., 2.]]]))


def test_bag_of_past_never_looks_ahead(load):
    torch.manual_seed(0)
    x = torch.randn(2, 6, 3)
    y = x.clone()
    y[:, 4:] += 1.0                            # change the future...
    bag_of_past = load("context.py").bag_of_past
    assert torch.equal(bag_of_past(x)[:, :4], bag_of_past(y)[:, :4])   # ...the past must not notice


def test_embedder_shape_and_parameters():
    emb = Embedder(vocab_size=10, block_size=8, n_embd=16)
    assert emb(torch.zeros((4, 8), dtype=torch.long)).shape == (4, 8, 16)
    assert sorted(tuple(p.shape) for p in emb.parameters()) == [(8, 16), (10, 16)]


def test_same_token_different_position_different_vector():
    torch.manual_seed(0)
    out = Embedder(10, 8, 16)(torch.tensor([[3, 3, 3]]))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_same_position_different_token_different_vector():
    torch.manual_seed(0)
    emb = Embedder(10, 8, 16)
    a, b = emb(torch.tensor([[3, 1]])), emb(torch.tensor([[3, 2]]))
    assert torch.equal(a[0, 0], b[0, 0])
    assert not torch.allclose(a[0, 1], b[0, 1])


def test_embedder_accepts_shorter_sequences_and_rejects_longer():
    emb = Embedder(10, 8, 16)
    assert emb(torch.zeros((1, 3), dtype=torch.long)).shape == (1, 3, 16)
    with pytest.raises(ValueError, match="(?s)(8.*9|9.*8)"):   # the message names both lengths
        emb(torch.zeros((1, 9), dtype=torch.long))
