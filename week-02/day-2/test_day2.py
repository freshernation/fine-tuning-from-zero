"""Week 2, Day 2 - self-attention, one head.

Run me:  pytest week-02/day-2 -v

attend goes in attend.py, in this folder. causal_mask and Head go in tunelab/attention.py.
"""

import torch

from tunelab.attention import Head, causal_mask


def test_causal_mask():
    mask = causal_mask(3)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, False, False], [True, True, False], [True, True, True]]


def test_equal_scores_give_a_running_mean(load):
    # zero queries and keys: every allowed position gets equal weight - yesterday's average
    v = torch.tensor([[[2., 0.], [4., 2.], [0., 4.]]])
    z = torch.zeros(1, 3, 2)
    out = load("attend.py").attend(z, z, v, causal_mask(3))
    assert torch.allclose(out, torch.tensor([[[2., 0.], [3., 1.], [2., 2.]]]))


def test_scores_are_scaled_by_root_head_size(load):
    # head_size is 4, so scores are divided by 2. Position 1 scores [0, 4] -> [0, 2].
    q = torch.tensor([[[0., 0., 0., 0.], [2., 0., 0., 0.]]])
    k = torch.tensor([[[0., 0., 0., 0.], [2., 0., 0., 0.]]])
    v = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
    out = load("attend.py").attend(q, k, v, causal_mask(2))
    weights = torch.softmax(torch.tensor([0., 2.]), dim=0)
    assert torch.allclose(out[0, 1, :2], weights)
    assert torch.allclose(out[0, 0, :2], torch.tensor([1., 0.]))   # position 0 hears only itself


def test_attend_respects_whatever_mask_it_is_given(load):
    # a mask that lets every position see everything: plain averaging, no causality
    v = torch.tensor([[[2., 0.], [4., 2.], [0., 4.]]])
    z = torch.zeros(1, 3, 2)
    out = load("attend.py").attend(z, z, v, torch.ones(3, 3, dtype=torch.bool))
    assert torch.allclose(out, torch.tensor([[[2., 2.]] * 3]))


def test_head_shape_and_no_trainable_mask():
    head = Head(n_embd=16, head_size=4, block_size=8)
    assert head(torch.randn(2, 8, 16)).shape == (2, 8, 4)
    assert sorted(tuple(p.shape) for p in head.parameters()) == [(4, 16)] * 3
    assert any(b.shape == (8, 8) for b in head.buffers()), "the mask should be a registered buffer"


def test_head_never_looks_ahead():
    torch.manual_seed(0)
    head = Head(16, 4, 8)
    x = torch.randn(2, 8, 16)
    y = x.clone()
    y[:, 5:] += 1.0                            # change positions 5, 6, 7
    out_x, out_y = head(x), head(y)
    assert torch.isfinite(out_x).all()
    assert torch.allclose(out_x[:, :5], out_y[:, :5], atol=1e-6)   # 0..4 must not notice
    assert not torch.allclose(out_x[:, 5:], out_y[:, 5:])


def test_head_dropout_drops_connections_not_outputs():
    # Dropout on the attention WEIGHTS cuts whole connections between positions.
    # Dropout on the output would instead blank single numbers inside a position's vector.
    torch.manual_seed(0)
    head = Head(16, 4, 8, dropout=0.5)
    head.train()
    x = torch.randn(64, 8, 16)
    out_a, out_b = head(x), head(x)
    assert not torch.equal(out_a, out_b), "dropout should make two training-mode calls differ"
    zeros = (out_a == 0)
    partly_blank = zeros.any(dim=-1) & ~zeros.all(dim=-1)
    assert not partly_blank.any(), "single numbers were zeroed: that is dropout on the output"
    head.eval()
    assert torch.equal(head(x), head(x))


def test_head_accepts_shorter_sequences():
    torch.manual_seed(0)
    head = Head(16, 4, 8)
    x = torch.randn(1, 8, 16)
    short = head(x[:, :3])
    assert short.shape == (1, 3, 4)
    assert torch.allclose(short, head(x)[:, :3], atol=1e-6)   # and it is the same answer
