"""Week 2, Day 3 - many heads, and the block.

Run me:  pytest week-02/day-3 -v

MultiHead goes in tunelab/attention.py. Block goes in tunelab/gpt.py.
"""

import pytest
import torch

from tunelab.attention import MultiHead
from tunelab.gpt import Block


def looks_ahead(module):
    """True if changing positions 5.. changes the module's output at positions 0..4."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    y = x.clone()
    y[:, 5:] += 1.0
    out_x, out_y = module(x), module(y)
    assert torch.isfinite(out_x).all()
    assert not torch.allclose(out_x[:, 5:], out_y[:, 5:]), "the changed positions should change"
    return not torch.allclose(out_x[:, :5], out_y[:, :5], atol=1e-5)


def test_multihead_shape():
    assert MultiHead(16, 4, 8)(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_multihead_is_heads_side_by_side_then_one_linear_layer():
    shapes = sorted(tuple(p.shape) for p in MultiHead(n_embd=16, n_head=4, block_size=8).parameters())
    # 4 heads x (key, query, value), each (4, 16); then a (16, 16) layer and its bias
    assert shapes == [(4, 16)] * 12 + [(16,), (16, 16)]


def set_mixing_layer(multihead, weight):
    """Overwrite MultiHead's final (n_embd, n_embd) layer; zero its bias."""
    with torch.no_grad():
        for p in multihead.parameters():
            if p.shape == weight.shape:
                p.copy_(weight)
            elif p.shape == (weight.shape[0],):
                p.zero_()


def test_multihead_output_goes_through_the_mixing_layer():
    multihead = MultiHead(16, 4, 8)
    set_mixing_layer(multihead, torch.zeros(16, 16))
    assert torch.count_nonzero(multihead(torch.randn(2, 8, 16))) == 0


def test_multihead_lays_heads_side_by_side():
    # with the mixing layer set to "pass straight through", the output is the heads
    # themselves, end to end - and four different heads give four different chunks
    torch.manual_seed(0)
    multihead = MultiHead(16, 4, 8)
    set_mixing_layer(multihead, torch.eye(16))
    out = multihead(torch.randn(2, 8, 16))
    chunks = out.split(4, dim=-1)
    assert not torch.allclose(chunks[0], chunks[1]) and not torch.allclose(chunks[2], chunks[3])


def test_multihead_rejects_indivisible_width():
    with pytest.raises(ValueError):
        MultiHead(n_embd=16, n_head=3, block_size=8)


def test_multihead_never_looks_ahead():
    assert not looks_ahead(MultiHead(16, 4, 8))


def test_block_keeps_shape():
    assert Block(16, 4, 8)(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_block_never_looks_ahead():
    assert not looks_ahead(Block(16, 4, 8))


def test_block_is_residual():
    # with every weight at zero both branches output zero, so only the skip path is left
    block = Block(16, 4, 8)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.randn(2, 8, 16)
    assert torch.equal(block(x), x)


def test_block_has_two_norms_and_a_four_times_wide_mlp():
    shapes = sorted(tuple(p.shape) for p in Block(16, 4, 8).parameters())
    assert shapes == sorted(
        [(4, 16)] * 12 + [(16, 16), (16,)]         # attention: 4 heads, then the mixing layer
        + [(64, 16), (64,), (16, 64), (16,)]       # mlp: out to 4x, and back
        + [(16,)] * 4                              # two LayerNorms, a weight and a bias each
    )


def test_block_normalises_before_attention_not_after():
    # pre-norm: a LayerNorm must be the first thing the block's input meets
    torch.manual_seed(0)
    block = Block(16, 4, 8)
    x = torch.randn(2, 8, 16)
    seen = []
    for module in block.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.register_forward_hook(lambda m, inputs, out: seen.append(inputs[0]))
    block(x)
    assert len(seen) == 2, "each LayerNorm should run exactly once"
    assert any(torch.equal(inp, x) for inp in seen), "no LayerNorm was applied to the block's input"
    assert not all(torch.equal(inp, x) for inp in seen), "the second norm should see x after attention was added"


def test_dropout_is_everywhere_it_should_be():
    # one in each of the 4 heads, one after the mixing layer, one at the end of the mlp
    block = Block(16, 4, 8, dropout=0.5)
    drops = [m for m in block.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(drops) == 6 and all(d.p == 0.5 for d in drops)


def test_dropout_is_off_in_eval_mode():
    torch.manual_seed(0)
    block = Block(16, 4, 8, dropout=0.5)
    x = torch.randn(2, 8, 16)
    block.eval()
    assert torch.equal(block(x), block(x))
    block.train()
    assert not torch.equal(block(x), block(x))
