"""Week 1, Day 4 - text as numbers, and a first language model.

Run me:  pytest week-01/day-4 -v

Today's code goes in tunelab/text.py and tunelab/bigram.py.
"""

import math

import pytest
import torch

from tunelab.bigram import Bigram
from tunelab.text import CharTokenizer, make_get_batch


def test_tokenizer_round_trip_and_order():
    tok = CharTokenizer("hello world")
    assert tok.vocab_size == 8
    assert tok.encode(" d") == [0, 1]          # ids follow sorted character order
    assert tok.decode(tok.encode("hello")) == "hello"


def test_encode_unknown_character_names_it():
    with pytest.raises(ValueError, match="z"):
        CharTokenizer("abc").encode("az")


def test_get_batch_shapes_and_shift():
    torch.manual_seed(0)
    data = torch.arange(100)
    x, y = make_get_batch(data, block_size=8, batch_size=4)("train")
    assert x.shape == y.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)               # y is x, one position later
    assert y.max() < 90                        # train never reads the validation tail


def test_get_batch_val_reads_only_the_tail():
    torch.manual_seed(0)
    x, _ = make_get_batch(torch.arange(100), block_size=4, batch_size=16)("val")
    assert x.min() >= 90


def test_get_batch_rejects_data_that_is_too_short():
    with pytest.raises(ValueError):
        make_get_batch(torch.arange(5), block_size=8, batch_size=2)


def test_bigram_forward():
    torch.manual_seed(0)
    model = Bigram(vocab_size=10)
    idx = torch.randint(0, 10, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 10) and loss is None
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(10)) < 1.0   # untrained: near ln(vocab_size)


def test_generate_appends_tokens():
    torch.manual_seed(0)
    start = torch.zeros((2, 1), dtype=torch.long)
    out = Bigram(10).generate(start, max_new_tokens=5)
    assert out.shape == (2, 6) and out.max() < 10
    assert torch.equal(out[:, :1], start)      # what you gave it is still at the front
