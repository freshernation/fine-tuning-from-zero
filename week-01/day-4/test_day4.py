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


def test_get_batch_train_never_reads_the_validation_tail():
    torch.manual_seed(0)
    _, y = make_get_batch(torch.arange(100), block_size=8, batch_size=512)("train")
    assert y.max() == 89                       # reaches the end of train, and stops there


def test_get_batch_val_reads_only_the_tail():
    torch.manual_seed(0)
    x, _ = make_get_batch(torch.arange(100), block_size=4, batch_size=512)("val")
    assert x.min() == 90


def test_get_batch_rejects_data_that_is_too_short():
    with pytest.raises(ValueError):
        make_get_batch(torch.arange(5), block_size=8, batch_size=2)


def test_get_batch_rejects_a_validation_split_that_is_too_short():
    # 50 tokens, 10% validation = 5 tokens: not enough for one example of 8 + 1
    with pytest.raises(ValueError):
        make_get_batch(torch.arange(50), block_size=8, batch_size=2)


def fill_table(model, table):
    """Overwrite the model's one trainable tensor - its lookup table."""
    params = list(model.parameters())
    assert len(params) == 1 and params[0].shape == table.shape, (
        "Bigram should hold exactly one trainable tensor: a (vocab_size, vocab_size) table"
    )
    with torch.no_grad():
        params[0].copy_(table)


def test_bigram_forward_shapes():
    model = Bigram(vocab_size=10)
    idx = torch.randint(0, 10, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 10) and loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()


def test_a_model_that_knows_nothing_scores_ln_vocab():
    model = Bigram(vocab_size=10)
    fill_table(model, torch.zeros(10, 10))     # every next character equally likely
    idx = torch.randint(0, 10, (4, 8))
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(10)) < 1e-4


def test_generate_follows_the_table():
    model = Bigram(vocab_size=3)
    table = torch.full((3, 3), -20.0)
    table[0, 1] = table[1, 2] = table[2, 0] = 20.0     # 0 -> 1 -> 2 -> 0, with certainty
    fill_table(model, table)
    out = model.generate(torch.tensor([[0], [2]]), max_new_tokens=5)
    assert out.tolist() == [[0, 1, 2, 0, 1, 2], [2, 0, 1, 2, 0, 1]]


def test_generate_samples_rather_than_always_taking_the_likeliest():
    torch.manual_seed(0)
    model = Bigram(vocab_size=4)
    fill_table(model, torch.zeros(4, 4))       # all equally likely: samples must vary
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=50)
    assert out.shape == (1, 51)
    assert len(set(out[0, 1:].tolist())) == 4
