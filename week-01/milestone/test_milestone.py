"""Week 1 milestone - a bigram model, trained by a loop you wrote.

Run me:  pytest week-01/milestone -v      (takes under a minute)
"""

from pathlib import Path

import torch

from tunelab.bigram import Bigram

TEXT = (Path(__file__).parents[2] / "data" / "tiny.txt").read_text()
TARGET = 2.59


def test_bigram_reaches_target_validation_loss(load):
    result = load("train_bigram.py").run(TEXT, steps=2000, seed=0)
    start, end = result["history"][0]["val"], result["val_loss"]
    assert start > 4.0, f"an untrained model should start near ln(65) = 4.17, got {start:.2f}"
    assert end < TARGET, f"validation loss {end:.3f} did not get under {TARGET}"
    assert len(result["sample"]) == 200


def test_same_seed_same_run(load):
    run = load("train_bigram.py").run
    first = run(TEXT[:20000], steps=50, seed=3)
    again = run(TEXT[:20000], steps=50, seed=3)
    assert first["sample"] == again["sample"]


def test_overfitting_one_batch_drives_loss_to_zero(load):
    torch.manual_seed(0)
    # a batch where every character has exactly one possible successor: a->b->c->a
    x = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]])
    y = torch.tensor([[1, 2, 0, 1, 2, 0, 1, 2]])
    assert load("train_bigram.py").overfit_one_batch(Bigram(3), x, y) < 0.01
