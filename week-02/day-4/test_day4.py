"""Week 2, Day 4 - the whole GPT, and a training run that survives being killed.

Run me:  pytest week-02/day-4 -v

GPT and count_params go in tunelab/gpt.py. The rest goes in tunelab/checkpoint.py.
"""

import math
import os

import pytest
import torch
from torch import nn

from tunelab.checkpoint import load, save, train_in_chunks
from tunelab.gpt import GPT, count_params

CFG = dict(vocab_size=11, block_size=8, n_embd=16, n_head=4, n_layer=2)


def test_gpt_forward_shapes():
    model = GPT(**CFG)
    idx = torch.randint(0, 11, (4, 8))
    logits, loss = model(idx)
    assert logits.shape == (4, 8, 11) and loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()


def test_gpt_starts_near_ln_vocab():
    # small starting weights make the first prediction almost perfectly even
    torch.manual_seed(0)
    model = GPT(**CFG)
    idx = torch.randint(0, 11, (16, 8))
    _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(11)) < 0.1


def test_gpt_starting_weights_are_small():
    torch.manual_seed(0)
    model = GPT(vocab_size=11, block_size=8, n_embd=32, n_head=4, n_layer=2)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            assert 0.012 < module.weight.std().item() < 0.028, f"{module} does not start at std 0.02"
        if isinstance(module, nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0, f"{module} has a bias that does not start at zero"


def test_gpt_has_the_parts_it_should():
    model = GPT(**CFG)
    norms = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    assert len(norms) == 2 * CFG["n_layer"] + 1, "two per block, and one after the last block"
    shapes = [tuple(p.shape) for p in model.parameters()]
    assert shapes.count((11, 16)) == 2, "the token table, and a final Linear(n_embd, vocab_size)"


def test_gpt_never_looks_ahead():
    torch.manual_seed(0)
    model = GPT(**CFG)
    a = torch.randint(0, 11, (2, 8))
    b = a.clone()
    b[:, 5:] = (b[:, 5:] + 1) % 11             # change the tokens at positions 5, 6, 7
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5)
    assert not torch.allclose(logits_a[:, 5:], logits_b[:, 5:])


def test_gpt_does_look_back():
    torch.manual_seed(0)
    model = GPT(**CFG)
    a = torch.randint(0, 11, (2, 8))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % 11               # change only the very first token
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert not torch.allclose(logits_a[:, 7], logits_b[:, 7], atol=1e-6), (
        "the last position's prediction should depend on the first token"
    )


def test_gpt_rejects_sequences_longer_than_block_size():
    with pytest.raises(ValueError):
        GPT(**CFG)(torch.zeros((1, 9), dtype=torch.long))


def test_generate_can_run_past_block_size():
    torch.manual_seed(0)
    out = GPT(**CFG).generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (2, 21) and out.max() < 11


def test_generate_keeps_what_it_was_given():
    torch.manual_seed(0)
    prompt = torch.tensor([[4, 5, 6]])
    out = GPT(**CFG).generate(prompt, max_new_tokens=4)
    assert torch.equal(out[:, :3], prompt)


def test_generate_predicts_from_the_last_position():
    torch.manual_seed(0)
    model = GPT(**CFG)
    with torch.no_grad():
        for p in model.parameters():
            if p.shape == (11, 16):            # the token table and the final layer:
                p.mul_(1000)                   # make every prediction all-but-certain
    # eleven prompts that share a first token and differ only in their last
    prompts = torch.stack([torch.zeros(11, dtype=torch.long), torch.arange(11)], dim=1)
    next_tokens = model.generate(prompts, max_new_tokens=1)[:, -1]
    assert len(set(next_tokens.tolist())) > 1, "every prompt got the same next token"


def test_generate_leaves_the_model_as_it_found_it():
    model = GPT(**CFG, dropout=0.5)
    model.train()
    before = [p.clone() for p in model.parameters()]
    modes = []
    model.register_forward_hook(lambda module, inputs, output: modes.append(module.training))
    model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=3)
    assert modes == [False] * 3, "dropout must be off while sampling"
    assert model.training, "and the model handed back in training mode, as it arrived"
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_count_params():
    assert count_params(nn.Linear(3, 2)) == 8


def test_gpt_can_memorise_one_batch():
    torch.manual_seed(0)
    model = GPT(**CFG)
    x = torch.randint(0, 11, (4, 8))
    y = torch.randint(0, 11, (4, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(300):
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert loss.item() < 0.3                   # from about 2.4


# ---- checkpoints -------------------------------------------------------------------

def get_batch(split):
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    x = torch.randint(0, 11, (4, 8), generator=g)
    return x, (x + 1) % 11


def same_weights(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_save_then_load_restores_weights_step_and_history(tmp_path):
    path = str(tmp_path / "ck.pt")
    torch.manual_seed(0)
    trained = GPT(**CFG)
    history = [{"step": 0, "train": 2.4, "val": 2.5}]
    save(path, trained, step=120, history=history)

    torch.manual_seed(1)
    fresh = GPT(**CFG)
    assert not same_weights(trained, fresh)
    assert load(path, fresh) == (120, history)
    assert same_weights(trained, fresh)


def test_load_with_no_file_starts_from_zero(tmp_path):
    model = GPT(**CFG)
    before = [p.clone() for p in model.parameters()]
    assert load(str(tmp_path / "nothing.pt"), model) == (0, [])
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_save_never_leaves_a_half_written_checkpoint(tmp_path, monkeypatch):
    path = str(tmp_path / "ck.pt")
    good = GPT(**CFG)
    save(path, good, step=10, history=[])
    assert os.listdir(tmp_path) == ["ck.pt"], "no temporary file should be left behind"

    # now the session dies in the middle of the next save
    def dies(obj, f, *args, **kwargs):
        with open(f, "wb") as handle:
            handle.write(b"half a checkpoint")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", dies)
    with pytest.raises(KeyboardInterrupt):
        save(path, GPT(**CFG), step=20, history=[])
    monkeypatch.undo()

    survivor = GPT(**CFG)
    assert load(path, survivor)[0] == 10       # the old checkpoint is intact
    assert same_weights(good, survivor)


def test_train_in_chunks_numbers_steps_straight_through(tmp_path):
    torch.manual_seed(0)
    history = train_in_chunks(GPT(**CFG), get_batch, total_steps=40, chunk=20, lr=1e-3,
                              path=str(tmp_path / "ck.pt"), eval_every=10)
    assert [h["step"] for h in history] == [0, 10, 19, 20, 30, 39]


def test_train_in_chunks_resumes_where_it_stopped(tmp_path):
    path = str(tmp_path / "ck.pt")
    torch.manual_seed(0)
    first = GPT(**CFG)
    train_in_chunks(first, get_batch, total_steps=40, chunk=20, lr=1e-3, path=path, eval_every=10)

    # the session died. A brand new model, the same path, a bigger target:
    torch.manual_seed(1)
    second = GPT(**CFG)
    calls = []

    def counting(split):
        calls.append(split)
        return get_batch(split)

    history = train_in_chunks(second, counting, total_steps=80, chunk=20, lr=1e-3, path=path, eval_every=10)
    assert [h["step"] for h in history] == [0, 10, 19, 20, 30, 39, 40, 50, 59, 60, 70, 79]
    # 40 new training steps, plus 6 measurements of 20 train batches each - not 80 steps' worth
    assert calls.count("train") == 40 + 6 * 20
    assert load(path, GPT(**CFG))[0] == 80


def test_train_in_chunks_does_nothing_when_already_done(tmp_path):
    path = str(tmp_path / "ck.pt")
    torch.manual_seed(0)
    model = GPT(**CFG)
    history = train_in_chunks(model, get_batch, total_steps=20, chunk=20, lr=1e-3, path=path, eval_every=10)
    again = GPT(**CFG)
    assert train_in_chunks(again, get_batch, total_steps=20, chunk=20, lr=1e-3, path=path, eval_every=10) == history
    assert same_weights(model, again)


def test_a_last_chunk_may_be_short(tmp_path):
    history = train_in_chunks(GPT(**CFG), get_batch, total_steps=30, chunk=20, lr=1e-3,
                              path=str(tmp_path / "ck.pt"), eval_every=10)
    assert history[-1]["step"] == 29
