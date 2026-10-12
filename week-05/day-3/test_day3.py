"""Week 5, Day 3 - reading a run, and four broken ones.

Run me:  pytest week-05/day-3 -v

diagnose.py is yours to write. broken_1.py to broken_4.py are somebody else's, each
with one bug: their tests fail as shipped and pass when you have fixed the bug.
"""

import json
import math
from pathlib import Path

import pytest

from tunelab.dataset import render_completion, render_prompt

HISTORIES = Path(__file__).parent / "histories"


def history(name):
    return json.loads((HISTORIES / f"{name}.json").read_text())


# ---- diagnose.py -------------------------------------------------------------------

@pytest.mark.parametrize("name, verdict", [
    ("run_a", "learning"), ("run_b", "overfitting"), ("run_c", "not learning"),
    ("run_d", "diverging"), ("run_e", "learning"), ("run_f", "diverging"),
])
def test_the_six_recorded_runs(load, name, verdict):
    assert load("diagnose.py").diagnose(history(name)) == verdict


def test_best_step(load):
    best_step = load("diagnose.py").best_step
    assert best_step(history("run_a")) == 100
    assert best_step(history("run_b")) == 40, "the best checkpoint is rarely the last"
    assert best_step(history("run_f")) == 20, "an eval loss of NaN is never the best"
    assert best_step([{"step": 20, "train": 1.0, "eval": 0.5}, {"step": 40, "train": 0.8, "eval": 0.5}]) == 20
    assert best_step([{"step": 20, "train": 1.0, "eval": float("nan")}, {"step": 40, "train": 0.8, "eval": 0.5}]) == 40


def rows(train, evals):
    return [dict({"step": i + 1, "train": t}, **({"eval": evals[i]} if evals[i] is not None else {}))
            for i, t in enumerate(train)]


def test_the_five_percent_line(load):
    diagnose = load("diagnose.py").diagnose
    assert diagnose(rows([2.00, 1.91], [None, None])) == "not learning"       # fell 4.5%
    assert diagnose(rows([2.00, 1.89], [None, None])) == "learning"           # fell 5.5%


def test_the_ten_percent_line(load):
    diagnose = load("diagnose.py").diagnose
    assert diagnose(rows([2.0, 1.0, 0.5], [1.00, 0.50, 0.54])) == "learning"          # 8% above its best
    assert diagnose(rows([2.0, 1.0, 0.5], [1.00, 0.50, 0.56])) == "overfitting"       # 12% above its best


def test_the_checks_run_in_order(load):
    diagnose = load("diagnose.py").diagnose
    # the train loss rose AND the eval loss is far above its best: diverging comes first
    assert diagnose(rows([1.0, 0.5, 3.0], [0.9, 0.5, 2.0])) == "diverging"
    # flat train loss with a wobbly eval: not learning comes before overfitting
    assert diagnose(rows([2.00, 1.99, 1.98], [1.0, 0.5, 0.9])) == "not learning"
    assert diagnose(rows([2.0, float("inf"), 1.0], [None, None, None])) == "diverging"


def test_a_run_with_no_evals_can_still_be_judged(load):
    assert load("diagnose.py").diagnose(rows([2.0, 1.0, 0.5], [None, None, None])) == "learning"


# ---- the four broken files ---------------------------------------------------------

SYSTEM = "You manage notes."
ADD = {"id": "s1", "seed_id": "s1", "kind": "add", "source": "seed",
       "messages": [{"role": "user", "content": "jot down milk — two litres"}],
       "expected": {"tool": "add_note", "arguments": {"title": "milk", "body": "two litres"}}}


def test_broken_1_trains_without_blowing_up(load):
    losses = load("broken_1.py").losses_over_20_steps()
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses), "the loss became NaN or infinite"
    assert max(losses) < 1.5 * losses[0], "the loss shot up at some point"
    assert losses[-1] < losses[0] - 0.05, "and by the end it should be going down"


def test_broken_2_learns_the_answer_and_only_the_answer(load, chat_tokenizer):
    row = load("broken_2.py").tokenize(chat_tokenizer, ADD, SYSTEM)
    learned = chat_tokenizer.decode([token for token in row["labels"] if token != -100])
    assert learned == render_completion(ADD)
    assert chat_tokenizer.decode(row["input_ids"]) == render_prompt(chat_tokenizer, ADD, SYSTEM) + render_completion(ADD)


def test_broken_3_keeps_train_and_eval_apart(load):
    data = [{"id": f"s{s:02d}-{v}", "seed_id": f"s{s:02d}", "kind": "add"} for s in range(60) for v in range(5)]
    train, held_out = load("broken_3.py").make_splits(data, eval_size=100, seed=0)
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in held_out}), "the same rows are on both sides"
    assert {r["seed_id"] for r in train}.isdisjoint({r["seed_id"] for r in held_out}), "the same seeds are on both sides"
    assert len(train) + len(held_out) == 300 and 90 <= len(held_out) <= 100


def test_broken_4_asks_at_inference_what_it_asked_in_training(load, chat_tokenizer):
    at_inference = load("broken_4.py").prompt_for_inference(chat_tokenizer, ADD, SYSTEM)
    assert at_inference == render_prompt(chat_tokenizer, ADD, SYSTEM)
