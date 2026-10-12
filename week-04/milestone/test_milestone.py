"""Week 4 milestone - your dataset, your frozen eval set, your baseline.

Run me:  pytest week-04/milestone -v

Unlike every test so far, these grade DATA: the files you made this week in
data/notes/. Nothing here can be passed by writing code.
"""

import json
from pathlib import Path

import pytest

from tunelab.dataset import (KINDS, balance_problems, check_frozen, find_conflicts, find_duplicates,
                             find_leaks, load_jsonl, validate)
from tunelab.toolcall import score_predictions

DATA = Path(__file__).parents[2] / "data" / "notes"


def need(name):
    path = DATA / name
    if not path.exists():
        pytest.fail(f"data/notes/{name} does not exist yet. See week-04/milestone/README.md.")
    return path


def train():
    return load_jsonl(need("train.jsonl"))


def held_out():
    return load_jsonl(need("eval.jsonl"))


def test_sizes():
    assert 500 <= len(train()) <= 1000, f"train has {len(train())} rows; the course asks for 500 to 1000"
    assert 90 <= len(held_out()) <= 120, f"eval has {len(held_out())} rows; the course asks for about 100"


def test_every_row_is_well_formed():
    for row in train() + held_out():
        assert validate(row) == [], f"{row.get('id')}: {validate(row)}"
    ids = [row["id"] for row in train() + held_out()]
    assert len(ids) == len(set(ids)), "every row needs its own id"


def test_no_duplicates_and_no_conflicts():
    for name, rows in (("train", train()), ("eval", held_out())):
        assert find_conflicts(rows) == [], f"{name} has rows that say the same thing and expect different answers"
        assert find_duplicates(rows) == [], f"{name} has duplicate rows"


def test_train_is_balanced_and_eval_has_every_kind():
    assert balance_problems(train()) == []
    assert {row["kind"] for row in held_out()} == set(KINDS)


def test_nothing_leaks_from_train_into_eval():
    leaks = find_leaks(train(), held_out())
    assert leaks == [], f"{len(leaks)} leaks, for example {leaks[:3]}"


def test_the_eval_set_is_frozen_and_untouched():
    need("eval.jsonl")
    check_frozen(DATA / "eval.jsonl", need("eval.sha256"))


def test_the_baseline_covers_every_eval_example():
    predictions = load_jsonl(need("baseline_predictions.jsonl"))
    assert sorted(p["id"] for p in predictions) == sorted(row["id"] for row in held_out())
    assert all(isinstance(p.get("output"), str) and "prediction" in p for p in predictions)


def test_the_baseline_score_is_the_one_the_predictions_give():
    recorded = json.loads(need("baseline.json").read_text())
    report = score_predictions(held_out(), load_jsonl(need("baseline_predictions.jsonl")))
    assert abs(recorded["accuracy"] - report["accuracy"]) < 0.0005, (
        f'baseline.json says {recorded["accuracy"]:.3f}; the predictions score {report["accuracy"]:.3f}')
    assert recorded["n"] == len(held_out())
