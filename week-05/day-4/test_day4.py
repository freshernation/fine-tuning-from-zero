"""Week 5, Day 4 - evaluate, and fix the data.

Run me:  pytest week-05/day-4 -v

Your code goes in tunelab/analysis.py. week-05/predict.py is given.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tunelab.analysis import changed, error_table, failure_type, paired_bootstrap
from tunelab.dataset import freeze, save_jsonl

ROOT = Path(__file__).parents[2]


def example(i, kind, tool, arguments=None):
    return {"id": i, "seed_id": i, "kind": kind, "source": "seed", "messages": [{"role": "user", "content": "..."}],
            "expected": {"tool": tool, "arguments": arguments or {}, **({"text": "ok"} if tool is None else {})}}


def row(i, tool=None, arguments=None, valid=True):
    return {"id": i, "output": "", "prediction": {"tool": tool, "arguments": arguments or {}, "text": "", "valid": valid}}


ADD = example("a", "add", "add_note", {"title": "milk"})
QUIET = example("q", "none", None)


# ---- failure_type ------------------------------------------------------------------

@pytest.mark.parametrize("ex, prediction, name", [
    (ADD, row("a", "add_note", {"title": "Milk"}), "correct"),
    (ADD, row("a", valid=False), "invalid"),
    (ADD, row("a", "find_notes", {"word": "milk"}), "wrong_tool"),
    (ADD, row("a", "add_note", {"title": "eggs"}), "wrong_arguments"),
    (ADD, row("a"), "did_not_call"),
    (QUIET, row("q"), "correct"),
    (QUIET, row("q", "list_titles"), "called_when_it_should_not"),
    (QUIET, row("q", valid=False), "invalid"),
])
def test_failure_types(ex, prediction, name):
    assert failure_type(ex, prediction["prediction"]) == name


# ---- error_table -------------------------------------------------------------------

EXAMPLES = [example("a1", "add", "add_note", {"title": "milk"}), example("a2", "add", "add_note", {"title": "eggs"}),
            example("a3", "add", "add_note", {"title": "jam"}), example("f1", "find", "find_notes", {"word": "milk"}),
            example("n1", "none", None), example("n2", "none", None)]
BEFORE = [row("a1", "add_note", {"title": "milk"}), row("a2", "add_note", {"title": "egg"}), row("a3"),
          row("f1", "list_titles"), row("n1", "list_titles"), row("n2")]
AFTER = [row("a1", "add_note", {"title": "milk"}), row("a2", "add_note", {"title": "eggs"}), row("a3", "add_note", {"title": "jam"}),
         row("f1", "list_titles"), row("n1"), row("n2", "list_titles")]


def test_error_table_counts_by_kind_then_by_failure():
    assert error_table(EXAMPLES, BEFORE) == {
        "add": {"correct": 1, "wrong_arguments": 1, "did_not_call": 1},
        "find": {"wrong_tool": 1},
        "none": {"called_when_it_should_not": 1, "correct": 1},
    }


def test_a_missing_prediction_is_its_own_row_in_the_table():
    table = error_table(EXAMPLES, BEFORE[:-1])
    assert table["none"] == {"called_when_it_should_not": 1, "missing": 1}


# ---- changed -----------------------------------------------------------------------

def test_changed_names_every_example_once():
    result = changed(EXAMPLES, BEFORE, AFTER)
    assert result == {
        "fixed": ["a2", "a3", "n1"],            # wrong before, right after
        "broken": ["n2"],                       # right before, wrong after
        "still_wrong": ["f1"],
        "still_right": ["a1"],
    }
    assert sum(len(ids) for ids in result.values()) == len(EXAMPLES)


# ---- paired_bootstrap --------------------------------------------------------------

def test_no_difference_gives_a_zero_interval():
    same = [True, False, True, True, False] * 20
    assert paired_bootstrap(same, same) == (0.0, 0.0)


def test_a_clear_improvement_gives_an_interval_above_zero():
    before = [False] * 60 + [True] * 40          # 40%
    after = [True] * 70 + [False] * 30           # 70%, and not the same examples
    low, high = paired_bootstrap(before, after, n=2000, seed=0)
    assert 0.0 < low < 0.30 < high < 0.60


def test_a_tiny_improvement_cannot_be_told_from_none():
    before = [True] * 50 + [False] * 50
    after = [True] * 51 + [False] * 49           # one more right, out of a hundred
    low, high = paired_bootstrap(before, after, n=2000, seed=0)
    assert low <= 0.0 <= high


def test_the_bootstrap_is_reproducible_and_seeded():
    before = [False] * 60 + [True] * 40
    after = [True] * 70 + [False] * 30
    assert paired_bootstrap(before, after, n=500, seed=1) == paired_bootstrap(before, after, n=500, seed=1)
    assert paired_bootstrap(before, after, n=500, seed=1) != paired_bootstrap(before, after, n=500, seed=2)


def test_pairing_matters():
    # the same 70 right before and after, plus 10 newly right: every pair is "same" or "better"
    before = [True] * 70 + [False] * 30
    after = [True] * 80 + [False] * 20
    low, high = paired_bootstrap(before, after, n=2000, seed=0)
    assert low > 0.0, "ten fixed and none broken is a real improvement; resampling the pairs together shows it"
    assert high < 0.20


def test_unequal_lengths_are_an_error():
    with pytest.raises(ValueError):
        paired_bootstrap([True, False], [True])


# ---- predict.py, on a toy model ----------------------------------------------------

def test_predict_writes_predictions_and_a_report_tied_to_the_run(tmp_path):
    rows = []
    for i in range(8):
        e = example(f"t{i}", "add", "add_note", {"title": f"note {i}"})
        e["messages"][0]["content"] = f"save note {i}"
        rows.append(e)
    save_jsonl(tmp_path / "train.jsonl", rows[:6])
    save_jsonl(tmp_path / "eval.jsonl", rows[6:])
    fingerprint = freeze(tmp_path / "eval.jsonl", tmp_path / "eval.sha256")
    (tmp_path / "prompt.md").write_text("You manage notes.")
    common = ["--smoke", "--eval", str(tmp_path / "eval.jsonl"), "--prompt", str(tmp_path / "prompt.md")]

    trained = subprocess.run([sys.executable, str(ROOT / "week-05/train.py"), *common, "--train", str(tmp_path / "train.jsonl"),
                              "--out", str(tmp_path / "run")], capture_output=True, text=True, cwd=ROOT)
    assert trained.returncode == 0, trained.stderr[-2000:]

    predicted = subprocess.run([sys.executable, str(ROOT / "week-05/predict.py"), *common, "--run", str(tmp_path / "run"),
                                "--out", str(tmp_path / "tuned")], capture_output=True, text=True, cwd=ROOT)
    assert predicted.returncode == 0, predicted.stderr[-2000:]

    predictions = [json.loads(line) for line in (tmp_path / "tuned_predictions.jsonl").read_text().splitlines()]
    assert [p["id"] for p in predictions] == ["t6", "t7"] and all("prediction" in p for p in predictions)
    report = json.loads((tmp_path / "tuned.json").read_text())
    assert report["n"] == 2 and 0.0 <= report["accuracy"] <= 1.0
    assert report["eval_fingerprint"] == fingerprint
    assert len(report["adapter_fingerprint"]) == 64
    assert report["run"]["eval_fingerprint"] == fingerprint, "the report carries the training run's own record"
