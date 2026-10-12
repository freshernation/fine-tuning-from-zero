"""Week 6, Day 3 - the agent, on your model.

Run me:  pytest week-06/day-3 -v

gap.py is in this folder.
"""

import pytest


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def test_expected_task_success(load):
    f = load("gap.py").expected_task_success
    assert f(0.9, 1) == pytest.approx(0.9)
    assert f(0.9, 4) == pytest.approx(0.6561)
    assert f(1.0, 10) == 1.0 and f(0.5, 0) == 1.0
    assert f(0.78, 3) == pytest.approx(0.474552)


def test_first_wrong_step(load):
    f = load("gap.py").first_wrong_step
    wanted = [call("list_titles"), call("delete_note", title="milk")]
    assert f(wanted, wanted) is None
    assert f([call("find_notes", word="milk"), call("delete_note", title="milk")], wanted) == 0      # wrong tool
    assert f([call("list_titles"), call("delete_note", title="eggs")], wanted) == 1                  # wrong arguments
    assert f([call("list_titles"), call("delete_note", title=" MILK ")], wanted) is None, "arguments are compared the way week 4 compares them"
    assert f([call("list_titles"), call("delete_note", title="milk"), call("list_titles")], wanted) == 2   # an extra call
    assert f([call("list_titles")], wanted) == 1                                                     # stopped early
    assert f([], wanted) == 0
    assert f([], []) is None


REPORT = {"cases": [
    {"id": "c1", "passed": True, "wanted": [call("add_note", title="milk")], "made": [call("add_note", title="milk")]},
    {"id": "c2", "passed": False, "wanted": [call("list_titles"), call("delete_note", title="milk")],
     "made": [call("list_titles"), call("delete_note", title="eggs")]},
    {"id": "c3", "passed": True, "wanted": [call("list_titles"), call("delete_note", title="milk"), call("add_note", title="oat milk")],
     "made": [call("list_titles"), call("delete_note", title="milk"), call("add_note", title="oat milk")]},
    {"id": "c4", "passed": False, "wanted": [], "made": [call("list_titles")]},
]}


def test_compare_with_expected(load):
    rows = load("gap.py").compare_with_expected(REPORT, step_accuracy=0.8)
    assert [r["id"] for r in rows] == ["c1", "c2", "c3", "c4"]
    # a task needs one decision per tool call, plus one more: the decision to stop and answer
    assert [r["decisions"] for r in rows] == [2, 3, 4, 1]
    assert [r["expected_success"] for r in rows] == pytest.approx([0.64, 0.512, 0.4096, 0.8])
    assert [r["passed"] for r in rows] == [True, False, True, False]
    assert [r["first_wrong_step"] for r in rows] == [None, 1, None, 0]


def test_summary(load):
    summary = load("gap.py").summarise(REPORT, step_accuracy=0.8)
    assert summary["cases"] == 4 and summary["passed"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["expected_pass_rate"] == pytest.approx((0.64 + 0.512 + 0.4096 + 0.8) / 4)
