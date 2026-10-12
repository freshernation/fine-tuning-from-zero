"""Week 6 milestone - Project 3: your model, in your agent.

Run me:  pytest week-06/milestone -v

These grade what you produced this week. The bar:

    the model you are actually SERVING scores above the baseline, and within
    5 points of last week's tuned score, with no template problems;
    the agent has run its fifteen cases on it;
    and the write-up has every number in it.
"""

import json
from pathlib import Path

import pytest

from tunelab.dataset import check_frozen, load_jsonl
from tunelab.toolcall import score_predictions

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "notes"
HERE = Path(__file__).parent

HEADINGS = [
    "## The task", "## Where prompting got to", "## The data", "## What training bought",
    "## What shipping cost", "## The agent, on my model", "## What it costs to run",
    "## Was fine-tuning worth it here", "## With another week",
]
RESULT_ROWS = ["Baseline accuracy", "Tuned accuracy", "Served accuracy", "Agent cases passed", "Model file size",
               "Latency p50", "Latency p95", "Hosted price per request"]


def need(path):
    if not path.exists():
        pytest.fail(f"{path.relative_to(ROOT)} does not exist yet. See week-06/milestone/README.md.")
    return path


def held_out():
    return load_jsonl(need(DATA / "eval.jsonl"))


def accuracy(name):
    return score_predictions(held_out(), load_jsonl(need(DATA / name)))["accuracy"]


def test_the_eval_set_is_still_the_one_you_froze():
    need(DATA / "eval.jsonl")
    check_frozen(DATA / "eval.jsonl", need(DATA / "eval.sha256"))


def test_the_served_predictions_cover_the_eval_set_and_give_the_reported_score():
    predictions = load_jsonl(need(DATA / "served_predictions.jsonl"))
    assert sorted(p["id"] for p in predictions) == sorted(ex["id"] for ex in held_out())
    recorded = json.loads(need(DATA / "served.json").read_text())
    assert abs(recorded["accuracy"] - accuracy("served_predictions.jsonl")) < 0.0005


def test_the_served_predictions_came_from_a_server():
    served = load_jsonl(need(DATA / "served_predictions.jsonl"))
    tuned = load_jsonl(need(DATA / "tuned_predictions.jsonl"))
    assert {r["id"]: r["output"] for r in served} != {r["id"]: r["output"] for r in tuned}, (
        "these are last week's predictions under a new name")
    for row in served:
        message = json.loads(row["output"])
        assert isinstance(message, dict) and message.get("role") == "assistant", (
            f'{row["id"]}: a served prediction\'s output is the assistant message the server sent, as JSON')


def test_no_template_problems():
    recorded = json.loads(need(DATA / "served.json").read_text())
    flagged = [r["id"] for r in load_jsonl(need(DATA / "served_predictions.jsonl")) if r["prediction"].get("template_problem")]
    assert recorded["template_problems"] == len(flagged) == 0, (
        f"{len(flagged)} tool calls came back as plain text: the server is not applying the chat template. Fix the serving command.")


def test_the_served_model_beats_the_baseline_and_keeps_what_training_bought():
    baseline, tuned, served = (accuracy(n) for n in ("baseline_predictions.jsonl", "tuned_predictions.jsonl", "served_predictions.jsonl"))
    assert served > baseline, f"served {served:.2f} is not above the baseline {baseline:.2f}"
    assert served >= tuned - 0.05 - 1e-9, (
        f"served {served:.2f} is more than 5 points below last week's {tuned:.2f}: merging, quantizing or serving broke something")


def test_the_agent_ran_its_fifteen_cases():
    report = json.loads(need(ROOT / "week-06" / "day-3" / "agent_report.json").read_text())
    assert len(report["cases"]) == 15
    assert all(set(case) >= {"id", "passed", "wanted", "made"} for case in report["cases"])
    assert report["passed"] == sum(case["passed"] for case in report["cases"])
    assert any(case["made"] for case in report["cases"]), "no tool was called in any case: was the agent pointed at your server?"


def test_the_write_up_is_complete():
    text = need(HERE / "WRITEUP.md").read_text()
    assert "____" not in text, "WRITEUP.md still has blanks"
    for heading in HEADINGS:
        assert heading in text, f"WRITEUP.md has lost its heading {heading!r}"
        body = text.split(heading, 1)[1].split("\n## ", 1)[0]
        assert len(body.strip()) > 60, f"nothing much is written under {heading!r}"
    for label in RESULT_ROWS:
        row = next((line for line in text.splitlines() if line.startswith(f"| {label}")), None)
        assert row and row.split("|")[2].strip(), f"the results table has no value for {label!r}"
