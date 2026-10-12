"""Week 5 milestone - Project 2: an adapter that beats the baseline.

Run me:  pytest week-05/milestone -v

Like week 4's, these grade what you PRODUCED: the predictions and reports in
data/notes/, your run log and your results page. The bar:

    tuned accuracy  >=  baseline accuracy + 0.10
    and the low end of a 95% paired-bootstrap interval on the gain is above zero
"""

import json
import re
from pathlib import Path

import pytest

from tunelab.analysis import paired_bootstrap
from tunelab.dataset import check_frozen, load_jsonl
from tunelab.toolcall import score_example, score_predictions

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "notes"
HERE = Path(__file__).parent


def need(path):
    if not path.exists():
        pytest.fail(f"{path.relative_to(ROOT)} does not exist yet. See week-05/milestone/README.md.")
    return path


def held_out():
    return load_jsonl(need(DATA / "eval.jsonl"))


def correct(name):
    by_id = {row["id"]: row["prediction"] for row in load_jsonl(need(DATA / name))}
    return [ex["id"] in by_id and score_example(ex, by_id[ex["id"]])["correct"] for ex in held_out()]


def test_the_eval_set_is_still_the_one_you_froze():
    need(DATA / "eval.jsonl")
    check_frozen(DATA / "eval.jsonl", need(DATA / "eval.sha256"))


def test_the_tuned_predictions_cover_the_eval_set_and_give_the_reported_score():
    predictions = load_jsonl(need(DATA / "tuned_predictions.jsonl"))
    assert sorted(p["id"] for p in predictions) == sorted(ex["id"] for ex in held_out())
    recorded = json.loads(need(DATA / "tuned.json").read_text())
    report = score_predictions(held_out(), predictions)
    assert abs(recorded["accuracy"] - report["accuracy"]) < 0.0005, (
        f'tuned.json says {recorded["accuracy"]:.3f}; the predictions score {report["accuracy"]:.3f}')


def test_the_score_is_tied_to_a_real_run_on_the_frozen_eval():
    recorded = json.loads(need(DATA / "tuned.json").read_text())
    frozen = need(DATA / "eval.sha256").read_text().strip()
    assert recorded["eval_fingerprint"] == frozen, "the tuned model was scored on a different eval set"
    assert recorded["run"]["eval_fingerprint"] == frozen, "the adapter was trained against a different eval set"
    assert len(recorded["adapter_fingerprint"]) == 64
    assert recorded["run"]["smoke"] is False, "this is the toy model's score, not your model's"
    assert recorded["run"]["trainable_parameters"] > 0


def test_tuned_and_baseline_are_not_the_same_predictions():
    tuned = {r["id"]: r["output"] for r in load_jsonl(need(DATA / "tuned_predictions.jsonl"))}
    base = {r["id"]: r["output"] for r in load_jsonl(need(DATA / "baseline_predictions.jsonl"))}
    assert tuned != base, "identical outputs: was the adapter attached when you scored it?"


def test_the_tuned_model_beats_the_baseline_by_ten_points():
    base, tuned = correct("baseline_predictions.jsonl"), correct("tuned_predictions.jsonl")
    gain = (sum(tuned) - sum(base)) / len(base)
    assert gain >= 0.10 - 1e-9, f"gain is {gain:+.3f}; Project 2 asks for at least +0.10"


def test_the_gain_is_not_a_coin_toss():
    low, high = paired_bootstrap(correct("baseline_predictions.jsonl"), correct("tuned_predictions.jsonl"), n=2000, seed=0)
    assert low > 0.0, f"95% interval on the gain is {low:+.3f} to {high:+.3f}: it includes zero"


def week5_rows():
    table = [line for line in need(ROOT / "logs" / "run-log.md").read_text().splitlines() if line.startswith("|")]
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
    return [row for row in rows if len(row) >= 8 and row[1] == "5" and "example" not in row[2].lower()]


def test_the_run_log_shows_the_work_including_a_run_that_failed():
    rows = week5_rows()
    assert len(rows) >= 2, "the run log needs at least two week-5 rows: nobody gets it right first time"
    assert any(re.search(r"\bfail", row[7], re.I) for row in rows), (
        'mark at least one week-5 run as failed in its Note ("failed: ..."), and say what it taught you')


def test_the_results_page_is_filled_in():
    text = need(HERE / "RESULTS.md").read_text()
    assert "____" not in text, "RESULTS.md still has blanks"
    for heading in ("## Baseline against tuned", "## Is the gain real", "## One run that failed", "## One data fix", "## What it is still bad at"):
        assert heading in text, f"RESULTS.md has lost its heading {heading!r}"
        body = text.split(heading, 1)[1].split("\n## ", 1)[0]
        assert len(body.strip()) > 40, f"nothing written under {heading!r}"
