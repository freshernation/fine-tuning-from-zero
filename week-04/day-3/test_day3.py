"""Week 4, Day 3 - cleaning.

Run me:  pytest week-04/day-3 -v

Everything today goes in tunelab/dataset.py (the Day 3 section).

dirty.jsonl, in this folder, is forty rows with problems planted in it. There is
no "delete" example anywhere in it: that is one of the problems.
"""

from pathlib import Path

import pytest

from tunelab.dataset import (balance, balance_problems, clean, find_conflicts, find_duplicates,
                             load_jsonl, normalise, validate)

def dirty():
    return load_jsonl(Path(__file__).parent / "dirty.jsonl")


INVALID = ["d20", "d21", "d22", "d23", "d24", "d25"]
CONFLICTS = [["d16", "d17"], ["d18", "d19"]]
DUPLICATES = [["d01", "d03"], ["d05", "d06"], ["d07", "d09"], ["d12", "d13"], ["d34", "d35"]]


def user(text, i, kind="add", tool="add_note", arguments=None, seed=None):
    return {"id": f"{seed or 'r'}{i}", "seed_id": seed or f"r{i}", "kind": kind, "source": "synthetic",
            "messages": [{"role": "user", "content": text}],
            "expected": {"tool": tool, "arguments": arguments if arguments is not None else {"title": "x"}}}


def test_normalise():
    assert normalise("  Call   the\tDentist \n") == "call the dentist"
    assert normalise("STRASSE") == normalise("strasse")


def test_the_planted_invalid_rows():
    assert [row["id"] for row in dirty() if validate(row)] == INVALID


def test_duplicates_ignore_case_and_spacing():
    rows = [user("save milk", 1), user("  Save   MILK ", 2), user("save eggs", 3), user("save milk", 4)]
    assert find_duplicates(rows) == [["r1", "r2", "r4"]]


def test_duplicates_compare_the_whole_conversation_not_just_the_last_message():
    first = user("delete it", 1)
    second = user("delete it", 2)
    second["messages"] = [{"role": "user", "content": "I have a note called milk"},
                          {"role": "assistant", "content": "Yes, you do."}, {"role": "user", "content": "delete it"}]
    assert find_duplicates([first, second]) == []


def test_conflicts_are_duplicates_that_disagree():
    same = [user("save milk", 1, arguments={"title": "milk"}), user("Save milk", 2, arguments={"title": " MILK "})]
    assert find_duplicates(same) == [["r1", "r2"]] and find_conflicts(same) == []
    different = [user("save milk", 1, arguments={"title": "milk"}), user("save milk", 2, arguments={"title": "the milk"})]
    assert find_conflicts(different) == [["r1", "r2"]]
    other_tool = [user("milk?", 1, "find", "find_notes", {"word": "milk"}), user("milk?", 2, "list", "list_titles", {})]
    assert find_conflicts(other_tool) == [["r1", "r2"]]


def test_an_empty_optional_argument_is_not_a_disagreement():
    rows = [user("save milk", 1, arguments={"title": "milk"}), user("save milk", 2, arguments={"title": "milk", "body": ""})]
    assert find_conflicts(rows) == []


def test_the_planted_duplicates_and_conflicts():
    valid = [row for row in dirty() if not validate(row)]
    assert find_conflicts(valid) == CONFLICTS
    assert find_duplicates(valid) == DUPLICATES[:4] + CONFLICTS + DUPLICATES[4:]


def test_balance_counts_every_kind_even_the_missing_one():
    assert balance(dirty())["delete"] == 0
    assert sum(balance(dirty()).values()) == 39          # one row has a kind that is not a kind
    assert set(balance([])) == {"add", "list", "find", "delete", "calculate", "none", "clarify"}


def kinds(**counts):
    rows, i = [], 0
    for kind, count in counts.items():
        for _ in range(count):
            i += 1
            rows.append(user(f"message {i}", i, kind=kind))
    return rows


EVEN = dict(add=20, list=14, find=14, delete=14, calculate=14, none=12, clarify=12)      # 100 rows


def test_a_balanced_set_has_no_problems():
    assert balance_problems(kinds(**EVEN)) == []


def test_a_kind_below_eight_percent_is_a_problem():
    problems = balance_problems(kinds(**dict(EVEN, delete=7, add=27)))
    assert len(problems) == 1 and "delete" in problems[0]


def test_too_few_refusals_is_a_problem():
    problems = balance_problems(kinds(**dict(EVEN, none=8, clarify=8, add=28)))      # 16% would pass; 8 + 8 = 16...
    assert problems == []
    problems = balance_problems(kinds(**dict(EVEN, none=8, clarify=6, add=30)))      # 14%: clarify is also under 8%
    assert any("none" in p and "clarify" in p for p in problems)


def test_one_seed_supplying_too_much_is_a_problem():
    rows = kinds(**EVEN)
    for row in rows[:6]:                       # six of a hundred rows from one seed
        row["seed_id"] = "greedy"
    problems = balance_problems(rows)
    assert len(problems) == 1 and "greedy" in problems[0]
    for row in rows[5:6]:
        row["seed_id"] = "other"
    assert balance_problems(rows) == []        # five of a hundred is allowed


def test_the_dirty_file_is_missing_a_kind():
    assert any("delete" in problem for problem in balance_problems(dirty()))


def test_clean_refuses_to_guess_at_conflicts():
    with pytest.raises(ValueError, match="d16.*d17|d17.*d16"):
        clean(dirty())


def test_clean_once_the_conflicts_are_decided():
    decided = [row for row in dirty() if row["id"] not in ("d17", "d19")]      # a person chose d16 and d18
    kept, counts = clean(decided)
    assert counts == {"invalid": 6, "duplicate": 5, "kept": 27}
    ids = [row["id"] for row in kept]
    assert len(ids) == 27 and "d01" in ids and "d03" not in ids, "the first of each group is the one kept"
    assert not any(i in ids for i in INVALID)
    assert find_duplicates(kept) == [] and all(validate(row) == [] for row in kept)
    assert len(decided) == 38, "clean returns new lists and leaves its input alone"
