"""Week 4, Day 2 - building examples.

Run me:  pytest week-04/day-2 -v

Everything today goes in tunelab/dataset.py (the Day 2 section).
"""

import copy
import json
import random

import pytest

from tunelab.dataset import expand_with_model, expand_with_templates, load_jsonl, save_jsonl, validate

ADD = {"id": "s001", "seed_id": "s001", "kind": "add", "source": "seed",
       "messages": [{"role": "user", "content": "jot down milk — two litres"}],
       "expected": {"tool": "add_note", "arguments": {"title": "milk", "body": "two litres"}}}
NONE = {"id": "s002", "seed_id": "s002", "kind": "none", "source": "seed",
        "messages": [{"role": "user", "content": "thanks!"}],
        "expected": {"tool": None, "arguments": {}, "text": "You're welcome."}}


def changed(example, **changes):
    new = copy.deepcopy(example)
    for path, value in changes.items():
        target = new
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        if value is KeyError:
            del target[last]
        else:
            target[last] = value
    return new


# ---- files -------------------------------------------------------------------------

def test_round_trip_keeps_everything_including_non_ascii(tmp_path):
    rows = [ADD, NONE, changed(ADD, id="s003", seed_id="s003", expected__arguments__title="café ☕ 買い物")]
    path = tmp_path / "rows.jsonl"
    save_jsonl(path, rows)
    assert load_jsonl(path) == rows
    text = path.read_text(encoding="utf-8")
    assert "café ☕ 買い物" in text, "non-ASCII characters are written as themselves"
    assert len(text.strip().splitlines()) == 3


def test_keys_are_written_in_a_fixed_order(tmp_path):
    scrambled = {"expected": ADD["expected"], "note": "extra", "messages": ADD["messages"],
                 "source": "seed", "kind": "add", "seed_id": "s001", "id": "s001"}
    path = tmp_path / "rows.jsonl"
    save_jsonl(path, [scrambled])
    assert list(json.loads(path.read_text())) == ["id", "seed_id", "kind", "source", "messages", "expected", "note"]


def test_blank_lines_are_skipped_and_bad_lines_are_named(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(ADD) + "\n\n" + json.dumps(NONE) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 4\b"):
        load_jsonl(path)
    path.write_text(json.dumps(ADD) + "\n\n" + json.dumps(NONE) + "\n", encoding="utf-8")
    assert len(load_jsonl(path)) == 2


# ---- validate ----------------------------------------------------------------------

def test_good_examples_have_no_problems():
    assert validate(ADD) == [] and validate(NONE) == []
    multi_turn = changed(ADD, kind="delete", messages=[
        {"role": "user", "content": "delete milk"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "list_titles", "arguments": {}}}]},
        {"role": "tool", "content": "1. milk"},
    ], expected={"tool": "delete_note", "arguments": {"title": "milk"}})
    assert validate(multi_turn) == []


@pytest.mark.parametrize("bad, word", [
    (changed(ADD, expected=KeyError), "missing"),
    (changed(ADD, kind="remember"), "kind"),
    (changed(ADD, source="scraped"), "source"),
    (changed(ADD, expected__tool="find_notes"), "kind"),
    (changed(NONE, expected__tool="list_titles"), "kind"),
    (changed(ADD, expected__arguments={"body": "two litres"}), "required"),
    (changed(ADD, expected__arguments={"title": "milk", "priority": "high"}), "unknown argument"),
    (changed(ADD, expected__arguments={"title": 7}), "string"),
    (changed(NONE, expected__text=KeyError), "text"),
    (changed(NONE, expected__text="   "), "text"),
    (changed(ADD, messages=[]), "messages"),
    (changed(ADD, messages=ADD["messages"] + [{"role": "assistant", "content": "ok"}]), "messages"),
    (changed(ADD, id="x001"), "id"),
    (changed(ADD, messages=[{"role": "narrator", "content": "Meanwhile"}, ADD["messages"][0]]), "message"),
    (changed(ADD, messages=[{"role": "user"}]), "message"),
    (changed(ADD, messages=[{"role": "user", "content": 5}]), "message"),
])
def test_each_rule_catches_its_mistake(bad, word):
    problems = validate(bad)
    assert len(problems) >= 1 and all(isinstance(p, str) for p in problems)
    assert any(word in problem for problem in problems), problems


def test_several_problems_are_all_reported():
    bad = changed(ADD, source="scraped", id="x001", expected__arguments={"title": 7, "priority": "high"})
    assert len(validate(bad)) >= 4


# ---- expand_with_templates ---------------------------------------------------------

TEMPLATED = dict(ADD, template={
    "messages": [{"role": "user", "content": "jot down {title} — {body}"}],
    "arguments": {"title": "{title}", "body": "{body}"},
})
SLOTS = {"note": [{"title": "eggs", "body": "a dozen"}, {"title": "passport", "body": "renew by June"},
                  {"title": "wifi", "body": "ask Sam {again}"}]}


def test_templates_fill_the_message_and_the_label_together():
    rows = expand_with_templates(TEMPLATED, SLOTS, n=12, rng=random.Random(0))
    assert len(rows) == 12
    assert [r["id"] for r in rows[:3]] == ["s001-t00", "s001-t01", "s001-t02"]
    for row in rows:
        assert validate(row) == []
        assert row["seed_id"] == "s001" and row["source"] == "synthetic" and row["kind"] == "add"
        assert "template" not in row
        title, body = row["expected"]["arguments"]["title"], row["expected"]["arguments"]["body"]
        assert row["messages"][0]["content"] == f"jot down {title} — {body}", "the label must match the text"
    assert {r["expected"]["arguments"]["title"] for r in rows} == {"eggs", "passport", "wifi"}


def test_templates_are_reproducible_and_leave_the_seed_alone():
    before = copy.deepcopy(TEMPLATED)
    first = expand_with_templates(TEMPLATED, SLOTS, n=5, rng=random.Random(7))
    again = expand_with_templates(TEMPLATED, SLOTS, n=5, rng=random.Random(7))
    assert first == again and TEMPLATED == before
    assert first != expand_with_templates(TEMPLATED, SLOTS, n=5, rng=random.Random(8))


def test_templates_reach_into_earlier_turns():
    seed = {"id": "s009", "seed_id": "s009", "kind": "delete", "source": "seed",
            "messages": [], "expected": {"tool": "delete_note", "arguments": {"title": "milk"}},
            "template": {"messages": [
                {"role": "user", "content": "bin the {title} note"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "find_notes", "arguments": {"word": "{title}"}}}]},
                {"role": "tool", "content": "1. {title}\n2. {word}"},
            ], "arguments": {"title": "{title}"}}}
    row = expand_with_templates(seed, {"note": [{"title": "eggs"}], "word": ["tax"]}, n=1, rng=random.Random(0))[0]
    assert row["messages"][0]["content"] == "bin the eggs note"
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"word": "eggs"}
    assert row["messages"][2]["content"] == "1. eggs\n2. tax"
    assert row["expected"] == {"tool": "delete_note", "arguments": {"title": "eggs"}}


def test_braces_that_are_not_holes_are_left_alone():
    seed = dict(ADD, template={
        "messages": [{"role": "user", "content": 'save {title} with body {"qty": 2} {unknown}'}],
        "arguments": {"title": "{title}", "body": '{"qty": 2}'},
    })
    row = expand_with_templates(seed, {"note": [{"title": "eggs"}]}, n=1, rng=random.Random(0))[0]
    assert row["messages"][0]["content"] == 'save eggs with body {"qty": 2} {unknown}'
    assert row["expected"]["arguments"] == {"title": "eggs", "body": '{"qty": 2}'}


def test_a_seed_without_a_template_gives_nothing():
    assert expand_with_templates(ADD, SLOTS, n=5, rng=random.Random(0)) == []


def test_text_for_none_and_clarify_is_carried_over():
    seed = dict(NONE, template={"messages": [{"role": "user", "content": "cheers for the {word}"}], "arguments": {}})
    row = expand_with_templates(seed, {"word": ["help"]}, n=1, rng=random.Random(0))[0]
    assert row["expected"] == {"tool": None, "arguments": {}, "text": "You're welcome."}


def test_reply_text_in_a_template_has_its_holes_filled_too():
    seed = dict(NONE, template={"messages": [{"role": "user", "content": "save {title}"}], "arguments": {},
                                "text": "Saved a note called {title}."})
    row = expand_with_templates(seed, {"note": [{"title": "eggs"}]}, n=1, rng=random.Random(0))[0]
    assert row["expected"] == {"tool": None, "arguments": {}, "text": "Saved a note called eggs."}
    assert validate(row) == []


# ---- expand_with_model -------------------------------------------------------------

def good_line(text, title, body):
    return json.dumps({"messages": [{"role": "user", "content": text}],
                       "expected": {"tool": "add_note", "arguments": {"title": title, "body": body}}})


def test_model_expansion_keeps_only_what_validates():
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return "\n".join([
            "Sure! Here are some examples:",
            good_line("note to self: eggs, a dozen", "eggs", "a dozen"),
            json.dumps({"messages": [{"role": "user", "content": "find eggs"}],
                        "expected": {"tool": "find_notes", "arguments": {"word": "eggs"}}}),   # wrong tool for this seed
            "{this is not json",
            good_line("pls save: passport — renew by June", "passport", "renew by June"),
            json.dumps(["not", "an", "object"]),
        ])

    rows = expand_with_model(ADD, ask, n=5, instructions="INSTRUCTIONS GO HERE")
    assert [r["id"] for r in rows] == ["s001-m00", "s001-m01"]
    assert [r["expected"]["arguments"]["title"] for r in rows] == ["eggs", "passport"]
    assert all(r["seed_id"] == "s001" and r["source"] == "synthetic" and r["kind"] == "add" for r in rows)
    assert all(validate(r) == [] for r in rows)
    assert len(prompts) == 1
    assert "INSTRUCTIONS GO HERE" in prompts[0] and "jot down milk" in prompts[0] and "5" in prompts[0]


def test_model_expansion_never_returns_more_than_asked_for():
    reply = "\n".join(good_line(f"save note {i}", f"note {i}", "") for i in range(9))
    assert len(expand_with_model(ADD, lambda prompt: reply, n=4, instructions="x")) == 4


def test_model_expansion_survives_a_useless_reply():
    assert expand_with_model(ADD, lambda prompt: "I'm sorry, I can't help with that.", n=3, instructions="x") == []
