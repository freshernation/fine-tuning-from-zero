"""Week 4, Day 4 - split, freeze, render.

Run me:  pytest week-04/day-4 -v

Everything today goes in tunelab/dataset.py (the Day 4 section).
"""

import copy
import random

import pytest

from tunelab.dataset import (KINDS, check_frozen, find_leaks, fingerprint, freeze, load_jsonl,
                             render_completion, render_prompt, save_jsonl, split_by_seed)
from tunelab.toolcall import parse_prediction, same_arguments

TOOL = {"add": "add_note", "list": "list_titles", "find": "find_notes", "delete": "delete_note",
        "calculate": "calculate", "none": None, "clarify": None}


def dataset():
    """35 seeds, 5 of each kind; each seed has itself plus 4 synthetic rewordings: 175 rows."""
    rows = []
    for s in range(35):
        kind = KINDS[s % 7]
        for v in range(5):
            expected = {"tool": TOOL[kind], "arguments": {}}
            if TOOL[kind] is None:
                expected["text"] = "OK."
            rows.append({"id": f"s{s:02d}" if v == 0 else f"s{s:02d}-t{v:02d}", "seed_id": f"s{s:02d}",
                         "kind": kind, "source": "seed" if v == 0 else "synthetic",
                         "messages": [{"role": "user", "content": f"request {s} worded way {v}"}],
                         "expected": expected})
    return rows


# ---- split -------------------------------------------------------------------------

def test_no_seed_is_on_both_sides():
    train, held_out = split_by_seed(dataset(), eval_size=50, rng=random.Random(0))
    assert {r["seed_id"] for r in train}.isdisjoint({r["seed_id"] for r in held_out})
    assert len(train) + len(held_out) == 175
    assert sorted(r["id"] for r in train + held_out) == sorted(r["id"] for r in dataset())


def test_eval_is_the_size_asked_for_and_has_every_kind():
    _, held_out = split_by_seed(dataset(), eval_size=50, rng=random.Random(0))
    assert len(held_out) == 50
    assert {r["kind"] for r in held_out} == set(KINDS)
    _, small = split_by_seed(dataset(), eval_size=10, rng=random.Random(0))
    assert len(small) == 35 and {r["kind"] for r in small} == set(KINDS), "every kind first, even if that goes over"


def test_eval_never_goes_over_once_every_kind_is_in():
    _, held_out = split_by_seed(dataset(), eval_size=52, rng=random.Random(0))
    assert len(held_out) == 50, "seeds are 5 rows each: 52 cannot be reached without going over"


def test_split_is_reproducible_and_depends_on_the_seed():
    first = split_by_seed(dataset(), 50, random.Random(3))
    assert first == split_by_seed(dataset(), 50, random.Random(3))
    assert first != split_by_seed(dataset(), 50, random.Random(4))


def test_rows_keep_their_order():
    train, held_out = split_by_seed(dataset(), 50, random.Random(0))
    order = {row["id"]: i for i, row in enumerate(dataset())}
    for side in (train, held_out):
        positions = [order[row["id"]] for row in side]
        assert positions == sorted(positions)


# ---- leaks -------------------------------------------------------------------------

def test_a_clean_split_has_no_leaks():
    train, held_out = split_by_seed(dataset(), 50, random.Random(0))
    assert find_leaks(train, held_out) == []


def test_a_shared_seed_is_a_leak():
    train, held_out = split_by_seed(dataset(), 50, random.Random(0))
    stray = held_out.pop()
    train.append(stray)
    leaks = find_leaks(train, held_out)
    assert leaks and all(a == stray["id"] and b.startswith(stray["seed_id"]) for a, b in leaks)


def test_the_same_words_under_another_seed_is_a_leak():
    train, held_out = split_by_seed(dataset(), 50, random.Random(0))
    copycat = copy.deepcopy(held_out[0])
    copycat.update(id="z99", seed_id="z99")
    copycat["messages"][0]["content"] = "  " + copycat["messages"][0]["content"].upper()
    assert find_leaks(train + [copycat], held_out) == [("z99", held_out[0]["id"])]


# ---- freeze ------------------------------------------------------------------------

def test_fingerprint_ignores_row_order_and_key_order():
    rows = dataset()[:10]
    shuffled = [dict(reversed(list(row.items()))) for row in reversed(rows)]
    assert fingerprint(rows) == fingerprint(shuffled)
    assert len(fingerprint(rows)) == 64 and set(fingerprint(rows)) <= set("0123456789abcdef")


def test_fingerprint_changes_with_one_character():
    rows = dataset()[:10]
    edited = copy.deepcopy(rows)
    edited[3]["messages"][0]["content"] += "."
    assert fingerprint(rows) != fingerprint(edited)
    assert fingerprint(rows) != fingerprint(rows[:9])


def test_fingerprint_sees_non_ascii_characters():
    a, b = dataset()[:1], dataset()[:1]
    a[0]["messages"][0]["content"] = "café"
    b[0]["messages"][0]["content"] = "cafe"
    assert fingerprint(a) != fingerprint(b)


def test_freeze_then_check(tmp_path):
    eval_path, hash_path = tmp_path / "eval.jsonl", tmp_path / "eval.sha256"
    rows = dataset()[:20]
    save_jsonl(eval_path, rows)
    digest = freeze(eval_path, hash_path)
    assert digest == fingerprint(rows) and hash_path.read_text().strip() == digest
    assert check_frozen(eval_path, hash_path) is None
    save_jsonl(eval_path, list(reversed(rows)))
    assert check_frozen(eval_path, hash_path) is None, "reordering the file is not a change"


def test_an_edited_eval_set_is_caught_and_not_refrozen(tmp_path):
    eval_path, hash_path = tmp_path / "eval.jsonl", tmp_path / "eval.sha256"
    rows = dataset()[:20]
    save_jsonl(eval_path, rows)
    recorded = freeze(eval_path, hash_path)
    rows[0]["expected"]["tool"] = "list_titles"                # "fixing" one unfair example
    save_jsonl(eval_path, rows)
    with pytest.raises(ValueError, match="(?is)restore.*freeze"):
        check_frozen(eval_path, hash_path)
    assert hash_path.read_text().strip() == recorded, "checking must never rewrite the recorded fingerprint"
    assert load_jsonl(eval_path) == rows, "and must never touch the eval file"


# ---- render ------------------------------------------------------------------------

SYSTEM = "You manage one person's notes."
ADD = {"id": "s1", "seed_id": "s1", "kind": "add", "source": "seed",
       "messages": [{"role": "user", "content": "jot down milk — two litres"}],
       "expected": {"tool": "add_note", "arguments": {"title": "milk", "body": "two litres"}}}
QUIET = {"id": "s2", "seed_id": "s2", "kind": "clarify", "source": "seed",
         "messages": [{"role": "user", "content": "delete it"}],
         "expected": {"tool": None, "arguments": {}, "text": "Which note do you want to delete?"}}
DELETE = {"id": "s3", "seed_id": "s3", "kind": "delete", "source": "seed",
          "messages": [
              {"role": "user", "content": "bin the milk note"},
              {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "list_titles", "arguments": {}}}]},
              {"role": "tool", "content": "1. milk\n2. eggs"}],
          "expected": {"tool": "delete_note", "arguments": {"title": "milk"}}}


def test_the_prompt_has_the_system_text_the_tools_and_the_conversation(chat_tokenizer):
    prompt = render_prompt(chat_tokenizer, ADD, SYSTEM)
    assert prompt.startswith("<|im_start|>system\nYou manage one person's notes.")
    for name in ("add_note", "list_titles", "find_notes", "delete_note", "calculate"):
        assert f'"name": "{name}"' in prompt
    assert "<|im_start|>user\njot down milk — two litres<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n"), "it must end where the model starts writing"


def test_the_prompt_includes_earlier_tool_calls_and_results(chat_tokenizer):
    prompt = render_prompt(chat_tokenizer, DELETE, SYSTEM)
    assert '<tool_call>\n{"name": "list_titles", "arguments": {}}\n</tool_call>' in prompt
    assert "<tool_response>\n1. milk\n2. eggs\n</tool_response>" in prompt


def test_the_completion_for_a_tool_call_and_for_words():
    assert render_completion(ADD) == (
        '<tool_call>\n{"name": "add_note", "arguments": {"title": "milk", "body": "two litres"}}\n</tool_call><|im_end|>')
    assert render_completion(QUIET) == "Which note do you want to delete?<|im_end|>"


UNICODE = dict(ADD, messages=[{"role": "user", "content": "note: café ☕ — 買い物"}],
               expected={"tool": "add_note", "arguments": {"title": "café ☕", "body": "買い物"}})


@pytest.mark.parametrize("example", [ADD, DELETE, QUIET, UNICODE])
def test_prompt_plus_completion_is_exactly_what_the_template_writes(chat_tokenizer, example):
    # The strongest check there is: put the right answer into the conversation as an
    # assistant turn, and ask the chat template itself to write the whole thing out.
    from tunelab.notes import SCHEMAS

    expected = example["expected"]
    if expected["tool"] is None:
        answer = {"role": "assistant", "content": expected["text"]}
    else:
        answer = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": expected["tool"], "arguments": expected["arguments"]}}]}
    whole = chat_tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}] + example["messages"] + [answer], tools=SCHEMAS, tokenize=False)
    assert render_prompt(chat_tokenizer, example, SYSTEM) + render_completion(example) + "\n" == whole


@pytest.mark.parametrize("arguments", [
    {"title": "milk", "body": "two litres"},
    {"title": "café ☕", "body": "買い物"},
    {"title": "config", "body": '{"retries": 3, "name": "x"}'},
    {"title": "syntax", "body": "write <tool_call> then </tool_call> around it"},
    {"title": 'the "big" one', "body": "line one\nline two"},
])
def test_a_completion_reads_back_as_the_call_it_came_from(arguments):
    example = dict(ADD, expected={"tool": "add_note", "arguments": arguments})
    text = render_completion(example)
    assert text.endswith("<|im_end|>")
    prediction = parse_prediction(text[: -len("<|im_end|>")])
    assert prediction["valid"] and prediction["tool"] == "add_note"
    assert prediction["arguments"] == arguments
    assert same_arguments(arguments, prediction["arguments"], "add_note")
