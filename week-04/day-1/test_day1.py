"""Week 4, Day 1 - the eval comes first.

Run me:  pytest week-04/day-1 -v

Everything today goes in tunelab/toolcall.py.
"""

import pytest

from tunelab.toolcall import parse_prediction, run_eval, same_arguments, score_example, score_predictions


def call(name, arguments):
    import json
    return '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (name, json.dumps(arguments))


# ---- parse_prediction --------------------------------------------------------------

def test_a_clean_tool_call():
    assert parse_prediction(call("add_note", {"title": "milk", "body": "two litres"})) == {
        "tool": "add_note", "arguments": {"title": "milk", "body": "two litres"}, "text": "", "valid": True,
    }


def test_plain_text_is_a_valid_prediction_with_no_tool():
    assert parse_prediction("  You have three notes.\n") == {
        "tool": None, "arguments": {}, "text": "You have three notes.", "valid": True,
    }


def test_text_around_a_call_is_kept_but_the_call_is_the_prediction():
    out = parse_prediction("Sure, saving that.\n" + call("add_note", {"title": "milk"}) + "\nDone!")
    assert out["tool"] == "add_note" and out["valid"]
    assert out["text"] == "Sure, saving that.\n\nDone!"


def test_the_first_of_two_calls_wins():
    out = parse_prediction(call("list_titles", {}) + "\n" + call("delete_note", {"title": "milk"}))
    assert out["tool"] == "list_titles" and out["arguments"] == {} and out["valid"]


def test_a_call_with_no_closing_tag_still_counts():
    out = parse_prediction('<tool_call>\n{"name": "find_notes", "arguments": {"word": "milk"}}')
    assert out["tool"] == "find_notes" and out["arguments"] == {"word": "milk"} and out["valid"]


def test_arguments_given_as_a_json_string_are_unpacked():
    out = parse_prediction('<tool_call>{"name": "find_notes", "arguments": "{\\"word\\": \\"milk\\"}"}</tool_call>')
    assert out["arguments"] == {"word": "milk"} and out["valid"]


def test_a_call_with_no_arguments_key_has_empty_arguments():
    out = parse_prediction('<tool_call>{"name": "list_titles"}</tool_call>')
    assert out == {"tool": "list_titles", "arguments": {}, "text": "", "valid": True}


@pytest.mark.parametrize("broken", [
    "<tool_call>{'name': 'list_titles', 'arguments': {}}</tool_call>",          # single quotes are not JSON
    '<tool_call>{"name": "add_note", "arguments": {"title": "milk",}}</tool_call>',   # trailing comma
    '<tool_call>add_note(title="milk")</tool_call>',                              # not JSON at all
    '<tool_call>{"arguments": {"title": "milk"}}</tool_call>',                    # no name
    '<tool_call>{"name": 7, "arguments": {}}</tool_call>',                        # name is not a string
    '<tool_call>{"name": "add_note", "arguments": ["milk"]}</tool_call>',         # arguments is not an object
    '<tool_call>["add_note"]</tool_call>',                                        # not an object
    "<tool_call></tool_call>",
])
def test_an_unparseable_call_is_invalid_not_a_refusal(broken):
    out = parse_prediction(broken)
    assert out["valid"] is False and out["tool"] is None and out["arguments"] == {}


def test_tool_call_text_inside_a_json_string_does_not_confuse_the_parser():
    out = parse_prediction(call("add_note", {"title": "syntax", "body": "write <tool_call> like this"}))
    assert out["valid"] and out["arguments"]["body"] == "write <tool_call> like this"


# ---- same_arguments ----------------------------------------------------------------

def test_strings_match_ignoring_case_and_spacing():
    assert same_arguments({"title": "Call the dentist"}, {"title": "  call  the\tDENTIST "}, "delete_note")
    assert not same_arguments({"title": "call the dentist"}, {"title": "call a dentist"}, "delete_note")


def test_an_absent_optional_argument_equals_an_empty_one():
    assert same_arguments({"title": "milk"}, {"title": "milk", "body": ""}, "add_note")
    assert same_arguments({"title": "milk", "body": ""}, {"title": "milk"}, "add_note")
    assert not same_arguments({"title": "milk"}, {"title": "milk", "body": "two litres"}, "add_note")


def test_a_missing_required_argument_is_a_mismatch():
    assert not same_arguments({"title": "milk"}, {}, "delete_note")


def test_an_argument_the_tool_does_not_have_is_a_mismatch():
    assert not same_arguments({"title": "milk"}, {"title": "milk", "priority": "high"}, "add_note")
    assert not same_arguments({}, {"all": "true"}, "list_titles")


def test_a_non_string_value_is_a_mismatch():
    assert not same_arguments({"title": "7"}, {"title": 7}, "add_note")


def test_calculate_ignores_all_spacing_but_nothing_else():
    assert same_arguments({"expression": "(2 + 3) * 4"}, {"expression": "(2+3)*4"}, "calculate")
    assert not same_arguments({"expression": "(2 + 3) * 4"}, {"expression": "2 + 3 * 4"}, "calculate")
    assert not same_arguments({"title": "a b"}, {"title": "ab"}, "delete_note"), "only calculate drops spaces"


# ---- score_example -----------------------------------------------------------------

def example(kind, tool, arguments=None):
    return {"id": "x", "kind": kind, "messages": [{"role": "user", "content": "..."}],
            "expected": {"tool": tool, "arguments": arguments or {}}}


def pred(tool=None, arguments=None, valid=True, text=""):
    return {"tool": tool, "arguments": arguments or {}, "text": text, "valid": valid}


ADD = example("add", "add_note", {"title": "milk"})


def test_a_right_call():
    assert score_example(ADD, pred("add_note", {"title": "Milk"})) == {
        "valid": True, "right_tool": True, "right_arguments": True, "correct": True}


def test_the_wrong_tool():
    assert score_example(ADD, pred("find_notes", {"word": "milk"})) == {
        "valid": True, "right_tool": False, "right_arguments": False, "correct": False}


def test_the_right_tool_with_wrong_arguments():
    assert score_example(ADD, pred("add_note", {"title": "eggs"})) == {
        "valid": True, "right_tool": True, "right_arguments": False, "correct": False}


def test_text_when_a_call_was_expected():
    assert score_example(ADD, pred(text="What should I call it?")) == {
        "valid": True, "right_tool": False, "right_arguments": False, "correct": False}


def test_an_invalid_call_is_never_correct():
    assert score_example(ADD, pred(valid=False))["correct"] is False
    assert score_example(example("none", None), pred(valid=False)) == {
        "valid": False, "right_tool": True, "right_arguments": True, "correct": False}


@pytest.mark.parametrize("kind", ["none", "clarify"])
def test_keeping_quiet_when_no_tool_fits(kind):
    quiet = example(kind, None)
    assert score_example(quiet, pred(text="Which note do you mean?"))["correct"] is True
    assert score_example(quiet, pred("list_titles")) == {
        "valid": True, "right_tool": False, "right_arguments": False, "correct": False}


# ---- score_predictions and run_eval ------------------------------------------------

EXAMPLES = [
    dict(example("add", "add_note", {"title": "milk"}), id="a1"),
    dict(example("add", "add_note", {"title": "eggs"}), id="a2"),
    dict(example("find", "find_notes", {"word": "milk"}), id="f1"),
    dict(example("delete", "delete_note", {"title": "milk"}), id="d1"),
    dict(example("none", None), id="n1"),
    dict(example("none", None), id="n2"),
    dict(example("clarify", None), id="c1"),
    dict(example("list", "list_titles"), id="l1"),
]
OUTPUTS = {
    "a1": call("add_note", {"title": "milk"}),                 # correct
    "a2": call("add_note", {"title": "egg"}),                  # right tool, wrong arguments
    "f1": call("list_titles", {}),                             # wrong tool
    "d1": "<tool_call>{'name': 'delete_note'}</tool_call>",    # invalid
    "n1": "You're welcome!",                                   # correct
    "n2": call("list_titles", {}),                             # called when it should not have
    "c1": "Which note?",                                       # correct
    # l1 is missing altogether
}


def rows():
    return [{"id": i, "output": out, "prediction": parse_prediction(out)} for i, out in OUTPUTS.items()]


def test_report_arithmetic():
    report = score_predictions(EXAMPLES, rows())
    assert report["n"] == 8
    assert report["accuracy"] == 3 / 8
    assert report["valid_rate"] == 6 / 8                       # d1 invalid, l1 missing
    assert report["right_tool_rate"] == 4 / 8                  # a1, a2, n1, c1
    assert report["right_arguments_rate"] == 3 / 8             # a1, n1, c1
    assert report["refusal_accuracy"] == 2 / 3                 # n1 and c1, out of n1, n2, c1
    assert report["by_kind"] == {
        "add": {"n": 2, "accuracy": 0.5}, "find": {"n": 1, "accuracy": 0.0},
        "delete": {"n": 1, "accuracy": 0.0}, "none": {"n": 2, "accuracy": 0.5},
        "clarify": {"n": 1, "accuracy": 1.0}, "list": {"n": 1, "accuracy": 0.0},
    }


def test_a_missing_prediction_counts_as_wrong_not_as_absent():
    only_one = [row for row in rows() if row["id"] == "a1"]
    report = score_predictions(EXAMPLES, only_one)
    assert report["n"] == 8 and report["accuracy"] == 1 / 8


def test_a_prediction_for_an_unknown_example_is_an_error():
    with pytest.raises(ValueError, match="zz9"):
        score_predictions(EXAMPLES, rows() + [{"id": "zz9", "output": "", "prediction": pred()}])


def test_two_predictions_for_one_example_is_an_error():
    with pytest.raises(ValueError, match="a1"):
        score_predictions(EXAMPLES, rows() + rows()[:1])


def test_refusal_accuracy_is_none_when_there_is_nothing_to_refuse():
    report = score_predictions(EXAMPLES[:1], rows()[:1])
    assert report["refusal_accuracy"] is None


def test_run_eval_asks_the_model_once_per_example_and_scores_the_answers():
    seen = []

    def generate(messages):
        seen.append(messages)
        return call("add_note", {"title": "milk"})

    predictions, report = run_eval(generate, EXAMPLES[:3])
    assert seen == [e["messages"] for e in EXAMPLES[:3]]
    assert [p["id"] for p in predictions] == ["a1", "a2", "f1"]
    assert predictions[0]["output"] == call("add_note", {"title": "milk"})
    assert predictions[0]["prediction"]["tool"] == "add_note"
    assert report == score_predictions(EXAMPLES[:3], predictions)
    assert report["accuracy"] == 1 / 3
