"""Week 6, Day 2 - serve it, and score what is actually served.

Run me:  pytest week-06/day-2 -v

Everything today goes in tunelab/served.py. `fake_server` is a small local server
that answers like a real one; nothing here needs a model.
"""

import json
import time

import pytest

from tunelab.notes import SCHEMAS
from tunelab.served import chat, eval_served, make_served_generate, to_api_messages, to_prediction, wait_until_ready


def tool_message(name, arguments, call_id="call_abc"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


# ---- to_api_messages ---------------------------------------------------------------

OURS = [
    {"role": "user", "content": "delete milk"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "list_titles", "arguments": {}}}]},
    {"role": "tool", "content": "1. milk"},
]


def test_our_messages_become_api_messages():
    assert to_api_messages(OURS) == [
        {"role": "user", "content": "delete milk"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "list_titles", "arguments": "{}"}}]},
        {"role": "tool", "content": "1. milk", "tool_call_id": "call_0"},
    ]


def test_each_tool_result_is_tied_to_its_own_call():
    twice = OURS + [
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "find_notes", "arguments": {"word": "café"}}}]},
        {"role": "tool", "content": "none"},
    ]
    api = to_api_messages(twice)
    assert [m.get("tool_call_id") for m in api] == [None, None, "call_0", None, "call_1"]
    assert api[3]["tool_calls"][0]["function"]["arguments"] == '{"word": "café"}'
    assert "tool_call_id" not in api[0]


def test_to_api_messages_leaves_its_input_alone():
    before = json.dumps(OURS)
    to_api_messages(OURS)
    assert json.dumps(OURS) == before


# ---- to_prediction -----------------------------------------------------------------

def test_a_tool_call_reply():
    assert to_prediction(tool_message("add_note", {"title": "milk"})) == {
        "tool": "add_note", "arguments": {"title": "milk"}, "text": "", "valid": True}


def test_a_text_reply():
    assert to_prediction({"role": "assistant", "content": " Which note? "}) == {
        "tool": None, "arguments": {}, "text": "Which note?", "valid": True}


def test_the_first_of_two_calls_wins():
    message = tool_message("list_titles", {})
    message["tool_calls"].append(tool_message("delete_note", {"title": "milk"})["tool_calls"][0])
    assert to_prediction(message)["tool"] == "list_titles"


def test_arguments_that_are_not_json_make_the_call_invalid():
    message = tool_message("add_note", {})
    message["tool_calls"][0]["function"]["arguments"] = "{title: milk"
    assert to_prediction(message) == {"tool": None, "arguments": {}, "text": "", "valid": False}


def test_a_tool_call_that_leaked_into_the_text_is_flagged():
    leaked = {"role": "assistant", "content": '<tool_call>\n{"name": "add_note", "arguments": {"title": "milk"}}\n</tool_call>'}
    prediction = to_prediction(leaked)
    assert prediction["tool"] == "add_note" and prediction["arguments"] == {"title": "milk"}
    assert prediction["template_problem"] is True
    assert "template_problem" not in to_prediction(tool_message("add_note", {"title": "milk"}))
    assert "template_problem" not in to_prediction({"role": "assistant", "content": "Hello"})


def test_a_leaked_call_that_cannot_even_be_read_is_flagged_and_invalid():
    leaked = {"role": "assistant", "content": "<tool_call>\n{'name': 'add_note'"}
    prediction = to_prediction(leaked)
    assert prediction["valid"] is False and prediction["tool"] is None
    assert prediction["template_problem"] is True


# ---- talking to a server -----------------------------------------------------------

def test_wait_until_ready_waits(fake_server):
    fake_server.not_ready = 2
    wait_until_ready(fake_server.url, timeout_s=10, interval_s=0.05)
    assert fake_server.not_ready == 0


def test_wait_until_ready_gives_up_and_names_the_url():
    started = time.time()
    with pytest.raises(TimeoutError, match="127.0.0.1:9/v1"):
        wait_until_ready("http://127.0.0.1:9/v1", timeout_s=0.5, interval_s=0.05)     # nothing listens on port 9
    assert time.time() - started < 5


def test_chat_sends_what_it_should_and_returns_the_message(fake_server):
    fake_server.default = tool_message("add_note", {"title": "milk"})
    reply = chat(fake_server.url, [{"role": "user", "content": "save milk"}], SCHEMAS, model="notes", max_tokens=64)
    assert isinstance(reply, dict) and reply["tool_calls"][0]["function"]["name"] == "add_note"
    sent = fake_server.requests[0]
    assert sent["model"] == "notes" and sent["temperature"] == 0 and sent["max_tokens"] == 64
    assert [t["function"]["name"] for t in sent["tools"]] == ["add_note", "list_titles", "find_notes", "delete_note", "calculate"]
    assert sent["messages"] == [{"role": "user", "content": "save milk"}]


def test_served_generate_puts_the_system_prompt_first_and_converts_history(fake_server):
    generate = make_served_generate(fake_server.url, SCHEMAS, system="You manage notes.")
    reply = generate(OURS)
    assert reply == {"role": "assistant", "content": "OK."} or reply["content"] == "OK."
    sent = fake_server.requests[0]["messages"]
    assert sent[0] == {"role": "system", "content": "You manage notes."}
    assert sent[1:] == to_api_messages(OURS)


def example(i, kind, tool, text, arguments=None):
    return {"id": i, "seed_id": i, "kind": kind, "source": "seed", "messages": [{"role": "user", "content": text}],
            "expected": {"tool": tool, "arguments": arguments or {}, **({"text": "ok"} if tool is None else {})}}


def test_eval_served_scores_what_the_server_said(fake_server):
    examples = [
        example("a", "add", "add_note", "save milk", {"title": "milk"}),
        example("b", "add", "add_note", "save eggs", {"title": "eggs"}),
        example("c", "none", None, "thanks"),
        example("d", "find", "find_notes", "find jam", {"word": "jam"}),
        example("e", "list", "list_titles", "list them"),
    ]
    fake_server.replies = {
        "save milk": tool_message("add_note", {"title": "milk"}),                       # correct
        "save eggs": tool_message("add_note", {"title": "egg"}),                        # wrong arguments
        "thanks": {"role": "assistant", "content": "You're welcome."},                  # correct
        "find jam": {"role": "assistant", "content": '<tool_call>{"name": "find_notes", "arguments": {"word": "jam"}}</tool_call>'},   # leaked
        "list them": {"role": "assistant", "content": "I can't do that."},              # did not call
    }
    predictions, report = eval_served(fake_server.url, examples, SCHEMAS, system="You manage notes.")
    assert [p["id"] for p in predictions] == ["a", "b", "c", "d", "e"]
    assert all(isinstance(p["output"], str) and "prediction" in p for p in predictions)
    assert json.loads(predictions[0]["output"])["tool_calls"][0]["function"]["name"] == "add_note"
    assert report["n"] == 5 and report["accuracy"] == 3 / 5, "the leaked call still counts as the call it is"
    assert report["template_problems"] == 1
    assert report["by_kind"]["add"] == {"n": 2, "accuracy": 0.5}
    assert len(fake_server.requests) == 5
