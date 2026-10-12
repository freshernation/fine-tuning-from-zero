"""One check that the starter agent works, with the fake model. No key, no network.

Run me:  pytest starter-agent -q
"""

import json


def test_the_agent_adds_a_note_and_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_FILE", str(tmp_path / "notes.json"))
    from agent import run_agent
    from kit import SCHEMAS, TOOLS
    from llm import FakeModel, say, use

    model = FakeModel([use("add_note", {"title": "milk", "body": "two litres"}), say("Saved.")])
    run = run_agent(model, "Remember milk: two litres", SCHEMAS, TOOLS)

    assert run["stopped"] == "done" and run["answer"] == "Saved." and run["steps"] == 2
    assert json.loads((tmp_path / "notes.json").read_text()) == [{"title": "milk", "body": "two litres"}]
    assert [tool["name"] for tool in SCHEMAS] == ["add_note", "list_titles", "find_notes", "delete_note", "calculate"]
