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


def test_the_eval_runner_passes_a_case_when_the_model_does_the_right_thing(tmp_path, monkeypatch):
    from evals.cases import CASES
    from evals.run import run_case
    from llm import FakeModel, say, use

    case = next(c for c in CASES if c["id"] == "delete")
    good = FakeModel([use("list_titles"), use("delete_note", {"title": "milk"}), say("Deleted milk.")])
    row = run_case(good, case)
    assert row["passed"] and row["titles_after"] == ["eggs"]
    assert row["made"] == [{"name": "list_titles", "arguments": {}}, {"name": "delete_note", "arguments": {"title": "milk"}}]

    careless = FakeModel([use("delete_note", {"title": "eggs"}), say("Done.")])
    assert run_case(careless, case)["passed"] is False


def test_there_are_fifteen_cases_and_each_is_well_formed():
    from evals.cases import CASES
    from kit import TOOLS

    assert len(CASES) == 15 and len({c["id"] for c in CASES}) == 15
    for case in CASES:
        assert all(step["name"] in TOOLS for step in case["wanted"])
        assert isinstance(case["titles_after"], list)
