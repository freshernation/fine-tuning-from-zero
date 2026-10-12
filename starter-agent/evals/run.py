"""Run the fifteen cases against whatever model the environment names.

    cd starter-agent
    python3 evals/run.py                      # the fake model: every case fails, harmlessly
    python3 evals/run.py --out ../week-06/day-3/agent_report.json

With AGENT_PROVIDER=openai and OPENAI_BASE_URL pointing at your own server, this is
your fine-tuned model doing the agent's whole job.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from agent import run_agent
from cli import SYSTEM
from kit import SCHEMAS, TOOLS
from llm import connect

from evals.cases import CASES


def run_case(model, case):
    """One case, in its own throwaway notes file. Returns its row of the report."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "notes.json"
        path.write_text(json.dumps(case["notes"]))
        previous = os.environ.get("NOTES_FILE")
        os.environ["NOTES_FILE"] = str(path)
        try:
            run = run_agent(model, case["task"], SCHEMAS, TOOLS, system=SYSTEM)
        finally:
            if previous is None:
                del os.environ["NOTES_FILE"]
            else:
                os.environ["NOTES_FILE"] = previous
        titles = [n["title"] for n in json.loads(path.read_text())]

    made = [{"name": step["name"], "arguments": step["args"]} for step in run["trace"] if step["kind"] == "call"]
    answer = run["answer"] or ""
    passed = (
        run["stopped"] == "done"
        and sorted(t.casefold() for t in titles) == sorted(t.casefold() for t in case["titles_after"])
        and all(word.casefold() in answer.casefold() for word in case.get("answer_has", []))
        and all(name in [m["name"] for m in made] for name in case.get("must_call", []))
        and ("must_call_any" not in case or any(m["name"] in case["must_call_any"] for m in made))
        and not (case.get("no_calls") and made)
    )
    return {"id": case["id"], "passed": passed, "wanted": case["wanted"], "made": made,
            "stopped": run["stopped"], "steps": run["steps"], "titles_after": titles, "answer": answer}


def run_all(model, cases=CASES):
    rows = [run_case(model, case) for case in cases]
    return {"cases": rows, "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="agent_report.json")
    args = parser.parse_args()
    report = run_all(connect())
    Path(args.out).write_text(json.dumps(report, indent=1))
    for row in report["cases"]:
        print(f'{"pass" if row["passed"] else "FAIL"}  {row["id"]:<20} {row["steps"]} steps, stopped: {row["stopped"]}')
    print(f'\n{report["passed"]} of {len(report["cases"])} passed. Wrote {args.out}')


if __name__ == "__main__":
    main()
