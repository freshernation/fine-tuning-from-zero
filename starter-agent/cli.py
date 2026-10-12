# Desk - an agent over a notes file. Run with:  python3 cli.py
import os

from agent import run_agent
from kit import SCHEMAS, TOOLS
from llm import connect
from show import format_trace, summary

SYSTEM = (
    "You are Desk, an agent that manages one person's notes. Use your tools to "
    "read and change the notes; never guess what is saved and never do arithmetic "
    "in your head. If a tool returns an error, read it, fix your call and try "
    "once more; if it fails again, tell the user plainly what went wrong. Before "
    "deleting, confirm the exact title with list_titles. Answer in one or two "
    "sentences."
)


def answer_task(model, task):
    result = run_agent(model, task, SCHEMAS, TOOLS, system=SYSTEM)
    text = f"{result['answer']}\n({summary(result)})"
    if os.environ.get("AGENT_TRACE") == "1":
        text = format_trace(result["trace"]) + "\n\n" + text
    return text


def main():
    model = connect()
    while True:
        task = input("Task: ")
        if task.strip().lower() == "quit":
            print("Bye.")
            break
        print(answer_task(model, task))


if __name__ == "__main__":
    main()
