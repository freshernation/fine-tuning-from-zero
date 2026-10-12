"""Desk's tools, their schemas, and the code that runs them - given, complete.

This is everything you built in week 5, in one file, so that this week is about
exactly one thing: the loop.

    SCHEMAS                        the five tool descriptions you send the model
    TOOLS                          the dispatch table: name -> function
    run_tool(tools, name, args)    run one tool; ALWAYS returns a string
    run_calls(tools, tool_calls)   one "tool" message per call, in order

`tools` is passed in rather than read from a global, so the same two functions
work for any dispatch table - including the one you design in week 8.

The notes live in the file named by the NOTES_FILE environment variable, or
notes.json in the current folder if that is not set.
"""

import json
import os
from pathlib import Path

SAFE_CHARACTERS = "0123456789+-*/(). "


def _path():
    return Path(os.environ.get("NOTES_FILE", "notes.json"))


def _load():
    path = _path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def _save(notes):
    _path().write_text(json.dumps(notes, indent=2))


def _find(notes, title):
    wanted = title.strip().lower()
    for note in notes:
        if note["title"].lower() == wanted:
            return note
    return None


def add_note(title, body=""):
    notes = _load()
    title = title.strip()
    if title == "":
        return "A note needs a title."
    existing = _find(notes, title)
    if existing is not None:
        return f"There is already a note called '{existing['title']}'."
    notes.append({"title": title, "body": body})
    _save(notes)
    return f"Saved '{title}'."


def list_titles():
    notes = _load()
    if not notes:
        return "No notes yet."
    lines = []
    for number, note in enumerate(notes, start=1):
        lines.append(f"{number}. {note['title']}")
    return "\n".join(lines)


def find_notes(word):
    wanted = word.lower()
    lines = []
    for note in _load():
        if wanted in note["title"].lower() or wanted in note["body"].lower():
            lines.append(f"{note['title']}: {note['body']}")
    if not lines:
        return f"No notes mention '{word}'."
    return "\n".join(lines)


def delete_note(title):
    notes = _load()
    note = _find(notes, title)
    if note is None:
        return f"No note called '{title}'."
    notes.remove(note)
    _save(notes)
    return f"Deleted '{note['title']}'."


def calculate(expression):
    for character in expression:
        if character not in SAFE_CHARACTERS:
            return "Error: Unsafe expression"
    if "**" in expression:
        return "Error: Unsafe expression"
    try:
        return str(eval(expression))
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except (SyntaxError, TypeError):
        return "Error: Not a calculation"

# --- their descriptions ------------------------------------------------------

def _text(description):
    return {"type": "string", "description": description}


def _schema(name, description, properties, required):
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


SCHEMAS = [
    _schema(
        "add_note",
        "Save a new note with a short title and an optional body. Use when the "
        "user asks you to remember, save, or write something down. Fails if a "
        "note with that title already exists.",
        {
            "title": _text("A short title, a few words, e.g. 'call the dentist'."),
            "body": _text("The detail of the note. May be left out."),
        },
        ["title"],
    ),
    _schema(
        "list_titles",
        "List the titles of every saved note, numbered. Use when the user asks "
        "what notes they have, or before deleting when you are unsure of a title.",
        {},
        [],
    ),
    _schema(
        "find_notes",
        "Search the saved notes for a word, in titles and bodies, ignoring "
        "capitals. Use to answer questions about what the user has saved.",
        {"word": _text("A single word or short phrase to look for, e.g. 'dentist'.")},
        ["word"],
    ),
    _schema(
        "delete_note",
        "Permanently delete the note with this exact title. This cannot be "
        "undone. Call list_titles first if you are not sure of the exact title.",
        {"title": _text("The exact title of the note to delete.")},
        ["title"],
    ),
    _schema(
        "calculate",
        "Evaluate a basic arithmetic expression such as '17 * 23'. Use for any "
        "arithmetic rather than working it out yourself. Supports only + - * / "
        "and parentheses.",
        {"expression": _text("The arithmetic to evaluate, e.g. '(2 + 3) * 4'.")},
        ["expression"],
    ),
]

# --- running them ------------------------------------------------------------

TOOLS = {
    "add_note": add_note,
    "list_titles": list_titles,
    "find_notes": find_notes,
    "delete_note": delete_note,
    "calculate": calculate,
}


def run_tool(tools, name, args):
    """Run one tool. Never raises - a failure comes back as a sentence."""
    if name not in tools:
        return f"Error: unknown tool '{name}'"
    try:
        return str(tools[name](**args))
    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"


def run_calls(tools, tool_calls):
    """One tool message per call, in order, each carrying its call's id."""
    messages = []
    for call in tool_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": run_tool(tools, call["name"], call["args"]),
        })
    return messages
