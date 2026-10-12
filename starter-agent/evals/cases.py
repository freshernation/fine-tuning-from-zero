"""Fifteen end-to-end tasks for the notes agent.

Each case is a starting set of notes, something a person asks, and what should be true
afterwards. "wanted" is the sequence of tool calls a careful agent would make; it is
there for analysis, not for grading. A case PASSES on its outcome: the notes that
exist at the end; where it matters, a number or a word in the final answer; and one
check on behaviour - "must_call" names tools that have to have been used (an agent
that answers "what notes do I have?" without looking has not passed), and
"no_calls" marks the cases where the right thing is to use no tool at all.
"""


def note(title, body=""):
    return {"title": title, "body": body}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


THREE = [note("gym", "Tuesdays at 7"), note("rent", "due on the 1st"), note("plumber", "Dave")]

CASES = [
    {"id": "add-simple", "notes": [], "task": "Remember: call the dentist",
     "wanted": [call("add_note", title="call the dentist")], "titles_after": ["call the dentist"]},
    {"id": "add-with-body", "notes": [], "task": "jot down milk — two litres",
     "wanted": [call("add_note", title="milk", body="two litres")], "titles_after": ["milk"]},
    {"id": "add-chatty", "notes": [note("milk")], "task": "hey, could you save a note called passport for me? thanks",
     "wanted": [call("add_note", title="passport")], "titles_after": ["milk", "passport"]},
    {"id": "add-duplicate", "notes": [note("milk", "two litres")], "task": "save a note called milk",
     "wanted": [call("add_note", title="milk")], "titles_after": ["milk"], "must_call": ["add_note"]},
    {"id": "list", "notes": [note("milk"), note("eggs")], "task": "what notes do I have?",
     "wanted": [call("list_titles")], "titles_after": ["milk", "eggs"], "answer_has": ["milk", "eggs"], "must_call": ["list_titles"]},
    {"id": "list-empty", "notes": [], "task": "show me everything",
     "wanted": [call("list_titles")], "titles_after": [], "must_call": ["list_titles"]},
    {"id": "find", "notes": [note("call the dentist", "before Friday"), note("milk")], "task": "is there anything about the dentist?",
     "wanted": [call("find_notes", word="dentist")], "titles_after": ["call the dentist", "milk"], "answer_has": ["dentist"], "must_call": ["find_notes"]},
    {"id": "find-nothing", "notes": [note("milk")], "task": "do I have a note on passport?",
     "wanted": [call("find_notes", word="passport")], "titles_after": ["milk"], "must_call": ["find_notes"]},
    {"id": "delete", "notes": [note("milk"), note("eggs")], "task": "delete the milk note",
     "wanted": [call("list_titles"), call("delete_note", title="milk")], "titles_after": ["eggs"]},
    {"id": "delete-of-three", "notes": THREE, "task": "I'm done with the gym one, get rid of it",
     "wanted": [call("list_titles"), call("delete_note", title="gym")], "titles_after": ["rent", "plumber"]},
    {"id": "delete-missing", "notes": [note("eggs")], "task": "delete the milk note",
     "wanted": [call("list_titles")], "titles_after": ["eggs"], "must_call": ["list_titles"]},
    {"id": "calculate", "notes": [], "task": "what's 17 times 23?",
     "wanted": [call("calculate", expression="17 * 23")], "titles_after": [], "answer_has": ["391"], "must_call": ["calculate"]},
    {"id": "calculate-brackets", "notes": [], "task": "(12 + 8) * 5 please",
     "wanted": [call("calculate", expression="(12 + 8) * 5")], "titles_after": [], "answer_has": ["100"], "must_call": ["calculate"]},
    {"id": "nothing-to-do", "notes": [note("milk")], "task": "thanks, that's all",
     "wanted": [], "titles_after": ["milk"], "no_calls": True},
    {"id": "missing-detail", "notes": [note("milk")], "task": "save a note",
     "wanted": [], "titles_after": ["milk"], "no_calls": True},
]
