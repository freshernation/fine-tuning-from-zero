"""Was that the right tool call? Written on Week 4 Day 1, used until the end of the course.

Two shapes appear everywhere in this file.

An EXAMPLE, one line of a .jsonl file:

    {"id": "s017-03", "kind": "add", ...,
     "messages": [{"role": "user", "content": "jot down: milk - two litres"}],
     "expected": {"tool": "add_note", "arguments": {"title": "milk", "body": "two litres"}}}

    kind is add, list, find, delete, calculate, none or clarify. For none and
    clarify the right thing is to answer in words: expected["tool"] is None.

A PREDICTION, what the model did, in a form that can be scored:

    {"tool": "add_note" or None, "arguments": {...}, "text": "...", "valid": True}

    valid is False only when the model TRIED to call a tool and what it wrote
    could not be read. Plain text with no tool call is valid.
"""

import json

from tunelab.notes import ARGUMENTS

OPEN, CLOSE = "<tool_call>", "</tool_call>"


def parse_prediction(text):
    """Turn the model's raw output into a prediction.

    The model writes a tool call as:   <tool_call>{"name": ..., "arguments": {...}}</tool_call>

    Rules, each of which has a test:
      - no <tool_call> at all: a valid prediction with tool None; text is the output, stripped
      - the FIRST <tool_call> is the prediction; later ones are ignored
      - a missing </tool_call> is forgiven: read to the end of the output
      - it must be a JSON object with a string "name". "arguments" may be an object,
        a string containing a JSON object, or absent (meaning {})
      - anything else is invalid: tool None, arguments {}, valid False
      - text is whatever was written outside the first call, stripped
    """
    raise NotImplementedError


def same_arguments(expected, got, tool):
    """Do the arguments the model gave match the ones we wanted, for this tool?

    - every value must be a string
    - strings are compared after trimming, collapsing runs of whitespace to one
      space, and case-folding
    - for the calculate tool only, ALL whitespace is removed before comparing
    - an argument that is absent equals the same argument given as ""
    - an argument the tool does not have (see tunelab.notes.ARGUMENTS) is a mismatch
    """
    raise NotImplementedError


def score_example(example, prediction):
    """Four yes/no answers about one prediction:

        valid             could the output be read?
        right_tool        the expected tool - or, for none/clarify, no tool
        right_arguments   right tool AND matching arguments (True for a correct no-tool)
        correct           all three
    """
    raise NotImplementedError


def score_predictions(examples, predictions):
    """The report for a whole eval set.

    predictions   rows of {"id", "output", "prediction"}, as run_eval makes them

    Returns:
        n                      number of EXAMPLES
        accuracy               fraction correct
        valid_rate, right_tool_rate, right_arguments_rate
        refusal_accuracy       accuracy over the none and clarify examples only
                               (None if there are none)
        by_kind                {kind: {"n": ..., "accuracy": ...}}

    Every rate is out of ALL the examples. An example with no prediction is wrong
    on every count. A prediction whose id matches no example, or an id predicted
    twice, is a ValueError naming the id: nothing is ever silently skipped.
    """
    raise NotImplementedError


def run_eval(generate, examples):
    """Ask the model about every example and score the answers.

    generate   a function from a list of chat messages to the model's raw text

    Returns (predictions, report), where predictions is a list of
    {"id": ..., "output": raw text, "prediction": parse_prediction(raw text)}.
    """
    raise NotImplementedError
