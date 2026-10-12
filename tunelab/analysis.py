"""Past the single number: what kind of wrong, which examples moved, and is the gain real.

Written on Week 5 Day 4. Everything here works on the examples and prediction rows
from tunelab/toolcall.py.
"""

import random

from tunelab.toolcall import score_example


def failure_type(example, prediction):
    """One word for what happened on one example:

        correct
        invalid                      the model tried to call a tool and it could not be read
        wrong_tool                   a tool was wanted; a different tool was called
        wrong_arguments              the right tool, with the wrong things in it
        did_not_call                 a tool was wanted; the model answered in words
        called_when_it_should_not    no tool was wanted (none / clarify); one was called

    `prediction` is the prediction dict itself, not the row that holds it.
    Check "invalid" first: an unreadable call is invalid whatever was wanted.
    """
    raise NotImplementedError


def error_table(examples, predictions):
    """{kind: {failure type: how many}}, with only the failure types that occurred.

    predictions   rows of {"id", "output", "prediction"}
    An example with no prediction counts under "missing".
    """
    raise NotImplementedError


def changed(examples, before, after):
    """Which examples moved between two sets of predictions.

    Returns four lists of ids, each in the order the examples are given:
        "fixed"         wrong before, correct after
        "broken"        correct before, wrong after
        "still_wrong"
        "still_right"
    A missing prediction is wrong.
    """
    raise NotImplementedError


def paired_bootstrap(before_correct, after_correct, n=2000, seed=0):
    """A 95% interval for (accuracy after - accuracy before).

    before_correct, after_correct   one True/False per example, in the SAME order

    n times: draw len(examples) positions at random WITH replacement, using one
    random.Random(seed) for the whole function, and take the difference in accuracy
    over those positions - the same positions for both lists. Sort the n differences.
    Return (the value at index int(0.025 * n), the value at index int(0.975 * n) - 1).

    Lists of different lengths are a ValueError.
    """
    raise NotImplementedError
