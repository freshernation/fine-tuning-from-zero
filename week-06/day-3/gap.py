"""Day 3 - why the agent succeeds less often than your eval score says the model is right."""

from tunelab.toolcall import same_arguments


def expected_task_success(step_accuracy, steps):
    """If each decision is right with probability step_accuracy, independently, how
    likely is a task that needs `steps` decisions to go right from start to finish?"""
    raise NotImplementedError


def first_wrong_step(made, wanted):
    """Where a run first left the path, or None if it never did.

    made, wanted   lists of tool calls, each {"name": ..., "arguments": {...}}

    Walk the two lists together. The answer is the index of the first position where
    the names differ, or the arguments do not match (use same_arguments from week 4),
    or one list has a call and the other has run out.
    """
    raise NotImplementedError


def compare_with_expected(agent_report, step_accuracy):
    """One row per case in the report: what the arithmetic predicted, and what happened.

    agent_report["cases"] is a list of {"id", "passed", "wanted", "made"}.

    Each row: {"id", "decisions", "expected_success", "passed", "first_wrong_step"}.
    "decisions" is len(wanted) + 1: one per tool call, plus the decision to stop and
    answer in words at the end.
    """
    raise NotImplementedError


def summarise(agent_report, step_accuracy):
    """{"cases", "passed", "pass_rate", "expected_pass_rate"} - the last is the mean of
    the cases' expected_success."""
    raise NotImplementedError
