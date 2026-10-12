"""Broken file 2 of 4.

"Training loss looks GREAT, way lower than my friend's. But the model is no better
at tool calls than before I trained it."

The run is fine. What it was asked to learn is not. Find the bug, fix it, and write
in NOTES.md how you found it.
"""

from tunelab.dataset import render_completion, render_prompt


def tokenize(tokenizer, example, system):
    """An example as token ids, ready for training."""
    text = render_prompt(tokenizer, example, system) + render_completion(example)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}
