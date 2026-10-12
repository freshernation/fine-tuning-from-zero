"""Broken file 4 of 4.

"Eval LOSS is low and still falling. But when I actually generate from the model
and score the answers, it is terrible - it writes a few words of rambling before
anything else. The loss and the score can't both be right?"

They are both right. They are measuring the model on two different prompts. Find
the bug, fix it, and write in NOTES.md how you found it.
"""

from tunelab.notes import SCHEMAS


def prompt_for_inference(tokenizer, example, system):
    """The text handed to the trained model when it is actually used."""
    chat = [{"role": "system", "content": system}] + list(example["messages"])
    return tokenizer.apply_chat_template(chat, tools=SCHEMAS, tokenize=False)
