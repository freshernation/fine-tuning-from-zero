"""Day 1 - a chat is one long string."""

ROLES = ("system", "user", "assistant")


def render_chatml(messages, add_generation_prompt=False):
    """Turn a list of chat messages into the single string the model actually reads.

    messages   [{"role": "user", "content": "Hello"}, ...]

    Each message becomes:    <|im_start|>ROLE\\nCONTENT<|im_end|>\\n
    With add_generation_prompt=True, finish with the opening of the assistant's
    next turn and nothing after it:    <|im_start|>assistant\\n

    Content goes in exactly as given: no stripping, no escaping.
    A role that is not in ROLES is a ValueError that names the role.
    """
    raise NotImplementedError
