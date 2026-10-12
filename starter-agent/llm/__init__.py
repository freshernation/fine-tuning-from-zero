"""The course's model layer - one way to talk to any model.

    from llm import connect

    model = connect()                      # whichever provider .env names
    reply = model.chat([{"role": "user", "content": "Hello"}])

    reply["text"]         "Hi there."
    reply["stop"]         "done" | "tools" | "length" | "refused"
    reply["tool_calls"]   [{"id": "call_1", "name": "add_note", "args": {...}}]
    reply["usage"]        {"input": 12, "output": 5}
    reply["message"]      the assistant turn, ready to append to your list

Three providers sit behind that one function:

    fake     no key, no network, no cost - the default, and what every test uses
    claude   Anthropic's API   (llm/claude_model.py)
    openai   OpenAI's API      (llm/openai_model.py)

You do not need to read this folder before Week 4, Day 4. On that day you
should: it is about 300 lines, and it is the whole difference between the
providers.
"""

import os

from llm.fake import FakeModel, InvalidRequest, ModelError, say, use, use_many

__all__ = [
    "connect", "FakeModel", "ModelError", "InvalidRequest", "say", "use", "use_many",
]

PROVIDERS = ("fake", "claude", "openai")


def connect(provider=None):
    """Return a model. With no argument, reads AGENT_PROVIDER (default: fake)."""
    _load_env_file()
    provider = (provider or os.environ.get("AGENT_PROVIDER") or "fake").strip().lower()

    if provider == "fake":
        return FakeModel()
    if provider == "claude":
        from llm.claude_model import ClaudeModel
        return ClaudeModel()
    if provider == "openai":
        from llm.openai_model import OpenAIModel
        return OpenAIModel()
    raise ValueError(
        f"Unknown provider {provider!r}. Set AGENT_PROVIDER to one of: "
        + ", ".join(PROVIDERS)
    )


def _load_env_file():
    """Read .env if python-dotenv is installed. Real environment variables win."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    load_dotenv(find_dotenv(usecwd=True))
