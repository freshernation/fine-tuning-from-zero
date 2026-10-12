"""Claude, behind the course's neutral chat() interface.

Needs ANTHROPIC_API_KEY in your environment (or .env). The model defaults to
claude-opus-5-5; set AGENT_CLAUDE_MODEL to use a different one.
"""

import os

from llm.fake import ModelError
from llm.wire import claude_messages, claude_reply, claude_tool

DEFAULT_MODEL = "claude-opus-5-5"

# Claude thinks before it answers, and that thinking is paid for out of
# max_tokens. A small limit can be used up before a single word of the answer
# is written, so the default is generous.
DEFAULT_MAX_TOKENS = 16000


class ClaudeModel:
    def __init__(self, model=None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self._client = client
        self.name = model or os.environ.get("AGENT_CLAUDE_MODEL") or DEFAULT_MODEL

    def chat(self, messages, system=None, tools=None, max_tokens=None):
        request = {
            "model": self.name,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "messages": claude_messages(messages),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [claude_tool(schema) for schema in tools]

        try:
            response = self._send(request)
        except ModelError as error:
            # Claude's replies carry reasoning that is only valid in the exact
            # conversation that produced it. If earlier turns were cut or
            # rewritten, the API rejects the replayed reasoning. The documented
            # recovery is to resend once without it.
            replayed = any("_raw" in message for message in messages)
            if not (replayed and "thinking" in str(error)):
                raise
            request["messages"] = claude_messages(messages, use_raw=False)
            response = self._send(request)

        return claude_reply(response)

    def _send(self, request):
        import anthropic

        try:
            return self._client.messages.create(**request)
        except anthropic.AuthenticationError as error:
            raise ModelError(
                "Claude rejected the API key. Check ANTHROPIC_API_KEY in .env."
            ) from error
        except anthropic.RateLimitError as error:
            raise ModelError("Claude is rate-limiting you.", retryable=True) from error
        except anthropic.BadRequestError as error:
            raise ModelError(f"Claude rejected the request: {error.message}") from error
        except anthropic.APIStatusError as error:
            raise ModelError(
                f"Claude returned an error ({error.status_code}): {error.message}",
                retryable=error.status_code >= 500,
            ) from error
        except anthropic.APIConnectionError as error:
            raise ModelError("Could not reach Claude.", retryable=True) from error
