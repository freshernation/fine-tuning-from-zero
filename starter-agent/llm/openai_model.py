"""OpenAI, behind the course's neutral chat() interface.

Needs OPENAI_API_KEY in your environment (or .env). Set AGENT_OPENAI_MODEL to
choose the model.
"""

import os

from llm.fake import ModelError
from llm.wire import openai_messages, openai_reply, openai_tool

DEFAULT_MODEL = "gpt-5-mini"


class OpenAIModel:
    def __init__(self, model=None, client=None):
        if client is None:
            import openai
            client = openai.OpenAI()
        self._client = client
        self.name = model or os.environ.get("AGENT_OPENAI_MODEL") or DEFAULT_MODEL

    def chat(self, messages, system=None, tools=None, max_tokens=None):
        request = {
            "model": self.name,
            "messages": openai_messages(messages, system),
        }
        if tools:
            request["tools"] = [openai_tool(schema) for schema in tools]
        if max_tokens:
            request["max_completion_tokens"] = max_tokens

        return openai_reply(self._send(request))

    def _send(self, request):
        import openai

        try:
            return self._client.chat.completions.create(**request)
        except openai.AuthenticationError as error:
            raise ModelError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
            ) from error
        except openai.RateLimitError as error:
            raise ModelError("OpenAI is rate-limiting you.", retryable=True) from error
        except openai.BadRequestError as error:
            raise ModelError(f"OpenAI rejected the request: {error.message}") from error
        except openai.APIStatusError as error:
            raise ModelError(
                f"OpenAI returned an error ({error.status_code}): {error.message}",
                retryable=error.status_code >= 500,
            ) from error
        except openai.APIConnectionError as error:
            raise ModelError("Could not reach OpenAI.", retryable=True) from error
