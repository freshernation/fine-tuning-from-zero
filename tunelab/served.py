"""Talking to a model that is running as a server. Written on Week 6 Day 2.

The server speaks the OpenAI chat protocol, as llama-server, Ollama and many others
do. So does the `openai` Python package, which is all that is used here: pointed at
your own machine, with a made-up API key, it never goes near OpenAI.
"""

import json
import time
import urllib.error
import urllib.request

from openai import OpenAI

from tunelab.toolcall import parse_prediction, score_predictions


def to_api_messages(messages):
    """The course's message format, turned into what the chat API requires.

    Three differences, all in tool calls:
      - every tool call needs an "id". Number them "call_0", "call_1", ... in the
        order they appear in the conversation.
      - a call's "arguments" must be a JSON STRING, not an object
        (json.dumps with ensure_ascii=False).
      - every {"role": "tool"} message needs "tool_call_id": the id of the most
        recent tool call before it.
    Everything else is copied as it is. The input is not modified.
    """
    raise NotImplementedError


def to_prediction(message):
    """An assistant message from the API, as a prediction that tunelab/toolcall.py can score.

      - message["tool_calls"] present and non-empty: the FIRST one is the prediction.
        Its arguments arrive as a JSON string; if that string is not a JSON object,
        the prediction is invalid (tool None, arguments {}, text "", valid False).
      - otherwise the content is text. But run it through parse_prediction first:
        if the "text" turns out to contain a tool call, the server did not apply the
        chat template properly. Return that call, and add "template_problem": True.
      - otherwise: tool None, the stripped content as text, valid True.

    "template_problem" is only present when there is one.
    """
    raise NotImplementedError


def wait_until_ready(base_url, timeout_s=60, interval_s=0.5):
    """Block until the server answers GET <base_url>/models with status 200.

    A server that is still loading its model answers 503, or refuses the connection
    altogether: keep trying every interval_s seconds. After timeout_s seconds, raise
    TimeoutError with a message that includes base_url.

    Use urllib.request.urlopen; a failed request raises urllib.error.URLError
    (HTTPError, for a 503, is a kind of URLError).
    """
    raise NotImplementedError


def chat(base_url, messages, tools, model="notes", max_tokens=128):
    """One request to the server. Returns the assistant's message as a plain dict.

    OpenAI(base_url=base_url, api_key="local", max_retries=0)
    client.chat.completions.create(model=..., messages=..., tools=..., temperature=0, max_tokens=...)
    The message is response.choices[0].message; .model_dump() makes it a dict.

    temperature=0 for the reason greedy decoding was used all along: a score
    measured twice should be the same score.
    """
    raise NotImplementedError


def make_served_generate(base_url, tools, system):
    """A function from the course's messages to the server's reply message (a dict):
    the system prompt first, then to_api_messages(messages), sent with chat()."""
    raise NotImplementedError


def eval_served(base_url, examples, tools, system):
    """Run a whole eval set through the server and score it.

    Returns (predictions, report). predictions are rows of
        {"id": ..., "output": the reply message as a JSON string, "prediction": to_prediction(reply)}
    report is score_predictions(...) with one more key, "template_problems":
    how many predictions carry that flag.
    """
    raise NotImplementedError
