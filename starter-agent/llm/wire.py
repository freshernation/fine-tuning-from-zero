"""Translation between the course's neutral shapes and each provider's own.

Every function here is pure: data in, data out, no network. That is what makes
an adapter testable, and it is why this file is the one worth reading.

The neutral shapes
------------------
A message is one of:

    {"role": "user",      "content": "text"}
    {"role": "assistant", "content": "text", "tool_calls": [call, ...]}
    {"role": "tool",      "tool_call_id": "call_1", "content": "result text"}

A tool call is     {"id": "call_1", "name": "add_note", "args": {"title": "x"}}
A tool schema is   {"name": ..., "description": ..., "parameters": {JSON Schema}}
"""

import json


def make_reply(text="", tool_calls=None, stop=None, input=0, output=0, raw=None):
    """Build the dict every provider's chat() returns."""
    tool_calls = list(tool_calls or [])
    if stop is None:
        stop = "tools" if tool_calls else "done"

    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if raw is not None:
        # The provider's own copy of this turn. Claude attaches reasoning to its
        # replies that must be sent back exactly as it arrived, so the adapter
        # keeps the original here and replays it. Leave keys starting with "_"
        # alone.
        message["_raw"] = raw

    return {
        "text": text,
        "tool_calls": tool_calls,
        "stop": stop,
        "usage": {"input": input, "output": output},
        "message": message,
    }


# --- Claude ------------------------------------------------------------------

CLAUDE_STOPS = {
    "end_turn": "done",
    "stop_sequence": "done",
    "pause_turn": "done",
    "tool_use": "tools",
    "max_tokens": "length",
    "refusal": "refused",
}


def claude_tool(schema):
    """Claude calls the JSON Schema `input_schema`."""
    return {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["parameters"],
    }


def claude_messages(messages, use_raw=True):
    """Neutral messages -> Claude's.

    Two real differences. Claude has no "tool" role: results travel inside a
    USER message, as tool_result blocks. And several results in a row must share
    ONE user message, not one each.
    """
    out = []
    for message in messages:
        role = message["role"]

        if role == "user":
            out.append({"role": "user", "content": message["content"]})

        elif role == "assistant":
            if use_raw and "_raw" in message:
                out.append({"role": "assistant", "content": message["_raw"]})
                continue
            blocks = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls", []):
                blocks.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["args"],
                })
            out.append({"role": "assistant", "content": blocks})

        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            previous = out[-1] if out else None
            carrying_results = (
                previous is not None
                and previous["role"] == "user"
                and isinstance(previous["content"], list)
            )
            if carrying_results:
                previous["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

    return out


def claude_reply(response):
    """Claude's response object -> the neutral reply.

    response.content is a LIST of blocks - text, tool_use, and thinking blocks
    this course never reads. Check .type before touching anything else.
    """
    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "args": block.input})

    return make_reply(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop=CLAUDE_STOPS.get(response.stop_reason, "done"),
        input=response.usage.input_tokens,
        output=response.usage.output_tokens,
        raw=response.content,
    )


# --- OpenAI ------------------------------------------------------------------

OPENAI_STOPS = {
    "stop": "done",
    "tool_calls": "tools",
    "length": "length",
    "content_filter": "refused",
}


def openai_tool(schema):
    """OpenAI wraps the same three fields one level deeper."""
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        },
    }


def openai_messages(messages, system=None):
    """Neutral messages -> OpenAI's.

    Two real differences. The system prompt is the first MESSAGE rather than a
    separate setting. And a tool call's arguments travel as a JSON *string*, not
    a dict.
    """
    out = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = message["role"]

        if role == "user":
            out.append({"role": "user", "content": message["content"]})

        elif role == "assistant":
            turn = {"role": "assistant", "content": message.get("content") or None}
            calls = message.get("tool_calls", [])
            if calls:
                turn["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"]),
                        },
                    }
                    for call in calls
                ]
            out.append(turn)

        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": message["content"],
            })

    return out


def openai_reply(response):
    """OpenAI's response object -> the neutral reply."""
    choice = response.choices[0]
    message = choice.message

    tool_calls = []
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # The model wrote JSON that does not parse. Hand the tool something
            # it will reject with a readable error rather than crashing here.
            args = {"_unparsed_arguments": call.function.arguments}
        tool_calls.append({"id": call.id, "name": call.function.name, "args": args})

    return make_reply(
        text=message.content or "",
        tool_calls=tool_calls,
        stop=OPENAI_STOPS.get(choice.finish_reason, "done"),
        input=response.usage.prompt_tokens,
        output=response.usage.completion_tokens,
    )
