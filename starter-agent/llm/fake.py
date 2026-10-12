"""A stand-in model that never touches the network.

Real model calls cost money, need a key, and answer differently every time -
none of which a test can work with. So every function you write takes a `model`
argument, and the tests hand it one of these.

That is not a testing trick. Passing the model in, rather than reaching for a
global, is called dependency injection. It is how every serious codebase handles
anything external, and it is the only reason your agent will be testable at all.

    from llm import FakeModel, say, use

    model = FakeModel([use("add_note", {"title": "milk"}), say("Saved.")])
    reply = model.chat([{"role": "user", "content": "Remember milk"}])
    reply["tool_calls"]             # [{"id": "call_1", "name": "add_note", ...}]
    model.requests[0]["messages"]   # exactly what your code sent

The fake also enforces the rules a real API enforces about what a conversation
may look like - so a broken loop fails here, loudly and for free, instead of the
first time it meets the real thing.
"""

import copy

from llm.wire import make_reply


class ModelError(Exception):
    """The model could not be reached, or refused the request.

    `retryable` says whether trying again could possibly help: True for a rate
    limit or a network blip, False for a bad key or a malformed request.
    """

    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class InvalidRequest(ModelError):
    """The request would have been rejected by a real API. Never retryable."""


# --- building scripted replies ------------------------------------------------

def say(text, stop="done", input=10, output=20):
    """A reply that is just text."""
    return make_reply(text=text, stop=stop, input=input, output=output)


def use(name, args=None, id="call_1", text="", input=10, output=20):
    """A reply in which the model asks for one tool."""
    call = {"id": id, "name": name, "args": dict(args or {})}
    return make_reply(text=text, tool_calls=[call], input=input, output=output)


def use_many(calls, text="", input=10, output=20):
    """A reply asking for several tools at once.

    use_many([("add", {"a": 1}), ("add", {"a": 2})])   ids are call_1, call_2
    use_many([("add", {"a": 1}, "my_id")])             or give your own
    """
    tool_calls = []
    for position, call in enumerate(calls, start=1):
        name, args = call[0], call[1]
        call_id = call[2] if len(call) > 2 else f"call_{position}"
        tool_calls.append({"id": call_id, "name": name, "args": dict(args or {})})
    return make_reply(text=text, tool_calls=tool_calls, input=input, output=output)


# --- the model ----------------------------------------------------------------

class FakeModel:
    """A scripted model.

    FakeModel([reply_1, reply_2])   hands those back in order
    FakeModel(reply)                hands the same one back every time
    FakeModel()                     repeats your last message back to you

    Put a ModelError in the list to make that call fail, which is how you test
    what your code does when the model is unavailable.
    """

    name = "fake"

    def __init__(self, script=None):
        self._repeat = script is not None and not isinstance(script, list)
        if script is None:
            self._script = []
        elif self._repeat:
            self._script = [script]
        else:
            self._script = list(script)
        self._unscripted = script is None
        self._position = 0
        self.requests = []

    def chat(self, messages, system=None, tools=None, max_tokens=None):
        check_messages(messages)
        check_tools(tools)
        if system is not None and not isinstance(system, str):
            raise InvalidRequest(f"system must be a string, not {type(system).__name__}.")

        self.requests.append({
            "messages": copy.deepcopy(messages),
            "system": system,
            "tools": copy.deepcopy(tools),
            "max_tokens": max_tokens,
        })

        if self._unscripted:
            return say(f"(fake model) You said: {_last_user_text(messages)}")

        if self._repeat:
            item = self._script[0]
        else:
            if self._position >= len(self._script):
                raise AssertionError(
                    f"The code called the model {self._position + 1} times but only "
                    f"{len(self._script)} replies were scripted. It should have "
                    "stopped by now."
                )
            item = self._script[self._position]
            self._position += 1

        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

    @property
    def call_count(self):
        return len(self.requests)

    @property
    def last_request(self):
        return self.requests[-1] if self.requests else None


def _last_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


# --- the rules a real API enforces -------------------------------------------

ROLES = ("user", "assistant", "tool")


def check_messages(messages):
    """Raise InvalidRequest if the conversation is not a legal one."""
    if not isinstance(messages, list) or not messages:
        raise InvalidRequest(
            "messages must be a list with at least one message in it. "
            "The model needs something to reply to."
        )

    waiting = []          # ids of tool calls that have not been answered yet
    asked = False         # has the most recent assistant turn asked for tools?
    previous = None

    for position, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise InvalidRequest(
                f"Message {position} is {message!r}. Every message is a dict "
                'with a "role".'
            )
        role = message["role"]

        if role == "system":
            raise InvalidRequest(
                f'Message {position} has the role "system". The system prompt is '
                "not part of the list - pass it as chat(messages, system=...)."
            )
        if role not in ROLES:
            raise InvalidRequest(
                f"Message {position} has the role {role!r}. Roles are: "
                + ", ".join(ROLES) + "."
            )
        if not isinstance(message.get("content"), str):
            raise InvalidRequest(
                f"Message {position} ({role}) has content "
                f"{message.get('content')!r}. Content is always a string - "
                "convert numbers and lists with str() first."
            )
        if position == 0 and role != "user":
            raise InvalidRequest(
                f"The conversation must start with a user message. Yours starts "
                f"with {role!r}."
            )

        if role == "tool":
            call_id = message.get("tool_call_id")
            if not asked:
                raise InvalidRequest(
                    f"Message {position} is a tool result, but the assistant turn "
                    "before it asked for no tools.\n"
                    "Append reply[\"message\"] - what the model said, tool calls "
                    "included - BEFORE you append the results. Otherwise the model "
                    "is handed an answer to a question it has no record of asking."
                )
            if call_id not in waiting:
                raise InvalidRequest(
                    f"Message {position} answers tool call {call_id!r}, but the "
                    f"calls still waiting for an answer are {waiting}.\n"
                    "Each call gets exactly one result, carrying that call's own "
                    "id. The id is the only thing connecting an answer to its "
                    "question."
                )
            waiting.remove(call_id)
        else:
            if waiting:
                raise InvalidRequest(
                    f"Message {position} ({role}) arrives while tool call(s) "
                    f"{waiting} have no result.\n"
                    "Every tool call the model makes must be answered before "
                    "anybody speaks again - even if the answer is an error message."
                )
            if role == previous:
                raise InvalidRequest(
                    f"Two {role} messages in a row at position {position}. "
                    "Did you append the same turn twice, or forget to append the "
                    "reply in between?"
                )
            asked = False
            if role == "assistant":
                calls = message.get("tool_calls", [])
                ids = [call.get("id") for call in calls]
                if len(set(ids)) != len(ids):
                    raise InvalidRequest(
                        f"Message {position} contains two tool calls with the "
                        f"same id: {ids}."
                    )
                waiting = list(ids)
                asked = bool(ids)

        previous = role

    if waiting:
        raise InvalidRequest(
            f"Tool call(s) {waiting} have no result.\n"
            "Every tool call the model makes must come back with a result before "
            "you call the model again - even if that result is an error message."
        )
    if previous == "assistant":
        raise InvalidRequest(
            "The last message is the model's own. It has nothing to reply to - "
            "the list must end with a user message or a tool result."
        )


def check_tools(tools):
    """Raise InvalidRequest if a tool schema is malformed."""
    if tools is None:
        return
    if not isinstance(tools, list):
        raise InvalidRequest("tools must be a list of schemas.")

    seen = set()
    for schema in tools:
        if not isinstance(schema, dict):
            raise InvalidRequest(
                f"tools contains {schema!r}. Pass the SCHEMAS (dicts describing "
                "each tool), not the functions themselves."
            )
        for key in ("name", "description", "parameters"):
            if key not in schema:
                raise InvalidRequest(
                    f"The tool schema {schema.get('name', schema)!r} has no "
                    f"{key!r}. A schema needs name, description and parameters."
                )
        name = schema["name"]
        if name in seen:
            raise InvalidRequest(f"Two tools are both called {name!r}.")
        seen.add(name)

        parameters = schema["parameters"]
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise InvalidRequest(
                f'The parameters of {name!r} must be a dict with "type": "object".'
            )
        properties = parameters.get("properties", {})
        for required in parameters.get("required", []):
            if required not in properties:
                raise InvalidRequest(
                    f"{name!r} lists {required!r} as required, but there is no "
                    "such property."
                )
