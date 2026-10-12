from guards import clip, is_repeating, signature
from kit import run_calls
from llm import ModelError


def finish(run, answer, stopped):
    """Fill in how the run ended and hand it back."""
    if stopped != "done":
        run["trace"].append({"step": run["steps"], "kind": "stop", "reason": stopped})
    run["answer"] = answer
    run["stopped"] = stopped
    return run


def run_agent(model, task, schemas, tools, system=None,
              max_steps=10, repeat_limit=3, result_limit=2000):
    messages = [{"role": "user", "content": task}]
    trace = []
    signatures = []
    run = {"answer": "", "stopped": "", "steps": 0,
           "usage": {"input": 0, "output": 0}, "trace": trace, "messages": messages}

    while run["steps"] < max_steps:
        try:
            reply = model.chat(messages, system=system, tools=schemas)
        except ModelError as error:
            return finish(run, f"The model is unavailable: {error}", "error")

        run["steps"] = run["steps"] + 1
        step = run["steps"]
        run["usage"]["input"] = run["usage"]["input"] + reply["usage"]["input"]
        run["usage"]["output"] = run["usage"]["output"] + reply["usage"]["output"]
        messages.append(reply["message"])

        if reply["stop"] == "length":
            return finish(run, reply["text"] + " [cut off]", "length")
        if reply["stop"] == "refused":
            return finish(run, "The model declined this request.", "refused")
        if reply["stop"] != "tools":
            trace.append({"step": step, "kind": "answer", "text": reply["text"]})
            return finish(run, reply["text"], "done")

        if reply["text"]:
            trace.append({"step": step, "kind": "say", "text": reply["text"]})
        for call in reply["tool_calls"]:
            trace.append({"step": step, "kind": "call",
                          "name": call["name"], "args": call["args"]})
            signatures.append(signature(call))
            if is_repeating(signatures, repeat_limit):
                return finish(run, "I kept repeating the same step, so I stopped.",
                              "repeating")

        results = run_calls(tools, reply["tool_calls"])
        for call, message in zip(reply["tool_calls"], results):
            message["content"] = clip(message["content"], result_limit)
            trace.append({"step": step, "kind": "result",
                          "name": call["name"], "content": message["content"]})
        messages.extend(results)

    return finish(run, "I ran out of steps before finishing.", "max_steps")
