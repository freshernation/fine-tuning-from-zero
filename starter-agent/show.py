import json


def format_event(event):
    step = f"[{event['step']}]"
    kind = event["kind"]
    if kind == "call":
        return f"{step} call {event['name']}({json.dumps(event['args'])})"
    if kind == "result":
        return f"{step} result {event['name']}: {event['content']}"
    if kind == "stop":
        return f"{step} stop: {event['reason']}"
    return f"{step} {kind}: {event['text']}"


def format_trace(trace):
    lines = []
    for event in trace:
        lines.append(format_event(event))
    return "\n".join(lines)


def summary(result):
    usage = result["usage"]
    return (f"{result['stopped']} in {result['steps']} steps "
            f"({usage['input']} tokens in, {usage['output']} out)")
