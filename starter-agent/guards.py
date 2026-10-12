import json


def signature(call):
    return call["name"] + ":" + json.dumps(call["args"], sort_keys=True)


def is_repeating(signatures, limit=3):
    if not signatures:
        return False
    return signatures.count(signatures[-1]) >= limit


def clip(text, limit=2000):
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more characters cut]"
