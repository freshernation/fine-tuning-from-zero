"""The notes agent's five tools, as the model is told about them.

Given, not an exercise. The descriptions are read straight from the starter
agent's kit.py, so the model you train is told exactly what the agent will tell it.
"""

import importlib.util
from pathlib import Path

_KIT = Path(__file__).parent.parent / "starter-agent" / "kit.py"


def _load_kit():
    spec = importlib.util.spec_from_file_location("starter_agent_kit", _KIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The agent's own schemas: [{"name", "description", "parameters"}, ...]
AGENT_SCHEMAS = _load_kit().SCHEMAS

# The same five, wrapped the way chat templates and OpenAI-compatible servers expect.
SCHEMAS = [{"type": "function", "function": schema} for schema in AGENT_SCHEMAS]

TOOL_NAMES = [schema["name"] for schema in AGENT_SCHEMAS]

# name -> (required argument names, all argument names)
ARGUMENTS = {
    schema["name"]: (
        list(schema["parameters"]["required"]),
        list(schema["parameters"]["properties"]),
    )
    for schema in AGENT_SCHEMAS
}
