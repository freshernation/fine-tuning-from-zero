"""The agent's files import each other by plain name (`from kit import ...`), the way
they did in Agent From Zero. This puts this folder on the path so that works here too."""

import sys
from pathlib import Path

HERE = str(Path(__file__).parent.resolve())
if HERE not in sys.path:
    sys.path.insert(0, HERE)
