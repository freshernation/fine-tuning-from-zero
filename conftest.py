"""Test harness for the course. Students never need to read this file.

Two jobs:

  1. Make the course's own `tunelab` package importable from every test, whether
     or not you ran `pip install -e .`.
  2. Give tests `load("shapes.py")`: import a student file that sits next to the
     test file, so two days can both have a file with the same name.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def load(request):
    folder = Path(request.fspath).parent

    def _load(filename):
        path = folder / filename
        if not path.exists():
            pytest.fail(f"{filename} does not exist in {folder.name}/.")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load
