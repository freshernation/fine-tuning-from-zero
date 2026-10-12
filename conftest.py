"""Test harness for the course. Students never need to read this file.

Two jobs:

  1. Make the course's own `tunelab` package importable from every test, whether
     or not you ran `pip install -e .`.
  2. Give tests `load("shapes.py")`: import a student file that sits next to the
     test file, so two days can both have a file with the same name.
  3. From week 3, hand tests a tiny pretrained-style model without downloading one.
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


@pytest.fixture
def tiny_llama():
    """A real Hugging Face model with the same architecture as the course's pretrained
    one, but tiny and with random weights. Built from a config: nothing is downloaded."""
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    ))


@pytest.fixture
def chat_tokenizer():
    """A tokenizer with the course base model's REAL chat template and a toy vocabulary
    of single bytes. Rendering a chat to text only needs the template, so tests can
    check prompts exactly without downloading the model."""
    from tunelab.tiny import tiny_tokenizer

    return tiny_tokenizer()
