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


class FakeServer:
    """A tiny stand-in for a model server that speaks the OpenAI chat protocol.

        server.url                 "http://127.0.0.1:<port>/v1"
        server.replies[text] = m   the assistant message to send back when the LAST
                                   user message in a request has this content
        server.default             the message sent back otherwise
        server.requests            every chat request body received, as dicts
        server.not_ready = 2       answer 503 to that many requests first
    """

    def __init__(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.replies, self.requests, self.not_ready = {}, [], 0
        self.default = {"role": "assistant", "content": "OK."}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _busy(self):
                if outer.not_ready > 0:
                    outer.not_ready -= 1
                    self._send(503, {"error": {"message": "Loading model", "type": "unavailable_error"}})
                    return True
                return False

            def do_GET(self):
                if not self._busy():
                    self._send(200, {"object": "list", "data": [{"id": "notes", "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self._busy():
                    return
                outer.requests.append(body)
                last_user = next((m.get("content") for m in reversed(body["messages"]) if m["role"] == "user"), None)
                message = outer.replies.get(last_user, outer.default)
                self._send(200, {
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body.get("model", "notes"),
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                })

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._http.server_address[1]}/v1"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def close(self):
        self._http.shutdown()
        self._http.server_close()


@pytest.fixture
def fake_server():
    server = FakeServer()
    yield server
    server.close()
