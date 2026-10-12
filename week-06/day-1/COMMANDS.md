# From adapter to a running server — the commands

Run these from the top of your repo. Each one is explained below it.

> llama.cpp renames its tools from time to time. If a command here does not exist in
> the version you installed, run it with `--help`, or check the llama.cpp README: the
> steps do not change, only their names.

---

## 0. Install llama.cpp

| | |
|---|---|
| macOS | `brew install llama.cpp` |
| Windows | `winget install llama.cpp` |
| Linux | a prebuilt release from <https://github.com/ggml-org/llama.cpp/releases>, unpacked somewhere on your `PATH` |

You also need the repository itself, for one Python script:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
```

```bash
pip install -r llama.cpp/requirements.txt
```

Check:

```bash
llama-server --version
```

## 1. Merge

Your adapter folder (the `adapter/` inside your week-5 run, downloaded from Drive)
into a full-precision copy of the base model. In a Python shell, using your own
`week-06/day-1/merge.py`:

```python
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer

spec = importlib.util.spec_from_file_location("merge", "week-06/day-1/merge.py")
merge = importlib.util.module_from_spec(spec); spec.loader.exec_module(merge)

MODEL, REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION)      # full precision, NOT 4-bit
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)

merged = merge.merge_adapter(base, "runs/run-01/adapter")
merge.save_merged(merged, tokenizer, "runs/merged")
```

`runs/merged/` is now a complete, ordinary Hugging Face model: about 2 GB.

## 2. Convert to GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py runs/merged --outfile runs/notes-f16.gguf --outtype f16
```

- `runs/merged` — the folder from step 1.
- `--outtype f16` — no quantization yet: 16-bit floats. About 1 GB.

The script reads the weights, the vocabulary and the chat template, and writes all
three into one file.

## 3. Quantize

```bash
llama-quantize runs/notes-f16.gguf runs/notes-q4_k_m.gguf Q4_K_M
```

Input file, output file, level. About 300 to 400 MB. To make another level, run it
again with `Q5_K_M` or `Q8_0`, always starting from the f16 file.

## 4. Serve

```bash
llama-server -m runs/notes-q4_k_m.gguf --port 8080 --alias notes --jinja -c 4096
```

- `-m` — the model file.
- `--port 8080` — the address is then `http://127.0.0.1:8080/v1`.
- `--alias notes` — the name clients use for the model. The agent's `.env` says `notes`.
- `--jinja` — **use the chat template stored in the file**, including its handling of
  tools. Without this, tool calls can come back as plain text. If your version says
  the flag is unknown, it is on by default.
- `-c 4096` — context length, in tokens. Your prompts are about nine hundred.

Leave it running in its own terminal. Check from another:

```bash
curl http://127.0.0.1:8080/v1/models
```

`runs/` is not in git, and should not be: add it to your `.gitignore` if it is not
already there.
