# Week 4 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session, so the model teaches you with the toolkit you actually have.

From this week the AI may write code with you. The fence still applies: it should not
reach for anything in "not yet".

## Allowed

- Everything in the fences for weeks 1 to 3
- JSON Lines files; `json.dumps(..., ensure_ascii=False, sort_keys=True)`
- The example format and the prediction format in `tunelab/toolcall.py`
- Parsing a model's raw output; treating unreadable output as invalid, not as silence
- Scoring: valid, right tool, right arguments, correct; rates by kind
- A prompted baseline; few-shot examples in a system prompt
- Seeds; expansion by templates, by a model through an API, by a model by hand
- Validating generated data and discarding what fails
- Normalising text; duplicates; conflicts; class balance
- Splitting by group (by seed); leakage
- `hashlib.sha256`; freezing an eval set
- `apply_chat_template` with `tools=` and `add_generation_prompt=`
- Prompt and completion; the end-of-turn marker
- `model.generate`, through `tunelab/generate.py` only
- `random.Random(seed)` for reproducible choices
- The starter agent's `llm` layer, for expansion, if you have a key

## Not yet

`Trainer`, `TrainingArguments`, `trl` · `peft` for training · `datasets` (the library)
· quantization, 4-bit loading, QLoRA · masking the loss to the assistant's reply ·
gradient accumulation · learning-rate schedules · generation settings beyond greedy
(temperature, top-p, constrained decoding, JSON mode) · model-graded evals ("LLM as
judge") · statistical tests on eval scores · embeddings for near-duplicate detection ·
merging, GGUF, serving
