# Week 6 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session.

## Allowed

- Everything in the fences for weeks 1 to 5
- Merging an adapter with `PeftModel.from_pretrained(...).merge_and_unload()`
- `save_pretrained` for a model and its tokenizer together
- GGUF as a single-file model format that carries weights, tokenizer and chat template
- llama.cpp's `convert_hf_to_gguf.py`, `llama-quantize` and `llama-server`
- Quantization levels by name (`f16`, `q8_0`, `q5_k_m`, `q4_k_m`) and bits per weight
- The OpenAI chat-completions request and reply shapes, including `tools` and `tool_calls`
- The `openai` Python package pointed at a local `base_url`
- Waiting for a server; timeouts; clear errors
- Template problems: tool calls arriving as text
- The agent's `.env` configuration: provider, base URL, key, model name
- Per-step accuracy compounding over a multi-step task
- Latency: warm-up, median, 95th percentile
- Token pricing per million; fixed against per-request cost; break-even
- Ollama, as an alternative you may choose, with its own template

## Not yet (and not in this course)

Containers and orchestration · GPU serving, batching and throughput tuning · vLLM, TGI
and other production servers · autoscaling · monitoring and drift detection in
production · A/B tests · guardrail services · caching layers · multi-adapter serving ·
further quantization research (imatrix, QAT) · everything on `CUT_LIST.md`
