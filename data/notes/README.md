# data/notes

Everything for the course's one real task: teaching a small model to be the
tool-calling brain of the notes agent.

| File | What it is | Who makes it |
|---|---|---|
| `chat_template.jinja` | The chat template of the course's base model, `Qwen/Qwen2.5-0.5B-Instruct`, copied unchanged from that model's `tokenizer_config.json` (Apache-2.0, © Alibaba Cloud). It is here so tests can render prompts without downloading the model. | given |
| `dev.jsonl` | Twenty hand-written cases for trying out your scorer and your prompt. | given |
| `seeds.jsonl`, `raw.jsonl`, `clean.jsonl` | Your examples, at three stages. | you, week 4 |
| `train.jsonl`, `eval.jsonl`, `eval.sha256` | Your dataset, split, with the eval set frozen. | you, week 4 |
| `baseline_predictions.jsonl`, `baseline.json` | What the untuned model did on your eval set. | you, week 4 |
