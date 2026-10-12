# Week 5 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session.

## Allowed

- Everything in the fences for weeks 1 to 4
- Quantization: levels, scale, codes, error; bits per weight
- 4-bit loading with `BitsAndBytesConfig`; QLoRA as "frozen 4-bit base, full-precision adapter"
- `peft`: `LoraConfig`, `get_peft_model`, `PeftModel.from_pretrained`, `save_pretrained`
- `transformers`: `Trainer`, `TrainingArguments`, a `Dataset` of tokenized rows, a
  custom collator
- Supervised fine-tuning on prompt/completion pairs; label masking with `-100`
- Padding on the right; attention masks
- Batch size against gradient accumulation; effective batch size; steps per epoch
- Learning rate, epochs, rank, max length as the four knobs
- The learning-rate schedule `Trainer` uses by default, as a fact
- Checkpoints and resuming, through `Trainer`
- Reading a run: learning, overfitting, not learning, diverging; the best checkpoint
- Eval loss against eval score, and why they can disagree
- Error analysis: failure types, by kind; what was fixed and what was broken
- The paired bootstrap and a 95% interval
- Fixing the training data in response to errors; never the eval set

## Not yet

`trl` / `SFTTrainer` · hyperparameter sweeps · custom learning-rate schedules · DoRA,
rsLoRA and other variants · adapters on every layer as a default · preference tuning
(DPO), RLHF · distillation as a named technique · merging adapters into the base ·
GGUF and llama.cpp · serving · constrained decoding or JSON mode · model-graded evals
· multi-GPU, `accelerate` configuration
