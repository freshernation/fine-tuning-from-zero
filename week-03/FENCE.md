# Week 3 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session, so the model teaches you with the toolkit you actually have.

## Allowed

- Everything in `week-01/FENCE.md` and `week-02/FENCE.md`
- "Pretrained": weights from someone else's long training run
- `AutoModelForCausalLM` and `AutoTokenizer`: `.from_pretrained` and `.from_config`,
  for **loading only**
- Calling a Hugging Face model with `input_ids=` and reading `.logits`
- The parts of a Llama-style model by name: `embed_tokens`, `layers`, `q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `mlp`, `lm_head`
- Subword tokenizers, used as given
- Chat templates; ChatML markers; the generation prompt
- Tied input and output embeddings, as a fact
- `named_parameters`, `named_modules`, `named_children`, `setattr` on a module
- `requires_grad_`, freezing and unfreezing by name
- `nn.Parameter`
- Counting trainable parameters; the memory cost of training with AdamW
- Catastrophic forgetting, and measuring it on held-out text of the old kind
- A smaller learning rate for fine-tuning
- Rank; a low-rank matrix as a product of two thin ones; `torch.linalg.matrix_rank`
- LoRA: `A`, `B`, rank, alpha, the scaling, zero-initialising `B`, dropout on the
  adapter path
- Which layers to adapt; injecting adapters into a model
- Saving and loading an adapter by itself; `.detach().clone()`
- Merging an adapter into the base weights
- `torch.cuda.max_memory_allocated`
- `model.generate`, in the notebook's given cells only
- `peft`, in `week-03/milestone/test_parity.py` only, as the answer key

## Not yet

`peft` for doing the work (`get_peft_model` in your own code) · `Trainer`,
`TrainingArguments`, `SFTTrainer`, `trl` · `datasets` · quantization, 4-bit loading,
`bitsandbytes`, QLoRA · DoRA, rsLoRA and other LoRA variants · adapters on the MLP
layers as a default · learning-rate schedules and warmup · gradient accumulation ·
gradient clipping · mixed precision · training on chat-formatted examples · masking
the loss to the assistant's reply · building a dataset · evals beyond a loss ·
generation settings (temperature, top-p) · merging into GGUF · serving
