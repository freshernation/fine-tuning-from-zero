# Week 1 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session, so the model teaches you with the toolkit you actually have.

## Allowed

- All the Python you arrived with: functions, classes, lists, dicts, files, pytest, git
- Tensors: creating them, `.shape`, `.dtype`, `int64` versus `float32`
- Indexing and slicing, including `x[:, -1, :]`
- `sum`, `mean` and friends with `dim=` and `keepdim=`
- Broadcasting, and its rule (line up from the right, stretch size 1)
- `view` / `reshape`, `torch.stack`, `torch.cat`, `torch.arange`, `torch.zeros`,
  `torch.randint`
- A loss as one number; squared error
- A gradient as a slope; the chain rule for one weight
- `requires_grad`, `.backward()`, `.grad`, `.item()`, zeroing gradients
- `torch.no_grad()`
- Gradient descent and the learning rate
- `nn.Module`, `forward`, `.parameters()`, `nn.Linear`, `nn.Embedding`
- `torch.optim.AdamW` as "the update step, done for you"
- The training loop: forward, loss, backward, step, zero
- Train and validation splits; `model.eval()` and `model.train()`
- Reading a loss curve: learning, overfitting, not learning
- A character-level tokenizer; vocabulary size
- Batches as `(B, T)`, with targets shifted by one
- Logits, softmax, cross-entropy, and the `ln(vocab_size)` starting loss
- A bigram language model; sampling with `torch.multinomial`
- `torch.manual_seed`
- Colab: opening a notebook, choosing a GPU runtime, cloning a repo

## Not yet

Attention · transformers · positional embeddings · layer norm · residual connections ·
MLP blocks · dropout · learning-rate schedules · gradient clipping · weight decay as a
topic · moving tensors to a GPU (`.to(device)`, `.cuda()`) · mixed precision ·
checkpoints · subword tokenizers (BPE) · pretrained models · the Hugging Face
libraries (`transformers`, `datasets`, `peft`, `trl`) · `Trainer` or any library that
runs the loop for you · fine-tuning · LoRA · quantization · evals beyond a validation
loss
