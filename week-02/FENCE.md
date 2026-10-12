# Week 2 — Concept fence

Every AI role in `ai/` reads this. Paste the two lists into the role prompt before you
start a session, so the model teaches you with the toolkit you actually have.

## Allowed

- Everything in `week-01/FENCE.md`
- Embeddings as learned vectors; `n_embd` as the width of the model
- Position embeddings; `block_size` as the longest sequence
- Matrix multiplication (`@`) as weighted averaging; `torch.tril`
- `.transpose(-2, -1)`, `masked_fill`, softmax along a chosen `dim`
- Query, key and value; attention scores and weights
- Scaling scores by the square root of the head size
- The causal mask
- `register_buffer`
- Multi-head attention; `nn.ModuleList`; `torch.cat` along the last dimension
- The MLP; `nn.Sequential`; `nn.GELU`
- Residual connections
- `nn.LayerNorm`, and pre-norm placement
- `nn.Dropout`, and why `model.eval()` now changes the answer
- Weight initialisation: `model.apply`, `nn.init.normal_`, `nn.init.zeros_`
- Counting parameters with `.numel()`
- Generating with a sliding window of `block_size`
- `.to(device)`; creating tensors on `idx.device`
- `state_dict`, `torch.save`, `torch.load`, `load_state_dict`
- `os.replace` for writing a file safely
- Training in chunks with a checkpoint between them
- Colab with a GPU; Google Drive mounted in Colab

## Not yet

Pretrained models · the Hugging Face libraries (`transformers`, `datasets`, `peft`,
`trl`) · `Trainer` or any library that runs the loop for you · subword tokenizers
(BPE) · fine-tuning · freezing layers · LoRA · quantization · learning-rate schedules
and warmup · gradient clipping · mixed precision · `torch.compile` ·
`F.scaled_dot_product_attention` and flash attention · `nn.MultiheadAttention` and
`nn.TransformerEncoderLayer` (you are writing these, not importing them) · weight
tying · KV caching · temperature, top-k and top-p sampling · encoder models and
cross-attention · evals beyond a validation loss
