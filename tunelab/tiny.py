"""A toy tokenizer and a toy model, for testing real pipelines in seconds on a CPU.

Given, not an exercise. Neither knows anything: the model's weights are random and
the tokenizer's "words" are single bytes. What they have is the right SHAPE - the
course base model's chat template, special tokens and architecture - so code that
works on them works on the real thing.
"""

from pathlib import Path

TEMPLATE = Path(__file__).parent.parent / "data" / "notes" / "chat_template.jinja"


def tiny_tokenizer():
    """One token per byte, so any text survives encode -> decode exactly. Carries the
    base model's real chat template and its three special tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    inner = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    inner.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=inner, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = TEMPLATE.read_text()
    tokenizer.model_input_names = ["input_ids", "attention_mask"]     # as the real tokenizer does
    return tokenizer


def tiny_model(vocab_size, seed=0):
    """A two-layer model with the base model's architecture and random weights."""
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(LlamaConfig(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=8192,
    ))
