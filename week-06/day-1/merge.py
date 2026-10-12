"""Day 1 - from an adapter and a base model to one ordinary model in a folder."""

from peft import PeftModel

# Roughly how many bits each weight takes in a GGUF file, by quantization level.
# The "k" levels mix precisions, which is why they are not whole numbers. Approximate
# figures from llama.cpp's own quantize tool; small models come out a little higher,
# because a few tensors are always kept at higher precision.
QUANT_BITS = {
    "f16": 16,
    "q8_0": 8.5,
    "q5_k_m": 5.7,
    "q4_k_m": 4.85,
}


def merge_adapter(base_model, adapter_dir):
    """The base model with the adapter folded into its weights, as a plain model.

    This is week 3's merge_lora, done by the library: attach the saved adapter with
    PeftModel.from_pretrained(base_model, adapter_dir), then .merge_and_unload().
    What comes back must be the ordinary model class, with no adapter layers left.
    """
    raise NotImplementedError


def save_merged(model, tokenizer, out_dir):
    """Write the model AND the tokenizer to out_dir with save_pretrained, so the folder
    is everything another tool needs: weights, vocabulary and the chat template."""
    raise NotImplementedError


def gguf_bytes(n_params, bits_per_weight):
    """About how big a file n_params weights make at this many bits each. A whole number."""
    raise NotImplementedError
