"""Day 1 - from an adapter and a base model to one ordinary model in a folder."""

from peft import PeftModel

# How many bits each weight takes in a GGUF file of THIS course's base model, by
# quantization level: measured by converting it (file size x 8 / 494 million weights).
# The "k" levels mix precisions, which is why they are not whole numbers. This model
# comes out well above q4_k_m's nominal 4.8: its layers are 896 wide, the 4-bit k
# format needs a width divisible by 256, and about half the tensors fall back to 5 bits.
QUANT_BITS = {
    "f16": 16,
    "q8_0": 8.6,
    "q5_k_m": 6.8,
    "q4_k_m": 6.4,
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
