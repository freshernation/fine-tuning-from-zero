"""Day 1 - peft's LoRA, read against yours.

From today peft is a tool, not just an answer key. This file is where you convince
yourself that using it changes nothing you built in week 3.
"""

from peft import LoraConfig, get_peft_model

from tunelab.finetune import count_trainable
from tunelab.lora import LoRALinear, inject_lora


def to_peft_config(rank, alpha, targets, dropout=0.0):
    """The LoraConfig that means what your inject_lora(model, targets, rank, alpha, dropout) means.

    Also set task_type="CAUSAL_LM": it tells peft what kind of model it is wrapping.
    """
    raise NotImplementedError


def same_as_mine(model_factory, rank, alpha, targets):
    """Put adapters into two fresh copies of a model - one with your inject_lora, one
    with peft - and report what each did.

    model_factory   call it to get a fresh model. Call it once per side.

    Returns {"mine": {...}, "peft": {...}}, each with
        "trainable"   how many parameters can train
        "layers"      sorted names of the layers that were wrapped, as the ORIGINAL
                      model names them, e.g. "model.layers.0.self_attn.q_proj"

    peft wraps the whole model, so its names gain a prefix ("base_model.model.").
    A wrapped layer on your side is a LoRALinear; on peft's side it is any module
    that has a `lora_A` attribute.
    """
    raise NotImplementedError
