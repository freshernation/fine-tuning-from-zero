"""Week 6, Day 1 - merge, convert, quantize.

Run me:  pytest week-06/day-1 -v

merge.py is in this folder. The tests use the toy model and tokenizer; the real
conversion to GGUF happens with llama.cpp's own tools, in COMMANDS.md.
"""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tunelab.tiny import tiny_model, tiny_tokenizer


@pytest.fixture
def trained(tmp_path):
    """A toy base model, and an adapter for it that has been trained a little and saved."""
    tokenizer = tiny_tokenizer()
    base = tiny_model(len(tokenizer))
    tuned = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    ids = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.AdamW([p for p in tuned.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(10):
        tuned(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tuned.save_pretrained(str(tmp_path / "adapter"))
    tuned.eval()
    return base, tuned, tokenizer, ids, str(tmp_path / "adapter")


def count(model):
    return sum(p.numel() for p in model.parameters())


def test_merged_model_gives_the_tuned_answers(load, trained):
    base, tuned, _, ids, adapter = trained
    merged = load("merge.py").merge_adapter(copy.deepcopy(base), adapter)
    merged.eval()
    assert torch.allclose(merged(input_ids=ids).logits, tuned(input_ids=ids).logits, atol=1e-5)
    assert not torch.allclose(merged(input_ids=ids).logits, base(input_ids=ids).logits, atol=1e-3), "these should be the TUNED answers"


def test_merged_model_is_a_plain_model_again(load, trained):
    base, _, _, _, adapter = trained
    merged = load("merge.py").merge_adapter(copy.deepcopy(base), adapter)
    assert type(merged) is type(base), "not a PeftModel: the ordinary model class"
    assert count(merged) == count(base)
    assert set(merged.state_dict()) == set(base.state_dict()), "no lora_A, no lora_B, no adapter names left"


def test_saved_folder_is_a_complete_model(load, trained, tmp_path):
    base, tuned, tokenizer, ids, adapter = trained
    merge = load("merge.py")
    out = str(tmp_path / "merged")
    merge.save_merged(merge.merge_adapter(copy.deepcopy(base), adapter), tokenizer, out)

    again = AutoModelForCausalLM.from_pretrained(out)
    again.eval()
    assert torch.allclose(again(input_ids=ids).logits, tuned(input_ids=ids).logits, atol=1e-5)
    reloaded = AutoTokenizer.from_pretrained(out)
    assert reloaded.chat_template == tokenizer.chat_template, "the chat template has to travel with the model"
    assert reloaded("save milk").input_ids == tokenizer("save milk").input_ids


def test_gguf_size_arithmetic(load):
    merge = load("merge.py")
    assert set(merge.QUANT_BITS) >= {"f16", "q8_0", "q5_k_m", "q4_k_m"}
    assert merge.QUANT_BITS["f16"] == 16
    assert merge.QUANT_BITS["q8_0"] > merge.QUANT_BITS["q5_k_m"] > merge.QUANT_BITS["q4_k_m"] > 4
    n = 494_032_768
    assert merge.gguf_bytes(n, 16) == 988_065_536
    assert merge.gguf_bytes(n, merge.QUANT_BITS["q4_k_m"]) < 0.35 * merge.gguf_bytes(n, 16)
    assert isinstance(merge.gguf_bytes(n, 4.85), int)
