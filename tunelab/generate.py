"""Getting text out of a real model. Given, not an exercise.

    generate = make_generate(model, tokenizer, SCHEMAS, system="You manage notes...")
    generate([{"role": "user", "content": "remember milk"}])
    # '<tool_call>\n{"name": "add_note", "arguments": {"title": "milk"}}\n</tool_call>'

It returns the raw text the model wrote, and nothing else: no prompt, no end marker.
Turning that text into a prediction is your job (tunelab/toolcall.py).
"""

import torch


def make_generate(model, tokenizer, schemas, system, max_new_tokens=128):
    """A function from a list of chat messages to the model's raw reply.

    model       a Hugging Face causal language model (not the week 3 wrapper)
    Greedy decoding: the same input always gives the same output, so a score
    measured twice is the same score.
    """

    @torch.no_grad()
    def generate(messages):
        chat = [{"role": "system", "content": system}] + list(messages)
        prompt = tokenizer.apply_chat_template(
            chat, tools=schemas, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate
