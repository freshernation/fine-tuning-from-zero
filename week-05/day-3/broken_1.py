"""Broken file 1 of 4.

Somebody's first fine-tuning run. "The loss started at 5.6, jumped to about 50 on
the second step and never came back down. I think the model is too small?"

It is not the model. Find the bug, fix it, and write in NOTES.md how you found it.
"""

import torch

from tunelab.tiny import tiny_model, tiny_tokenizer

LEARNING_RATE = 2e0       # "the default in the course is 2e-something, I typed it from memory"


def losses_over_20_steps():
    """Train the toy model for 20 steps on one fixed batch. Returns the 20 losses."""
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    model = tiny_model(len(tokenizer))
    ids = torch.randint(0, 256, (4, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    for _ in range(20):
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses
