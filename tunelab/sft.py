"""Turning examples into what a training library eats. Written on Week 5 Day 2.

"SFT" is supervised fine-tuning: show the model a prompt and the answer you wanted,
and train it to produce that answer.
"""

import math

import torch

from tunelab.dataset import render_completion, render_prompt


def tokenize_example(tokenizer, example, system, max_len):
    """One example, as token ids, with the loss switched off everywhere but the answer.

    Build the text with render_prompt and render_completion - the SAME two functions
    that built the baseline's prompts. Tokenize the two pieces separately, with
    add_special_tokens=False, and join the ids: that is exactly what happens at
    inference, where the prompt is tokenized alone and the model writes the rest.

    Returns {"input_ids": [...], "attention_mask": [1, 1, ...], "labels": [...]},
    all plain lists of the same length. labels is -100 for every prompt token and
    the token id itself for every answer token. (-100 is the value the loss
    function is told to ignore.)

    If the whole thing is longer than max_len tokens, return None. Never truncate:
    cutting the end cuts the answer, and cutting the start cuts the instructions.
    """
    raise NotImplementedError


def collate(rows, pad_id):
    """A batch of tokenized examples, as three int64 tensors of shape (B, longest).

    Shorter rows are padded on the RIGHT: input_ids with pad_id, attention_mask
    with 0, labels with -100.
    """
    raise NotImplementedError


def effective_batch_size(batch_size, accumulation):
    """How many examples contribute to one optimizer step."""
    raise NotImplementedError


def steps_per_epoch(n_examples, batch_size, accumulation):
    """How many optimizer steps one pass over the data takes. The last batch may be
    short, and so may the last group of batches; both still count."""
    raise NotImplementedError
