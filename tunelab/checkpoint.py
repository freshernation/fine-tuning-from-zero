"""Saving a training run, so that a dead Colab session costs minutes, not hours.

Written on Week 2 Day 4. Uses train() from tunelab/loop.py exactly as it is.
"""

import os

import torch

from tunelab.loop import train


def save(path, model, step, history):
    """Write the model's weights, the step reached and the history to `path`.

    A session can die at any moment - including halfway through this function. So
    never write straight onto `path`: write to path + ".tmp", and only when that
    has finished, os.replace() it onto `path`. A crash then costs you the new
    checkpoint, never the old one.
    """
    raise NotImplementedError


def load(path, model):
    """Put the saved weights back into `model`. Return (step, history).

    If there is no file at `path`, change nothing and return (0, []).
    """
    raise NotImplementedError


def train_in_chunks(model, get_batch, total_steps, chunk, lr, path, eval_every=100):
    """Train to `total_steps`, saving every `chunk` steps, resuming if there is a checkpoint.

    1. load(path, model). If it says some steps are already done, carry on from there.
    2. Until total_steps are done: call train() for one chunk (the last one may be
       shorter), add its history to the history so far, save.
    3. Return the whole history, including what was loaded.

    train() numbers its steps from 0 every time it is called. Yours must not: a
    measurement taken at step 10 of the third chunk of 20 is step 50.

    One thing this costs: train() builds a new AdamW each time, so the optimizer's
    memory of recent gradients restarts at every chunk. At this scale you will not
    see it. The library you use in week 5 saves the optimizer too.
    """
    raise NotImplementedError
