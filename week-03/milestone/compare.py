"""Week 3 milestone, part 2 - three ways to fine-tune, measured the same way.

Everything you need is already written, by you:
    freeze, unfreeze, inject_lora         tunelab/lora.py
    count_trainable, training_memory_bytes  tunelab/finetune.py
    train, estimate_loss                  tunelab/loop.py     (week 1, unchanged)
"""

from tunelab.finetune import count_trainable, training_memory_bytes
from tunelab.loop import estimate_loss, train
from tunelab.lora import freeze, inject_lora, unfreeze

METHODS = ("full", "last_block", "lora")


def run(method, model, get_batch_new, get_batch_old, steps, lr,
        rank=8, alpha=16, targets=("q_proj", "v_proj"), last_block="layers.1."):
    """Fine-tune `model` on the new task by one of three methods, and report the cost.

    method   "full"         every parameter trains
             "last_block"   freeze everything, then unfreeze parameters whose name
                            contains `last_block`
             "lora"         inject_lora(model, targets, rank, alpha)
             anything else is a ValueError

    model    follows the course convention - model(idx, targets) -> (logits, loss)

    Set the model up for the method, then count and measure, then train on the new
    task only, then measure again. Returns a dict:

        "trainable", "total"      from count_trainable, after setting up
        "memory_bytes"            from training_memory_bytes, after setting up
        "new_before", "new_after"   validation loss on the new task
        "old_before", "old_after"   validation loss on the old task
    """
    raise NotImplementedError
