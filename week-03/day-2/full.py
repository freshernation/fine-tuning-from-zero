"""Day 2 - fine-tune the obvious way, and measure what it does."""

from tunelab.loop import estimate_loss, train


def fine_tune_and_measure(model, get_batch_new, get_batch_old, steps, lr):
    """Fine-tune `model` on the new task and report what happened to both tasks.

    get_batch_new   batches of the thing you want the model to learn
    get_batch_old   batches of something it could already do

    Train on the NEW task only, with train() from week 1, unchanged.
    Measure the validation loss of BOTH tasks, before training and again after.

    Returns four floats:
        {"new_before": ..., "new_after": ..., "old_before": ..., "old_after": ...}
    """
    raise NotImplementedError
