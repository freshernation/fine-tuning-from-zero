# Day 2 — The obvious way

> **By the end of today** you have fine-tuned a model the obvious way, and measured the
> two things wrong with it.

Fine-tuning is not a new technique. **It is training, started from someone else's
weights instead of random ones.** Same loop, same loss, same five lines. You could
have done it on the first Friday of the course if you had had a model worth starting
from.

So today's code is short. The lesson is in the two numbers you measure around it.

`fine_tune_and_measure` goes in `full.py`, in this folder. `count_trainable` and
`training_memory_bytes` go in `tunelab/finetune.py`.

---

## Read first

- [ ] Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — again, as promised in week 1. This time read step 2 onward · 20 min

---

## What you need to know

### Same loop, two differences

**The starting point.** Not random numbers: weights that already know a great deal.
Your job is to nudge them, not to build them.

**A smaller learning rate.** In week 2 you trained from scratch at `1e-3`. Fine-tuning
a pretrained model is usually done ten to a hundred times more gently. Think about
what a step does: it moves every weight a little in the direction that helps *this
batch*. Random weights have nothing to lose. Pretrained weights are carrying
everything the model knows, and a large step toward your handful of examples is a
large step away from all of that.

Which brings you to the first thing wrong with the obvious way.

### Problem one: it forgets

You are training on your data only. The loss you are minimising says nothing about
everything else the model could do. The weights that held that knowledge are **the
same weights you are now overwriting**, and nothing in the loop protects them.

This is called catastrophic forgetting, and you do not have to take it on trust,
because it is easy to measure:

1. Keep some text of the *old* kind — something the model was already good at.
2. Measure the loss on it before you fine-tune.
3. Fine-tune on the new thing.
4. Measure the old loss again.

Never train on the old text. It is a thermometer.

Today's tests do this in miniature. Your own GPT is pretrained, inside the test, on a
toy language where each token is followed by the next one up: `3 4 5 6`. Then you
fine-tune it on a language where each is followed by the one *two* up: `3 5 7 9`. You
will watch it become perfect at the second and helpless at the first, in a hundred
steps.

A real model on real text forgets less dramatically than that. It still forgets.

### Problem two: the bill

Count what has to be in memory to train a model with AdamW.

| | Numbers |
|---|---|
| The weights | 1 × the model |
| A gradient for every weight you train | 1 × |
| AdamW's running average of the gradient | 1 × |
| AdamW's running average of its square | 1 × |

Four copies of the model, each number a 4-byte float, before the model has looked at
a single token. Then the activations — every intermediate result, kept so that
backward can use it — which grow with batch size and sequence length.

For this week's model:

```
135 million × 4 bytes × 4 copies  ≈  2.2 GB
```

Fine. A free Colab GPU has about 15 GB.

Now a model you would actually want to deploy, at 7 billion parameters:

```
7 billion × 4 bytes × 4 copies  ≈  112 GB
```

No free GPU, and not many paid ones. And at the end you would have a complete
7-billion-parameter copy of the model to store, for every task you fine-tuned it on.

Look at the table again, though. **Three of the four rows are per weight you train.**
A weight you do not train still has to be stored, but it needs no gradient and nothing
from AdamW. If you could get away with training very few of them, three quarters of
the bill would nearly vanish.

Hold that thought. It is the whole of tomorrow.

### Freezing, in one line

A parameter is trained if its `requires_grad` is `True`, and every parameter in an
`nn.Module` starts that way. You can switch it off:

```python
layer = nn.Linear(4, 4)
layer.requires_grad_(False)                   # for a whole module
layer.weight.requires_grad                    # False
```

A frozen parameter gets no gradient, and the optimizer leaves it alone. Today you only
need this to count; tomorrow you use it.

---

## Exercises

1. `count_trainable(model)` in `tunelab/finetune.py`.
2. `training_memory_bytes(model)` in `tunelab/finetune.py` — the table above, as
   arithmetic.
3. `fine_tune_and_measure(...)` in `full.py`. The loop is `train()`, from week 1, as
   it is.

```bash
pytest week-03/day-2 -v
```

These take a little longer than usual: several of them pretrain a small GPT first.

---

## Then, in a shell

When they pass, try the forgetting experiment at a gentler learning rate. In
`test_day2.py` the fine-tuning runs at `lr=1e-2`. Copy the setup into a Python shell
and run it at `1e-3` and at `1e-4`, for the same hundred steps. Write three rows in
`logs/run-log.md`: the learning rate, the new-task loss afterwards, the old-task loss
afterwards.

You will find you can forget less — by learning less. That trade does not go away by
choosing the right number. It is why tomorrow's idea exists.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.
