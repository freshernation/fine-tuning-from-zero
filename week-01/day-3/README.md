# Day 3 — The loop

> **By the end of today** you have written the five-line loop every model in this course
> is trained with.

Yesterday you took steps downhill by hand. Today you put that in a loop, let an
optimizer do the update, and — the part people skip — **measure honestly whether it is
working.**

Today's code does not live in this folder. It goes in `tunelab/loop.py`, because you
will use it again in Week 2 to train a transformer and in Week 3 to fine-tune one.
Unchanged. That is the point of this week: the loop that trains a billion-parameter
model is the loop you write today.

---

## Read first

- [ ] [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — the whole page · 20 min
- [ ] [Build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — the first half, up to "Model layers" · 10 min

---

## What you need to know

### A model is an `nn.Module`

```python
from torch import nn

class Line(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)        # holds a weight and a bias

    def forward(self, x, targets=None):
        out = self.lin(x)
        loss = None if targets is None else ((out - targets) ** 2).mean()
        return out, loss
```

`model(x, y)` calls `forward`. `model.parameters()` hands back every tensor inside it
that has `requires_grad=True` — yesterday's `w` and `b`, found for you.

**The convention for this whole course:** a model takes `(x, targets)` and returns
`(output, loss)`. If you give it no targets, the loss is `None`. Every model you write
until Week 4 follows it, so one loop can train all of them.

### The optimizer

Yesterday:

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
w.grad.zero_()
b.grad.zero_()
```

Today, for any number of parameters:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

optimizer.step()         # the update
optimizer.zero_grad()    # the zeroing
```

`AdamW` is a smarter version of the step you wrote: it keeps a running memory of
recent gradients and scales each parameter's step separately. You do not need its
formula this week. You need to know it does the same job as your two lines.

### The five lines

```
        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
   1. forward     out, loss = model(x, y)              │
   2. loss        (it came back from the model)        │
   3. backward    loss.backward()                      │
   4. step        optimizer.step()                     │
   5. zero        optimizer.zero_grad()  ──────────────┘
```

Say them in order until you can do it without looking: **forward, loss, backward,
step, zero.** On Friday you will be asked to.

### Why you hold data back

A model can get the training loss down by memorising the training data. That is
useless: you wanted it to learn the pattern, not the examples.

So you split the data in two before you start. The model trains on one part and
**never sees** the other. The loss on the part it never saw — the **validation loss** —
is the only number that tells you whether it learned anything that transfers.

### Measuring without disturbing

Measuring the loss is not training. Three things make sure it leaves no trace:

```python
model.eval()                  # some layers behave differently when training; turn that off
with torch.no_grad():         # do not build the machinery for gradients
    _, loss = model(x, y)
model.train()                 # put it back
```

Forget `model.train()` and the next thousand steps train a model that thinks it is
being tested. Nothing crashes. It just learns worse, and you will not know why.

One batch is a noisy measurement, so `estimate_loss` averages over several.

### Reading a loss curve

You will look at thousands of these. There are three shapes worth knowing by sight.

```
 learning                 overfitting               not learning

 loss                     loss                      loss
  │\                       │\                        │
  │ \                      │ \      val  __/         │~~~~~~~~~~~~~  train
  │  \___  val             │  \____/                 │~~~~~~~~~~~~~  val
  │   \____ train          │   \________ train       │
  └────────── steps        └────────── steps         └────────── steps
```

- **Learning:** both go down together, validation a little above training.
- **Overfitting:** training keeps falling, validation turns round and climbs. The
  model has started memorising. Stop at the turn.
- **Not learning:** both flat. Something is broken — usually the learning rate, the
  data, or a missing `zero_grad()`.

---

## Exercises

Open `tunelab/loop.py`.

1. `estimate_loss(model, get_batch, split, batches=20)` — the average loss over some
   batches, as a plain float, leaving no trace.
2. `train(model, get_batch, steps, lr, eval_every=100)` — the five lines, in a loop,
   recording the train and validation loss as it goes.

```bash
pytest week-01/day-3 -v
```

Write `estimate_loss` first. `train` calls it.

---

## Before tomorrow

In `logs/signal-log.md`: without looking, write the five lines in order, then one
sentence on what would happen if you swapped lines 4 and 5.
