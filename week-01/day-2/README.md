# Day 2 — Downhill

> **By the end of today** you can say what a gradient is, compute one by hand, and take
> one step downhill.

Training a model is one idea repeated a few million times: **measure how wrong you
are, find which way is less wrong, move a little that way.** Today you do it once, on
one number, with a pencil. Everything after today is the same thing with more numbers.

---

## Read first

- [ ] [Automatic differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — the whole page · 20 min
- [ ] 3Blue1Brown, [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 20 min. Watch for the picture, not the notation.

---

## What you need to know

### Loss: one number that says how wrong

A model with a single weight:

```python
prediction = w * x
```

Suppose `x = 3`, the right answer is `y = 10`, and the weight is currently `w = 2`.
The prediction is 6. It is wrong by 4.

We want one number that is big when the model is badly wrong and zero when it is
right. Squaring the error does that:

```python
loss = (w * x - y) ** 2        # (6 - 10) ** 2 = 16
```

The data (`x`, `y`) is fixed. The only thing you are allowed to change is `w`. So the
loss is really a function of `w`: try a different `w`, get a different loss.

### Gradient: which way is downhill

Nudge `w` up a hair, from 2 to 2.001. The loss goes from 16 to about 15.976. It went
**down** by 0.024 for a nudge of 0.001 — a slope of about **−24**.

That slope is the gradient. Its sign tells you the direction (negative: increasing `w`
reduces the loss) and its size tells you how steep.

You do not have to nudge. Calculus gives the slope exactly, and you already have what
you need: the loss is *something squared*, and the something is `w*x - y`. The chain
rule says: the slope of the outside, times the slope of the inside.

Derive it on paper now. Check it against the nudge: at `w = 2`, `x = 3`, `y = 10` your
formula must give **−24**. That formula is exercise 1.

### One step

The gradient points **uphill**. So you go the other way, a little:

```python
w = w - lr * grad
```

`lr` is the learning rate: how big "a little" is. Too small and you crawl. Too big and
you jump clean over the valley and land higher up the other side.

### Making torch do the calculus

Deriving a slope by hand is fine for one weight. A model has millions. Autograd does it
for you:

```python
a = torch.tensor(5.0, requires_grad=True)     # "track what happens to this"
out = a * a + 1                               # 26
out.backward()                                # work out d(out)/d(a)
a.grad                                        # tensor(10.)   <- 2 * a
```

Three things to hold on to:

- `requires_grad=True` marks a tensor as something you want slopes for.
- `.backward()` is called on the **final number**, and fills in `.grad` on everything
  that was marked.
- `.item()` turns a one-number tensor into a plain Python float.

### Two traps

**Gradients add up.** `.backward()` does not replace what is in `.grad`; it adds to it.
So on the second trip round a loop — a fresh loss, a fresh `.backward()` — `.grad`
holds the sum of both trips. That is deliberate, and it is almost never what you want.
After every update you zero it:

```python
a.grad.zero_()
```

**The update itself must not be tracked.** Changing `a` is not part of the model's
arithmetic; it is you, reaching in from outside. Tell torch to look away:

```python
with torch.no_grad():
    a -= 0.1 * a.grad
```

---

## Exercises

Open `descent.py`.

1. `grad_by_hand(w, x, y)` — the formula you derived, in plain Python.
2. `autograd_grad(w, x, y)` — the same slope, from `.backward()`. The test checks the
   two agree. When they do, you have checked torch's calculus against your own. Do
   that once and you can trust it for the rest of your life.
3. `step(w, grad, lr)` — one line.
4. `fit_line(xs, ys, lr, steps)` — two weights now, `w` and `b`, and fifty data points.
   The loss is the **mean** squared error. No `torch.optim` today: you write the update.

```bash
pytest week-01/day-2 -v
```

### Then break it

When `fit_line` passes, do these two and write what you saw in `logs/stuck-log.md`:

- Call it with `lr=1.5`. Print the loss every step for the first ten steps.
- Put `lr` back, and delete the two lines that zero the gradients. Print `w` every
  fifty steps.

You will meet both of these again, by accident, on a real model. Better to have seen
them on a line first.
