# Day 3 — The block

> **By the end of today** you can draw a transformer block from memory, and you have
> built one.

There is one diagram on this page that matters more than everything else this week.
Every large language model you have ever used is that diagram, repeated. On Friday you
will be asked to draw it on a blank page and say what each box is for.

Today has four small new ideas. None is as hard as yesterday's. The work is in how
they fit together.

`MultiHead` goes in `tunelab/attention.py`. `Block` goes in `tunelab/gpt.py`.

---

## Read first

- [ ] 3Blue1Brown, [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the rest, from multiple heads onward · 10 min
- [ ] [`nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) — the first paragraph and the formula · 5 min

---

## The diagram

```
                      x   (B, T, n_embd)
                      │
            ┌─────────┤
            │         ▼
            │    ┌───────────┐
            │    │ LayerNorm │
            │    └─────┬─────┘
            │          ▼
            │    ┌───────────┐
            │    │ Attention │     gather: each position collects
            │    │ (n heads) │     from the positions before it
            │    └─────┬─────┘
            │          ▼
            └────────▶(+)
                      │
            ┌─────────┤
            │         ▼
            │    ┌───────────┐
            │    │ LayerNorm │
            │    └─────┬─────┘
            │          ▼
            │    ┌───────────┐
            │    │    MLP    │     think: each position, alone,
            │    │ (4x wide) │     works on what it collected
            │    └─────┬─────┘
            │          ▼
            └────────▶(+)
                      │
                      ▼
                      x   (B, T, n_embd)        same shape as it came in
```

Two halves, and they are the same pattern twice: **normalise, do something, add the
result back to what you had.**

Copy it onto paper now. Then read on, and by the end each box should have a reason.

---

## What you need to know

### Many heads

One head has one set of queries and keys: one way of looking. A character may want to
know several things at the same moment — where the word started, what the last vowel
was, whether a line just broke.

So run several heads side by side. Each has its own query, key and value layers, and
each is free to learn to look for something different.

To keep the model the same width throughout, the heads share it out. With
`n_embd = 16` and four heads, each head produces 4 numbers per position. Lay the four
outputs end to end and you are back at 16:

```
 head 0        head 1        head 2        head 3
 (B, T, 4)     (B, T, 4)     (B, T, 4)     (B, T, 4)
     └─────────────┴──── end to end ───┴───────┘
                     (B, T, 16)
```

That only works if `n_embd` divides evenly by `n_head`. If it does not, say so the
moment the model is built — not three calls later, as a shape error from somewhere
inside a matrix multiply.

Then one more linear layer, `n_embd → n_embd`, to mix what the separate heads found.

### The MLP

Attention moves information **between** positions. It does not do much with it: the
output is only ever a weighted average of values.

The MLP is where each position works on what it gathered. It is applied to every
position separately, with the same weights, and nothing in it looks sideways:

```
 n_embd  ──▶  4 × n_embd  ──▶  GELU  ──▶  n_embd
```

Out to four times the width, a non-linear function, back to the width. `GELU` is a
smooth cousin of "set negatives to zero"; without something non-linear in the middle,
two linear layers collapse into one and the MLP could not compute anything a single
layer cannot.

A shorthand worth keeping: **attention gathers, the MLP thinks.**

`nn.Sequential` runs a list of layers one after another, which is all an MLP is:

```python
net = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
net(torch.zeros(7, 3)).shape        # torch.Size([7, 2])
```

And when you need a *list* of layers that PyTorch can see — so their weights show up
in `.parameters()` — a plain Python list will not do. `nn.ModuleList` will:

```python
layers = nn.ModuleList(nn.Linear(3, 3) for _ in range(4))
```

### Add it back

The `(+)` in the diagram: the output of each half is **added to** its input, not
substituted for it.

```
 without        x  ──▶  f(x)
 with           x  ──▶  x + f(x)
```

Why it matters. Stack twelve blocks without it and the gradient has to pass *through*
every one on its way back to the first. Each can shrink it. By the time it arrives
there is nothing left, and the early layers do not learn: last week's "not learning"
curve, flat from the start.

With the addition there is a straight road from the output to every layer. The
gradient can always travel down it. Each block only has to learn a small
**correction** to what is already there — and a block that has nothing useful to add
can output zero and do no harm.

One of today's tests sets every weight in your block to zero and checks that what
comes out is exactly what went in. Work out why that must be true of the diagram
before you run it.

### Keep the numbers in range

Add things to `x` twelve times over and its numbers can drift far from where they
started — large in one block, tiny in another. Layers train badly when their inputs
keep changing scale.

`LayerNorm` takes each position's vector and rescales it to mean 0 and spread 1, then
applies a learned scale and shift:

```python
norm = nn.LayerNorm(4)
norm(torch.tensor([[10., 20., 30., 40.]]))      # tensor([[-1.34, -0.45, 0.45, 1.34]])
```

It works on each position by itself. It never mixes positions and never mixes
examples in a batch.

**Where it goes.** Look at the diagram: the norm sits at the *start* of each half,
inside the branch, and the straight road around it is left untouched. That is called
pre-norm. The original paper put it after the addition; almost every model since has
moved it, because it trains more steadily. The tests check that yours is pre-norm.

There are two norms in a block and they are separate layers with separate weights.

### Dropout

During training, `nn.Dropout(0.2)` sets a random 20% of the numbers passing through
it to zero. It stops the model leaning too hard on any one number, which makes it
harder to memorise the training set.

During evaluation it does nothing at all.

```python
drop = nn.Dropout(0.5)
x = torch.ones(8)

drop.train();  drop(x)      # tensor([2., 0., 2., 2., 0., 0., 2., 0.])   different every call
drop.eval();   drop(x)      # tensor([1., 1., 1., 1., 1., 1., 1., 1.])   always
```

Which is why, on the Wednesday of week 1, you were made to write `model.eval()` and
`model.train()` around a measurement in a model where it made no difference. **From
today it makes a difference.** Measure a model with dropout while it is still in
training mode and the loss you read is the loss of a model with a fifth of its numbers
missing.

It goes in three places: on the attention weights (you did that yesterday), after the
mixing layer in `MultiHead`, and at the end of the MLP.

---

## Exercises

1. `MultiHead` in `tunelab/attention.py`.
2. `Block` in `tunelab/gpt.py`.

```bash
pytest week-02/day-3 -v
```

`torch.cat(list_of_tensors, dim=-1)` lays tensors end to end along the last dimension.
You met `cat` in week 1.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Then close this page and draw the block. Check it against the diagram. Do it again
until you get it right twice running. Tomorrow you stack several of them and it trains.
