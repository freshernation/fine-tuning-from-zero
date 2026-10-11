# Day 1 — Shapes

> **By the end of today** you can read the shape of any tensor and predict the shape an
> operation will give back.

Today's real lesson is not PyTorch. It is this: **almost every bug you hit in this
course will be a shape bug.** A model is a pile of numbers arranged in boxes, and
training is arithmetic between boxes. If you can say the shape of every box out loud,
you can debug anything in the next six weeks.

---

## Read first

- [ ] [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) — the whole page · 20 min
- [ ] [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) — "General semantics" only · 10 min

---

## What you need to know

### A tensor is a box of numbers with a shape

```python
import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

x.shape      # torch.Size([2, 3])   <- 2 rows, 3 columns
x.dtype      # torch.float32
```

Read a shape from the outside in. `(2, 3)` is two things, each of which is three
numbers. `(4, 8, 16)` is four things, each of which is eight things, each of which is
sixteen numbers.

In this course three letters come back every day:

| Letter | Means | Example |
|---|---|---|
| `B` | batch — how many examples at once | 32 sentences |
| `T` | time — how many positions in each | 8 characters |
| `C` | channels — how many numbers describe each position | 65 scores |

So `(B, T, C)` is "a batch of sequences of vectors". Say it that way.

### Indexing

```python
x[0]         # tensor([1., 2., 3.])     first row            shape (3,)
x[:, 0]      # tensor([1., 4.])         first column         shape (2,)
x[1, 2]      # tensor(6.)               one number           shape ()
x[:, -1]     # tensor([3., 6.])         last column          shape (2,)
```

`:` means "all of this dimension". A plain number **removes** that dimension. That is
why `x[0]` has shape `(3,)` and not `(1, 3)`.

### `dim=` — which dimension disappears

```python
x.sum()            # tensor(21.)             everything
x.sum(dim=0)       # tensor([5., 7., 9.])    shape (3,)  <- dim 0 is gone
x.sum(dim=1)       # tensor([ 6., 15.])      shape (2,)  <- dim 1 is gone
```

`dim=1` does not mean "sum the columns". It means **"dimension 1 is the one that gets
squashed"**. Before you run a line with `dim=` in it, cross that number out of the
shape and say what is left.

`keepdim=True` squashes it to size 1 instead of removing it:

```python
x.sum(dim=1, keepdim=True)     # tensor([[ 6.], [15.]])   shape (2, 1)
```

You will want that in about ten minutes.

### Broadcasting

When two tensors have different shapes, PyTorch lines the shapes up **from the right**
and stretches any dimension of size 1 to fit.

```python
prices = torch.tensor([[10., 20., 30.],
                       [40., 50., 60.]])       # (2, 3)
discount = torch.tensor([[0.5],
                         [0.1]])               # (2, 1)

prices * discount
# tensor([[ 5., 10., 15.],
#         [ 4.,  5.,  6.]])                    # (2, 3)
```

The `(2, 1)` was stretched across three columns: each row got its own discount.

Now the trap. Drop the inner brackets so the discount is shape `(2,)`:

```python
torch.tensor([[10., 20.], [40., 50.]]) * torch.tensor([0.5, 0.1])
# tensor([[ 5.,  2.],
#         [20.,  5.]])
```

No error. Wrong answer. Lined up from the right, `(2,)` matched the **columns**, so
each column got a discount instead of each row. Broadcasting never complains when the
sizes happen to fit. **A shape bug that crashes is a gift; the ones that run are the
ones that cost you an evening.**

### dtype

```python
torch.tensor([1, 2]).dtype        # torch.int64    <- whole numbers: positions, ids
torch.tensor([1., 2.]).dtype      # torch.float32  <- decimals: everything a model learns
torch.zeros(2, 3).dtype           # torch.float32
```

Whole-number tensors point at things. Float tensors get trained. Mixing them up is the
second most common bug after shapes.

---

## Exercises

Open `shapes.py`. There are four functions, each a line or two. For each one, **write
the shape you expect as a comment before you run anything.**

1. `row_means(x)` — the mean of each row.
2. `normalize_rows(x)` — divide each row by its own sum.
3. `one_hot(idx, n)` — a row of zeros with a single `1.0`, for each index.
4. `last_column(x)` — the last time step of every sequence in a batch.

```bash
pytest week-01/day-1 -v
```

Every test fails before you start. That is the starting line, not a problem.

`normalize_rows` is the one to slow down on. If your first attempt runs but the test
says the rows do not sum to one, re-read the trap above before you ask anyone.

---

## Before tomorrow

Write one line in `logs/signal-log.md`: which of the four did you get wrong first, and
what shape did you think it was?
