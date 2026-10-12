# Day 2 — Attention

> **By the end of today** a position can choose which earlier positions to listen to.

Yesterday every position took a flat average of its past. Today each one decides for
itself where to look. That decision is called **attention**, and it is the single idea
that separates the model you finish this week from the one you built last week.

It is also less than you have been led to expect. It is yesterday's weighted average,
with the weights computed instead of fixed. Four lines.

`attend` goes in `attend.py`, in this folder. `causal_mask` and `Head` go in
`tunelab/attention.py`.

---

## Read first

- [ ] 3Blue1Brown, [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — up to where he introduces multiple heads · 15 min
- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **section 3.2.1 only**, "Scaled Dot-Product Attention". Half a page and one equation. By this evening you will have implemented that equation · 10 min

---

## What you need to know

### From equal weights to chosen weights

Yesterday's matrix, for three positions:

```
 [ 1    0    0 ]
 [ ½    ½    0 ]
 [ ⅓    ⅓    ⅓ ]
```

Keep the shape. Keep the zeros. Change only the numbers that are not zero, so that
they still add up to one along each row but are no longer equal:

```
 [ 1     0     0   ]
 [ 0.1   0.9   0   ]
 [ 0.7   0.1   0.2 ]      <- position 2 mostly listens to position 0
```

Everything else — the multiplication, the shapes, the fact that nothing looks ahead —
is exactly as it was. So the only new question is: **where do those numbers come
from?**

### Query, key, value

Each position produces three vectors from its own embedding.

| | In words | Used for |
|---|---|---|
| **query** | "Here is what I am looking for." | scoring |
| **key** | "Here is what I contain." | scoring |
| **value** | "Here is what I will say, if you listen to me." | the answer |

The **score** between position `i` and position `j` is the dot product of `i`'s query
with `j`'s key: multiply them number by number and add up. If they point the same way
the score is large; if they are unrelated it is near zero.

Do that for every pair and you have a `(T, T)` table of scores. Row `i` is "how
interested is position `i` in each other position". That is the weight matrix, almost.

Each of the three is made by a plain linear layer, with no bias, from the position's
vector. Nobody tells those layers what to look for. Training does.

### Turning scores into weights

Scores can be any size and any sign. Weights have to be positive and add up to one.
You know the function that does that: softmax, along each row.

But first, the future has to be removed. Set every score for a future position to
**negative infinity**. Softmax turns negative infinity into exactly zero:

```python
torch.softmax(torch.tensor([1.0, 2.0, float("-inf")]), dim=0)
# tensor([0.2689, 0.7311, 0.0000])
```

That is why the mask goes **before** the softmax and not after. Zero out the weights
afterwards and each row would no longer add up to one.

`masked_fill` does the setting. It wants a boolean tensor that is `True` in the places
to overwrite:

```python
t = torch.tensor([5., 6., 7.])
t.masked_fill(torch.tensor([False, True, False]), 0.0)     # tensor([5., 0., 7.])
```

### Why divide by √head_size

Softmax exaggerates. Feed it big numbers and it gives nearly everything to the
largest:

```python
torch.softmax(torch.tensor([1.0, 2.0]), dim=0)       # tensor([0.27, 0.73])   a preference
torch.softmax(torch.tensor([10.0, 20.0]), dim=0)     # tensor([0.00, 1.00])   a dictatorship
```

A dot product adds up `head_size` terms, so the wider the vectors, the bigger the
scores, and the more the softmax collapses onto a single position. A model that starts
out listening to exactly one position has nothing to adjust. Dividing by
`√head_size` keeps the scores the same size whatever the width.

### All of it, by hand

Two positions, vectors of width 1 so the arithmetic fits on a line. So `√head_size`
is 1 and can be ignored this once.

```
               query    key    value
 position 0      1       2      10
 position 1      3       1      20
```

Scores, query of the row times key of the column:

```
               key 0    key 1
 query 0       1·2=2    1·1=1
 query 1       3·2=6    3·1=3
```

Mask the future (position 0 may not see position 1), then softmax each row:

```
 row 0   [ 2, -inf ]   ->   [ 1.00, 0.00 ]
 row 1   [ 6,   3  ]   ->   [ 0.95, 0.05 ]
```

Weights times values:

```
 position 0   1.00·10 + 0.00·20  =  10.0
 position 1   0.95·10 + 0.05·20  =  10.5
```

Position 1's query matched position 0's key far better than its own, so it mostly
copied position 0's value. That is attention. Work one more row yourself, with numbers
of your own, before you write any code.

### Shapes

For one head, with `C = n_embd` and `H = head_size`:

| | Shape |
|---|---|
| input `x` | `(B, T, C)` |
| query, key, value | `(B, T, H)` each |
| scores | `(B, T, T)` |
| weights | `(B, T, T)` |
| output | `(B, T, H)` |

To get `(B, T, T)` scores out of two `(B, T, H)` tensors, one of them has to be turned
on its side. `.transpose(-2, -1)` swaps the last two dimensions:

```python
torch.zeros(4, 8, 16).transpose(-2, -1).shape      # torch.Size([4, 16, 8])
```

### A buffer: saved with the model, never trained

The mask is the same every time, so build it once. But a plain attribute is invisible
to PyTorch: it will not be saved, and on Thursday it will not move to the GPU with the
rest of the model. `register_buffer` fixes both:

```python
class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(2.0))     # now reachable as self.factor
```

A buffer is part of the model that training does not touch.

---

## Exercises

1. `attend(q, k, v, mask)` in `attend.py` — the arithmetic, on tensors you are handed.
2. `causal_mask(T)` in `tunelab/attention.py`.
3. `Head` in `tunelab/attention.py` — the same arithmetic, but the head makes its own
   queries, keys and values from `x`.

```bash
pytest week-02/day-2 -v
```

One test gives `attend` queries and keys that are all zero and checks the answer
against **yesterday's running mean**. Before you run it, say why that has to be true.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

A head has one set of weights per position: one way of looking. But a character might
want to know several things at once — what the last vowel was, whether a word just
ended, who is speaking. Tomorrow: more than one head, and what to do with what they
find.
