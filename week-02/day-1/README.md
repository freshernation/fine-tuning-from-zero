# Day 1 — What, where, and before

> **By the end of today** every character knows what it is and where it is, and you have
> a way for a position to hear from the ones before it.

Last week's model stopped at a loss of about 2.5 and would not go lower. You worked
out why: it guesses the next character from **one** character. Everything this week is
about fixing that, and it takes three ideas. Today is the first two, plus the trick
that makes the third possible.

`bag_of_past` goes in `context.py`, in this folder. `Embedder` goes in `tunelab/gpt.py`.

---

## Read first

- [ ] 3Blue1Brown, [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — the section on word embeddings · 15 min
- [ ] [`torch.tril`](https://docs.pytorch.org/docs/stable/generated/torch.tril.html) — the examples · 3 min

No Karpathy video today until **after** the exercises. See the end of this page.

---

## What you need to know

### A row of scores is not enough

The bigram's table had one row per character, and the row *was* the answer: 65 scores
for what comes next. That leaves the model nowhere to keep anything else. It cannot
note "I am a vowel", or "I am the start of a word", because every number in the row is
already spoken for.

So split the job in two. First turn each character into a vector of numbers that
*describes* it. Later, turn a description into scores.

```python
table = nn.Embedding(65, 32)          # 65 characters, each described by 32 numbers
table(torch.tensor([[7, 2, 7]])).shape        # torch.Size([1, 3, 32])
```

`(B, T)` whole numbers go in. `(B, T, C)` floats come out, where `C` is now called
`n_embd`: the width of the model. Nobody decides what the 32 numbers mean. Training
does.

### Where am I?

Look at what that table gives the two `7`s above: the same vector. The model cannot
tell the first from the third. To it, `"ab"` and `"ba"` are the same bag of letters.

The fix is almost embarrassingly simple. Make a **second** table, with one row per
*position* instead of per character, and add the two together:

```
              what I am                where I am               what goes on
 position 0   token row for 'h'    +   position row 0     =     vector for "h, at 0"
 position 1   token row for 'e'    +   position row 1     =     vector for "e, at 1"
 position 2   token row for 'h'    +   position row 2     =     vector for "h, at 2"
```

Now the two `h`s differ, because their position rows differ. Both tables are learned.

The position table has `block_size` rows — the longest stretch of text the model will
ever look at. That is a hard limit. Hand the model a longer sequence and there is no
row for the extra positions. Yours should say so clearly rather than crash somewhere
deep inside PyTorch.

Shapes, because today is where you start needing them: the token lookup gives
`(B, T, C)`. The position lookup, done once for positions `0 … T-1`, gives `(T, C)`.
Line those up from the right, the way you learned on the first Monday, and they add.

### Hearing from the past

Each position now knows what and where it is. It still knows nothing about its
neighbours. The simplest way to tell it: **replace each position with the average of
itself and everything before it.**

```
 input       [ 6 ]      [ 0 ]      [ 3 ]
 output      [ 6 ]      [ 3 ]      [ 3 ]
               ↑          ↑          ↑
             6 / 1    (6+0) / 2   (6+0+3) / 3
```

Only *before*, never after: when the model is predicting character 5, character 6 is
the answer, and it must not be allowed to see it.

You could write that with two `for` loops. Do not. There is a way to do it for every
position and every batch in one operation, and it is the trick the whole week is built
on.

### Matrix multiplication is weighted averaging

Multiply a matrix of weights by a column of values, and each output is a weighted sum
of the values, using one row of weights:

```
   weights             values       output

 [ 1    0    0 ]       [ 6 ]       [ 1·6 + 0·0 + 0·3 ]     [ 6 ]
 [ ½    ½    0 ]   @   [ 0 ]   =   [ ½·6 + ½·0 + 0·3 ]  =  [ 3 ]
 [ ⅓    ⅓    ⅓ ]       [ 3 ]       [ ⅓·6 + ⅓·0 + ⅓·3 ]     [ 3 ]
```

Read the weights row by row. Row 0 says "position 0 listens only to itself". Row 2
says "position 2 listens equally to all three". The zeros in the top-right corner are
what stop a position hearing the future.

That is the same output as the picture above. So the question "how do I average the
past?" has become "how do I build that matrix?" — and you have just read the
documentation for the function that builds its shape.

In PyTorch, `@` is matrix multiplication, and it broadcasts: a `(T, T)` matrix times a
`(B, T, C)` tensor applies the same weights to every batch.

---

## Exercises

1. `bag_of_past(x)` in `context.py`. No loops.
2. `Embedder` in `tunelab/gpt.py`.

```bash
pytest week-02/day-1 -v
```

---

## After the exercises

Now watch Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY),
from where he starts averaging past context with a loop, up to — **not including** —
the part he calls the crux of the video, self-attention. He solves `bag_of_past` three
ways. You have already solved it once, which is the only reason watching him do it is
now worth your time.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`. Then think about this, because it is
tomorrow's whole lesson:

In your weight matrix, row 2 is `⅓ ⅓ ⅓`. Every earlier position gets an equal say.
That is a strange way to read. In the sentence you are reading now, some earlier
words matter far more than others.

**What if the weights were not equal — and what if each position could choose its
own?**
