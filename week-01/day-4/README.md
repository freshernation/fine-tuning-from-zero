# Day 4 — A model that predicts the next character

> **By the end of today** a model you wrote predicts the next character, badly, and you
> know exactly why it is bad.

Every language model you have ever called through an API does one thing: given some
text, it scores every possible next piece. Today you build the smallest thing that
does that. It is a lookup table. It is also, structurally, a language model — same
inputs, same outputs, same loss, same loop as the ones in Week 5.

Today's code goes in `tunelab/text.py` and `tunelab/bigram.py`.

---

## Read first

- [ ] Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **from the start until he begins "the mathematical trick in self-attention"** (about 40 minutes in: data, tokenizer, batches, bigram). Stop there. The rest is next week · 40 min
- [ ] [`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) — the "Shape" section only · 5 min

Watch, then close the video and write it yourself. Typing along with a video is
copying, and weeks 1–3 are for writing.

---

## What you need to know

### Characters to numbers and back

A model does arithmetic. It cannot do arithmetic on `"h"`. So every distinct character
gets a whole number:

```python
text = "hello world"
sorted(set(text))        # [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
                         #   0    1    2    3    4    5    6    7
```

`"hello"` becomes `[3, 2, 4, 4, 5]`. The count of distinct characters is the
**vocabulary size**. That is a tokenizer. Real ones cut text into chunks bigger than a
character, but they are the same idea: a table from pieces of text to numbers, and
back.

### A batch is text and the same text shifted by one

The task is "predict what comes next". So for every position, the right answer is
simply the character after it:

```
text      h  e  l  l  o  _  w  o  r  l  d

x         h  e  l  l            <- what the model sees
y         e  l  l  o            <- what it should have predicted
```

`y` is `x`, one position later. No labelling, no humans. The text is its own answer
key. That one trick is why language models can be trained on the whole internet.

Each row of a batch is a stretch like this cut from a random place. `block_size` is
how long the stretch is; `batch_size` is how many you cut. So `x` and `y` are both
`(B, T)` whole numbers.

### Logits

For each position the model outputs one score per character in the vocabulary. Those
scores are called **logits**. They are not probabilities: they can be negative, and
they do not add up to one. Softmax turns them into probabilities:

```python
logits = torch.tensor([2.0, 1.0, 0.1])
torch.softmax(logits, dim=-1)       # tensor([0.659, 0.242, 0.099])
```

Shapes: `(B, T)` goes in, `(B, T, C)` comes out, where `C` is the vocabulary size.
Yesterday's letters, all three at once.

### Cross-entropy: how surprised was the model?

For each position, look at the probability the model gave to the character that
actually came next. Cross-entropy is the negative log of that probability, averaged
over every position.

| Probability given to the right answer | Loss |
|---|---|
| 1.0 (certain, and right) | 0 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |

A model that knows nothing spreads its bets evenly: probability `1/C` on everything.
Its loss is `-ln(1/C)`, which is **`ln(C)`**. With 65 characters that is about 4.17.

Remember that number. A fresh model should start near `ln(vocab_size)`. If yours
starts at 9, it is not "bad", it is broken. That check costs nothing and you should
make it every time you build a model.

`F.cross_entropy` wants logits as `(N, C)` and targets as `(N,)`. Yours are
`(B, T, C)` and `(B, T)`. Reshaping one into the other is today's shape exercise.

### The bigram model

"Bigram" means pairs. The model predicts the next character from the current one and
nothing else.

That makes it a table with one row per character. Row `h` holds the scores for what
follows `h`. Looking up a row by its number is exactly what `nn.Embedding` does:

```python
table = nn.Embedding(8, 8)                # 8 rows of 8 numbers
table(torch.tensor([3, 2])).shape         # torch.Size([2, 8])
```

Those 64 numbers start random and are the only thing that gets trained.

### Sampling

To generate: get the logits for the **last** position, softmax them, and draw one
character according to those probabilities.

```python
probs = torch.tensor([0.7, 0.2, 0.1])
torch.multinomial(probs, num_samples=1)    # usually tensor([0]), sometimes not
```

Glue the new character on the end. Repeat.

---

## Exercises

1. `CharTokenizer` in `tunelab/text.py` — `encode`, `decode`, `vocab_size`.
2. `make_get_batch` in `tunelab/text.py` — returns a `get_batch(split)` function of the
   kind yesterday's loop expects.
3. `Bigram` in `tunelab/bigram.py` — `forward` and `generate`.

```bash
pytest week-01/day-4 -v
```

Do them in that order. Each one is testable alone.

---

## Why it is bad

When it passes, try it for real — in a Python shell, not a test:

```python
import torch
from tunelab.text import CharTokenizer, make_get_batch
from tunelab.bigram import Bigram
from tunelab.loop import train

text = open("data/tiny.txt").read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
model = Bigram(tok.vocab_size)

history = train(model, make_get_batch(data, 8, 32), steps=2000, lr=1e-2)
print(history[0], history[-1])
print(tok.decode(model.generate(torch.zeros((1, 1), dtype=torch.long), 300)[0].tolist()))
```

You wrote every line that just ran: the tokenizer, the batches, the model, the loop.

Look at the output. It has the texture of English — spaces in roughly the right
places, plausible pairs of letters — and no words. That is exactly what a model with a
memory of **one character** can do. It knows `q` is followed by `u`. It cannot know
that `th` is followed by `e`, because by the time it sees the `h` it has forgotten
the `t`.

Nothing is wrong with your loop, your loss, or your data. The model cannot see far
enough. Fixing that is the whole of next week.
