# Day 4 — A GPT, and a run that survives

> **By the end of today** you have a GPT, and a training run that survives being killed.

Two halves today, and the second matters as much as the first.

The first is short: you have every part, and today you stack them. The second is the
thing nobody teaches until it has cost them a night. A free Colab session **will** die
in the middle of a run. Today you make that cost five minutes.

`GPT` and `count_params` go in `tunelab/gpt.py`. `save`, `load` and `train_in_chunks`
go in `tunelab/checkpoint.py`.

---

## Read first

- [ ] [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — "What is a `state_dict`?" and "Saving & Loading a General Checkpoint" only · 15 min
- [ ] Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **after** the exercises, from self-attention to the end. You have now written everything in it.

---

## Part one — the model

### The stack

```
 idx                                   (B, T)              whole numbers
  │
  ▼
 Embedder        what + where          (B, T, n_embd)
  │
  ▼
 dropout
  │
  ▼
 Block  ┐
 Block  │  n_layer of them             (B, T, n_embd)      shape never changes
 Block  ┘
  │
  ▼
 LayerNorm       one last tidy-up      (B, T, n_embd)
  │
  ▼
 Linear          description → scores  (B, T, vocab_size)  logits
```

Then the loss, exactly as in the bigram: reshape, `cross_entropy`. And the same
convention as ever: `model(idx, targets)` returns `(logits, loss)`.

Compare the top and bottom of that picture with last week's model. The bigram went
from `(B, T)` straight to `(B, T, vocab_size)` in one lookup. This model does the same
journey, with a stretch in the middle where positions can talk to each other. That
stretch is the whole difference.

### How it should start

Last week your bigram's first loss was around 4.6, a little above `ln(65) = 4.17`,
because a table of random numbers is confidently wrong about some characters.

A deep model with careless starting weights is much worse than that, and a model that
starts badly spends its first thousand steps just unlearning the noise. So set the
starting weights small: drawn from a normal distribution with standard deviation
**0.02**, and every bias at zero. Small weights mean small logits, small logits mean
nearly equal probabilities, and the first loss lands almost exactly on
`ln(vocab_size)`.

`model.apply(fn)` calls `fn` on every layer inside a model, however deeply nested:

```python
def describe(module):
    if isinstance(module, nn.Linear):
        print("a Linear with weight", tuple(module.weight.shape))

nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).apply(describe)
# a Linear with weight (5, 3)
# a Linear with weight (2, 5)
```

`nn.init.normal_(tensor, mean=0.0, std=1.0)` and `nn.init.zeros_(tensor)` overwrite a
tensor in place.

The check you learned in week 1 now has teeth. If your GPT's first loss is not within
a whisker of `ln(vocab_size)`, something is wrong before you have trained a step.

### Generating

Two things the bigram never had to think about.

**The model has a longest sequence.** Ask it to continue a text longer than
`block_size` and there is no position row for the extra characters. So each time
round, hand it only the last `block_size` characters of what you have so far. It
writes with a sliding window.

```python
t = torch.arange(10)
t[-4:]          # tensor([6, 7, 8, 9])     the last four
```

**Dropout must be off.** You want the model's real prediction, not the prediction of
a model with some of its numbers knocked out. But do not leave it switched to eval
mode: if generate is called in the middle of training, the model has to go back the
way it came. `model.training` is `True` or `False`, and `model.train(flag)` sets it.

### Where the parameters are

`p.numel()` is how many numbers are in a tensor. Add that up over
`model.parameters()` and you have the size of the model.

When yours works, count the Colab-sized model from tomorrow's milestone, then count
its pieces separately. Most people guess the embedding tables are the bulk of it.
Find out. You will be asked on Friday.

---

## Part two — the GPU, and staying alive

### Moving to the GPU

Three lines of setup, and the loop you wrote in week 1 does not change at all:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)        # every parameter and every buffer
data = data.to(device)          # the encoded text, before you build get_batch
```

A model on the GPU can only work on tensors that are on the GPU. Batches cut from a
tensor on the GPU are already there, which is why moving `data` is enough.

**The one bug the tests cannot catch for you.** Any tensor you *create* inside
`forward` is born on the CPU unless you say otherwise. On your laptop that is
invisible. On Colab it is

```
RuntimeError: Expected all tensors to be on the same device
```

There are exactly two places in your model where this can happen, and you wrote both
this week:

- **the positions** in `Embedder` — make them where the input lives:
  `torch.arange(T, device=idx.device)`
- **the mask** in `Head` — which is why it had to be a registered buffer. Buffers move
  with `model.to(device)`. A mask built fresh inside `forward` does not.

Go and look at both now. Tomorrow's notebook has a cell that checks.

### What a checkpoint is

A model's learned numbers live in its `state_dict`: a dictionary from names to
tensors.

```python
net = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
for name, tensor in net.state_dict().items():
    print(name, tuple(tensor.shape))
# 0.weight (5, 3)
# 0.bias (5,)
# 1.weight (2, 5)
# 1.bias (2,)

torch.save({"weights": net.state_dict(), "note": "anything else you like"}, "toy.pt")

saved = torch.load("toy.pt")
net.load_state_dict(saved["weights"])       # the numbers go back into the same shapes
```

`torch.save` will write any dictionary. So a checkpoint is the weights **plus
whatever you need to carry on**: for you, how many steps are done, and the history so
far.

Notice what `load_state_dict` needs: a model of the same shape, already built. A
checkpoint is the numbers, not the architecture.

One trap for later. A checkpoint saved on a GPU remembers that its tensors lived on a
GPU, and will not open on a laptop that has none. `torch.load(path, map_location="cpu")`
opens it anywhere; `load_state_dict` then copies the numbers to wherever the model is.

### Never write on top of your only copy

Saving takes time. A session can die halfway through a save. If you were writing
straight onto your one checkpoint file, you now have half a file and no checkpoint.

So write somewhere else first, and swap only when it is complete:

```python
os.replace("report.tmp", "report.txt")      # report.txt is now the new file, all at once
```

`os.replace` either happens entirely or not at all. A crash before it costs you the
new save. It can never cost you the old one.

### Training in chunks

Your loop trains for a number of steps and returns. To checkpoint without touching
it: call it for 500 steps, save, call it for 500 more, save. If the session dies, the
next run loads the last save and carries on from there.

Two details, and both are today's tests:

- `train()` numbers its steps from 0 every time. Your history must not restart.
- If the run is already finished, running it again should do nothing.

One honest cost: `train()` builds a new AdamW on every call, so the optimizer's memory
of recent gradients restarts at each chunk. At this scale you will not see it. Real
training code saves the optimizer too, and in week 5 you will see a library do it.

### Drive

Colab's disk is wiped when the session ends. Google Drive is not. Tomorrow's notebook
mounts your Drive, and your checkpoint path points into it:

```python
from google.colab import drive
drive.mount("/content/drive")
# then save to "/content/drive/MyDrive/tunelab/week-02.pt"
```

---

## Exercises

1. `GPT` and `count_params` in `tunelab/gpt.py`.
2. `save` and `load` in `tunelab/checkpoint.py`.
3. `train_in_chunks` in `tunelab/checkpoint.py`.

```bash
pytest week-02/day-4 -v
```

One test trains your GPT on a single batch until it has memorised it. You know that
test: you wrote it yourself last Friday. If it fails while the shape tests pass, the
bug is in how the parts connect, and the block diagram is where to look.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Then do tomorrow's setup tonight: open `week-02/colab.ipynb` in Colab, pick the T4,
and run it as far as the GPU check. A GPU is not always free the moment you want one.
