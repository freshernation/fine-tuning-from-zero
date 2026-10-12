# Day 3 — LoRA

> **By the end of today** you have written LoRA.

This is the day the course was built around. Everything before it was so that today
would contain nothing unfamiliar, and everything after it uses what you write today.

It is about thirty lines. There is one idea in it, and the idea is arithmetic you
already know. Take the day slowly anyway. On Friday you will be asked to derive it on
a blank page, and in three weeks a model you tuned with it will be running inside your
agent.

The counting functions go in `lowrank.py`, in this folder. `freeze`, `unfreeze` and
`LoRALinear` go in `tunelab/lora.py`.

---

## Read first

- [ ] [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — the abstract, **section 4.1**, and Figure 1. Nothing else yet · 20 min

Read it before the rest of this page, not after. It is three paragraphs and one
equation, and it is more readable than its reputation. Then come back and build it.

---

## Where yesterday left you

Full fine-tuning has a bill — four copies of the model in memory, three of them
*per weight you train* — and it forgets.

Both problems come from the same place: you are changing every weight.

### First answer: train fewer weights

Freeze the model, then switch a few layers back on:

```python
for name, p in net.named_parameters():
    print(name, p.requires_grad)
```

`named_parameters()` gives every tensor with its full dotted name, so you can choose
by name: only the last block, say.

That fixes the bill. A frozen weight is stored once and costs nothing more.

But look at what you gave up. You can now only change what the last block does. Every
earlier block is exactly as it was. If what your task needs is a slightly different
way of attending in block 3, you cannot have it.

So the question becomes sharper: **can you change every layer, a little, while
training almost no numbers?**

---

## The idea

### Learn the change, not the weights

Take one linear layer in the model. Its weight matrix is `W`, shaped
`(d_out, d_in)`, and it computes `W x`.

Fine-tuning turns `W` into some new matrix. Call the difference `ΔW`:

```
after fine-tuning:   W + ΔW
```

Now do something that sounds like it changes nothing. **Freeze `W`, and train `ΔW` as
a separate thing.** The layer computes:

```
W x  +  ΔW x
```

`W` is never touched, so nothing it knew can be overwritten. So far this saves
nothing: `ΔW` is the same size as `W`.

### A change does not need a whole matrix

Here is the observation the paper rests on. When you fine-tune a big model on a narrow
task, the change `ΔW` turns out to be **simple**. Not small in size — simple in
structure. It can be written, nearly, as the product of two thin matrices:

```
      ΔW            =          B          @          A
 (d_out, d_in)            (d_out, r)             (r, d_in)

  ┌───────────┐            ┌──┐
  │           │            │  │              ┌───────────┐
  │           │     =      │  │       @      │           │
  │           │            │  │              └───────────┘
  └───────────┘            └──┘
```

`r` is a small number you choose: 4, 8, 16. It is called the **rank**.

Count the numbers. For one 768 × 768 layer:

| | Numbers to train |
|---|---|
| `ΔW` as a full matrix | 768 × 768 = 589,824 |
| `B` and `A` with `r = 8` | 768 × 8 + 8 × 768 = 12,288 |

Two per cent. And you can do that to every layer you care about.

That is LoRA. **Low-Rank Adaptation**: adapt the model by learning a low-rank change
to its weights.

### What "low rank" costs you

`A` squeezes the input from `d_in` numbers down to `r`. `B` expands those `r` back out
to `d_out`. So whatever the change is, it has to pass through a bottleneck `r` numbers
wide. A full `ΔW` could do anything to `x`; yours can only do things that survive that
squeeze.

The bet is that fine-tuning for one task does not need more. It is a bet, and on
Friday you will measure it: you will train at ranks 1, 4 and 16 and see what each buys.

Your first exercise builds `B @ A` in full and asks PyTorch for its rank. It will never
be more than `r`, however large the matrix. That is the meaning of the word.

---

## The layer

```
                         x   (…, d_in)
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
     ┌─────────────┐            ┌───────────┐
     │      W      │            │     A     │   (r, d_in)      trained
     │   frozen    │            └─────┬─────┘
     │(d_out, d_in)│                  │   (…, r)
     └──────┬──────┘                  ▼
            │                   ┌───────────┐
            │                   │     B     │   (d_out, r)     trained
            │                   └─────┬─────┘
            │                         │   (…, d_out)
            │                         ▼
            │                   × (alpha / r)
            │                         │
            └──────────▶ (+) ◀────────┘
                         │
                         ▼
                   (…, d_out)
```

Two roads. The left is the layer exactly as it came, frozen. The right is yours.
They are added.

You have seen this shape before. It is a residual connection, turned on its side: an
untouched road, plus a learned correction.

### By hand

A 2-in, 2-out layer, rank 1. Take `alpha = 1`, so the scale is 1 and can be ignored
for now.

```
 W = [ 1  0 ]        A = [ 2  1 ]          B = [ 1 ]
     [ 0  1 ]            (1, 2)                [ 3 ]
                                               (2, 1)
 x = [ 4 ]
     [ 5 ]
```

Left road: `W x = [4, 5]`.

Right road, in two steps. `A x` squeezes two numbers into one:
`2·4 + 1·5 = 13`. Then `B` expands that one number back into two:
`[1·13, 3·13] = [13, 39]`.

Add: `[4 + 13, 5 + 39] = [17, 44]`.

Now do it the other way, to convince yourself they are the same thing. Multiply `B @ A`
out first:

```
 B @ A = [ 2  1 ]
         [ 6  3 ]
```

and `(W + B @ A) x = [3·4 + 1·5, 6·4 + 4·5] = [17, 44]`. Same answer.

Look at that `B @ A`. Its second row is three times its first. However you choose the
four numbers in `A` and `B`, the rows will always be multiples of each other. That is
rank 1, and that is the bottleneck made visible.

Work one more yourself, with your own numbers, both ways, before you write any code.

### How it starts

**`B` starts as all zeros.** Then `B @ A` is zero, the right-hand road contributes
nothing, and on step 0 the layer behaves *exactly* as the pretrained layer did. You
begin from the model as it was — not from the model plus some random disturbance.

**`A` starts random.** Small random numbers, the same way PyTorch initialises an
ordinary `Linear`: `nn.init.kaiming_uniform_(A, a=math.sqrt(5))`.

Why not both at zero? One of today's tests checks which of `A` and `B` receives a
gradient on the very first backward pass. Run it, look at what it asserts, and work
out the answer to this question from that. You will be asked on Friday.

### The volume knob

The right-hand road is multiplied by `alpha / r` before it is added.

`alpha` is a number you choose alongside the rank. It sets how loudly the adapter
speaks relative to the frozen layer. Dividing by `r` means that if you double the
rank you do not also have to retune everything else: the overall size of the update
stays roughly where it was.

In practice people set `alpha` to once or twice the rank and leave it. It is a knob
you need to know exists, and to put in the right place.

### Three practical details

- **`nn.Parameter`.** A tensor only becomes part of a module — visible to
  `.parameters()`, moved by `.to(device)`, saved in the `state_dict` — if it is wrapped:

  ```python
  class Scale(nn.Module):
      def __init__(self):
          super().__init__()
          self.factor = nn.Parameter(torch.ones(3))
  ```

- **Make `A` and `B` where the base layer lives.** By the time you wrap a layer, the
  model may already be on the GPU. `torch.zeros(…, device=w.device, dtype=w.dtype)`
  builds a tensor that matches another one.

- **Dropout** goes on `x` as it enters the right-hand road only. The frozen road must
  be left exactly as it was.

`x @ M.T` applies a matrix `M` shaped `(out, in)` to a batch of row vectors `x` shaped
`(…, in)`. That is what `nn.Linear` does inside.

---

## Exercises

1. `full_update_params`, `lora_params` and `update_matrix` in `lowrank.py`.
2. `freeze` and `unfreeze` in `tunelab/lora.py`.
3. `LoRALinear` in `tunelab/lora.py`.

```bash
pytest week-03/day-3 -v
```

The attribute names `base`, `A`, `B` and `scaling` are fixed. Friday's tests compare
your layer, number for number, with the library everybody uses. Conventions matter
there: which of the two matrices is the zero one, which way round they multiply, where
the scale goes. Today's tests check the same conventions, so that Friday is not a
surprise.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Then, on paper, from a blank page: the diagram, with every shape. Then the one-line
formula for the output. Then the count of trainable numbers for a layer with
`d_in = 576`, `d_out = 192`, `r = 8`.

You have a layer. Tomorrow you put it inside a real model, in every place it belongs,
and train through it.
