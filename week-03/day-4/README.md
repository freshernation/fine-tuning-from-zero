# Day 4 — Inside the model

> **By the end of today** a pretrained model is learning through thirty lines you wrote,
> and the adapter that holds everything it learned is a file smaller than a photo.

Yesterday you built one layer. Today is plumbing: getting that layer into every place
it belongs inside a real model, training through it, and then deciding what to do with
the result. None of it is hard. All of it is where real bugs live.

Everything today goes in `tunelab/lora.py`.

---

## Read first

- [ ] [LoRA](https://arxiv.org/abs/2106.09685) — **section 4.2**, "Applying LoRA to Transformer", the first two paragraphs · 10 min

---

## What you need to know

### Which layers

A transformer block has a handful of linear layers. Which of them get an adapter?

The paper tried the combinations and found that adapting the attention **query and
value** projections gave most of the benefit for the fewest parameters. That is a
finding about the models they tested, not a law, and people now often adapt more. It
is the usual starting point, and it is yours: `q_proj` and `v_proj`.

Notice what that choice means in the terms you learned in week 2. You are adjusting
*what each position looks for* and *what it says when listened to*, in every block,
and leaving the rest alone.

### Replacing a layer that is buried in a model

The layers you want are nested several levels down:

```
model.layers.0.self_attn.q_proj
```

`named_modules()` walks the whole tree and gives every module with its dotted name:

```python
net = nn.Sequential(nn.Linear(3, 3), nn.Sequential(nn.Linear(3, 3), nn.ReLU()))

for name, module in net.named_modules():
    print(repr(name), type(module).__name__)
# ''     Sequential
# '0'    Linear
# '1'    Sequential
# '1.0'  Linear
# '1.1'  ReLU
```

To *replace* one, you cannot assign to the name. You need its **parent** — the module
that holds it — and the short name it has there. `named_children()` gives a module's
direct children only:

```python
class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 3)
        self.right = nn.Linear(3, 3)

pair = Pair()
setattr(pair, "left", nn.Identity())       # the same as:  pair.left = nn.Identity()
```

`setattr` with a string is what lets you do that for a name you only know at run time.

Two traps, and both have a test.

**Match the whole name.** You are looking for layers called `q_proj`. If you check
whether the target *appears in* the name, then a target of `"q"` matches `q_proj`, and
also anything else with a q in it. Match the last part of the dotted name, exactly.

**Do not change a collection while you walk it.** If you replace modules while looping
over `named_modules()`, you are editing the tree under the loop. Collect what you want
to replace first; replace afterwards.

### Freeze first

Putting adapters in does not, by itself, stop the rest of the model training. If you
forget to freeze it, you are doing full fine-tuning with some extra parameters
attached: every weight moves, the memory bill is back, and nothing looks wrong. The
loss goes down nicely.

So `inject_lora` freezes the whole model before it wraps anything, and one of today's
tests trains through your adapters and then compares **every base weight, bit for
bit,** with a copy taken beforehand.

### The adapter is its own thing

After training, everything the model learned is in the `A`s and `B`s. Nothing else
changed — you have just proved that. So that is all you need to save.

For the real model, with rank 8 on `q_proj` and `v_proj` in all thirty blocks:

```
 q_proj   576 → 576     A (8, 576) + B (576, 8)     9,216
 v_proj   576 → 192     A (8, 576) + B (192, 8)     6,144
                                         per block  15,360
                                       × 30 blocks  460,800 numbers  ≈ 1.8 MB
```

against 135 million numbers and 540 MB for the model. A third of one per cent.

Three consequences worth saying out loud:

- You can keep **many adapters for one base model** — one per task — and each costs
  megabytes.
- An adapter is **useless without the exact base model** it was trained on. It is a
  change; a change to something else is noise.
- An adapter only fits a model with adapters **in the same places, at the same rank**.
  Load a rank-8 adapter into rank-4 slots and the shapes do not match. That must be an
  error, loudly. The dangerous version is a loader that quietly skips what does not
  fit and leaves you evaluating an untrained adapter.

When you save tensors that are still being trained, save **copies**:

```python
snapshot = p.detach().clone()       # a separate tensor that no longer follows p
```

### Merging

Yesterday you checked by hand that these give the same answer:

```
 W x  +  (alpha / r) · B A x          two roads
 ( W + (alpha / r) · B A ) x          one matrix
```

So once training is done, you can do the multiplication once, add the result to `W`,
and put an ordinary `nn.Linear` back where the adapter was. The model is then exactly
the shape it was when you downloaded it, with no trace of LoRA in it, and it gives the
tuned answers.

| | Keep the adapter separate | Merge |
|---|---|---|
| Speed at inference | a little slower: two roads | same as the original model |
| Size on disk | base + a few MB | a whole new model |
| Swap tasks | change a small file | change the whole model |
| Works with tools that expect a plain model | no | yes |

In week 6 you will merge, because the tool that turns your model into a single file
for serving wants a plain model.

Can you un-merge? Think about what you would need to still have.

---

## Exercises

1. `LoRALinear.merged()` — a new method on yesterday's class.
2. `inject_lora(model, targets, rank, alpha, dropout=0.0)`.
3. `lora_state_dict(model)` and `load_lora(model, state)`.
4. `merge_lora(model)`.

```bash
pytest week-03/day-4 -v
```

Do them in that order; the tests are grouped the same way. `isinstance(m, nn.Linear)`
tells you whether a module is a linear layer; note that your `LoRALinear` is *not* one,
though it holds one.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Then open `week-03/colab.ipynb` in Colab, pick the T4, and run it as far as the GPU
check, as you did last Thursday. Tomorrow you run three fine-tunes of a real model and
compare them, and you want the GPU to be there.
