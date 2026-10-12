# Day 1 — Four bits, and a library you already wrote

> **By the end of today** you know what is lost when a model is loaded in four bits,
> and you have read `peft`'s configuration as a description of code you already wrote.

This is the week you train the model that goes into your agent. You have the data, the
frozen eval and a baseline. Today is the last piece of setup, and it is also where the
first half of the course pays out.

From today you use libraries: `peft` for LoRA, `transformers`' `Trainer` for the loop.
You will find very little in them that is new.

`quant.py` and `bridge.py` are in this folder.

---

## Read first

- [ ] [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — the abstract and the first paragraph of the introduction. That is enough · 10 min
- [ ] [LoRA conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — `peft`'s own. Read it as a review of week 3, and notice how little of it is news · 15 min

---

## Part one — four bits

### A weight is a 32-bit float

Every number in the model is stored as a float32: 32 bits, good for about seven
significant digits. Half a billion of them is 2 GB.

Most of that precision is wasted. A weight of `0.0372841` and a weight of `0.04` do
almost the same job. If you could store each weight in 4 bits instead of 32, the model
would be an eighth the size.

Four bits can tell apart only sixteen different values. So **quantizing** means
choosing sixteen levels and moving every weight to the nearest one.

### By hand, on six numbers

```
 weights     -1.0   -0.6   0.0   0.2   0.6   2.0
```

Two bits: four levels. Spread them evenly from the smallest weight to the largest:

```
 levels      -1      0      1      2
 code         0      1      2      3
```

The gap between levels — the **scale** — is `(2 − (−1)) / 3 = 1`. Each weight becomes
the code of its nearest level:

```
 weights     -1.0   -0.6   0.0   0.2   0.6   2.0
 code           0      0     1     1     2     3
 comes back  -1.0   -1.0   0.0   0.0   1.0   2.0
```

You store the codes (2 bits each), plus two floats for the whole tensor: the scale and
the lowest value. To use the weights again: `code × scale + low`.

Look at what was lost. `−0.6` came back as `−1.0`, `0.2` as `0.0`. No weight moved by
more than half a step, and with sixteen levels instead of four the step is five times
smaller. Your first exercise measures exactly how much.

Real 4-bit loading is a refined version of this. It quantizes in small blocks, each
with its own scale, and it places its sixteen levels where weights actually cluster
instead of spacing them evenly. Same idea.

### Why a model survives it

A model's answer comes from millions of small numbers added together. Nudge each one
by a small random amount and the errors mostly cancel. The model gets slightly worse.
You will measure how much, on your own eval, in week 6.

### Why you can still train it

Here is the problem. You cannot train a quantized weight: gradient descent needs to
make tiny adjustments, and a weight that can only sit on one of sixteen levels cannot
move by a tiny amount.

You already have the answer, and you built it in week 3.

**LoRA never trains the base weights.** They are frozen. They can be rough. The
adapter's `A` and `B` are separate, small, and kept in full precision, and they do all
the learning.

```
 W        frozen, 4-bit        stored cheaply, never changed
 A, B     trainable, float     a fraction of a per cent of the model
```

That combination is called **QLoRA**, and it is why a model too large to fine-tune on
your hardware can be fine-tuned on your hardware.

An honest note. This week's model is half a billion parameters. A T4 can hold it in
full precision with room to spare; you do not *need* four bits. You are loading it
that way because the next model you meet will be ten times larger, and you should
have done it once on something small enough to check.

---

## Part two — `peft`, read against your own code

Open `tunelab/lora.py` in one window and this table in the other.

| `peft.LoraConfig(...)` | In your `tunelab/lora.py` |
|---|---|
| `r=16` | `rank`: `A` is `(rank, in)`, `B` is `(out, rank)` |
| `lora_alpha=32` | `alpha`: `self.scaling = alpha / rank` |
| `target_modules=["q_proj", "v_proj"]` | `targets`, matched against the last part of each layer's name |
| `lora_dropout=0.05` | `dropout`, on the input of the adapter path only |
| `task_type="CAUSAL_LM"` | nothing: it tells `peft` what kind of model it is wrapping |

| `peft` function | Yours |
|---|---|
| `get_peft_model(model, config)` | `inject_lora(model, targets, rank, alpha, dropout)` |
| `model.save_pretrained("adapter/")` | `torch.save(lora_state_dict(model), ...)` |
| `PeftModel.from_pretrained(model, "adapter/")` | `inject_lora(...)` then `load_lora(model, state)` |
| `model.merge_and_unload()` | `merge_lora(model)` |

You proved in week 3 that the arithmetic is identical, to five decimal places. Today's
second exercise shows the plumbing is too: the same layers get wrapped and the same
number of parameters train.

So why use the library at all? Because the things around the arithmetic are tedious
and easy to get wrong: saving in a format other tools read, working with 4-bit base
weights, cooperating with `Trainer`. You are not delegating the idea. You are
delegating the paperwork, and you know what the paperwork is for.

### Loading in four bits

You will meet this tomorrow, in `week-05/train.py`. It is one configuration object:

| `BitsAndBytesConfig(...)` | Meaning |
|---|---|
| `load_in_4bit=True` | quantize the model's linear layers as they are loaded |
| `bnb_4bit_quant_type="nf4"` | put the sixteen levels where normally-distributed weights are, not evenly |
| `bnb_4bit_compute_dtype=torch.float16` | turn codes back into 16-bit floats when computing |
| `bnb_4bit_use_double_quant=True` | quantize the per-block scales too; a little more saved |

It needs a GPU and the `bitsandbytes` library, which is why none of today's tests use
it. Everything you write today runs on your laptop.

---

## Exercises

1. `quant.py`: `quantize`, `dequantize`, `quantization_error`, `model_bytes`.
2. `bridge.py`: `to_peft_config` and `same_as_mine`.

```bash
pytest week-05/day-1 -v
```

Then, in a shell, with your `quant.py`: take one weight matrix (any `q_proj` from
`tunelab.tiny.tiny_model(259)` will do), and print `quantization_error` at 8, 4, 3 and
2 bits, next to the tensor's own standard deviation. One row in `logs/run-log.md`: at
how many bits is the error a tenth of the spread? At how many is it half?

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Tomorrow a library runs your loop for you. Before it does, list from memory the five
lines of the loop, and for each one, write where you expect to find it in a
configuration object.
