# Day 1 — One file

> **By the end of today** your model is one file that runs without Python.

At the moment "your model" is two things in two places: a base model on the Hugging
Face Hub, and an adapter on your Drive. To use it you need PyTorch, `transformers`,
`peft`, the right versions of each, and code that knows how to put the two together.

Today you turn that into a single file that a small C++ program can run on a laptop.
Three steps: **merge**, **convert**, **quantize**.

`merge.py` is in this folder. The conversion commands are in `COMMANDS.md`.

---

## Read first

- [ ] `tunelab/lora.py` — your own `merged()` and `merge_lora`, from week 3 · 5 min
- [ ] The [llama.cpp README](https://github.com/ggml-org/llama.cpp) — the opening
      description, and the section on preparing and quantizing models · 15 min

---

## What you need to know

### Merge

You did this by hand in week 3: `W + (alpha / r) · B A` is just another matrix, so
compute it once and put an ordinary layer back where the adapter was.

On Thursday of that week the trade was laid out: keep the adapter separate and you can
swap tasks cheaply; merge and you get a plain model that every other tool understands.
This week you need the second. The converter in the next step knows nothing about
adapters.

With `peft` it is two steps, and you have written both by hand:

| Step | In `peft` | What you called it |
|---|---|---|
| attach the saved adapter to a base model | `PeftModel.from_pretrained` | `inject_lora`, then `load_lora` |
| fold it in and hand back a plain model | `merge_and_unload` | `merge_lora` |

**Merge into a full-precision base.** Last week you trained on a 4-bit base because it
was cheap. Do not merge into that. Load the base model again in ordinary precision and
merge the adapter into *that*: the adapter is a small, exact correction, and it
deserves an exact thing to correct. Quantizing comes afterwards, once, on the merged
result.

Then save the model **and the tokenizer** to one folder. The tokenizer is not an
afterthought: it carries the vocabulary and the chat template, and the next step reads
both.

### Convert: a different kind of file

A Hugging Face model folder is a set of files that Python libraries know how to read.
**GGUF** is one file that holds everything — the weights, the vocabulary, the chat
template, the settings — laid out so that a program can map it straight into memory
and start.

It is the format of **llama.cpp**, a C++ program that runs language models on
ordinary hardware with no Python anywhere. It is what you will serve with tomorrow.

The conversion is done by a script that ships with llama.cpp. Notice what goes into
the file: the chat template travels with the weights. Tomorrow that turns out to
matter a great deal.

### Quantize, again

You quantized six numbers by hand last Monday. Now a tool does it to half a billion.

The levels have names. The ones worth knowing:

| Name | Nominal bits per weight | This model's file | Its real bits per weight |
|---|---|---|---|
| `f16` | 16 | 994 MB: no quantization, half precision | 16 |
| `q8_0` | about 8.5 | 531 MB | 8.6 |
| `q5_k_m` | about 5.7 | 420 MB | 6.8 |
| `q4_k_m` | about 4.8 | 398 MB: the usual default | 6.4 |

The "k" levels are cleverer than last week's even spacing: they give more bits to the
tensors that suffer most from losing them. That is why the numbers are not whole.

Look at the last column. On most models `q4_k_m` really does come to under five bits a
weight. On yours it is over six, and the tool tells you why if you read what it
prints: this model's layers are 896 wide, the 4-bit "k" format needs a width that
divides by 256, and so about half the tensors **fall back** to a 5-bit format. Very
little of your "4-bit" file is 4-bit. The saving from `q5_k_m` to `q4_k_m` is twenty
megabytes. Remember that when you choose, and read your tools' output.

The usual advice is to start with `q4_k_m`. **For a model this small, start with
`q8_0`.** The next paragraph is why.

One warning that applies to you in particular: **small models are hurt more by
quantization than large ones.** A 70-billion-parameter model barely notices four bits.
A half-billion one has less redundancy to hide the damage in. The untuned base model,
served with the course's reference prompt, scores 44% unquantized and 36% at `q4_k_m`:
eight points. A *tuned* model can fare worse, because what fine-tuning added is a
small, precise correction, and that is exactly what coarse rounding destroys: the
course's reference adapter scores 67% in PyTorch, 66% served at `q8_0`, and 45% at
`q4_k_m` — below the untuned baseline — to save 133 megabytes. Make both files, score
both tomorrow, and see what yours does.

### Nothing is shipped until it is re-scored

After today you will have a `.gguf` file. You will not yet know whether it is any
good. Between last Friday's score and this file there are three places things can go
wrong: the merge, the conversion, the quantization. Tomorrow you put the file behind a
server and run the frozen eval through it, and only that number counts.

---

## Exercises

In `merge.py`:

1. `merge_adapter(base_model, adapter_dir)`
2. `save_merged(model, tokenizer, out_dir)`
3. `gguf_bytes(n_params, bits_per_weight)`

```bash
pytest week-06/day-1 -v
```

The tests merge a toy adapter into a toy model and check that the result gives the
tuned answers, is a plain model again, and reloads from its folder with the chat
template intact.

### Then, for real

1. Install llama.cpp. `COMMANDS.md` has the one-line install for each platform.
2. Download your adapter folder from Drive.
3. Merge it into the base model with your `merge.py`, and save to a folder.
4. Convert and quantize, following `COMMANDS.md`.
5. Row in `logs/run-log.md`: the file's size at each level you made, next to what
   `gguf_bytes` predicted.

If llama.cpp will not install on your machine, `week-06/colab.ipynb` does steps 3 and
4 on Colab and gives you a `q8_0` file. You still need the `llama-server` program to
serve it: a prebuilt release from llama.cpp's GitHub page is one download.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

You have a file. Tomorrow it answers the phone.
