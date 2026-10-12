# Day 2 — The first real run

> **By the end of today** a library has run your loop for you, on your data, and you can
> point at every part of it you have written yourself.

Today you train the model. It is, deliberately, an anticlimax: one new idea, forty
lines of your own code, and then a script somebody else wrote, which you read rather
than write.

The reading is the exercise. `week-05/train.py` is about two hundred lines, and there
should be nothing in it you cannot map to something you built in the first three
weeks.

Your code goes in `tunelab/sft.py`.

---

## Read first

- [ ] [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) — the opening paragraphs, before the class reference · 10 min
- [ ] `tunelab/loop.py` — your own, from week 1. Twenty lines. Have it open all day.

---

## The map

| `week-05/train.py` | What you called it |
|---|---|
| `Trainer(...).train()` | `train(model, get_batch, steps, lr)` — week 1 |
| inside it: forward, loss, backward, step, zero | the five lines — week 1 |
| `eval_strategy="steps"`, `eval_steps=20` | `eval_every`, and `estimate_loss` — week 1 |
| `save_steps`, `resume_from_checkpoint` | `train_in_chunks`, `save`, `load` — week 2 |
| refusing to resume under changed settings, or to write over a finished run | new: a changed run is a new run |
| `get_peft_model(model, config)` | `inject_lora` — week 3 |
| `model.save_pretrained("adapter")` | `lora_state_dict` — week 3 |
| `check_frozen(...)`, first thing | yours — week 4 |
| `tokenize_example`, `collate` | yours — today |

Two things `Trainer` does that your loop did not, both small: it changes the learning
rate as it goes (starting high and sliding to zero), and it saves the optimizer's
state in each checkpoint, which your week 2 version honestly admitted it did not.

---

## The one new idea: train on the answer, not the question

In weeks 1 to 3 you trained on running text, and every position was a lesson: predict
the next token, everywhere.

Now each example has two parts, and they are not equally interesting:

```
 ┌──────────────────────── prompt ────────────────────────┐┌──── completion ────┐
  system text · five tool descriptions · user's message ·   <tool_call>{...}
  "<|im_start|>assistant\n"                                 </tool_call><|im_end|>
```

With a reasonable system prompt, the prompt is about nine hundred tokens. The answer
is about twenty.

If you train on every position, about ninety-eight per cent of the loss is the model
learning to recite your system prompt and the tool descriptions — text that is always
given to it and that it never has to produce. The part you care about is a rounding
error.

So you tell the loss function to **ignore** the prompt. The model still *reads* every
token — attention needs them — but it is only *graded* on the answer.

```
 input_ids    [ p  p  p  p  p  p  p  a  a  a  a ]
 labels       [ –  –  –  –  –  –  –  a  a  a  a ]        – is -100
```

`-100` is a special value: `cross_entropy` skips any position whose label is `-100`.
You have used `cross_entropy` since week 1; this is a feature of it you have not
needed until now.

Three things to get exactly right, and each has a test:

- **The mask ends exactly where the prompt ends.** One token late and the model never
  learns to produce the first token of its answer. One token early and it is graded on
  the last token of the prompt.
- **The answer includes the end-of-turn token.** Otherwise the model is never taught
  to stop.
- **The prompt is built by `render_prompt`.** The same function the baseline used.
  One function, used everywhere: you were told why on Thursday.

### A trap you have met from the other side

In week 3 you were warned not to pass `labels=` to a Hugging Face model, because it
shifts them by one position and your batches were already shifted.

`Trainer` *does* pass `labels=`, and here that is right: `input_ids` and `labels` line
up position for position, unshifted, and the model does the shifting. Same fact, other
side. If you find yourself shifting anything by hand today, stop.

### Padding

Examples have different lengths; a batch is a rectangle. Shorter rows are filled out
with a padding token.

- Pad on the **right**. The answer must stay attached to its prompt.
- Padded positions get `attention_mask` 0: do not look at these.
- Padded positions get label `-100`: do not learn these. Forget this and the model is
  trained to produce padding.

---

## The numbers that matter on a T4

**Batch size and accumulation.** A T4 cannot hold many nine-hundred-token examples at
once. So you run a small batch, do *not* step, run another, and step only after
several: the gradients add up, which you discovered by accident in week 1 and are now
using on purpose. `batch_size=4` with `accumulation=4` behaves like a batch of 16 and
costs the memory of 4.

**The four you might change**, with the direction that hurts:

| | Default | Too much |
|---|---|---|
| learning rate | `2e-4` | the loss jumps around or goes up |
| epochs | `2` | eval loss turns and climbs: memorising |
| rank | `16` | more to store, rarely better on a task this narrow |
| max length | `1536` | out of memory. Too *little*: examples silently dropped — the script counts them, so read the count |

Leave them alone for the first run. A first run with the defaults is a measurement. A
first run with your own guesses is a guess.

---

## Exercises

In `tunelab/sft.py`:

1. `tokenize_example(tokenizer, example, system, max_len)`
2. `collate(rows, pad_id)`
3. `effective_batch_size` and `steps_per_epoch`

```bash
pytest week-05/day-2 -v
```

The last three tests run `week-05/train.py --smoke`: three steps, on a toy model, with
your `tokenize_example` and `collate` and yesterday's `to_peft_config`. They take a
minute. They check that your pieces and the script fit together before you spend GPU
time finding out that they do not.

### Then read `train.py`

Top to bottom, with `tunelab/loop.py` beside it. Answer these in
`logs/signal-log.md`, a line each:

1. Which line makes sure the eval set is the one you froze, and what happens if it is
   not?
2. Where is the system prompt read from? Is it the same file the baseline used?
3. Where would you look, after a run, to learn how many examples were dropped for
   being too long?
4. If Colab disconnects at step 60, what does the next run of the same command do?
5. Which three files does a finished run leave behind, and which of them is the model?

### Then run it

```bash
python3 week-05/train.py --smoke --out runs/smoke
```

on your laptop, against your real data. (On a Mac, the script trains on the CPU even
without `--smoke`: Apple's GPU is not reliable for this. Real runs belong on Colab.) Then open `week-05/colab.ipynb` and run it as
far as **"the first run"**. Before the training cell, it asks you to write a row in
`logs/run-log.md`. Write it first.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

You have a `history.json`. Tomorrow you learn to read it — and four others that went
wrong.
