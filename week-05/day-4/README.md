# Day 4 — Evaluate, and fix the data

> **By the end of today** you know not just whether your model got better, but which
> examples changed, and what to do about the ones that did not.

This is the day the course has been pointed at since Monday of last week. Your tuned
model takes the frozen eval. You get a number, and you put it beside the baseline.

Then the real work starts, because a number is not a result.

Your code goes in `tunelab/analysis.py`.

---

## Read first

- [ ] Your `DATASHEET.md` from last Friday, and the prediction you wrote down last night · 5 min

---

## What you need to know

### Scoring the tuned model

`week-05/predict.py` loads the base model, attaches your adapter, and runs the frozen
eval through it, using the same prompt file, the same generate function and the same
scorer as the baseline. Read it; it is short. One line in it has a comment worth
taking seriously: the line that attaches the adapter. Leave it out and you are scoring
the untuned model, and the result will look exactly like your baseline, and you will
conclude that fine-tuning did nothing.

That is why its report records an **adapter fingerprint** and an **eval fingerprint**.
A score should always be traceable to the exact model and the exact data that produced
it. On Friday the tests check that yours is.

### One number is not a result

Suppose the baseline was 55% and the tuned model is 78%.

**By kind.** Is that 78% everywhere, or 95% on four kinds and 20% on `clarify`? The
second is a model with a hole in it, and you would rather know now.

**By failure type.** Of the wrong answers: unreadable output, the wrong tool, the right
tool with wrong arguments, a call when it should have kept quiet, silence when it
should have called? Five failures, five different causes. `error_table` crosses the
two: for each kind, how many of each failure.

**By what moved.** 78% could be "fixed 23 and broke none". It could be "fixed 35 and
broke 12". Same number, very different models: the second has learned some things and
*unlearned* others, and the twelve it broke are the most interesting examples you
have. `changed` lists the ids in each direction.

### Is it real?

Your eval set is a hundred examples. Move ten of them and you have moved ten points.
How much of a 23-point gain is the model, and how much is which hundred examples you
happened to pick?

You cannot draw a new eval set from the world. You can do the next best thing, which
is to **resample the one you have**. Draw a hundred examples from your hundred, at
random, *with replacement* — some twice, some not at all. Compute the gain on that
draw. Do it two thousand times. You now have two thousand plausible gains, and the
range the middle 95% of them fall in is an honest "somewhere between":

```
 gain: +23 points   (95% interval: +12 to +33)
```

If that interval sits clear of zero, you have an improvement. If it straddles zero —
*+3 points (−6 to +11)* — you have a coin toss, however pleased you were.

It is called the **bootstrap**, and it needs nothing but a loop and a random number
generator.

**Paired** matters. Both models answered the *same* examples, so resample the examples
once and score both models on that same draw. Resample the two separately and you
throw away the fact that on most examples they agree, and the interval comes out far
too wide.

For this course, Project 2's bar is: **at least ten points above the baseline, with
the low end of the interval above zero.**

### Then: the loop that improves a model

1. Read the **still-wrong** examples. All of them. In full.
2. Ask what they have in common. A phrasing? A kind? Requests with two details in
   them? Titles with more than three words?
3. Go and look for that pattern **in your training data**. Usually one of three things
   is true: there are no examples of it, there are very few, or the ones there are
   disagree with each other.
4. **Fix the data.** Write the seeds that were missing. Resolve the disagreement.
5. Train again. One change. Logged first.

That loop — not the choice of rank, not a third epoch — is what makes fine-tuned
models good. When this week's milestone asks for "one data fix, with its before and
after", this is what it means.

### What you may not do

The still-wrong examples will include one or two where you look at the expected answer
and think: *that is unfair. My model's answer is arguably better.*

You may be right. **You still do not edit the eval set.** Write the id down, say why
in `RESULTS.md`, and argue it on Friday. If the eval set changes whenever your model
disagrees with it, it is measuring your patience. `check_frozen` will stop you; do not
go round it.

You may fix the *training* data as much as you like. That is the whole idea.

### When a hyperparameter is the right answer

Sometimes it is. The test is whether **yesterday's curve** says so:

- eval loss turned upward: fewer epochs.
- loss barely moved: check the learning rate, and check that anything was trainable.
- loss still falling steeply at the end: another epoch is a reasonable experiment.

If the curve looks healthy and the score is poor, the answer is not in the
hyperparameters. Changing them anyway is a way of feeling busy.

---

## Exercises

In `tunelab/analysis.py`:

1. `failure_type(example, prediction)`
2. `error_table(examples, predictions)`
3. `changed(examples, before, after)`
4. `paired_bootstrap(before_correct, after_correct, n=2000, seed=0)`

```bash
pytest week-05/day-4 -v
```

The last test trains and scores a toy model end to end. It takes about a minute.

### Then, for real

In `week-05/colab.ipynb`, the section **"score it"**: it runs `predict.py` on your
run, then your four functions on the result against your baseline, and prints the
still-wrong examples in full.

Then do one turn of the loop above. One.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`, and two rows to `logs/run-log.md`: the run you
scored, and the run after your data fix. If either failed — worse than the baseline,
diverged, scored the wrong file — **mark it failed and leave it in.** Tomorrow's
milestone asks to see one.
