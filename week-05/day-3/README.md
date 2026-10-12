# Day 3 — Reading a run

> **By the end of today** you can look at a run that went wrong and say which of four
> things it was.

Yesterday's run finished, and you have a `history.json`. Maybe it looks fine. Today is
about the runs that do not, because most do not, and the skill that separates people
who can fine-tune from people who can run a fine-tuning script is knowing **where to
look first**.

No new machinery today, and almost no teaching in the live hour. You write one small
function, and then you fix four broken things that somebody else built.

`diagnose.py` and the four broken files are in this folder.

---

## Read first

- [ ] Your own `history.json` from yesterday's run. Plot it or just read the numbers · 10 min
- [ ] Karpathy's [Recipe](https://karpathy.github.io/2019/04/25/recipe/) — the list of
      checks under "Set up the end-to-end training/evaluation skeleton". Count how many
      you have already done in this course · 10 min

---

## What you need to know

### The shapes, again

Week 1, Wednesday: three curves worth knowing by sight. They are the same three, and
there is a fourth.

```
 learning             overfitting            not learning          diverging

 │\                   │\                     │                     │        /
 │ \___ eval          │ \    eval __/        │~~~~~~~~  train      │       /
 │  \____ train       │  \____/              │~~~~~~~~  eval       │    __/
 │                    │   \______ train      │                     │___/
 └──────── steps      └──────── steps        └──────── steps       └──────── steps
```

**Diverging** is new: the loss goes *up*, or becomes `NaN`. Something is throwing the
weights around rather than nudging them.

`diagnose` turns the picture into rules, checked in order. The thresholds — 5% for
"did it learn at all", 10% for "has eval turned" — are this course's, chosen to be
easy to reason about. The order matters: a run that is diverging is not usefully
described as overfitting.

### When to stop

In an overfitting run, the model you want is not the last one. It is the one at the
bottom of the eval curve. That is what checkpoints are for, and `best_step` finds it.

"The best checkpoint is rarely the last" is worth saying aloud, because the script
saves the adapter from the last step. If yesterday's run overfitted, the fix is fewer
epochs — and now you know roughly how many.

### What actually goes wrong, in order

When a fine-tuning run disappoints, people reach for the hyperparameters. It is almost
never the hyperparameters. In rough order of how often each is the cause:

1. **The data.** Wrong labels, conflicting labels, a kind that is missing, a leak.
2. **The template.** Training text and inference text laid out differently.
3. **The labels.** Graded on the wrong tokens: the prompt, the padding, an off-by-one.
4. **The learning rate.** Usually a typo, usually by a factor of ten or more.
5. Everything else, a long way behind.

Each of today's four broken files is one of the first four. Notice which of them a
loss curve would reveal — and which would produce a *beautiful* loss curve.

### A good loss with a bad score

That last case deserves its own name, because it is the one that costs days.

The eval **loss** is measured the way training is: the model is shown the prompt *and*
the right answer, and asked how surprised it was by each next token. The eval
**score** is measured the way the model is used: it is shown the prompt alone and has
to write the answer itself.

A low loss with a poor score means those two are not the same situation. The model is
good at the task it was trained on, and the task it is being used for is slightly
different. Nearly always: the prompt is being built differently in the two places.

### The method

Change **one thing**. Write it in `logs/run-log.md` **before** you look at the result.
Then look.

It feels slow. It is the only way to end the week knowing which change helped. Change
three things at once, get a better number, and you know nothing — including whether
two of them made it worse.

---

## Exercises

### 1. `diagnose.py`

`best_step(history)` and `diagnose(history)`. Six recorded runs are in `histories/`;
look at them before you write anything, and decide by eye what each one is.

### 2. Four broken files

`broken_1.py` to `broken_4.py`. Each is a small, complete piece of somebody's
fine-tuning pipeline, with their complaint at the top and **one mistake**. (In
`broken_3` the one mistake shows up in more than one line. Fixing the line you see
first is not enough, and the test will tell you so.)

For each one:

1. Read the complaint. Before opening the code, write down what you would look at
   first.
2. Find the bug. Run things. Print things. The test tells you what "fixed" means, not
   where the bug is.
3. Fix it, in the file.
4. Write it up in `NOTES.md`: the bug in one sentence, and **what you looked at that
   gave it away**.

```bash
pytest week-05/day-3 -v
```

The four broken tests fail as shipped. That is the starting state.

Run `diagnose` on your own `history.json` from yesterday and put the verdict, and the
best step, in `logs/run-log.md`.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Your model has a loss. It does not yet have a score. Tomorrow it takes the frozen
eval, and you find out whether a week and a half of work beat a paragraph of prompt.
Write your guess down tonight.
