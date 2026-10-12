# Week 4 — Data is the model

> **Destination**
> Decide what "better" means, measure where you start, and build the examples that
> will get you there — before a single training step.

The second half of the course is one job, done properly: take the notes agent from
Agent From Zero, whose brain is a large hosted model, and replace that brain with a
small model you trained yourself.

You will not train anything this week. That is deliberate, and it is the most
transferable habit in the course. Fine-tuning projects rarely fail in the training
loop. They fail because nobody defined success before starting, because the data was
never read, or because the test set leaked. This week is the three things that prevent
those.

**The AI rule changes.** For three weeks the AI was a tutor only. From now on you may
write code with it. The rule that replaces the old one: **you may not commit a line
you could not have written and cannot explain on Friday.**

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Score a model on a tool-calling task, four ways, and measure an untuned one |
| Tue | `day-2/` | Write seeds, multiply them three ways, and reject what does not validate |
| Wed | `day-3/` | Find duplicates, conflicts and imbalance, and decide what to do about each |
| Thu | `day-4/` | Split by seed, freeze the eval set, and render examples exactly as the model sees them |
| Fri | `milestone/` | Deliver the dataset, the frozen eval and the baseline, then defend them |

Each day folder has its own `README.md`. Start there, every time.

Code goes in `tunelab/toolcall.py` (Monday) and `tunelab/dataset.py` (the rest). Data
goes in `data/notes/`.

---

## The task, the model, the agent

**The task:** given a conversation, call one of the notes agent's five tools with the
right arguments — or answer in words.

**The model:** `Qwen/Qwen2.5-0.5B-Instruct`, half a billion parameters, about a
thousandth the size of the model the agent normally uses. See `SETUP.md`.

**The agent:** `starter-agent/`. If you took Agent From Zero you may use your own.

---

## How a day goes

1. **Read** the day's README.
2. **Write** the functions. Tests are in the day folder, as always.
3. **Then do the day's real work**, which is on your own data, and which no test in
   the day folder checks. Friday's tests check it.
4. **Log**: `logs/stuck-log.md`, `logs/signal-log.md`, and from this week
   `logs/run-log.md` matters.
5. **Commit and push.** Every day.

---

## Milestone

A training set of 500 to 1000 examples, an eval set of about 100 that shares no seed
with it and whose fingerprint is committed, and the untuned model's score on that eval
set with your best prompt. Spec in `milestone/README.md`.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *How does a driving examiner make sure every candidate is judged by the same
> standard, and what would go wrong if examiners could adjust the test afterwards?*

Also this week: `ai/planner.md`. It was written for this Monday.

---

## What this week is not about

Training. A big dataset: a few hundred rows you have read beat ten thousand you have
not. Clever generation pipelines. Anything in the "not yet" column of `FENCE.md`.
