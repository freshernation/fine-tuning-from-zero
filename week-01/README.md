# Week 1 — Training is not magic

> **Destination**
> Train a language model with a loop you wrote yourself, and explain every number it
> prints.

By Friday you will have a model that predicts the next character of Shakespeare. It
will be a bad model — a lookup table with a one-character memory. That is not the
point. The point is the **loop** around it: forward, loss, backward, step, zero. Hold
on to that shape. In week 3 you will use the same loop, unchanged, to fine-tune a real
pretrained model, and in week 5 a library will run it for you and you will know
exactly what it is doing.

Weeks 1–3 are **AI-off for writing code**. The model tutors you (`ai/tutor.md`) and
never authors for you.

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Read the shape of any tensor, and predict the shape an operation gives back |
| Tue | `day-2/` | Compute a gradient by hand, check it against autograd, and take a step downhill |
| Wed | `day-3/` | Write the training loop, and measure honestly whether it is working |
| Thu | `day-4/` | Turn text into numbers and build a model that predicts the next character |
| Fri | `milestone/` | Train it, log it, run it on Colab, then defend it |

Each day folder has its own `README.md`. Start there, every time.

Days 3 and 4 put their code in `tunelab/`, not in the day folder, because you will
import it again next week.

---

## How a day goes

1. **Read** the day's README. All of it, before writing anything.
2. **Predict, then run.** Before you run any line, say out loud what **shape** you
   think comes back. Being wrong here is worth more than being right.
3. **Write** the exercises. In order — they build.
4. **Test** with `pytest week-01/day-N -v`. Green means correct behaviour, nothing more.
5. **Log** anything that stuck you in `logs/stuck-log.md`, and your numbers in
   `logs/signal-log.md`.
6. **Commit and push.** Every day.

---

## Milestone

A bigram model trained on `data/tiny.txt` to a validation loss under 2.59, and a
second run that deliberately overfits a single batch to zero. Spec in
`milestone/README.md`. It must pass its tests **and** survive Friday's defence, which
is a different and harder bar.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *How does a music teacher tell a student who has learned a piece from one who has
> only memorised it?*

It is not on any test. It is how the idea of a validation set stops being about
PyTorch.

---

## What this week is not about

A good model. Speed. GPUs. Anything in the "not yet" column of `FENCE.md` — attention
and transformers above all, however tempting the rest of Thursday's video is. You
will train something small and bad this week and that is correct. A bad model you
understand completely beats a good one you downloaded, and it is not close.
