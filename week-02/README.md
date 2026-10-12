# Week 2 — A transformer you wrote

> **Destination**
> Build a GPT from parts you wrote, train it on a GPU, and draw it from memory.

Last Friday your model stopped at a loss of about 2.5 and you could say exactly why:
it could see one character. This week you give it eyes. By Friday the same loop, on
the same text, with a model you built, gets well under that — and writes something
that looks like a play.

Nothing about *training* changes this week. `tunelab/loop.py` is not edited. What
changes is the thing inside the loop. That is worth noticing: the loop that trained a
lookup table trains a transformer, and in week 3 it will fine-tune a real one.

Weeks 1–3 are **AI-off for writing code**. The model tutors you (`ai/tutor.md`) and
never authors for you. This is the week that rule is hardest to keep, because the
whole solution is one search away. A transformer you copied is worth nothing to you
in week 3.

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Give every token a vector that says what it is and where it is |
| Tue | `day-2/` | Let a position choose which earlier positions to listen to |
| Wed | `day-3/` | Build a transformer block, and draw it |
| Thu | `day-4/` | Stack a GPT, move it to a GPU, and checkpoint a run |
| Fri | `milestone/` | Train it on Colab, kill it, resume it, then defend it |

Each day folder has its own `README.md`. Start there, every time.

The model goes in `tunelab/attention.py` and `tunelab/gpt.py`; checkpoints in
`tunelab/checkpoint.py`. You will import all three next week.

---

## How a day goes

1. **Read** the day's README. All of it, before writing anything.
2. **Predict, then run.** Before you run any line, say what **shape** comes back.
3. **Write** the exercises. In order — they build.
4. **Test** with `pytest week-02/day-N -v`.
5. **Log** anything that stuck you in `logs/stuck-log.md`, and your numbers in
   `logs/signal-log.md`.
6. **Commit and push.** Every day.

The Karpathy video is assigned **after** each day's exercises, not before. Watching
him write the thing you are about to write is the fastest way to finish the week
having learned nothing.

---

## Milestone

Two runs. A small GPT trained on your laptop to a validation loss under 2.26, graded
by tests. And a bigger one — just under two million parameters — trained on a Colab
GPU to a validation loss under 1.66, with checkpoints on your Drive, killed once on
purpose and resumed. Spec in
`milestone/README.md`.

On Friday you will also draw the block, on a blank page, from memory.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *How does an editor decide which earlier sentences a new sentence has to agree
> with?*

---

## What this week is not about

A model that writes well. Speed tricks. Pretrained anything. Everything in the "not
yet" column of `FENCE.md`. Your GPT will produce Shakespeare-shaped nonsense, and that
is exactly as good as two million parameters and a million characters can do. The
point is that you can open any transformer after this week and recognise every part.
