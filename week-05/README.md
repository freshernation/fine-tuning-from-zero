# Week 5 — The real run

> **Destination**
> Train an adapter that beats your own best prompt on examples it has never seen, and
> be able to say why you believe the number.

This is the week the model gets trained. After four weeks of preparation, the training
itself takes about ten minutes of GPU time and one command.

That ratio is the lesson. The work was the loop you understand, the adapter you wrote,
the eval you froze and the data you read. What is left this week is the part people
think fine-tuning *is* — and then the part that actually decides whether it worked:
reading the run, scoring it honestly, and going back to fix the data.

You use libraries from now on: `peft` and `transformers`' `Trainer`. You may write
code with an AI. The rule: **you may not commit a line you could not have written and
cannot explain on Friday.**

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Say what 4-bit loading loses, and read `peft`'s config as your own week-3 code |
| Tue | `day-2/` | Mask the loss to the answer, and run a real fine-tune you can explain line by line |
| Wed | `day-3/` | Read a loss curve, pick the best checkpoint, and diagnose four broken runs |
| Thu | `day-4/` | Score the tuned model, see what moved, test whether the gain is real — and fix the data |
| Fri | `milestone/` | **Project 2** — beat the baseline by ten points, with the evidence on one page |

Each day folder has its own `README.md`. Start there, every time.

Code goes in `tunelab/sft.py` (Tuesday) and `tunelab/analysis.py` (Thursday). Two
scripts are **given**, for reading: `week-05/train.py` and `week-05/predict.py`.

---

## The run log

`logs/run-log.md` matters this week more than any other file. One row per run,
**written before you look at the result**, saying the one thing that changed. By
Friday you will have done six or eight runs, and the log is the only way to know which
change did what.

Failed runs stay in. The milestone asks to see one.

---

## Milestone — Project 2

On your frozen eval set: tuned accuracy at least ten points above your baseline, with
a 95% interval on the gain that excludes zero. Spec in `milestone/README.md`.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *How does a sports coach tell a real improvement from a good day?*

---

## What this week is not about

Finding the perfect hyperparameters. A bigger model. Trying random seeds until the
number clears the bar. Anything in the "not yet" column of `FENCE.md`. The defaults in
`train.py` are good enough to pass; if you are not passing, the answer is in your data,
and Thursday tells you how to find it.
