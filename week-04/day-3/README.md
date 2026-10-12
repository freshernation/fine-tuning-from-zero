# Day 3 — Cleaning

> **By the end of today** your dataset is smaller and better, and you can say what you
> removed and why.

Yesterday you made data. Today you take some of it away. It will feel like going
backwards, and it is the most reliable way there is to improve a fine-tuned model:
more reliable than a bigger model, a better rank, or another epoch.

Everything today goes in `tunelab/dataset.py`, in the Day 3 section.

---

## Read first

- [ ] `week-04/day-3/dirty.jsonl` — forty rows with problems planted in them. Before
      you write any code, find as many as you can by eye, and write the ids down · 20 min

---

## What you need to know

### Why a tuned model is less forgiving than a prompted one

A prompted model reads your instructions and uses judgement. A fine-tuned model has no
instructions. It has examples, and it reproduces them — including the mistakes, and
with more conviction than you would like. If 5% of your `delete` examples have the
wrong title, you have trained a model to get the title wrong about 5% of the time, on
purpose.

### Duplicates

Templates draw the same values twice. A generating model repeats itself. Two rows that
are the same conversation, give or take capitals and spacing, are one example counted
twice.

In the training set, that tilts the model toward whatever happened to be repeated. If
one of the pair lands in training and the other in the eval set, it is worse: the eval
is then measuring memory. So duplicates go, before the split.

"The same", here, means the **whole conversation** is the same after normalising —
not only the last message. "Delete it", as an opening message, and "delete it" after
the assistant has just listed three notes are different situations with different
right answers.

### Conflicts

A conflict is a duplicate where the two rows **disagree about the answer**.

```
"do I have a note on tax"   ->   find_notes  word="tax"
"do I have a note on tax"   ->   list_titles
```

These are the most valuable thing you will find today. Each one is a place where
*you* have not decided what the assistant should do — and if you have not decided,
the model certainly cannot learn it.

That is why `clean` refuses to resolve them. Dropping one at random would make the
file tidy and leave the question unanswered. It raises an error with the ids, and a
person — you — opens the file, decides what the rule is, and fixes every row the rule
touches. Then writes the rule down, because it will come up again.

### Balance

Count the rows of each kind. Three things go wrong.

**A kind is nearly missing.** `dirty.jsonl` has no `delete` examples at all. A model
trained on it would not be bad at deleting; it would not know deleting existed.

**Too few refusals.** If nearly every row ends in a tool call, the model learns that
the answer to any message is a tool call. Say "thanks" and it will list your notes.
`none` and `clarify` together need to be a real share of the data.

**One seed dominates.** If a single template supplied a fifth of your rows, a fifth of
your dataset is one sentence.

The thresholds in `balance_problems` — 8% per kind, 15% for the refusals together, 5%
per seed — are this course's choices for this task, not laws. You should be able to
say why each exists.

### The pass no function makes

After the code has run, read again. Fifty more rows, from the cleaned file. You are
looking for what code cannot see: a title that is not what the user meant, a body the
user never gave, a `clarify` that should have been a `find`.

---

## Exercises

In `tunelab/dataset.py`:

1. `normalise`
2. `find_duplicates` and `find_conflicts`
3. `balance` and `balance_problems`
4. `clean`

```bash
pytest week-04/day-3 -v
```

Then clean your own data.

1. Run `clean` on `data/notes/raw.jsonl`. It will probably raise. Resolve each
   conflict by hand, and note the rule you chose.
2. Save the result as `data/notes/clean.jsonl`.
3. Run `balance_problems` on it. If a kind is short, go back to yesterday: write more
   seeds of that kind, expand, clean again.
4. One paragraph in `logs/run-log.md`: how many rows in, how many out, how many of
   each problem, and every rule you had to decide.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Your file is clean. Tomorrow you cut it in two, and one of the halves gets locked away
from you for the rest of the course.
