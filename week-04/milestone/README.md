# Week 4 milestone — a dataset, a frozen eval, a baseline

There is no code to write today. The week's code is done. Today you produce the three
things the rest of the course stands on, and you get the number your fine-tuned model
will have to beat.

---

## Deliver

All in `data/notes/`:

| File | What |
|---|---|
| `train.jsonl` | 500 to 1000 examples |
| `eval.jsonl` | about 100, sharing no seed with train |
| `eval.sha256` | the eval set's fingerprint, committed with it |
| `baseline_predictions.jsonl` | what the untuned model wrote for each eval example |
| `baseline.json` | its report |

The first three you made yesterday. The last two come from Colab.

```bash
pytest week-04/milestone -v
```

These tests grade your **data**: sizes, well-formedness, no duplicates, no conflicts,
balance, no leaks, the frozen fingerprint, and that the score in `baseline.json` is
really the score your predictions give. None of them can be passed by writing code.

### The baseline run

Open `week-04/colab.ipynb` in Colab and run it to the end. It loads the untuned model,
gives it your `prompt.md` and the five tools, runs your frozen eval through it, scores
it with your `tunelab/toolcall.py`, and offers the two files for download. Put them in
`data/notes/` and commit.

**Before you run it, write your guess in `logs/run-log.md`.** The overall accuracy,
and which kind will be worst.

Then the row: the baseline accuracy, the four rates, and the worst kind.

### The datasheet

Fill in `DATASHEET.md`. One page: what the data is, how it was made, what you checked,
what you decided, what it does not cover.

---

## Three questions for Friday

**1. Which kind does the baseline fail most, and what would you add to the prompt
before reaching for training?**
Try it, if you have time. If a sentence in the prompt fixes it, that is a finding.

**2. Show me two rows a reasonable person would label differently.**
There are some. If you cannot find them you have not read your data.

**3. Next week your tuned model scores 100% on this eval. Name three ways that could
be a bug rather than a triumph.**

---

## Done means

- [ ] `pytest week-04 -v` — 111 passed
- [ ] The five files in `data/notes/`, committed
- [ ] `eval.jsonl` and `eval.sha256` in a commit of their own, made on Thursday
- [ ] `DATASHEET.md` filled in
- [ ] Your guess, then the baseline, in `logs/run-log.md`
- [ ] Pushed
