# Week 5 milestone — Project 2: an adapter that beats the baseline

The second of the course's three projects. It proves: **you can make a model
measurably better at one thing, and say why you believe the number.**

Nothing new to write. Today you finish the loop you started yesterday, and you put
your results on one page.

---

## The bar

On your frozen eval set:

- tuned accuracy is **at least 10 points** above your baseline, and
- the low end of a **95% paired-bootstrap interval** on the gain is above zero.

Both are checked by a test, from your prediction files. Neither is typed in by hand.

## Deliver

| | |
|---|---|
| `data/notes/tuned_predictions.jsonl`, `tuned.json` | written by `week-05/predict.py` from your best run |
| `logs/run-log.md` | every week-5 run, including at least one marked **failed** |
| `week-05/milestone/RESULTS.md` | one page |

```bash
pytest week-05/milestone -v
```

The tests check that your eval set is still the frozen one; that `tuned.json` is the
score your predictions really give; that the score is tied to a real training run on
that eval set, and not to the toy model; that the tuned predictions are not simply the
baseline's; the two halves of the bar; that your run log has a failed run in it; and
that `RESULTS.md` has no blanks.

### About the failed run

It is required, and that is not a trick. Nobody gets this right first time, and a log
with only successes in it is a log that has been tidied. What a failed run taught you
is usually the most useful thing on the page. In the Note column, start with
`failed:` and say what happened.

If, honestly, nothing failed: train one run for six epochs, score it, and write up
what overfitting did to which kinds.

### If you cannot clear the bar

Then you have a finding, and you write it up as one. Say what you tried, what the
still-wrong examples have in common, and what you believe is in the way. The test will
fail; your instructor reads `RESULTS.md`. A clear account of why fine-tuning did not
beat a prompt on this task is worth more than a 10.1-point gain obtained by trying
seeds until one worked.

---

## Three questions for Friday

**1. Your eval went up by N points. How do you know it is not leakage?**
Not "I ran `find_leaks`". What would a leak look like in *these* results, and why is
that not what you see?

**2. What is the model still worst at, and is that a data problem or a capacity
problem?**
What experiment would tell the two apart?

**3. Which single training example would you most like to add, and why that one?**

---

## Done means

- [ ] `pytest week-05 -v` — 68 passed
- [ ] The three deliverables, committed
- [ ] Your adapter saved on Drive, with its `run.json` and `history.json`
- [ ] Pushed
