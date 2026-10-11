# Week 1 milestone — a language model, trained by a loop you wrote

No new ideas today. You have written every piece already. The milestone is putting
them together and being able to explain the number that comes out.

---

## Deliver

Open `train_bigram.py`. Two functions.

**1. `run(text, steps=2000, seed=0)`**

Tokenize the text, build the batches, make a `Bigram`, train it with your loop,
generate 200 characters. Return the history, the final validation loss and the sample.

The test trains on `data/tiny.txt` — about a million characters of Shakespeare, 65
distinct ones — and asks for a validation loss **under 2.59**.

**2. `overfit_one_batch(model, x, y, steps=500, lr=0.1)`**

Train on one batch, the same one, every step. Return the final loss.

The test hands you a batch where `a` is always followed by `b`, `b` by `c`, `c` by
`a`, and asks for a loss **under 0.01**.

```bash
pytest week-01/milestone -v                    # the grader, under a minute
python3 week-01/milestone/train_bigram.py      # watch it yourself
```

Then open `week-01/colab.ipynb` in Colab and run it top to bottom. Nothing this week
needs a GPU; you are doing it now so that it is not new in Week 2.

---

## Log it

One row in `logs/run-log.md`: the date, the steps, the final train and validation
loss, and in the note column the first forty characters your model generated.

From Week 4 on, a run that is not in that log did not happen. Start the habit on a run
that takes five seconds.

---

## Three questions for Friday

You will be asked all three. Work out the answers; do not look them up.

**1. Why does the loss start between 4 and 5?**
You know where 4.17 comes from. Yours starts a little above it. What about a table of
random numbers makes it *worse* than knowing nothing?

**2. Why does it stop around 2.5 and go no lower, however long you train?**
Try it: `steps=20000`. Nothing is broken. What would the model need that it does not
have?

**3. Why can the loss reach zero on the `a → b → c` batch, but not on a real one?**
Call `overfit_one_batch` on a real batch from `data/tiny.txt` and see where it stops.
Then say exactly what that floor is measuring.

And the reason the function exists at all: **when a training run misbehaves, this is
the first thing you try.** A model that cannot memorise one batch has a bug in the
model, the loss or the loop — not a tuning problem — and you have just saved yourself
an hour of changing the learning rate.

---

## Done means

- [ ] `pytest week-01 -v` — 20 passed
- [ ] The Colab notebook ran top to bottom
- [ ] A row in `logs/run-log.md`
- [ ] Committed and pushed
