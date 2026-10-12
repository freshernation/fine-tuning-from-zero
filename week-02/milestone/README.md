# Week 2 milestone — your GPT, trained on a GPU, killed and resumed

No new ideas today. You wrote every part this week. Today you train it properly, on
hardware that will not wait for you, and you prove that losing the session costs you
almost nothing.

---

## Deliver

### 1. `run()` in `train_gpt.py`

One function. Tokenize, move to the device, build the model, train **through
`train_in_chunks`**, generate. Its docstring is the spec.

```bash
pytest week-02/milestone -v                    # the grader, about half a minute
python3 week-02/milestone/train_gpt.py         # watch the small one train
```

The tests train the `SMALL` model — about 64,000 parameters — for 400 steps on your
laptop, and ask for a validation loss **under 2.26**. Last week's model could not get
under about 2.5 however long you trained it. Yours now can, in ten seconds.

### 2. The real run, on Colab

Open `week-02/colab.ipynb` in Colab with a T4 and run it top to bottom. It trains the
`BIG` model — about 1.8 million parameters — for 3000 steps.

- Target: a validation loss **under 1.66**.
- **Kill it once, on purpose**, after the notebook has printed its line for step
  1000, and run the training cell again. It must carry on, not start over. If it starts over, your checkpointing has
  a bug that your tests did not find; fix it before you go on.

Start early in the day. A GPU is not always free the moment you ask.

### 3. Write it down

- A row in `logs/run-log.md`: parameters, steps, final train and validation loss, and
  in the note column the step you killed it at.
- `week-02/milestone/sample.txt`: the 300 characters your big model generated.

---

## Three questions for Friday

Work out the answers; do not look them up.

**1. Where are most of the parameters?**
Count the big model's pieces separately: the two embedding tables, one block, the
final layer. Then, inside one block: attention against the MLP. Which is biggest, and
could you have predicted it from the diagram?

**2. What would the sample look like with `block_size=8`?**
Do not guess. Change `BIG` (keep the rest), train the small number of steps you can
afford, and compare the text. Use a **different checkpoint path**: a checkpoint from
one shape of model will not load into another. Then say what the model can no longer do.

**3. Your run was killed at step 1000-and-something. What exactly was lost?**
Be precise: which steps had to be done again, and what else restarted that is not in
your checkpoint at all?

And one that is not a question: **you will draw the block from memory.** Practise.

---

## Done means

- [ ] `pytest week-02 -v` — 55 passed
- [ ] `pytest week-01 -v` — still 27 passed
- [ ] The Colab run finished under the target, having been killed and resumed
- [ ] A row in `logs/run-log.md`, and `sample.txt`
- [ ] Committed and pushed
