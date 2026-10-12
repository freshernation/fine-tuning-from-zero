# Week 3 milestone — Project 1: LoRA, by hand

The first of the course's three projects. It proves one thing: **you understand
fine-tuning rather than calling it.**

There are three parts. The first two run on your laptop and are graded by tests. The
third is a real model on a real GPU, graded by what you write about it.

---

## Part 1 — Is your LoRA the library's LoRA?

Nothing to write. `tunelab/lora.py` is already finished; this is its exam.

```bash
pytest week-03/milestone/test_parity.py -v
```

`peft` is the library everybody uses for LoRA. Each test builds the same adapters
twice — once with `peft`, once with your thirty lines — gives both the same `A` and
`B`, and checks that the outputs, the gradients and the merged weights agree to five
decimal places.

If one fails, the message says which. The cause is always a **convention**: which
matrix starts at zero, which way round they multiply, where `alpha / rank` goes. Your
layer can be internally consistent, pass Wednesday's tests and still differ from the
world's. Fix `tunelab/lora.py`, not the test.

This is the only place in the first three weeks where `peft` appears. It is the answer
key, not a tool.

## Part 2 — Three ways, one function

`run()` in `compare.py`. Its docstring is the spec. It sets a model up for one of
three methods — full fine-tuning, last block only, LoRA — then counts, measures,
trains and measures again. Every piece it needs you have already written.

```bash
pytest week-03/milestone/test_compare.py -v
```

## Part 3 — The real model

Open `week-03/colab.ipynb` in Colab with a T4 and run it top to bottom, about twenty
minutes. It downloads the real 135-million-parameter model and uses **your**
`compare.py` and **your** `tunelab/lora.py` to teach it Shakespeare three ways, then
LoRA at four ranks.

Before you run cell 7, write your prediction in `logs/run-log.md`. It asks for one.

Then fill in `WRITEUP.md`: two tables, the adapter's size, two samples, three
questions. One page.

---

## Three questions for Friday

**1. Why is `B` zero at the start, and why is `A` not?**
Both halves. The second half is the one people cannot answer.

**2. What exactly is in the adapter file, and what is not?**
If I gave you only that file, what else would you need before it was any use?

**3. After merging, could you un-merge?**

And, on a blank page: **the LoRA diagram, with every shape, drawn inside the block
diagram from last week.** Show where in the block the adapters sit.

---

## Done means

- [ ] `pytest week-03 -v` — 62 passed
- [ ] `pytest week-01 week-02 -v` — still 82 passed
- [ ] The Colab notebook ran top to bottom, including the parity tests in cell 3
- [ ] `WRITEUP.md` filled in, with no blanks
- [ ] A row per method in `logs/run-log.md`, and your prediction, written first
- [ ] Committed and pushed
