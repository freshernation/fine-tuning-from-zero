# Day 4 — Split, freeze, render

> **By the end of today** there is a set of examples your model will never be trained
> on, and you could not cheat on it if you wanted to.

Three short jobs, each of which protects you from a particular way of fooling
yourself. In week 5 you will be tired, your model will be at 78%, and one of these
will be the only thing standing between you and a number that is not true.

Everything today goes in `tunelab/dataset.py`, in the Day 4 section.

---

## Read first

- [ ] Karpathy's [Recipe](https://karpathy.github.io/2019/04/25/recipe/) — the first
      section, "Become one with the data". You have now done what it describes · 10 min

---

## What you need to know

### Leakage

You know why a model is tested on data it has not seen. Here is how that gets broken
without anyone intending it.

You wrote a seed: *"jot down milk — two litres"*. You expanded it into fifteen
rewordings. You shuffled all your rows and took 100 for the eval set. Eight of the
fifteen went to training and seven to eval.

Your model now scores beautifully on those seven. It has not learned to add notes. It
has seen eight near-copies of the question. The eval is measuring memory, the number
is high, and **nothing about it looks wrong.**

So you do not split rows. **You split seeds.** Every example made from one seed goes
to the same side, whole. The eval set then asks: does the model handle *kinds of
request it was never shown*? That is the question you care about.

`find_leaks` is the check, and it checks both ways a leak can happen: a shared
`seed_id`, and two rows that turn out to say the same thing under different seeds.

One more requirement: the eval set must contain every kind. An eval with no `clarify`
examples cannot tell you your model has stopped asking questions.

### Freezing

Once the eval set exists, it must not change. Not when you find an example in it that
seems unfair. Not when your model keeps failing one row and you are sure the row is
wrong.

Especially not then. Every edit made after seeing a result moves the eval toward what
your model happens to do, and you will always have a good reason. A hundred good
reasons later you have an eval your model passes and nobody else's would.

A **fingerprint** makes the promise checkable: one short string computed from the
whole file. Change a character anywhere and the string changes.

```python
import hashlib
hashlib.sha256("delete it".encode("utf-8")).hexdigest()[:16]     # 'cb00afa235d8f25d'
hashlib.sha256("delete it.".encode("utf-8")).hexdigest()[:16]    # 'bd6edfe3c01615e8'   one full stop later
```

You record the fingerprint today and commit it. From now on every script that trains
or scores checks it first.

The fingerprint must depend on **what the file says**, not on how it happens to be
laid out. Re-saving the file with its rows in a different order is not a change. So
sort before you hash.

And when the check fails, it must fail in the right direction. A check that responds
to a changed file by quietly recording the new fingerprint is not a lock. The message
it gives tells you to put the file back.

If you find a genuinely wrong example in the eval set later: write it down, leave it
in, and argue it on Friday. A fixed, slightly imperfect ruler beats a perfect one that
moves.

### Rendering

Monday of week 3: a chat is one long string. Now you make the string for real, with
the five tools in it.

```python
tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True, tokenize=False)
```

does the flattening. Look at what comes back for one of your examples: the system
text, then the five tool descriptions in a block the model was trained to read, then
the turns, then the opening of the assistant's turn with nothing after it.

That is the **prompt**. What the model should write next is the **completion**: a tool
call in the model's own syntax, or a sentence, and then the end-of-turn marker.

Two rules.

**The completion ends with the end-of-turn marker.** A model trained on answers with
no ending learns to answer and then carry on writing.

**There is one function that builds the prompt, and everything uses it.** Training,
the baseline, the eval, and in week 6 the server. The classic silent failure in
fine-tuning is a prompt that is laid out one way when training and slightly another
way when the model is used: a missing newline, the tools in a different place. The
loss is fine. The eval is terrible. It can take days to find.

Today's strongest test guards against it: it puts the right answer into the
conversation, asks the chat template itself to write the whole exchange out, and
checks that *your prompt plus your completion* is the same string, character for
character.

---

## Exercises

In `tunelab/dataset.py`:

1. `split_by_seed`
2. `find_leaks`
3. `fingerprint`, `freeze`, `check_frozen`
4. `render_prompt` and `render_completion`

```bash
pytest week-04/day-4 -v
```

Then, for your own data:

1. `split_by_seed` on `data/notes/clean.jsonl`, with `eval_size=100`. Save
   `data/notes/train.jsonl` and `data/notes/eval.jsonl`.
2. `find_leaks`. It must be empty.
3. `balance_problems` on the training side. It must be empty.
4. **`freeze`.** Commit `eval.jsonl` and `eval.sha256` together, in a commit of their
   own. That commit is the moment the ruler was made.
5. Print `render_prompt` and `render_completion` for three of your examples and read
   them as the model will.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Tomorrow the untuned model takes your frozen eval. Before it does, write down what you
think it will score.
