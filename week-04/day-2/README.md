# Day 2 — Building examples

> **By the end of today** you have several hundred examples, and you know which of them
> to distrust.

A fine-tuned model is a compressed copy of its training data. Not a metaphor: whatever
your examples do, consistently, it will do. Whatever they do inconsistently, it will
do at random. Whatever they never show, it has no reason to learn.

So the week's real work is here, in a text file, and it is not glamorous. People who
are good at fine-tuning are mostly people who are willing to read their data.

Everything today goes in `tunelab/dataset.py`, in the Day 2 section.

---

## Read first

- [ ] `data/notes/dev.jsonl` again, all twenty, now as a writer rather than a reader · 10 min
- [ ] Your own wrong answers from yesterday's notebook run · 10 min

---

## What you need to know

### Seeds first, by hand

A **seed** is an example you wrote yourself. You will write at least sixty before you
generate anything.

That order matters. You cannot judge a thousand examples of a task you have not
written ten of: you do not yet know where it is hard. Writing them is how you find
out. By the thirtieth you will have hit three or four questions with no obvious
answer — *is "do I have a note on tax?" a search, or a request to list everything?* —
and those questions are the task. A model trained on data from someone who never
noticed them will answer them by coin toss.

What a good seed set has:

- **Every kind**, including `none` and `clarify`. At least eight of each.
- **The hard pairs.** Requests that look like one tool and are another. "What notes do
  I have?" is `list_titles`. "What notes do I have about the dentist?" is not.
- **The agent's own rules.** The agent is told to confirm a title with `list_titles`
  before deleting. So "delete my shopping note", as a first message, is a `list`
  example. `delete_note` is only right once the list of titles is in the conversation.
- **Later turns, not only first ones.** After a tool has returned its result, the
  right move is usually to answer in words. Those are `none` examples, and an agent
  spends half its life in them.
- **Ugly input.** Real people type "pls delete", "NOTE milk", "um, save this". If
  every seed is a polite sentence, that is all your model will understand.

A seed's `seed_id` is its own `id`.

### Three ways to multiply a seed

Sixty examples will not train a model. You need several hundred, and three routes get
you there. All three are free; they cost different things.

**Templates.** Put holes in a seed and fill them from lists:

```
"jot down {title} — {body}"     +     title: milk, eggs, passport …
```

Free, instant, and the label is right *by construction*, because the same value goes
into the text and into the expected arguments. The cost: the wording never changes.
Fifty examples from one template teach one phrasing fifty times.

**A bigger model, through an API**, if you have a key from Agent From Zero. The
starter agent's `llm` layer gives you `model.chat(...)`; wrap it as a function from a
prompt to a reply. You ask it to reword a seed twenty ways. The wording varies
beautifully. The cost: it makes mistakes, and it makes them fluently.

**A bigger model, by hand.** The same prompt, pasted into any chat assistant you have
access to, and the reply saved to a file. Slower, free, and exactly as untrustworthy.

`expand_prompt.md`, in this folder, is the instruction text for the last two.

### The rule that keeps labels true

**The label is made together with the text, never guessed afterwards.**

With templates that is automatic. With a model, the model writes both the request and
the expected call, and you check what comes back before a single row is kept: is it
JSON, does it validate, is it the kind of example you asked for. `expand_with_model`
throws away everything that fails. A generated dataset that was never filtered is a
generated dataset with a few per cent of nonsense in it, and a few per cent is enough.

`validate` is the gatekeeper. It cannot tell you that a label is *right* — that "milk"
is truly the title the user meant. It can tell you that a row is *well-formed*: the
tool exists, the arguments are ones the tool has, a `none` example has words to say.
Everything it cannot check, you check, by reading.

### Read fifty

When you have a few hundred rows, pick fifty at random and read every one, slowly,
asking of each: *if I were the assistant, is this what I should do?*

Count the ones where the answer is no. Write that number down. If it is three of
fifty, your dataset is about 6% wrong and you now know it. Nobody who skips this step
knows theirs.

---

## Exercises

In `tunelab/dataset.py`:

1. `load_jsonl` and `save_jsonl`
2. `validate`
3. `expand_with_templates`
4. `expand_with_model`

```bash
pytest week-04/day-2 -v
```

Then build your data. This is most of the day.

1. Write at least **60 seeds** in `data/notes/seeds.jsonl`, with a `"template"` on the
   ones that have details worth varying. Run `validate` over all of them.
2. Expand them, by any mix of the three routes, to somewhere between **700 and 1100
   rows**. Save as `data/notes/raw.jsonl`. Do not use the twenty rows of `dev.jsonl`.
3. Read fifty. Log the count in `logs/run-log.md`.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

You almost certainly have duplicates, and you almost certainly have at least one pair
of rows that say the same thing and expect different answers. Tomorrow you find them.
