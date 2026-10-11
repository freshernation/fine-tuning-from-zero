# Fine-Tuning From Zero

Six weeks from your first tensor to a fine-tuned model you trained, measured, served
and can explain.

You need working Python, and you need to have called a model through an API. You do
not need to have trained anything, ever. You do need about four hours a day.

Everything in this repo is aimed at one outcome: **you can write LoRA on a whiteboard,
and a model you tuned is running inside your own agent, beating the prompted baseline
on an eval you froze before training.** (The agent is the notes agent from Agent From
Zero. If you did not take that course, a starter copy arrives in week 4.) If something looks missing, it was cut on
purpose — see `CUT_LIST.md`.

---

## The three rules

**1. The twenty-minute rule.**
Stuck? Struggle alone for twenty minutes. Then open the **tutor** (`ai/tutor.md`) —
never a plain chat, never "write this for me". Then, and only then, bring it to the
live hour. Log every stuck event in `logs/stuck-log.md`.

**2. You write every line for the first three weeks.**
Weeks 1–3 are AI-off for writing code. The AI is a tutor, not an author. Not a moral
position — the training loop, the transformer block and the LoRA layer are each about
thirty lines, and the entire value of this course is that you have written those
thirty lines yourself. A model will happily write them for you. Then you will own a
repo you cannot explain.

From week 4 the AI may write code with you, and the rule that replaces it is:
**you may not commit a line you could not have written and cannot explain on Friday.**

**3. Nothing counts until the tests pass, the number is logged, and it's pushed.**
A training run without a logged score did not happen. `logs/run-log.md`.

---

## The six weeks

| Week | | Milestone |
|---|---|---|
| 1 | Training is not magic | A bigram language model, trained by a loop you wrote |
| 2 | A transformer you wrote | A tiny character-level GPT; you can draw the block |
| 3 | **Fine-tuning by hand** | **Project 1** — your own LoRA layer on a pretrained model, matching the library's |
| 4 | Data is the model | A task, a measured baseline, a dataset, and an eval set you froze |
| 5 | The real run | **Project 2** — an adapter that beats the baseline on the frozen eval |
| 6 | Ship and defend | **Project 3** — merged, quantized, served, and running inside your agent |

Week 3 is the keystone. Weeks 1 and 2 exist so that week 3 contains nothing
unfamiliar, and week 3 exists so that week 5 does not either: the loop is week 1, the
model is week 2, the adapter is week 3.

> Week 1 is in this repo now. Weeks 2–6 arrive with the course.

---

## Two acts

```
BUILD IT   week 1   a loop that makes a number go down
           week 2   a model worth training with it
           week 3   a way to change a big model by training a small piece    <- LoRA

USE IT     week 4   decide what "better" means, before you train anything
           week 5   train it, with the libraries you can now read
           week 6   put it to work, and say honestly whether it was worth it
```

In the first act you use no library that trains for you. In the second act you use
all of them, and nothing they do is a mystery.

---

## Where it runs

On your laptop and on a **free Colab GPU**. No paid compute, no paid API.

Weeks 1–3: the tests run on your laptop's CPU in seconds. Colab is where you train
anything that takes longer than a minute. Weeks 4–6: training happens on Colab.

---

## Daily rhythm

| | |
|---|---|
| **Live hour** | One hour with your instructor, five days a week |
| **Independent** | Four or more hours a day — where the learning actually happens |
| **Friday** | No new content. Defend, ship, retro. |

```bash
source .venv/bin/activate          # every session

pytest week-01/day-1 -v            # one day
pytest week-01 -v                  # one week
```

Tests are the grader for behaviour. A green test means "this behaves correctly", not
"this is good code" — the **editor** (`ai/editor.md`) judges the second thing, and
your instructor judges both on Friday. Every test fails before you write anything.

From week 4 there is a second grader: **the frozen eval.** A model is as good as its
score on examples it has never seen, and no better.

---

## Map

| Path | What it is |
|---|---|
| `SETUP.md` | Day zero. Do this before anything else. |
| `week-01/` … `week-06/` | One folder per week: four days, a milestone, a fence, a defence |
| `tunelab/` | The pieces you write once and reuse: the loop, the tokenizer, the models |
| `data/` | The text you train on in weeks 1 and 2 |
| `ai/` | Your AI roles — planner, source finder, tutor, editor, outside view, defend |
| `SOURCES.md` | Every reading in the course, and why it was chosen |
| `CUT_LIST.md` | What this course deliberately does not teach |
| `logs/stuck-log.md` | Every time you get stuck. Non-negotiable. |
| `logs/signal-log.md` | Your daily numbers. Two minutes at end of day. |
| `logs/run-log.md` | Every training run and what it scored. |

Each week has a `FENCE.md` listing what you have learned and what you have not. Paste
both lists into any AI role before you use it — it is what stops the model reaching
for a library that does the week's work for you.

---

## The three projects

| | Week | Proves |
|---|---|---|
| **LoRA, by hand** | 3 | you understand fine-tuning rather than calling it |
| **An adapter that wins** | 5 | you can make a model measurably better at one thing |
| **Your model, in your agent** | 6 | you can ship it, and say what it cost |
