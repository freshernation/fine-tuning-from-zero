# Week 3 — Fine-tuning by hand

> **Destination**
> Change what a real pretrained model does by training thirty lines you wrote, and
> prove your thirty lines are the same as the library's.

This is the keystone week. The first two weeks existed so that this one would contain
nothing unfamiliar: the loop is from week 1, the model is the one you built in week 2
with more layers, and the only new idea is a way to change a large model by training
a small piece attached to it.

It also closes the first half of the course. From Monday you use libraries — and
after this week there is nothing inside them that you have not written yourself.
`peft`'s LoRA is your `tunelab/lora.py`. On Friday you prove that, number for number.

Weeks 1–3 are **AI-off for writing code.** This is the last week of that rule, and the
week it matters most. LoRA is thirty lines, and you could have them in ten seconds.
Then you would have a file called `lora.py` and nothing else.

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Open a pretrained model, find your block diagram inside it, and train it with your own loop |
| Tue | `day-2/` | Fine-tune the obvious way, and measure what it costs and what it forgets |
| Wed | `day-3/` | **Write LoRA** |
| Thu | `day-4/` | Put it inside a model, train through it, save the adapter, merge it |
| Fri | `milestone/` | **Project 1** — match the library, compare three methods on a real model, write it up |

Each day folder has its own `README.md`. Start there, every time.

This week's code goes in `tunelab/pretrained.py`, `tunelab/finetune.py` and
`tunelab/lora.py`. Nothing from weeks 1 and 2 is edited.

---

## The model

`HuggingFaceTB/SmolLM2-135M-Instruct`: 135 million parameters, small enough for a free
GPU. You do not download it until Friday. Every test this week runs on your laptop,
offline, against a miniature with the same architecture.

---

## How a day goes

1. **Read** the day's README. All of it, before writing anything.
2. **Predict, then run.** Shapes, still. And now: which parameters will have changed?
3. **Write** the exercises. In order — they build.
4. **Test** with `pytest week-03/day-N -v`.
5. **Log** anything that stuck you in `logs/stuck-log.md`, and your numbers in
   `logs/signal-log.md`.
6. **Commit and push.** Every day.

---

## Milestone — Project 1

Your LoRA matches `peft`'s to five decimal places; one function runs full
fine-tuning, last-block-only and LoRA and measures each the same way; and on Colab you
teach a real model Shakespeare all three ways and write one page on what each cost and
what each forgot. Spec in `milestone/README.md`.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *When a master craftsman takes on a trained journeyman from another workshop, what
> do they retrain, and what do they leave alone?*

---

## What this week is not about

A useful model. Teaching a chat model to write Shakespeare is a pointless thing to do
to it, chosen because the text is already in the repo and the change is easy to hear.
The real task starts on Monday. Nor is it about using `peft`, `Trainer`, quantization,
or anything else in the "not yet" column of `FENCE.md` — all of which arrive in week 5,
and all of which will look smaller than you expected.
