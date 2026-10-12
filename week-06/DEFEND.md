# Week 6 — The final defence

Forty minutes per student. It covers the whole course. The student has their repo,
their running server, and a blank sheet of paper.

---

## Phase 0 — From memory (8 min)

On paper, no screen.

1. *"The five lines of the training loop, in order."* (Week 1)
2. *"A transformer block, with shapes."* (Week 2)
3. *"Now put LoRA into it. Where do the adapters sit, what is trained, what is frozen,
   and what is the formula for one adapted layer's output?"* (Week 3)
4. *"Why is `B` zero at the start and `A` not?"*

A student who cannot do these has not finished the course, whatever their tests say.

## Phase 1 — Trace a number (8 min)

Open `WRITEUP.md`. **The examiner** picks one number from the results table.

*"Where did this come from? Show me the file. Show me the command that wrote the file.
Show me the function, in code you wrote, that computed it. Now show me the test that
would fail if that function were wrong."*

Then a second number, from a different week.

Looking for: every number has a chain back to code they can explain, and they can walk
it without searching.

## Phase 2 — Mutate (8 min)

*"The agent gets a sixth tool tomorrow: `rename_note(old_title, new_title)`. List
everything you would redo, in order. Then everything you would not touch."*

Looking for, in order: the tool and its schema in the agent; new seeds, including the
hard pairs (rename against delete-then-add) and the refusals; expansion, cleaning,
balance; a **new eval set, a new fingerprint, a new baseline**; a training run; error
analysis; merge, convert, quantize; serve; the served eval; the agent cases, with new
ones. Not touched: the loop, the LoRA layer, the scorer, the parser, the split, the
renderer, `train.py`, `predict.py`, `served.py`. And they should say roughly how long
it would take, which is the cost of maintaining a fine-tuned model.

## Phase 3 — Break (8 min)

*"I re-quantize your model one level lower and the served score drops four points.
Walk me through what you do, with your own tools."*

Looking for: `eval_served` again; `changed` against the previous served predictions to
get the ids that broke; read them; ask whether they share a kind or a pattern; check
template problems are still zero, to rule out the plumbing; then decide whether four
points matter, **using the agent cases and the cost arithmetic, not a feeling.**

Then: *"Now I tell you the four points were all in `clarify`. Does that change your
answer?"*

## Phase 4 — The question (8 min)

*"A colleague comes to you. They have a task, a hosted model that does it about 80%
right, and a manager who has heard that fine-tuning fixes things. What do you ask them
first? And second? And what would have to be true before you said yes?"*

There is no script for the answer. It should contain, in the student's own words:
what "right" means and whether they can measure it; whether prompting has genuinely
been tried; how much data they have and whether anyone has read it; volume, latency,
privacy; and who will maintain it when the task changes.

This phase is the course. Score it as such.
