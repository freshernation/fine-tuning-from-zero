# Week 3 — Friday defence: Project 1

Thirty minutes per student. It starts with a blank sheet of paper.

---

## Phase 0 — Draw (6 min)

*"Draw the LoRA layer: both roads, every shape."* Then: *"Now draw last week's block,
and show me where in it these sit when the targets are the query and value
projections."* Then: *"Write the formula for the output in one line."*

## Phase 1 — Explain (7 min)

The student shares `tunelab/lora.py`.

1. *"`B` starts at zero. Why? And why does `A` not?"* The second half is the question.
2. *"What does `alpha / rank` do? If I double the rank and change nothing else, what
   happens to the size of the update, and why was it designed that way?"*
3. *"Point at the line that freezes the base layer. What would training look like if
   it were missing, and which of your tests would tell you?"*
4. *"Which tensors are in the adapter file? If I hand you only that file, what else
   do you need?"*
5. *"On the first backward pass, one of `A` and `B` gets a gradient and the other does
   not. Which, and why?"*

## Phase 2 — Mutate (8 min)

Words first, keyboard second.

- **A** — *"Put adapters on the MLP's linear layers as well as on query and value."*
  Looking for: it is a change to one argument, not to any code; and they can say how
  the trainable count changes for the real model before running anything.
- **B** — *"I have two adapters trained separately on the same base model, for two
  different tasks. I want to switch between them without reloading 540 MB. Describe
  how, using only functions you have written."* Looking for: inject once,
  `load_lora` to swap — and the realisation that this is impossible after merging.

## Phase 3 — Break (4 min)

*"Without running it: I save an adapter at rank 8 and load it into a model injected
at rank 4. What happens in your code? What would happen in a loader that skipped
whatever did not fit, and how long would it take you to notice?"*

Then the seeded break, from the runbook.

## Phase 4 — The write-up (5 min)

Pick one number from `WRITEUP.md` and ask where it came from: which cell, which
function, which line. Then: *"Which method forgot the most? Was that your prediction?
Show me where you wrote the prediction down."*
