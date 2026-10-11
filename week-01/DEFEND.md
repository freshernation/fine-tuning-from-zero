# Week 1 — Friday defence

Twenty minutes per student. The student shares their screen with `tunelab/loop.py`,
`tunelab/bigram.py` and `milestone/train_bigram.py`.

---

## Phase 1 — Explain (5 min)

1. *"Point at the five lines of the loop and name each one. What does the model look
   like after line 3 but before line 4?"*
2. *"Which line would you delete to make the weights stop changing but everything else
   still run? Which line would you delete to make training quietly get worse?"*
3. *"Your first validation loss was about 4.5. Where does 4.17 come from, and why is
   yours above it?"*
4. *"In `forward`, your logits are `(B, T, C)`. Point at where they stop being that
   shape, and say why they have to."*
5. *"`estimate_loss` — show me the three things in it that make it a measurement and
   not training."*

## Phase 2 — Mutate (8 min)

Words first, keyboard second.

- **A** — *"The loss sometimes jumps. Add gradient clipping: no gradient may be larger
  than 1.0."* They have not met clipping; tell them what it means in one sentence.
  Looking for: it goes **between backward and step**, and they can say why it cannot
  go anywhere else.
- **B** — *"Change the bigram so the table is 65 rows of 16 numbers instead of 65 rows
  of 65, and still predicts 65 characters."* Looking for: a second layer that maps 16
  back to 65, and the realisation that the logits' shape has not changed so the loss
  and the loop do not change at all.

## Phase 3 — Break (5 min)

*"Without running it: I delete `optimizer.zero_grad()`. Draw the loss curve."* Then
have them run it and compare with their drawing.

Then seed one: move `model.train()` out of `estimate_loss`. Ask what a user of the
function would notice, and when. (For this model: nothing, ever — a bigram has no
layer that behaves differently in eval mode. The right answer is "nothing yet, and
that is why it is dangerous": it becomes a real bug the first time the model contains
dropout, in week 2.)
