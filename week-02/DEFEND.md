# Week 2 — Friday defence

Twenty-five minutes per student. It starts with a blank sheet of paper, not a screen.

---

## Phase 0 — Draw (5 min)

*"Draw a transformer block. Label every box and every arrow with its shape."*

No notes, no screen. Then: *"Now draw the whole model around it."*

This is scored with Phase 1. A student who cannot draw the block has not finished the
week, whatever their tests say.

## Phase 1 — Explain (5 min)

The student shares `tunelab/attention.py` and `tunelab/gpt.py`.

1. *"Point at the one line that stops a position seeing the future. What would the
   training loss do if you deleted it? What would the generated text look like?"*
2. *"Why is the mask a buffer and not a plain attribute? And why not a parameter?"*
3. *"Scores are divided by something before the softmax. By what, and what goes wrong
   without it?"*
4. *"`x = x + ...` appears twice in your block. What does the `x +` buy you? Which of
   your tests would fail without it, and why does that test work?"*
5. *"Where are most of your big model's parameters? Show me how you found out."*

## Phase 2 — Mutate (8 min)

Words first, keyboard second.

- **A** — *"Turn it into a model where every position can see every other position,
  before and after."* Looking for: it is one change, in one place; and then — *"so
  why can that model not be trained to predict the next character?"*
- **B** — *"I want it to read 256 characters instead of 128, starting from your
  trained checkpoint rather than from scratch. What breaks when you load it, and what
  could you do about it?"* Looking for: exactly which tensors change shape (the
  position table and the mask), which do not (everything else), and an idea for the
  new position rows.

## Phase 3 — Break (5 min)

*"Without running it: I delete the position embedding, so the model only gets token
embeddings. Does it still train? Is the validation loss better than the bigram's,
worse than your GPT's, or the same as one of them — and why?"* Then have them run the
small model both ways and compare.

Then seed one, from the runbook.

## Phase 4 — The run (2 min)

*"Show me the run log. At what step did you kill it? What exactly was lost when you
did, and what was not?"*
