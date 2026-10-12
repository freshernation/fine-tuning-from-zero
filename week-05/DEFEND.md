# Week 5 — Friday defence: Project 2

Thirty minutes per student. The student shares `RESULTS.md`, their run log and
`tunelab/sft.py`.

---

## Phase 1 — Explain (8 min)

1. *"Tokenize one of your examples and print the labels. Point at the first position
   that is not −100. What token is it, and why is the one before it masked?"*
2. *"In week 3 I told you never to pass `labels=` to a Hugging Face model. `Trainer`
   passes `labels=`. Why is it right now and wrong then?"*
3. *"Open `train.py`. Show me the five lines of your week-1 loop."* (They are inside
   `trainer.train()`; the right answer says so, and says what `Trainer` adds.)
4. *"Read me your `LoraConfig`, field by field, pointing at the line of your own
   `tunelab/lora.py` each one corresponds to."*
5. *"The base model was loaded in 4 bits and you trained it. How, if a 4-bit weight
   cannot move by a small amount?"*

## Phase 2 — Mutate (8 min)

Words first, keyboard second.

- **A** — *"The model must now be able to make two tool calls in one reply: 'save milk
  and eggs'. What changes in the data, the eval and the parser? What does not change
  in training at all?"* Looking for: the definition of success moves, so the eval set,
  its fingerprint and the baseline are all new; `tokenize_example`, `collate` and
  `train.py` do not change by a character.
- **B** — *"You have half the GPU memory. Keep the same training behaviour."* Looking
  for: halve the batch size, double the accumulation; and what `max_len` would cost.

## Phase 3 — Break (5 min)

*"I run `predict.py` with the line that attaches the adapter deleted. What do I see?
Name everything in your pipeline that would tell me something was wrong."* Looking
for: a score equal to the baseline; `test_tuned_and_baseline_are_not_the_same_predictions`.

Then the seeded break, from the runbook.

## Phase 4 — The results (9 min)

- *"Your gain is N points, interval A to B. What does that let you claim? What would a
  sceptic say, and what is your answer?"*
- *"How do you know it is not leakage? Not which function you ran: what would a leak
  look like in these results?"*
- *"Show me the failed run. What did it cost you, and what did you change?"*
- *"Show me the data fix. What did the wrong answers have in common, and how did you
  find that pattern in the training set?"*
- *"Which eval examples do you think are wrong? Why did you leave them alone?"*
