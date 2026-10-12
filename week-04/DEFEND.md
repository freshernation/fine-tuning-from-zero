# Week 4 — Friday defence

Twenty-five minutes per student. The student shares `data/notes/`, their datasheet and
`tunelab/toolcall.py`.

---

## Phase 1 — Explain (7 min)

1. *"Your scorer answers four questions about each prediction. Which one does your
   baseline fail most? What would the fix be for each of the other three, if it had
   been that one instead?"*
2. *"Show me an output your parser calls invalid. Why is it not simply scored as 'the
   model said nothing'? Which of your examples would it have got credit for?"*
3. *"Why did you split by seed? Tell me the story of what goes wrong otherwise, with
   one of your own seeds."*
4. *"`eval.sha256` — what is in it, when did you commit it, and what happens if I
   change one character of `eval.jsonl` right now?"* Then do it, and run the test.
5. *"Print `render_prompt` and `render_completion` for one example. Point at the first
   character the model has to produce. Point at the last."*

## Phase 2 — Mutate (8 min)

Words first, keyboard second.

- **A** — *"The agent gets a sixth tool tomorrow: `rename_note(old_title, new_title)`.
  Walk me through everything in this week's work that changes, in order — and
  everything that does not."* Looking for: schemas, a new kind, seeds, balance, a new
  eval set **with a new fingerprint and a new baseline**; and that the scorer, the
  parser, the split and the renderer do not change at all.
- **B** — *"The model should be allowed to call two tools in one reply. What breaks?"*
  Looking for: the prediction format, the first-call-wins rule, `expected`, and that
  it is a change to the definition of success, so the old baseline is void.

## Phase 3 — Break (4 min)

*"I am going to take three rows from your eval set and paste them into your training
set with new ids. Which test catches me, and what does it compare?"* Then the seeded
break, from the runbook.

## Phase 4 — The data (6 min)

- *"Show me a conflict you found and the rule you chose."*
- *"Show me two rows a reasonable person would label differently."*
- *"Open `train.jsonl` at a row I pick. Is it right?"* Pick three, at random, live.
- *"What does this dataset deliberately not cover? What will your model do when it
  meets one of those?"*
- *"Was your baseline prompt a fair attempt? Convince me."*
