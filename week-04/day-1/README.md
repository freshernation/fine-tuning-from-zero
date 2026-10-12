# Day 1 — The eval comes first

> **By the end of today** you can put a number on how good a model is at this job, and
> you have the number for a model nobody has trained.

The second half of the course starts here, and it starts in an odd place: you will not
train anything this week.

For three weeks the question was *how does training work*. From today it is *did
training help*, and you cannot answer that after the fact. If you train first and
decide what "good" means afterwards, you will decide that good means whatever your
model does. Everybody does. So the measurement is built first, on a model you have not
touched, and then it is locked.

**The rule about AI changes this week.** You may now write code with an AI. What
replaces the old rule: *you may not commit a line you could not have written and
cannot explain on Friday.* You have earned that; the first three weeks were so that
"could have written" is true.

Everything today goes in `tunelab/toolcall.py`.

---

## Read first

- [ ] [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic. You may have read it in Agent From Zero. Read it again for the kinds of grader · 20 min

---

## The task, for the next three weeks

The notes agent from Agent From Zero has five tools. (If you did not take that course,
it is in `starter-agent/`; its `README.md` is five minutes.)

| Tool | Arguments | For |
|---|---|---|
| `add_note` | `title`, optional `body` | saving something |
| `list_titles` | — | "what notes do I have?" |
| `find_notes` | `word` | "is there anything about…?" |
| `delete_note` | `title` | removing one, by its exact title |
| `calculate` | `expression` | arithmetic |

An agent is a loop around one decision, made over and over: **given the conversation
so far, call a tool — which one, with what arguments — or answer in words.** A large
hosted model makes that decision well. You are going to teach a model two thousand
times smaller to make it, and in week 6 you will swap yours in.

So one example of the task is one decision:

```json
{"id": "dev-02", "kind": "add",
 "messages": [{"role": "user", "content": "jot down milk — two litres, semi-skimmed"}],
 "expected": {"tool": "add_note", "arguments": {"title": "milk", "body": "two litres, semi-skimmed"}}}
```

`kind` says what sort of decision it is. Five kinds are named after the tool. Two are
not:

- **`none`** — nothing needs doing. "Thanks!" The right move is to answer in words.
- **`clarify`** — something needs doing but a piece is missing. "Save a note." The
  right move is to ask.

For both, `expected.tool` is `null`. They are the examples everybody forgets, and a
model that has never seen one learns that every message deserves a tool call.

Open `data/notes/dev.jsonl` and read all twenty. Three are worth a second look:
`dev-09`, `dev-11` and `dev-20`.

---

## What "right" means

Not one question. Four.

| | Question | Fails when |
|---|---|---|
| **valid** | Could the output be read at all? | the model tried to call a tool and wrote something that is not JSON |
| **right tool** | Did it choose the right tool — or rightly choose none? | `list_titles` when you wanted `find_notes`; a tool call in reply to "thanks" |
| **right arguments** | Right tool, *and* the right things in it? | `add_note` with the title and body swapped |
| **correct** | All three. | |

Why keep four numbers when the fourth is the one you care about? Because **one
accuracy number tells you that the model is failing and nothing about how.** "62%"
could be a model that understands everything and cannot write JSON, or one that writes
perfect JSON for the wrong tool. Those need opposite fixes. You will be very glad of
the breakdown in week 5.

### Reading what a model wrote

The model writes text. For this model, a tool call looks like this:

```
<tool_call>
{"name": "add_note", "arguments": {"title": "milk"}}
</tool_call>
```

Real output is messier, and your parser has to have an opinion about each mess:

- chat before or after the call
- two calls in one reply
- a call that is never closed, because the model stopped
- single quotes, a trailing comma: almost JSON
- `"arguments"` as a string with JSON inside it

`parse_prediction`'s docstring says what to do with each, and each has a test. The
principle behind all of them: **be forgiving about packaging, strict about content,
and never guess.** An unreadable call is recorded as *invalid*. It is not quietly
turned into "the model said nothing", because then a model that produces garbage would
score perfectly on the `none` examples.

### When are two arguments the same?

The model says `{"title": "Call the dentist "}` and you wanted `"call the dentist"`.
Wrong? No: the agent would have done the right thing.

So strings are compared loosely — trimmed, runs of spaces collapsed, case ignored —
and nothing else is forgiven. "call a dentist" is wrong. An extra argument the tool
does not have is wrong. A number where a string belongs is wrong.

You are deciding, here, what counts as success. Decide it on purpose, write it down,
and do not change it later because a model you like fell on the wrong side of it.

---

## The baseline

Fine-tuning is the expensive way to make a model better at something. The cheap way is
to tell it what you want. So before you spend a week building data, you give the cheap
way an honest try: the untuned model, the five tool descriptions, and the best system
prompt you can write.

Whatever that scores is the **baseline**, and it is the number your fine-tuned model
has to beat. If your best prompt already gets 95%, you have learned that this task did
not need fine-tuning, on Monday, for free. That is a good outcome. It is also unlikely
with a model this small.

Write your prompt in `week-04/day-1/prompt.md`. Take it seriously: say what the
assistant is for, when to ask instead of acting, and anything the tool descriptions
leave unsaid. A lazy baseline makes fine-tuning look good, and you will be asked on
Friday whether yours was lazy.

---

## Exercises

In `tunelab/toolcall.py`, in this order:

1. `parse_prediction(text)`
2. `same_arguments(expected, got, tool)`
3. `score_example(example, prediction)`
4. `score_predictions(examples, predictions)`
5. `run_eval(generate, examples)`

```bash
pytest week-04/day-1 -v
```

Then write `prompt.md`, open `week-04/colab.ipynb` in Colab, and run it as far as
**"the dev cases"**. It loads the real model, runs your prompt on the twenty dev cases
with your scorer, and prints every wrong answer in full.

Read the wrong answers. Improve the prompt. Run it again. Three rounds, no more.

Then one row in `logs/run-log.md`: the dev accuracy, and in the note, which kind the
model was worst at.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

You now know where the model fails. Tomorrow you start writing the examples that will
teach it not to — and the twenty in `dev.jsonl` are not among them. Why not?
