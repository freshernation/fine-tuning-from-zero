# Day 3 — The agent, on your model

> **By the end of today** the agent you built in another course is running on a model
> you trained in this one.

This is the moment the course's first sentence promised. It takes three environment
variables.

Then it gets interesting, because your agent will succeed less often than your eval
score says your model is right, and today is about understanding exactly why.

`gap.py` is in this folder.

---

## Read first

- [ ] `starter-agent/README.md`, the section "Point it at a model" · 3 min
- [ ] `starter-agent/evals/cases.py` — fifteen small tasks, each with a starting set of
      notes, a request, and what should be true afterwards · 10 min

---

## What you need to know

### No code change

With your server from yesterday still running, in `starter-agent/.env`:

```
AGENT_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
AGENT_OPENAI_MODEL=notes
```

```bash
cd starter-agent
AGENT_TRACE=1 python3 cli.py
```

Type "what notes do I have?". The loop you wrote in Agent From Zero asks a model what
to do, and the model that answers is yours.

Stop and take in why that worked. The agent was written against an **interface** —
"something with a `chat` method that takes messages and tools" — and not against a
provider. Your model sits behind the same interface. Nothing above it can tell.

### One decision, many times

Your eval measures one decision at a time. An agent makes several in a row, and a task
only succeeds if **all** of them are right.

Take "delete the milk note". Three decisions:

1. call `list_titles`
2. call `delete_note` with the exact title
3. stop, and say it is done

If your model is right 90% of the time, and each decision were independent:

```
 0.9 × 0.9 × 0.9  =  0.73
```

A 90% model completes a three-decision task 73% of the time. A five-decision task,
59%. This is the single most useful piece of arithmetic for anyone building agents:
**per-step accuracy compounds, and it compounds against you.**

It is also the honest answer to "your eval says 85%, why does the agent keep failing?"

Note the count. A task that needs two tool calls needs **three** decisions: the last
one is deciding to stop and answer. In your dataset those were the `none` examples
that came after a tool result. This is where they pay for themselves.

### The arithmetic is a baseline, not a law

Real runs differ from that calculation, in both directions.

**Worse**, because errors are not independent. Call the wrong tool at step one and the
conversation now contains something your training data never showed: an irrelevant
tool result. The model is off the path it was trained on, and its accuracy there is
anyone's guess.

**Better**, because the agent has guards — you wrote them — and because tools answer
back. Delete a title that does not exist and the tool says so; a model that reads the
error and tries again has recovered. Your dataset has a few examples of that. More
would help.

So you compare: what the arithmetic predicts for each case, against what happened.
Where they disagree is where the interesting behaviour is.

### Where a run left the path

For each failed case you want one fact first: **at which step did it first go wrong?**
Everything after that step is noise. `first_wrong_step` finds it, comparing the calls
the agent made with the calls a careful agent would have made, using the same argument
matching as week 4.

Then you read the trace at that step. Is it a decision your eval would have caught —
the wrong tool for a plain request? Or a situation your dataset never covered — the
second turn after an error?

The first kind means your eval already told you. The second kind is a finding: it
names the examples to write next.

### What the hosted model did that yours does not

Be straight about this in your write-up. The model the agent was built with is
thousands of times larger. It recovers from odd situations, handles requests nobody
anticipated, and writes better final answers. Yours handles one narrow job and was
taught it from eight hundred examples.

Whether that trade is worth making is tomorrow's question. Today's job is to describe
it accurately.

---

## Exercises

In `gap.py`:

1. `expected_task_success(step_accuracy, steps)`
2. `first_wrong_step(made, wanted)`
3. `compare_with_expected(agent_report, step_accuracy)` and `summarise(...)`

```bash
pytest week-06/day-3 -v
```

### Then, for real

With your server running and `.env` set:

```bash
cd starter-agent
python3 evals/run.py --out ../week-06/day-3/agent_report.json
```

Fifteen cases. Then run your `compare_with_expected` on the report, with
`step_accuracy` set to yesterday's served accuracy.

For every failed case: the first wrong step, and one sentence on which kind of failure
it was. That table goes in Friday's write-up.

If you have a key for a hosted model, run the same fifteen cases against it by
changing `.env`, and keep that report too. It is optional, and it is the fairest
comparison you can make.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`, and a row in `logs/run-log.md`: cases passed,
what the arithmetic predicted, and the commonest first wrong step.
