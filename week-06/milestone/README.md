# Week 6 milestone — Project 3: your model, in your agent

The third project, and the end of the course. It proves: **you can ship a model you
trained, and say what it cost.**

---

## The bar

- The model you are **serving** — merged, quantized, running behind `llama-server` —
  scores **above your baseline** and **within 5 points of last week's tuned score** on
  the frozen eval, with **zero template problems**.
- The agent has run its **fifteen cases** on it.
- The write-up has every number.

## Deliver

| | From |
|---|---|
| `data/notes/served_predictions.jsonl`, `served.json` | Tuesday: `eval_served` |
| `week-06/day-3/agent_report.json` | Wednesday: `starter-agent/evals/run.py` |
| `week-06/milestone/WRITEUP.md` | started Thursday, finished today |

```bash
pytest week-06/milestone -v
```

Your GGUF file does not go in git: it is hundreds of megabytes. Keep it on your
machine and on Drive, and write its size and quantization level in the table.

## The write-up

`WRITEUP.md` has nine headings. It is the story of the last three weeks told with
numbers: the task, what a prompt could do, the data, what training bought, what
shipping cost, what the agent does on it, what it costs to run.

Then the heading the course exists for:

> **Was fine-tuning worth it here — and what would have been cheaper?**

There is no right answer to that, and "no, it was not" is a perfectly good one if you
can show why. What is not acceptable is not having an opinion. You have built the
whole thing by hand; you are one of very few people entitled to one.

---

## Friday

The final defence covers the whole course: `week-06/DEFEND.md`. Forty minutes. It
starts with a blank page and ends with a question from a colleague.

---

## Done means

- [ ] `pytest week-06 -v` — 34 passed
- [ ] `pytest` at the top of the repo: everything in six weeks, green
- [ ] The three deliverables, committed
- [ ] Your agent, running on your model, on your machine
- [ ] Pushed
