# Day 4 — What it costs

> **By the end of today** you can say, with numbers, whether this was worth doing.

You have a fine-tuned model running your agent. Before you call that a success, do the
thing almost nobody does at this point: add up what it cost, and ask what else would
have got you the same result for less.

That question is the reason this course is six weeks long instead of one afternoon
with a hosted fine-tuning API. Anybody can fine-tune a model. Knowing **whether to**
is the skill.

`cost.py` is in this folder.

---

## Read first

- [ ] Your own `logs/run-log.md`, from week 4 to today. Count the runs · 10 min
- [ ] The pricing page of one hosted model provider. Find the price per million input
      tokens and per million output tokens for a small model, and **write down the
      date you looked** · 5 min

---

## What you need to know

### The ledger

Two columns. Fill in both honestly.

**What fine-tuning cost you**

- The hours building and cleaning data. This is the big one. Count them.
- GPU time. Free, on Colab; not free in general.
- A model to **host**: a machine that is on, a server process that stays up, and you,
  when it falls over.
- A model to **maintain**: add a sixth tool and the dataset, the eval, the baseline
  and the training run all have to be redone. You worked that out in week 4.

**What it bought**

- Accuracy on this one task: your numbers, baseline against tuned against served.
- Latency: a small local model can answer in a fraction of the time.
- Price per request: after the fixed cost, close to nothing.
- Privacy: the notes never leave the machine.
- Independence: no provider can change the model under you, raise the price, or
  retire it.

For a hobby notes app, the right-hand column may be short. For a product handling a
million requests a day, or data that may not leave a building, it can be the whole
business. The same arithmetic gives different answers in different places, which is
why you need to be able to do it.

### Measuring latency properly

Three ways people get this wrong:

- **Timing the first request.** It includes loading the model. Make a warm-up call and
  throw its time away.
- **Reporting the average.** One slow request hides in an average. Report the
  **median** (p50, what a typical request feels like) and the **95th percentile**
  (p95, what the unlucky one in twenty feels like).
- **Timing one request.** Run at least twenty.

`measure_latency` does all three. Its `clock` argument is there so that a test can
hand it a pretend clock: a function you can test without waiting is worth the extra
parameter.

### Pricing a hosted call

Providers price per **million tokens**, separately for input and output. One of your
requests is about nine hundred tokens in — the system prompt and five tool
descriptions, every single time — and about twenty out.

```
 cost of one call  =  (900 × price_in  +  20 × price_out) / 1,000,000
```

Notice which side dominates. It is the prompt you send, not the answer you get. That
matters for the alternatives below.

Prices change. A cost comparison without a date on it is not worth reading; put the
date in your write-up.

### Break-even

A local model has a fixed cost up front and a small cost per request. A hosted one has
no fixed cost and a larger cost per request. They cross where:

```
 requests  =  fixed cost / (hosted cost per request − local cost per request)
```

Run the numbers for your own case. For a notes app with one user, be ready for the
answer to be "never, or in several decades". That is not a failure of the project. It
is the project's most useful finding.

### Three cheaper things you should have tried

You are now qualified to judge these, which you were not five weeks ago.

1. **A better prompt.** You gave it three rounds on Monday of week 4. Was that a fair
   fight? If the baseline's worst kind could have been fixed with one more sentence,
   say so.
2. **Examples in the prompt.** Ten well-chosen examples of the task, in the system
   prompt, cost nothing to build and often close most of the gap. They also make every
   request longer, forever — and you have just seen that the prompt is where the money
   goes.
3. **A slightly bigger model.** The next size up, untuned, with your prompt. If it
   gets your tuned model's score, you have paid in data and maintenance for what you
   could have had by changing one word in a config file.

If you can, try at least one of them today and put the number in the write-up. A
write-up that says "fine-tuning took us from 47% to 85%" is a claim. One that adds
"and the next model up, untuned, gets 81%" is an analysis.

### When fine-tuning wins

It does, often. Roughly when several of these are true:

- the task is narrow, and stays the same;
- the volume is high;
- latency or cost per request really matters;
- the data cannot leave;
- the behaviour you want is easier to **show** than to **describe**;
- prompting has genuinely been tried and has stalled.

---

## Exercises

In `cost.py`:

1. `measure_latency(call, runs, clock, warmup)`
2. `hosted_cost(...)` and `monthly_cost(...)`
3. `break_even_requests(...)`

```bash
pytest week-06/day-4 -v
```

### Then, for real

1. With your server running, time twenty requests through `chat` (pick one eval
   example). p50 and p95.
2. Price one request at a hosted provider's listed price for a small model. Note the
   provider, the model and the date.
3. Your fixed cost: hours × whatever you think an hour of yours is worth, plus zero
   for Colab. Break-even, in requests, and in years at a hundred requests a day.
4. One of the three cheaper things, if you can.

Everything goes in tomorrow's write-up. Start it today: `week-06/milestone/WRITEUP.md`.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Tomorrow you defend the whole course. The last question will be the one in the
write-up's second-to-last heading. Have an answer you believe.
