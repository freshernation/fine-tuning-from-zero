# Week 6 — Ship and defend

> **Destination**
> Put a model you trained to work inside something real, and tell the truth about what
> it cost.

Last week ended with an adapter on your Drive and a number in a JSON file. Neither is
of any use to anybody. This week the model leaves Python: it becomes one file, then a
service on a port, then the brain of an agent.

And then you add up the bill.

---

## The week

| Day | Folder | What you'll be able to do |
|---|---|---|
| Mon | `day-1/` | Merge your adapter, convert the model to one GGUF file, and quantize it |
| Tue | `day-2/` | Serve it behind an OpenAI-compatible endpoint, and score what is actually served |
| Wed | `day-3/` | Run the notes agent on it with no code change, and explain the gap between step accuracy and task success |
| Thu | `day-4/` | Measure latency, price the alternative, and work out when fine-tuning pays |
| Fri | `milestone/` | **Project 3** — the agent on your model, and the write-up; then the final defence |

Each day folder has its own `README.md`. Start there, every time.

Code goes in `tunelab/served.py` (Tuesday) and small files in the day folders.

---

## What you need on your machine

Until now everything heavy ran on Colab. This week the model runs **locally**, because
that is the point. You need:

- **llama.cpp**, for converting, quantizing and serving. `day-1/README.md` covers
  installing it. A model this size runs comfortably on a laptop CPU.
- Your **adapter** from week 5, downloaded from Drive.

If your laptop cannot build or install llama.cpp, `week-06/colab.ipynb` does Monday's
merge and conversion on Colab and hands you the finished file. Serving it is still
local.

---

## Milestone — Project 3

The served model above your baseline and within five points of last week's score; the
agent's fifteen cases run on it; and a write-up that answers, with numbers, whether
fine-tuning was worth it. Spec in `milestone/README.md`.

---

## Outside-view question

Once this week, open `ai/outside-view.md` and ask it this:

> *When a restaurant decides whether to bake its own bread or buy it in, what do they
> actually count?*

---

## What this week is not about

Deployment in the grown-up sense: containers, autoscaling, monitoring, GPUs in a data
centre. You are running one small model on one machine. The decisions you practise
this week are the ones that come before any of that — and they are the ones that
decide whether the rest is worth doing.
