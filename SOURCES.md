# Sources

The job of a reading list is not to include everything. It is to say which handful of
sources deserve your attention, and which thousands you can ignore.

This is every reading in the course so far. The rule used to choose them: **the
primary source, or the person who built the thing, and nothing else.** No listicles,
no "ultimate guides", no course adverts. When a day's README links a page, it names
the section to read; the rest of the page can wait.

Documentation moves. If a link is dead, search the site for the page title.

---

## The one worth reading twice

| | Why |
|---|---|
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy | The habits that separate people whose models train from people whose models mysteriously do not: look at the data, check the starting loss, overfit one batch, change one thing at a time. Read it at the end of week 1, when you have done three of those. Read it again in week 5, when you need the rest. |

---

## Week 1 — Training is not magic

| Day | Source | Read |
|---|---|---|
| 1 | [Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) — PyTorch | All of it |
| 1 | [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) — PyTorch | "General semantics" only |
| 2 | [Automatic differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — PyTorch | All of it |
| 2 | [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — 3Blue1Brown | All of it, for the picture |
| 3 | [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — PyTorch | All of it |
| 3 | [Build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — PyTorch | Up to "Model Layers" |
| 4 | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy | The first forty minutes: data, tokenizer, batches, bigram. The rest is week 2 |
| 4 | [`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) — PyTorch | "Shape" only |

**Ignore for now:** any tutorial that begins by importing `transformers`, and any
"fine-tune a model in ten lines" post. They are not wrong. They start at week 5.

---

## Weeks 2–6

Listed here as each week arrives.
