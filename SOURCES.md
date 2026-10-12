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

## Week 2 — A transformer you wrote

| Day | Source | Read |
|---|---|---|
| 1 | [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown | The section on word embeddings |
| 2, 3 | [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown | Up to multiple heads on Tuesday; the rest on Wednesday |
| 2 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. | Section 3.2.1 only. Half a page, one equation, and you implement it that day |
| 4 | [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) — PyTorch | "What is a `state_dict`?" and "Saving & Loading a General Checkpoint" |
| every day, **after** the exercises | [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy | The part that matches what you just wrote |

The Karpathy video is the complete solution to this week. That is why it is assigned
after each day's exercises and never before. Watched afterwards, it is the best
explanation there is of why your code looks the way it does.

**Ignore for now:** "The Illustrated Transformer" and anything else that explains the
2017 encoder-decoder model. It is a good piece about a different architecture; you are
building the decoder-only kind, and the extra half will confuse you this week.

---

## Week 3 — Fine-tuning by hand

| Day | Source | Read |
|---|---|---|
| 1 | [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) — Hugging Face | The summary and "How to use" |
| 1 | [Chat templates](https://huggingface.co/docs/transformers/chat_templating) — Hugging Face | The introduction and first example |
| 2 | [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy | Again, as promised in week 1 |
| 3, 4 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. | Wednesday: the abstract, section 4.1 ("Low-Rank-Parametrized Update Matrices") and Figure 1. Thursday: section 4.2 ("Applying LoRA to Transformer"), first two paragraphs |

The LoRA paper is the one paper in this course you read properly. It is short, it is
clear, and by Wednesday evening you will have implemented its central equation.

**Ignore for now:** every "LoRA explained" blog post and video, and the `peft`
documentation. The first are longer than the paper and say less. The second is week 5,
when it will read as a description of code you have already written.

---

## Week 4 — Data is the model

| Day | Source | Read |
|---|---|---|
| 1 | [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic | All of it, for the kinds of grader |
| 4 | [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy | The first section, "Become one with the data" |

Two sources for a whole week, and one of them for the third time. That is not an
oversight. This week's reading is your own data: `dev.jsonl`, then the model's wrong
answers, then fifty of your own rows, then fifty more.

**Ignore for now:** tools and papers about generating synthetic datasets at scale.
They solve the problem of making a hundred thousand rows. Your problem this week is
making eight hundred you have actually read.

---

## Week 5 — The real run

| Day | Source | Read |
|---|---|---|
| 1 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. | The abstract and the first paragraph of the introduction |
| 1 | [LoRA conceptual guide](https://huggingface.co/docs/peft/conceptual_guides/lora) — `peft` | All of it, as a review of week 3 |
| 2 | [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) — `transformers` | The opening paragraphs, before the class reference |
| 3 | [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy | The checks under "Set up the end-to-end training/evaluation skeleton" |

The two scripts you read this week, `week-05/train.py` and `week-05/predict.py`, are
sources too. Read them the way you read the LoRA paper.

**Ignore for now:** fine-tuning "recipes" that list twenty hyperparameters to tune. If
your run is not working, Thursday's lesson tells you where to look, and it is not
there.

---

## Week 6 — Ship and defend

| Day | Source | Read |
|---|---|---|
| 1 | [llama.cpp](https://github.com/ggml-org/llama.cpp) — README | The description, and the section on preparing and quantizing models |

One source. By now you can read a tool's own documentation and know what every option
is doing on your behalf, which was the point.

---

## Read next

The course's cut list (`CUT_LIST.md`) is the reading list for afterwards. In the order
that will make most sense:

1. A hosted fine-tuning API's documentation, any provider. Read every parameter and
   name the thing in this course it corresponds to.
2. The QLoRA paper, in full this time.
3. The DPO paper (Direct Preference Optimization), for when "the right answer" is a
   matter of better and worse.
4. Anything on distillation: using a large model's outputs to train a small one. You
   did a little of it in week 4.
