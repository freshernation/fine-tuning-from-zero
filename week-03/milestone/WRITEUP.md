# Project 1 — LoRA, by hand

One page. Numbers from your own Colab run; nothing here is filled in for you.

## Results: three ways to fine-tune

Model: SmolLM2-135M-Instruct. 300 steps of Shakespeare. Before any training:
Shakespeare loss `____`, general-English loss `____`.

| Method | Trainable parameters | % of model | Peak GPU memory | Shakespeare loss after | General-English loss after |
|---|---|---|---|---|---|
| Full fine-tuning | | | | | |
| Last block only | | | | | |
| LoRA, rank 8 | | | | | |

## What rank buys

| Rank | Trainable parameters | Shakespeare loss after | General-English loss after |
|---|---|---|---|
| 1 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

## The adapter

Size on disk: `____` MB, against `____` MB for the model.

## Two samples, same prompt

As downloaded:

> 

Tuned:

> 

## Three questions

**1. What did full fine-tuning cost that LoRA did not, and what did it buy for that?**

**2. What did rank buy, and where did it stop paying?**

**3. Which method forgot the most plain English? Is that what you predicted before
you ran it, and if not, why not?**
