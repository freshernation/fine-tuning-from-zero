# What this course does not teach

The most valuable part of any course plan is the cut list. Every item below is a real
thing people who fine-tune models know, and every one of them is out.

| Cut | Why |
|---|---|
| RLHF, DPO, preference tuning | A second course; needs the supervised case to be solid first |
| Distillation | Same |
| Multi-GPU, full fine-tuning of large models | Does not fit a T4; week 3 does full fine-tuning once, small |
| Vision and audio | Different data problems |
| Hosted fine-tuning APIs | Hide exactly the parts this course exists to show |
| Hyperparameter sweeps | Week 5 teaches reading one run well, not launching fifty |
| Pretraining theory, BPE from scratch | Week 2 uses characters; week 3 uses the model's tokenizer as given |
| "RAG vs fine-tuning" as a topic | Week 4's baseline and week 6's write-up answer it for one case, with numbers |

---

## Why cut so much

Six weeks is short. The course spends them on one thing: you can take a model, change
what it does by training it, and prove that you did. Every item above is either a
variation on that (learn it afterwards, quickly, because you have the base) or a way
of not having to understand it (which is the opposite of the point).

## What to learn next

When you finish, the cut list is your reading list. In roughly this order:

1. **Hosted fine-tuning APIs.** You will now be able to read their documentation and
   know what every parameter is doing on your behalf.
2. **Hyperparameter sweeps.** One afternoon, once you can read a single run.
3. **Preference tuning (DPO).** The natural next step when "the right answer" is a
   matter of better and worse rather than correct and incorrect.
4. **Distillation.** Using a large model's outputs as the dataset for a small one —
   which you will already have done, a little, in week 4.
