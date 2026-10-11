# Run log

Every training run, one row. Newest at the bottom.

**A run that is not in this table did not happen.** Not because anyone is checking,
but because in three weeks you will have done forty runs, and the only way to know
which change made the model better is to have written down what each one was.

Fill in the row **before** you look at whether it was good. Failed runs go in too —
they are the ones you learn from, and week 5's milestone asks to see one.

| Date | Week | What was run | Steps | Train loss | Val loss | Eval score | Note |
|---|---|---|---|---|---|---|---|
| 2026-10-16 | 1 | bigram on tiny.txt, seed 0 *(example — delete this one)* | 2000 | 2.50 | 2.54 | — | "By, d ravine homoreZANo f k, l." |
|  |  |  |  |  |  |  |  |

**Eval score** stays empty until week 4. From then on it is the number that matters,
and it is always measured on the frozen eval set.

**Note** is for the one thing you changed since the last run. One thing. If you
changed two, you will not know which one did it.
