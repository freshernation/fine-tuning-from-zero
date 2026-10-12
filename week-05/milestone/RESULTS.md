# Project 2 — an adapter that beats the baseline

One page. Every number here must be one a test or a script can reproduce.

## Baseline against tuned

Model: `____`. Adapter: rank `____`, alpha `____`, `____` epochs, `____` training examples.

| Kind | n | Baseline | Tuned |
|---|---|---|---|
| add | ____ | ____ | ____ |
| list | ____ | ____ | ____ |
| find | ____ | ____ | ____ |
| delete | ____ | ____ | ____ |
| calculate | ____ | ____ | ____ |
| none | ____ | ____ | ____ |
| clarify | ____ | ____ | ____ |
| **all** | ____ | ____ | ____ |

Fixed: `____` examples. Broken: `____`. Still wrong: `____`.

## Is the gain real

Gain: `____` points. 95% paired-bootstrap interval: `____` to `____`.

In one sentence, what that interval lets you claim and what it does not:
____

## One run that failed

Which run (its row in `logs/run-log.md`), what went wrong, how you found out, and what
you changed:
____

## One data fix

What the still-wrong examples had in common, what you found in the training data,
what you changed, and the score before and after:
____

## What it is still bad at

The kind or pattern it is worst at now. Is that a data problem or a limit of a model
this size? What is your evidence?
____

## Examples in the eval set I think are wrong

(Ids and reasons. You did not change them. Argue them on Friday.)
____
