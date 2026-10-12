# Datasheet — notes tool-calling dataset

One page. A reader who has never seen your data should be able to tell, from this,
what it is good for and what it is not.

## What it is

Task, in one sentence:

Rows: train `____`, eval `____`. Seeds: `____`.

| Kind | Train | Eval |
|---|---|---|
| add | | |
| list | | |
| find | | |
| delete | | |
| calculate | | |
| none | | |
| clarify | | |

## How it was made

Seeds written by hand: `____`.
Expanded by (templates / a model through an API / a model by hand), in what proportion:

## What I checked

Rows read by hand: `____`. Rows I judged wrong among them: `____`.

Dropped while cleaning — invalid: `____`, duplicate: `____`.

Conflicts I had to decide, and the rule I chose for each:

## What it deliberately does not cover

(Requests this dataset has no examples of, on purpose. A model trained on it should
not be expected to handle them.)

## The baseline

Model: Prompt: `week-04/day-1/prompt.md`.
Accuracy on the frozen eval: `____`. Worst kind: `____`, at `____`.

## What I would fix with another day
