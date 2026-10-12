# Day 2 — Serve it

> **By the end of today** your model answers HTTP requests, and you have scored the
> thing that is actually running.

Yesterday your model became a file. Today it becomes a service: a program that sits on
a port, takes a conversation, and returns the next turn. From this point nothing that
uses your model needs Python, PyTorch, or to know that it was ever fine-tuned.

And then you score it again. Not because you doubt yesterday's file, but because of a
rule worth keeping for life: **the number that counts is the one measured on the thing
that is deployed.**

Everything today goes in `tunelab/served.py`.

---

## Read first

- [ ] `week-06/day-1/COMMANDS.md` — the serving command at the end, and what each flag does · 5 min
- [ ] Your own `tunelab/toolcall.py` from week 4. Today you feed it from a new source · 5 min

---

## What you need to know

### "OpenAI-compatible"

There is no standard for talking to a language model. There is a de facto one: the
shape of request that OpenAI's chat API takes, which nearly every server now accepts.

```json
POST /v1/chat/completions
{
  "model": "notes",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "save milk"}],
  "tools": [{"type": "function", "function": {"name": "add_note", "...": "..."}}],
  "temperature": 0
}
```

and back comes:

```json
{"choices": [{"message": {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "add_note", "arguments": "{\"title\": \"milk\"}"}}]}}]}
```

`llama-server` speaks it. So does the `openai` Python package, which is just a client
for that shape: give it your own machine's address and a made-up key and it talks to
your model, and never goes near OpenAI.

That is also why, tomorrow, the notes agent will run on your model with **no code
change**. It was written against this shape.

### What is different from your own format

Look closely at that reply. Three things differ from the examples you have been
writing since week 4, and all three are in tool calls:

- Every call has an **`id`**.
- `arguments` is a **string** containing JSON, not an object.
- A tool result must say which call it answers: **`tool_call_id`**.

Your examples carry earlier turns — "delete milk", then a `list_titles` call, then its
result. Before those can be sent to a server they have to be put in the API's form.
That is `to_api_messages`, and it is the first exercise.

### The template travels in the file

In week 4 you called `apply_chat_template` yourself to turn messages and tools into
one string. Now the **server** does it. It reads the chat template that yesterday's
conversion stored inside the GGUF file, builds the same string, runs the model, and
then parses the model's `<tool_call>` text back into the `tool_calls` field for you.

That is the best case, and it is why the course serves with `llama-server`: one
template, written once by the model's authors, used in training and in serving.

It can go wrong, and it is worth knowing what wrong looks like. If the server does not
apply the template's tool handling — started without the right flag, or a GGUF that
lost its template — the model still writes its `<tool_call>…</tool_call>`, but the
server hands it back to you as plain text in `content`:

```json
{"message": {"role": "assistant", "content": "<tool_call>\n{\"name\": \"add_note\", ...}\n</tool_call>"}}
```

A naive scorer sees text, records "the model did not call a tool", and reports a
collapse. The model is fine. The plumbing is wrong.

So `to_prediction` checks: if the "text" parses as a tool call, it scores the call for
what it is **and raises a flag**. Your report counts the flags. The right number of
template problems is zero, and if it is not zero you fix the serving command, not the
scorer.

### Waiting for a server

A server that has just started is not ready: it is loading a file of several hundred
megabytes. Ask too early and it refuses the connection, or answers "503, still
loading". Code that talks to servers has to wait politely, and has to give up
eventually with a message a human can act on:

```
no model server answered at http://127.0.0.1:8080/v1 within 60 seconds. Is it running, and on that port?
```

is worth ten of a stack trace from inside an HTTP library.

### Scoring what is served

`eval_served` runs your frozen eval through the server and scores it with the same
`score_predictions` as ever. Then compare with last week, example by example, using
your own `changed` from week 5.

Between last week's score and today's, three things changed: the adapter was merged,
the weights were quantized, and the prompt is now built by the
server. If the score moved, `changed` tells you which examples moved, and that tells
you which of the three to suspect.

---

## Exercises

In `tunelab/served.py`:

1. `to_api_messages(messages)`
2. `to_prediction(message)`
3. `wait_until_ready(base_url, timeout_s, interval_s)`
4. `chat(...)`, `make_served_generate(...)`, `eval_served(...)`

```bash
pytest week-06/day-2 -v
```

The tests use a small fake server that runs inside the test. Nothing needs a model.

### Then, for real

1. Start your server with the command from `COMMANDS.md`. Leave it running in its own
   terminal.
2. In a Python shell: `wait_until_ready`, then `eval_served` on your frozen eval, with
   the same system prompt file as always.
3. Save `data/notes/served_predictions.jsonl` and `data/notes/served.json`.
4. `changed(examples, tuned_predictions, served_predictions)`. Read every example in
   `broken`.
5. Stop the server, start it again on your `q4_k_m` file, and score that too. Keep
   the better one as `served_predictions.jsonl`; put both numbers in your run log.

Row in `logs/run-log.md`: the served accuracy, the number of template problems, how
many examples broke and how many were fixed against last week.

---

## A note on Ollama

Ollama is the other common way to run a GGUF file, and it is pleasant to use. You may
use it instead. Know one thing first: Ollama applies **its own** template, written in
its own template language, taken from its library or from a `Modelfile` you write. It
does not use the one in your file. For a fine-tuned model that is a second copy of the
most fragile string in the project. If you go that way, your count of template
problems is the first thing to look at.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Your model is a service on port 8080. Tomorrow the agent you built in another course
gets its address.
