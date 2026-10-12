# Starter agent

The notes agent from **Agent From Zero**, finished and working: a loop you wrote by
hand in that course, five tools over a JSON file of notes, and a small layer that
talks to a model.

If you took that course, this is your week-6 project and you may use your own copy
instead. If you did not, this is the thing your fine-tuned model is going to run
inside.

> This folder contains finished Agent From Zero code. If you are also taking that
> course, do not read it until you have passed week 6 there.

| File | What it is |
|---|---|
| `kit.py` | The five tools, their descriptions (`SCHEMAS`), and the code that runs them |
| `agent.py` | The loop: ask the model, run the tools it asks for, go round again |
| `guards.py` | A step cap, repeat detection, clipping long tool results |
| `cli.py` | Talk to it: `python3 cli.py` |
| `llm/` | One way to talk to any model: `fake`, `claude`, `openai` |

## Try it

```bash
cd starter-agent
AGENT_TRACE=1 python3 cli.py
```

With no configuration it uses the **fake** model, which repeats your message back.
That proves the loop runs, and costs nothing.

## Point it at a model

Copy `.env.example` to `.env` in this folder.

**A hosted model**, if you have a key (optional in this course):

```
AGENT_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Your own fine-tuned model**, in week 6. No code changes: the `openai` backend talks
to any server that speaks the same protocol, and the server you start in week 6 does.

```
AGENT_PROVIDER=openai
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
AGENT_OPENAI_MODEL=notes
```

## What the course uses from here

- **Weeks 4 and 5:** only `kit.py`'s `SCHEMAS`: the exact words the model is told
  about each tool. `tunelab/notes.py` reads them from this folder.
- **Week 6:** the whole agent, running on your model.
