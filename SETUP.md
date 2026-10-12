# Day Zero — Setup

Do this before Monday of week 1. It takes about forty minutes, most of it waiting for
one download.

You have set up a Python project before, so this is brief. If any step fails, that is
a stuck-log entry and a question for the first live hour — not a reason to start late.

---

## 1. Python 3.10 or newer

```bash
python3 --version
```

## 2. Get your own copy of the course

Make your **own** repository from the course repo — your instructor will tell you
whether to fork it or use it as a template — then clone *yours*:

```bash
git clone https://github.com/YOUR-NAME/YOUR-REPO.git
cd YOUR-REPO
```

You push to your copy every day. Your instructor reads it.

Keep the repository **public**, or Colab will not be able to clone it without extra
setup. If it has to be private, tell your instructor before Friday of week 1.

## 3. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs PyTorch, which is large. Expect several minutes.

## 4. Prove it works

```bash
python3 -c "import torch, tunelab; print(torch.__version__)"
pytest week-01
```

The first line prints a version number. The second prints **27 failed**.

That is correct. Every test in this course fails until you write the code it is
testing. Twenty-seven failures with `NotImplementedError` means your setup is perfect.

If you see errors about imports instead of `NotImplementedError`, stop and ask.

## 5. Colab

You need a Google account. Nothing else — the free tier is enough for the whole
course.

1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. **File → Open notebook → GitHub**, paste the address of your repo, and open
   `week-01/colab.ipynb`.
3. **Runtime → Change runtime type → T4 GPU → Save.**
4. Stop there. You will run it on Friday of week 1.

Three things about free Colab that will matter later:

- **It forgets everything when the session ends.** Files you create on it are gone.
  Your code lives in git; from week 2, your checkpoints live in Google Drive.
- **Sessions time out** — after a stretch of no activity, and after some hours
  regardless.
- **The GPU is not always available.** If Colab says none is free, try again in an
  hour. Do not leave Thursday's training run until Thursday night.

## 6. Tell git who you are

If you have not on this machine:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Before week 2

Nothing to install. Week 2 Day 4 walks you through mounting Google Drive in Colab, for
checkpoints.

## Before week 3

`pip install -r requirements.txt` again: it now brings in `transformers` and `peft`.
The week's pretrained model, `HuggingFaceTB/SmolLM2-135M-Instruct`, downloads itself
inside Friday's notebook. No account is needed.

## Before week 4

`pip install -r requirements.txt` again, for the starter agent's two small
dependencies.

**The course's base model for weeks 4 to 6 is `Qwen/Qwen2.5-0.5B-Instruct`**, pinned at
revision `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0, about 1 GB). It
downloads itself in Colab; you never need it on your laptop until week 6.

Optional: if you have an API key from Agent From Zero, copy `starter-agent/.env.example`
to `starter-agent/.env` and fill it in. Week 4 can use a hosted model to help write
examples. It is never required; every step has a free route.

## Not yet

| Week | You will need | Do it now? |
|---|---|---|
| 6 | A local model server | No |

Resist setting things up early. Each one is ten minutes on the day you need it, and an
hour of confusion three weeks before.
