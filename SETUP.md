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

The first line prints a version number. The second prints **20 failed**.

That is correct. Every test in this course fails until you write the code it is
testing. Twenty failures with `NotImplementedError` means your setup is perfect.

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

## Not yet

| Week | You will need | Do it now? |
|---|---|---|
| 2 | Google Drive mounted in Colab, for checkpoints | No — Week 2 Day 4 walks through it |
| 3 | A free Hugging Face account, and the course's pinned model | No — Week 3 Day 1 |
| 6 | A local model server | No |

Resist setting these up early. Each one is ten minutes on the day you need it, and an
hour of confusion three weeks before.
