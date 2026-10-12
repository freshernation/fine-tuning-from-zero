"""Week 5, Day 2 - the first real run.

Run me:  pytest week-05/day-2 -v

Your code goes in tunelab/sft.py. week-05/train.py is given: the last tests here run
it, for three steps, on a toy model, to check that your code and it fit together.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tunelab.dataset import freeze, render_completion, render_prompt, save_jsonl
from tunelab.sft import collate, effective_batch_size, steps_per_epoch, tokenize_example

ROOT = Path(__file__).parents[2]
SYSTEM = "You manage notes."
ADD = {"id": "s1", "seed_id": "s1", "kind": "add", "source": "seed",
       "messages": [{"role": "user", "content": "jot down café — two litres"}],
       "expected": {"tool": "add_note", "arguments": {"title": "café", "body": "two litres"}}}
QUIET = {"id": "s2", "seed_id": "s2", "kind": "clarify", "source": "seed",
         "messages": [{"role": "user", "content": "delete it"}],
         "expected": {"tool": None, "arguments": {}, "text": "Which note do you want to delete?"}}


# ---- tokenize_example --------------------------------------------------------------

@pytest.mark.parametrize("example", [ADD, QUIET])
def test_the_model_is_shown_prompt_then_answer(chat_tokenizer, example):
    row = tokenize_example(chat_tokenizer, example, SYSTEM, max_len=8000)
    assert set(row) == {"input_ids", "attention_mask", "labels"}
    assert chat_tokenizer.decode(row["input_ids"]) == render_prompt(chat_tokenizer, example, SYSTEM) + render_completion(example)
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"])
    assert set(row["attention_mask"]) == {1}


@pytest.mark.parametrize("example", [ADD, QUIET])
def test_the_loss_is_on_the_answer_and_nothing_else(chat_tokenizer, example):
    row = tokenize_example(chat_tokenizer, example, SYSTEM, max_len=8000)
    learned = [token for token in row["labels"] if token != -100]
    assert chat_tokenizer.decode(learned) == render_completion(example), "the unmasked labels must spell exactly the answer"
    first = row["labels"].index(learned[0]) if -100 not in row["labels"][:1] else next(i for i, t in enumerate(row["labels"]) if t != -100)
    assert all(token == -100 for token in row["labels"][:first]), "every prompt token is masked"
    assert row["labels"][first:] == row["input_ids"][first:], "from the first answer token on, labels are the tokens themselves"
    assert chat_tokenizer.decode(row["input_ids"][:first]) == render_prompt(chat_tokenizer, example, SYSTEM), "the mask starts exactly where the prompt ends"


def test_the_answer_ends_with_the_end_of_turn_token(chat_tokenizer):
    row = tokenize_example(chat_tokenizer, ADD, SYSTEM, max_len=8000)
    assert row["labels"][-1] == chat_tokenizer.eos_token_id, "the model must be taught to stop"


def test_no_extra_special_tokens_are_added(chat_tokenizer):
    row = tokenize_example(chat_tokenizer, ADD, SYSTEM, max_len=8000)
    prompt_ids = chat_tokenizer(render_prompt(chat_tokenizer, ADD, SYSTEM), add_special_tokens=False).input_ids
    assert row["input_ids"][:len(prompt_ids)] == prompt_ids, "the prompt is tokenized exactly as it will be at inference"


def test_an_example_that_does_not_fit_is_dropped_not_truncated(chat_tokenizer):
    full = tokenize_example(chat_tokenizer, ADD, SYSTEM, max_len=8000)
    n = len(full["input_ids"])
    assert tokenize_example(chat_tokenizer, ADD, SYSTEM, max_len=n) is not None
    assert tokenize_example(chat_tokenizer, ADD, SYSTEM, max_len=n - 1) is None


# ---- collate -----------------------------------------------------------------------

ROWS = [
    {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 7, 8]},
    {"input_ids": [5, 9], "attention_mask": [1, 1], "labels": [-100, 9]},
]


def test_collate_pads_on_the_right_to_the_longest_row():
    batch = collate(ROWS, pad_id=0)
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert all(t.dtype == torch.int64 and t.shape == (2, 4) for t in batch.values())
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8], [5, 9, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_padding_is_never_something_to_learn():
    batch = collate(ROWS, pad_id=0)
    assert batch["labels"].tolist() == [[-100, -100, 7, 8], [-100, 9, -100, -100]]
    batch = collate(ROWS, pad_id=9)            # even when the pad id is a real token in the data
    assert batch["labels"].tolist() == [[-100, -100, 7, 8], [-100, 9, -100, -100]]
    assert batch["input_ids"].tolist()[1] == [5, 9, 9, 9]


def test_collate_leaves_its_input_alone():
    before = json.dumps(ROWS)
    collate(ROWS, pad_id=0)
    assert json.dumps(ROWS) == before


# ---- arithmetic --------------------------------------------------------------------

def test_effective_batch_size():
    assert effective_batch_size(batch_size=4, accumulation=4) == 16
    assert effective_batch_size(8, 1) == 8


def test_steps_per_epoch_counts_optimizer_steps():
    assert steps_per_epoch(n_examples=800, batch_size=4, accumulation=4) == 50
    assert steps_per_epoch(798, 4, 4) == 50                 # 200 batches (the last one short) -> 50 steps
    assert steps_per_epoch(801, 4, 4) == 51                 # 201 batches -> 50 full groups and one short
    assert steps_per_epoch(10, 4, 1) == 3
    assert steps_per_epoch(3, 4, 4) == 1


# ---- train.py, three steps on a toy model ------------------------------------------

def make_data(folder):
    rows = []
    for i in range(12):
        example = json.loads(json.dumps(ADD if i % 2 else QUIET))
        example["id"] = example["seed_id"] = f"t{i:02d}"
        example["messages"][0]["content"] += f" {i}"
        rows.append(example)
    save_jsonl(folder / "train.jsonl", rows[:8])
    save_jsonl(folder / "eval.jsonl", rows[8:])
    freeze(folder / "eval.jsonl", folder / "eval.sha256")
    (folder / "prompt.md").write_text(SYSTEM)


def run_train(folder, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "week-05" / "train.py"), "--smoke",
         "--train", str(folder / "train.jsonl"), "--eval", str(folder / "eval.jsonl"),
         "--prompt", str(folder / "prompt.md"), "--out", str(folder / "run"), *extra],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_smoke_run_trains_and_writes_everything(tmp_path):
    make_data(tmp_path)
    done = run_train(tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    run = tmp_path / "run"
    assert (run / "adapter" / "adapter_config.json").exists()
    history = json.loads((run / "history.json").read_text())
    assert [row["step"] for row in history] == [1, 2, 3]
    assert all(isinstance(row["train"], float) for row in history) and isinstance(history[-1]["eval"], float)
    record = json.loads((run / "run.json").read_text())
    assert record["eval_fingerprint"] == (tmp_path / "eval.sha256").read_text().strip()
    assert record["dropped"] == {"train": 0, "eval": 0} and record["smoke"] is True
    assert record["trainable_parameters"] > 0


def test_a_finished_run_is_not_overwritten_by_accident(tmp_path):
    make_data(tmp_path)
    assert run_train(tmp_path).returncode == 0
    again = run_train(tmp_path)
    assert again.returncode != 0 and "--force" in again.stderr
    assert run_train(tmp_path, "--force").returncode == 0


def test_training_refuses_to_start_on_an_edited_eval_set(tmp_path):
    make_data(tmp_path)
    with open(tmp_path / "eval.jsonl", "a") as f:
        f.write(json.dumps(dict(ADD, id="zz", seed_id="zz")) + "\n")
    done = run_train(tmp_path)
    assert done.returncode != 0 and "restore" in done.stderr.lower()
    assert not (tmp_path / "run" / "adapter").exists()
