"""Week 2 milestone - a GPT you wrote, trained with checkpoints.

Run me:  pytest week-02/milestone -v      (about half a minute)

These tests train the SMALL model on your laptop. The BIG one is trained on Colab
and is graded by your run log and by Friday.
"""

from pathlib import Path

TEXT = (Path(__file__).parents[2] / "data" / "tiny.txt").read_text()
TARGET = 2.26


def test_small_gpt_beats_the_bigram_floor(load, tmp_path):
    m = load("train_gpt.py")
    result = m.run(TEXT, m.SMALL, steps=400, path=str(tmp_path / "ck.pt"))
    start, end = result["history"][0]["val"], result["val_loss"]
    assert abs(start - 4.17) < 0.15, f"should start very near ln(65) = 4.17, got {start:.2f}"
    # last week's model could not get under about 2.5 however long it trained
    assert end < TARGET, f"validation loss {end:.3f} did not get under {TARGET}"
    assert len(result["sample"]) == 300
    assert result["params"] == 64193


def test_big_config_is_under_the_size_cap(load):
    assert load("train_gpt.py").count_big_params() <= 2_000_000


def test_run_resumes_from_its_checkpoint(load, tmp_path):
    m = load("train_gpt.py")
    path = str(tmp_path / "ck.pt")
    first = m.run(TEXT[:20000], m.SMALL, steps=40, path=path, chunk=20)
    assert [row["step"] for row in first["history"]] == [0, 19, 20, 39]
    # the session died; the same call again with a bigger target carries on
    second = m.run(TEXT[:20000], m.SMALL, steps=80, path=path, chunk=20)
    assert [row["step"] for row in second["history"]] == [0, 19, 20, 39, 40, 59, 60, 79]
    assert second["history"][:4] == first["history"]


def test_same_seed_same_run(load, tmp_path):
    m = load("train_gpt.py")
    a = m.run(TEXT[:20000], m.SMALL, steps=20, path=str(tmp_path / "a.pt"), seed=3)
    b = m.run(TEXT[:20000], m.SMALL, steps=20, path=str(tmp_path / "b.pt"), seed=3)
    assert a["sample"] == b["sample"]
