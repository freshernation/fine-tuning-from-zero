"""Fine-tune the base model on your dataset, with LoRA. Given - read it, do not rewrite it.

    python3 week-05/train.py --smoke --out runs/smoke        # 3 steps, toy model, your laptop
    python3 week-05/train.py --four-bit --out /content/drive/MyDrive/tunelab/run-01     # Colab

Everything that decides WHAT the model learns is code you wrote:

    tunelab/dataset.py   render_prompt, render_completion, check_frozen      (week 4)
    tunelab/sft.py       tokenize_example, collate                           (today)
    week-05/day-1/bridge.py   to_peft_config                                 (yesterday)

This file is the loop around them. Read it against tunelab/loop.py from week 1:
the five lines are all here, inside `Trainer`.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from tunelab.dataset import check_frozen, load_jsonl
from tunelab.sft import collate, tokenize_example

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--train", default=str(ROOT / "data/notes/train.jsonl"))
    p.add_argument("--eval", default=str(ROOT / "data/notes/eval.jsonl"))
    p.add_argument("--prompt", default=str(ROOT / "week-04/day-1/prompt.md"), help="the system prompt file")
    p.add_argument("--out", required=True, help="a folder for this run; on Colab, put it on Drive")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--max-len", type=int, default=1536, help="examples longer than this many tokens are dropped")
    p.add_argument("--eval-every", type=int, default=20, help="optimizer steps between evaluations and checkpoints")
    p.add_argument("--four-bit", action="store_true", help="load the base model in 4 bits (needs a GPU and bitsandbytes)")
    p.add_argument("--smoke", action="store_true", help="3 steps on a toy model: checks the plumbing, learns nothing")
    p.add_argument("--force", action="store_true", help="start again in a folder that already holds a finished run")
    return p.parse_args()


def load_bridge():
    """to_peft_config lives in a day folder, not in the package: load it by path."""
    spec = importlib.util.spec_from_file_location("bridge", ROOT / "week-05/day-1/bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(args):
    if args.smoke:
        from tunelab.tiny import tiny_model, tiny_tokenizer
        tokenizer = tiny_tokenizer()
        return tiny_model(len(tokenizer)), tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.four_bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, quantization_config=quantization, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    return model, tokenizer


class Examples(Dataset):
    """Your tokenized examples, in the shape Trainer wants: something with a length
    that can be indexed."""

    def __init__(self, path, tokenizer, system, max_len):
        rows = [tokenize_example(tokenizer, example, system, max_len) for example in load_jsonl(path)]
        self.rows = [row for row in rows if row is not None]
        self.dropped = len(rows) - len(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def history_from(log):
    """Trainer logs train loss and eval loss as separate entries; join them by step."""
    by_step = {}
    for entry in log:
        row = by_step.setdefault(entry["step"], {"step": entry["step"]})
        if "loss" in entry:
            row["train"] = float(entry["loss"])
        if "eval_loss" in entry:
            row["eval"] = float(entry["eval_loss"])
    return [by_step[step] for step in sorted(by_step) if "train" in by_step[step] or "eval" in by_step[step]]


def main():
    args = arguments()
    out = Path(args.out)

    # 1. The eval set must be the one you froze. Nothing happens until that is true.
    eval_hash = Path(args.eval).with_suffix(".sha256")
    try:
        check_frozen(args.eval, eval_hash)
    except ValueError as error:
        sys.exit(f"Not starting: {error}")

    # 2. A finished run is evidence. Do not write over it by accident.
    if (out / "run.json").exists() and not args.force:
        sys.exit(f"{out} already holds a finished run. Choose another --out, or pass --force to start it again.")
    if args.force and out.exists():
        import shutil
        shutil.rmtree(out)

    # 3. Model, tokenizer, data. The same prompt file the baseline used.
    system = Path(args.prompt).read_text().strip()
    model, tokenizer = load_model(args)
    max_len = 8192 if args.smoke else args.max_len      # the toy tokenizer spends a token per byte
    train = Examples(args.train, tokenizer, system, max_len)
    held_out = Examples(args.eval, tokenizer, system, max_len)
    if len(train) == 0:
        sys.exit("Every training example was longer than --max-len. Raise it.")

    # 4. Adapters, through peft, configured by your own function.
    from peft import get_peft_model
    model = get_peft_model(model, load_bridge().to_peft_config(args.rank, args.alpha, tuple(args.targets), args.dropout))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{len(train)} train examples ({train.dropped} dropped), {len(held_out)} eval ({held_out.dropped} dropped)")
    print(f"trainable parameters: {trainable:,} of {total:,}")

    # 5. The loop. Every argument here is one you can now explain.
    every = 1 if args.smoke else args.eval_every
    settings = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=2 if args.smoke else args.batch_size,
        per_device_eval_batch_size=2 if args.smoke else args.batch_size,
        gradient_accumulation_steps=1 if args.smoke else args.accumulation,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,
        eval_strategy="steps", eval_steps=every,
        save_strategy="steps", save_steps=every, save_total_limit=2,
        logging_steps=1 if args.smoke else 5,
        fp16=torch.cuda.is_available() and not args.smoke,
        use_cpu=args.smoke,
        report_to="none",
        seed=0,
    )
    trainer = Trainer(
        model=model, args=settings, train_dataset=train, eval_dataset=held_out,
        data_collator=lambda rows: collate(rows, tokenizer.pad_token_id),
    )

    # 6. If Colab killed an earlier attempt, carry on from its last checkpoint.
    checkpoints = sorted((out / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)

    # 7. Keep the adapter, the curve, and a record of exactly what was run.
    model.save_pretrained(str(out / "adapter"))
    (out / "history.json").write_text(json.dumps(history_from(trainer.state.log_history), indent=1))
    record = {
        **{key: value for key, value in vars(args).items() if key != "force"},
        "eval_fingerprint": eval_hash.read_text().strip(),
        "train_examples": len(train), "eval_examples": len(held_out),
        "dropped": {"train": train.dropped, "eval": held_out.dropped},
        "trainable_parameters": trainable, "total_parameters": total,
        "steps": trainer.state.global_step,
        "minutes": round((time.time() - started) / 60, 2),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
    }
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(f"done: {record['steps']} steps in {record['minutes']} min. Adapter in {out / 'adapter'}")


if __name__ == "__main__":
    main()
