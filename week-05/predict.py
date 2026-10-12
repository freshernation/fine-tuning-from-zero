"""Score a trained adapter on the frozen eval set. Given - read it, do not rewrite it.

    python3 week-05/predict.py --run /content/drive/MyDrive/tunelab/run-01
    python3 week-05/predict.py --smoke --run runs/smoke --out runs/smoke-tuned

Writes <out>_predictions.jsonl and <out>.json (default: data/notes/tuned...).
The report records which adapter and which eval set produced it, so that a score
can always be traced back to the run it came from.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from tunelab.dataset import check_frozen, load_jsonl, save_jsonl
from tunelab.generate import make_generate
from tunelab.notes import SCHEMAS
from tunelab.toolcall import run_eval

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the folder train.py wrote: it holds adapter/ and run.json")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--eval", default=str(ROOT / "data/notes/eval.jsonl"))
    p.add_argument("--prompt", default=str(ROOT / "week-04/day-1/prompt.md"))
    p.add_argument("--out", default=str(ROOT / "data/notes/tuned"), help="prefix for the two files written")
    p.add_argument("--smoke", action="store_true", help="use the toy model that train.py --smoke trained")
    return p.parse_args()


def fingerprint_of(folder):
    """A SHA-256 over the adapter's weight file(s): which adapter, exactly, was scored."""
    digest = hashlib.sha256()
    for path in sorted(Path(folder).glob("adapter_model.*")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    args = arguments()
    run = Path(args.run)
    eval_hash = Path(args.eval).with_suffix(".sha256")
    try:
        check_frozen(args.eval, eval_hash)
    except ValueError as error:
        sys.exit(f"Not scoring: {error}")
    if not (run / "adapter" / "adapter_config.json").exists():
        sys.exit(f"{run} has no adapter/ in it. Has that run finished?")

    if args.smoke:
        from tunelab.tiny import tiny_model, tiny_tokenizer
        tokenizer = tiny_tokenizer()
        base = tiny_model(len(tokenizer))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
        if torch.cuda.is_available():
            base = base.to("cuda")

    # The step people forget: attach the adapter. Without this line you are scoring
    # the untuned model, and it will look exactly like your baseline.
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(run / "adapter"))
    model.eval()

    system = Path(args.prompt).read_text().strip()
    generate = make_generate(model, tokenizer, SCHEMAS, system, max_new_tokens=16 if args.smoke else 128)
    examples = load_jsonl(args.eval)
    predictions, report = run_eval(generate, examples)

    save_jsonl(args.out + "_predictions.jsonl", predictions)
    record = {
        "model": args.model, "revision": args.revision,
        "adapter_fingerprint": fingerprint_of(run / "adapter"),
        "eval_fingerprint": eval_hash.read_text().strip(),
        "run": json.loads((run / "run.json").read_text()),
        **report,
    }
    Path(args.out + ".json").write_text(json.dumps(record, indent=1))
    print(f'accuracy {report["accuracy"]:.3f} on {report["n"]} examples. Wrote {args.out}_predictions.jsonl and {args.out}.json')


if __name__ == "__main__":
    main()
