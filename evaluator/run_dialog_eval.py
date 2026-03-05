"""
Dialog evaluation: load dialogs.jsonl, build prompt from conversation, call evaluator, save raw predictions.
"""
import argparse
import json
from pathlib import Path

from data_utils import load_dialogs_jsonl, build_dialog_eval_prompt
from model_utils import call_evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument("--eval_prompt", type=str, required=True)
    p.add_argument("--event_corpus", type=str, required=True)
    p.add_argument("--mode", type=str, default="dialog_first")
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--dialog_file", type=str, default="output/dialog/dialogs.jsonl",
                   help="Input dialogs.jsonl (profile_id, corpus_id, conversation).")
    p.add_argument("--verbose", action="store_true", help="Print each raw response to terminal")
    p.add_argument("--limit", type=int, default=None, help="Max number of dialogs to process")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    prompt_path = repo_root / "evaluator" / "prompt" / f"{args.eval_prompt}.txt"
    dialog_path = repo_root / args.dialog_file
    output_path = repo_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    dialogs = load_dialogs_jsonl(dialog_path)
    if args.limit is not None:
        dialogs = dialogs[: args.limit]
    n = len(dialogs)
    print(f"Loaded {n} dialogs from {dialog_path}, writing to {output_path}", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(dialogs):
            prompt = build_dialog_eval_prompt(prompt_template, record, mode=args.mode)
            raw = call_evaluator(prompt, model=args.eval_model)
            sample_id = record.get("corpus_id", str(i))
            rec = {"sample_id": sample_id, "raw_prediction": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.verbose:
                preview = raw[:500] + "..." if len(raw) > 500 else raw
                print(f"[{sample_id}] {preview}", flush=True)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1}/{n}", flush=True)

    print(f"Done. Wrote raw predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
