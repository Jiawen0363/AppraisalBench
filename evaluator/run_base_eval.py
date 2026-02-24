"""
Base evaluation: load corpus + prompt, call evaluator per row, save raw predictions.
"""
import argparse
import json
from pathlib import Path

from data_utils import load_corpus_tsv, DataBuilder
from model_utils import call_evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument("--eval_prompt", type=str, required=True)
    p.add_argument("--event_corpus", type=str, required=True)
    p.add_argument("--mode", type=str, default="base")
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--verbose", action="store_true", help="Print each raw response to terminal")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # Resolve paths
    prompt_path = repo_root / "evaluator" / "prompt" / f"{args.eval_prompt}.txt"
    corpus_path = repo_root / "emotion_appraisal_corpus.tsv"

    p = Path(args.output_file)
    raw_output_path = p.parent / (p.stem + "_raw.jsonl")
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load
    prompt_template = prompt_path.read_text(encoding="utf-8")
    rows = load_corpus_tsv(corpus_path)
    n = len(rows)
    print(f"Loaded {n} rows, writing to {raw_output_path}", flush=True)

    # Run evaluator per row, write raw predictions
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            builder = DataBuilder(prompt_template, row)
            prompt = builder.build_eval_base_prompt()
            raw = call_evaluator(prompt, model=args.eval_model)
            rec = {"sample_id": row["Sentence_id"], "raw_prediction": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.verbose:
                preview = raw[:500] + "..." if len(raw) > 500 else raw
                print(f"[{row['Sentence_id']}] {preview}", flush=True)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1}/{n}", flush=True)

    print(f"Done. Wrote raw predictions to {raw_output_path}", flush=True)


if __name__ == "__main__":
    main()
