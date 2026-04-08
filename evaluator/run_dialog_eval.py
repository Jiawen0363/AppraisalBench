"""
Dialog evaluation: load dialogs.jsonl, build prompt from conversation, call evaluator, save raw predictions.
"""
import argparse
import json
from pathlib import Path

from data_utils import load_corpus_tsv, load_dialogs_jsonl, build_dialog_eval_prompt
from model_utils import call_evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_endpoint", type=str, default=None, help="vLLM/OpenAI API base URL (default: VLLM_BASE_URL or http://localhost:8000/v1)")
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument("--eval_prompt", type=str, default=None,
                   help="Prompt name (e.g. eval_base, eval_given_appraisal). Auto-set from mode when mode ends with _given_appraisal.")
    p.add_argument("--event_corpus", type=str, required=True)
    p.add_argument("--mode", type=str, default="dialog_first",
                   help="dialog_first | dialog_full | dialog_first_given_appraisal | dialog_full_given_appraisal")
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--dialog_file", type=str, default="output/dialog/dialogs.jsonl",
                   help="Input dialogs.jsonl (profile_id, corpus_id, conversation).")
    p.add_argument("--verbose", action="store_true", help="Print each raw response to terminal")
    p.add_argument("--limit", type=int, default=None, help="Max number of dialogs to process")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # When mode is *_given_appraisal, always use eval_given_appraisal prompt (only output emotion);
    # otherwise prompt is eval_base (output appraisal + emotion). Explicit --eval_prompt is ignored in _given_appraisal mode.
    if args.mode.endswith("_given_appraisal"):
        eval_prompt = "eval_given_appraisal"
    else:
        eval_prompt = args.eval_prompt or "eval_base"
    prompt_path = repo_root / "evaluator" / "prompt" / f"{eval_prompt}.txt"
    dialog_path = repo_root / args.dialog_file
    corpus_path = repo_root / "emotion_appraisal_corpus.tsv"
    output_path = repo_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    dialogs = load_dialogs_jsonl(dialog_path)
    if args.limit is not None:
        dialogs = dialogs[: args.limit]
    n = len(dialogs)

    corpus_by_id = None
    if args.mode.endswith("_given_appraisal"):
        corpus_rows = load_corpus_tsv(corpus_path)
        corpus_by_id = {str(row["Sentence_id"]): row for row in corpus_rows}
        print(f"Loaded {n} dialogs and {len(corpus_by_id)} corpus rows (appraisal gold from {corpus_path})", flush=True)
    else:
        print(f"Loaded {n} dialogs from {dialog_path}, writing to {output_path}", flush=True)

    with output_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(dialogs):
            corpus_row = None
            if corpus_by_id is not None:
                corpus_row = corpus_by_id.get(str(record.get("corpus_id", "")))
            prompt = build_dialog_eval_prompt(prompt_template, record, mode=args.mode, corpus_row=corpus_row)
            raw = call_evaluator(prompt, model=args.eval_model, base_url=args.vllm_endpoint)
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
