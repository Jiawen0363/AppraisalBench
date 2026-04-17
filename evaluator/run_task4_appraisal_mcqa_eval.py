"""
Task 4: same 4-option appraisal MCQA as Task 2, but the prompt also includes the gold emotion label.

Each output line matches Task 2 structure except task.given_information is dialog_emotion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from openai import APIConnectionError

from dialog_roles import relabel_dialog_block
from model_utils import call_evaluator


def format_emotion_sentence(raw_emotion: str | None) -> str:
    """
    Natural English for the prompt: title-case emotion nouns; anger -> 'angry ...'.
    """
    e = str(raw_emotion or "").strip().lower()
    if not e:
        return "The Experiencer's emotion label is not provided."
    if e == "anger":
        return "The Experiencer is feeling angry about what happened."
    return f"The Experiencer is feeling {e.title()} towards the event."


def format_full_question(row: dict) -> str:
    stem = str(row.get("question_stem", "")).strip()
    parts = [
        f"Question: {stem}",
        "",
        f"A. {row.get('option_A', '')}",
        f"B. {row.get('option_B', '')}",
        f"C. {row.get('option_C', '')}",
        f"D. {row.get('option_D', '')}",
    ]
    return "\n".join(parts)


def fill_prompt(template: str, row: dict) -> str:
    dialog_text = relabel_dialog_block(str(row.get("dialog", "")))
    text = template.replace("{Dialogue}", dialog_text)
    text = text.replace("{Emotion_sentence}", format_emotion_sentence(row.get("emotion")))
    text = text.replace("[DIMENSION-SPECIFIC QUESTION]", str(row.get("question_stem", "")))
    text = text.replace("{option A}", str(row.get("option_A", "")))
    text = text.replace("{option B}", str(row.get("option_B", "")))
    text = text.replace("{option C}", str(row.get("option_C", "")))
    text = text.replace("{option D}", str(row.get("option_D", "")))
    return text


def parse_option_letter(raw: str) -> str | None:
    t = (raw or "").strip()
    if not t:
        return None
    m = re.search(r"\b([ABCD])\b", t.upper())
    if m:
        return m.group(1)
    if len(t) == 1 and t.upper() in "ABCD":
        return t.upper()
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_endpoint", type=str, default=None)
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument(
        "--eval_prompt",
        type=str,
        default="task4/given_dialog_emotion_infer_appraisal",
        help="Stem under evaluator/prompt, e.g. task4/given_dialog_emotion_infer_appraisal",
    )
    p.add_argument(
        "--question_file",
        type=str,
        required=True,
        help="JSONL under repo root (same format as task2_question, e.g. .../attention.jsonl)",
    )
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0, help="Skip first N question rows.")
    p.add_argument("--append", action="store_true", help="Append to output file instead of overwrite.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def resolve_eval_base_url(cli: str | None) -> str:
    if cli is not None and str(cli).strip():
        return str(cli).strip()
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
        or "http://35.164.11.19:3887/v1"
    )


def main():
    args = parse_args()
    base_url = resolve_eval_base_url(args.vllm_endpoint)
    repo_root = Path(__file__).resolve().parent.parent
    prompt_path = repo_root / "evaluator" / "prompt" / f"{args.eval_prompt}.txt"
    question_path = repo_root / args.question_file
    output_path = repo_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    rows: list[dict] = []
    with question_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    start_idx = max(0, args.offset)
    if start_idx >= len(rows):
        print(f"Offset out of range: offset={start_idx}, total={len(rows)}", flush=True)
        return
    rows = rows[start_idx:]
    if args.limit is not None:
        rows = rows[: args.limit]

    n = len(rows)
    print(f"Loaded {n} questions from {question_path} -> {output_path}", flush=True)
    print(f"API base_url={base_url}", flush=True)

    write_mode = "a" if args.append else "w"
    with output_path.open(write_mode, encoding="utf-8", buffering=1) as out_f:
        for i, row in enumerate(rows, start=start_idx):
            gold = str(row.get("correct_option", "")).strip().upper()[:1]
            if gold not in "ABCD":
                print(f"[warn] bad correct_option for row {i}, skipping", flush=True)
                continue

            prompt = fill_prompt(prompt_template, row)
            try:
                raw = call_evaluator(
                    prompt,
                    model=args.eval_model,
                    base_url=base_url,
                    max_tokens=32,
                )
            except APIConnectionError:
                print(
                    "Connection refused or unreachable API. "
                    "Check OPENAI_API_KEY / run_tasks/.env and base URL.",
                    flush=True,
                )
                raise SystemExit(1)

            pred = parse_option_letter(raw)
            ok = pred is not None and pred == gold
            full_question = format_full_question(row)

            out = {
                "sample": {
                    "scenario_id": row.get("scenario_id"),
                    "appraisal_dimension": row.get("appraisal_dimension"),
                    "emotion": row.get("emotion"),
                },
                "task": {
                    "given_information": "dialog_emotion",
                    "question": full_question,
                },
                "choice": {
                    "model": pred,
                    "gold": gold,
                    "is_correct": ok,
                },
                "model": {
                    "raw_text": raw,
                },
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()

            if args.verbose:
                print(
                    f"[{row.get('scenario_id')}] raw={raw!r} -> pred={pred} gold={gold} ok={ok}",
                    flush=True,
                )
            if (i + 1) % 20 == 0 or i == start_idx:
                print(f"  {i + 1}/{n}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
