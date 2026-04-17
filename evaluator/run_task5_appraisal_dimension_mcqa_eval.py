"""
Task 5: 7-option MCQA — match the appraisal_expansion for the target dimension, given seed event
(seed event + seven appraisal-dimension sentences; no per-dimension definition text).

Input (choose one):
  - --question_file: prebuilt JSONL (same rows as the build script).
  - --scenarios_file + --appraisal_dimension: build each row on the fly (no task5_question export needed).

Each output line matches Task 2 structure, except:
  - task.given_information is \"seed_event\"
  - choice letters are A–G.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from openai import APIConnectionError

from model_utils import call_evaluator
from task5_mcqa_records import DIM_ORDER, iter_task5_question_rows, read_jsonl


def _pick_line_instruction(row: dict) -> str:
    disp = str(row.get("appraisal_dimension_display", "")).strip()
    return f"Pick the line that best captures the Experiencer's appraisal along {disp}."


def format_full_question(row: dict) -> str:
    parts = [
        _pick_line_instruction(row),
        "",
    ]
    for letter in "ABCDEFG":
        parts.append(f"{letter}. {row.get(f'option_{letter}', '')}")
    return "\n".join(parts)


def fill_prompt(template: str, row: dict) -> str:
    disp = str(row.get("appraisal_dimension_display", ""))
    text = template.replace("{Appraisal_dimension_display}", disp)
    text = text.replace("{Seed_event}", str(row.get("seed_event", "")))
    for letter in "ABCDEFG":
        text = text.replace(f"{{option {letter}}}", str(row.get(f"option_{letter}", "")))
    return text


def parse_option_letter(raw: str) -> str | None:
    t = (raw or "").strip()
    if not t:
        return None
    m = re.search(r"\b([A-G])\b", t.upper())
    if m:
        return m.group(1)
    if len(t) == 1 and t.upper() in "ABCDEFG":
        return t.upper()
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Task 5 MCQA eval: --question_file OR (--scenarios_file + --appraisal_dimension)."
    )
    p.add_argument("--vllm_endpoint", type=str, default=None)
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument(
        "--eval_prompt",
        type=str,
        default="task5/given_seed_definition_match_appraisal_expansion",
        help="Stem under evaluator/prompt, e.g. task5/given_seed_definition_match_appraisal_expansion",
    )
    p.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="Prebuilt JSONL under repo root (e.g. output/evaluation/task5_question/attention.jsonl).",
    )
    p.add_argument(
        "--scenarios_file",
        type=str,
        default=None,
        help="If set with --appraisal_dimension, build questions on the fly (no pre-export needed).",
    )
    p.add_argument(
        "--appraisal_dimension",
        type=str,
        default=None,
        help="Target dimension key, e.g. attention (use with --scenarios_file).",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Only used with --scenarios_file; must match run_task5_build_7way_mcq.py --seed for identical options.",
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
    output_path = repo_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.question_file:
        if args.scenarios_file or args.appraisal_dimension:
            raise SystemExit(
                "Use either --question_file alone OR --scenarios_file with --appraisal_dimension, not both."
            )
        question_path = repo_root / args.question_file
        rows: list[dict] = []
        with question_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        source_desc = str(question_path)
    elif args.scenarios_file and args.appraisal_dimension:
        dim = args.appraisal_dimension.strip().lower()
        if dim not in DIM_ORDER:
            raise SystemExit(f"Invalid --appraisal_dimension {dim!r}. Valid: {DIM_ORDER}")
        scenarios_path = repo_root / args.scenarios_file
        scenarios = read_jsonl(scenarios_path)
        rows = list(
            iter_task5_question_rows(
                scenarios,
                dim,
                args.shuffle_seed,
                limit=args.limit,
            )
        )
        source_desc = f"{scenarios_path} dim={dim} (on-the-fly)"
        # When reading from file, limit truncates loaded rows; when on-the-fly, iter already applied limit.
        args.limit = None
    else:
        raise SystemExit(
            "Provide --question_file OR both --scenarios_file and --appraisal_dimension "
            "(see evaluator/run_task5_appraisal_dimension_mcqa_eval.py docstring)."
        )

    start_idx = max(0, args.offset)
    if start_idx >= len(rows):
        print(f"Offset out of range: offset={start_idx}, total={len(rows)}", flush=True)
        return
    rows = rows[start_idx:]
    if args.limit is not None:
        rows = rows[: args.limit]

    prompt_template = prompt_path.read_text(encoding="utf-8")

    n = len(rows)
    print(f"Loaded {n} questions from {source_desc} -> {output_path}", flush=True)
    print(f"API base_url={base_url}", flush=True)

    write_mode = "a" if args.append else "w"
    with output_path.open(write_mode, encoding="utf-8", buffering=1) as out_f:
        for i, row in enumerate(rows, start=start_idx):
            gold = str(row.get("correct_option", "")).strip().upper()[:1]
            if gold not in "ABCDEFG":
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
                    "given_information": "seed_event",
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
