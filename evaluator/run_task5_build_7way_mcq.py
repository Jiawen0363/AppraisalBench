#!/usr/bin/env python3
"""
Build Task 5 question JSONL (one file per appraisal dimension).

Same rows as on-the-fly eval; optional export for inspection. The prompt embeds the task line
(no question_stem JSON field). See evaluator/task5_mcqa_records.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from task5_mcqa_records import DIM_ORDER, build_task5_question_row, read_jsonl

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Build Task 5 7-way MCQA JSONL (7 files).")
    p.add_argument(
        "--scenarios",
        type=Path,
        default=_REPO_ROOT / "output" / "seed2scenario" / "scenarios.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "evaluation" / "task5_question",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed; combined per (scenario_id, dimension).")
    p.add_argument("--limit", type=int, default=None, help="Max scenario rows to process.")
    p.add_argument(
        "--dimensions",
        type=str,
        default="all",
        help="Comma-separated dimensions or 'all'.",
    )
    args = p.parse_args()

    scenarios = read_jsonl(args.scenarios)

    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    if args.dimensions.strip().lower() == "all":
        selected_dims = list(DIM_ORDER)
    else:
        raw = [d.strip().lower() for d in args.dimensions.split(",") if d.strip()]
        invalid = [d for d in raw if d not in DIM_ORDER]
        if invalid:
            raise SystemExit(f"Invalid dimensions: {invalid}. Valid: {DIM_ORDER}")
        selected_dims = [d for d in DIM_ORDER if d in set(raw)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_files = {
        dim: (args.output_dir / f"{dim}.jsonl").open("w", encoding="utf-8")
        for dim in selected_dims
    }

    try:
        for s in scenarios:
            scenario_id = str(s.get("id", "")).strip()
            for dim in selected_dims:
                rec = build_task5_question_row(s, dim, args.seed)
                if rec is None:
                    if scenario_id:
                        print(
                            f"[warn] skip {scenario_id} for dim={dim}: missing data",
                            flush=True,
                        )
                    continue
                out_files[dim].write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_files[dim].flush()

        print(f"Done. Wrote {len(selected_dims)} files under {args.output_dir}", flush=True)
    finally:
        for f in out_files.values():
            f.close()


if __name__ == "__main__":
    main()
