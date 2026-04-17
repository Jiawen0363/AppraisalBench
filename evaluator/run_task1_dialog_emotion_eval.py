"""
Task 1: emotion-only eval from dialog; gold label from scenarios.jsonl.

Each output line:
  sample_id, given_information ("dialog"), predicted_emotion, golden_emotion, prediction (bool)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openai import APIConnectionError

from data_utils import load_dialogs_jsonl
from model_utils import call_evaluator

EMOTION_LABELS = (
    "anger",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "sadness",
    "shame",
)


def load_scenarios_emotions(path: str | Path) -> dict[str, str]:
    """Map scenario id -> emotion string (as stored, usually lowercase)."""
    path = Path(path)
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            em = row.get("emotion")
            if sid and em is not None:
                out[sid] = str(em).strip()
    return out


def normalize_golden_emotion(gold: str) -> str:
    return gold.strip().lower()


def dialog_to_context_text(dialog_record: dict) -> str:
    """Format full multi-turn conversation for the prompt."""
    lines: list[str] = []
    for turn in dialog_record.get("conversation", []):
        for role, content in turn.items():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def parse_predicted_emotion(raw: str) -> str | None:
    """
    Map free-form model output to one canonical label in EMOTION_LABELS (lowercase), or None.
    """
    t = (raw or "").strip()
    if not t:
        return None
    simple = t.lower().rstrip(".")
    if simple in EMOTION_LABELS:
        return simple
    for label in EMOTION_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", simple):
            return label
    return None


def emotion_prediction_correct(predicted: str | None, golden_emotion: str) -> bool:
    """True iff parsed prediction matches gold (case-insensitive)."""
    gold = normalize_golden_emotion(golden_emotion)
    if not gold or predicted is None:
        return False
    return predicted == gold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_endpoint", type=str, default=None)
    p.add_argument("--eval_model", type=str, required=True)
    p.add_argument(
        "--eval_prompt",
        type=str,
        default="task1/eval_emotion_base",
        help="Path stem under evaluator/prompt, e.g. task1/eval_emotion_base (uses {context_text})",
    )
    p.add_argument(
        "--dialog_file",
        type=str,
        default="output/dialog/gpt4o/dialog_advanced.jsonl",
    )
    p.add_argument(
        "--scenarios_file",
        type=str,
        default="output/seed2scenario/scenarios.jsonl",
    )
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0, help="Skip first N dialog rows.")
    p.add_argument("--append", action="store_true", help="Append to output file instead of overwrite.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    prompt_path = repo_root / "evaluator" / "prompt" / f"{args.eval_prompt}.txt"
    dialog_path = repo_root / args.dialog_file
    scenarios_path = repo_root / args.scenarios_file
    output_path = repo_root / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    dialogs = load_dialogs_jsonl(dialog_path)
    gold_by_id = load_scenarios_emotions(scenarios_path)
    start_idx = max(0, args.offset)
    if start_idx >= len(dialogs):
        print(f"Offset out of range: offset={start_idx}, total={len(dialogs)}", flush=True)
        return
    dialogs = dialogs[start_idx:]
    if args.limit is not None:
        dialogs = dialogs[: args.limit]

    n = len(dialogs)
    print(f"Loaded {n} dialogs, {len(gold_by_id)} scenarios -> {output_path}", flush=True)

    # Line-buffered + flush each row so output file is visible while the run is in progress.
    write_mode = "a" if args.append else "w"
    with output_path.open(write_mode, encoding="utf-8", buffering=1) as f:
        for i, record in enumerate(dialogs, start=start_idx):
            sample_id = str(record.get("scenario_id", record.get("corpus_id", str(i))))
            golden = gold_by_id.get(sample_id)
            if golden is None:
                print(f"[warn] no scenario for sample_id={sample_id}, skipping", flush=True)
                continue

            prompt = prompt_template.replace(
                "{context_text}", dialog_to_context_text(record)
            )
            try:
                raw = call_evaluator(
                    prompt,
                    model=args.eval_model,
                    base_url=args.vllm_endpoint,
                    max_tokens=None,
                )
            except APIConnectionError as e:
                print(
                    "Connection refused or unreachable API. "
                    "Check OPENAI_API_KEY / run_tasks/.env and base URL, or start local vLLM.",
                    flush=True,
                )
                raise SystemExit(1) from e
            parsed = parse_predicted_emotion(raw)
            pred_ok = emotion_prediction_correct(parsed, golden)

            # predicted_emotion: prefer canonical parsed label; else raw strip (unparseable)
            display_pred = parsed if parsed is not None else (raw or "").strip()

            out = {
                "sample_id": sample_id,
                "given_information": "dialog",
                "predicted_emotion": display_pred,
                "golden_emotion": golden,
                "prediction": pred_ok,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            if args.verbose:
                print(f"[{sample_id}] raw={raw!r} -> pred={display_pred} gold={golden} ok={pred_ok}", flush=True)
            if ((i + 1) % 20 == 0) or (i == start_idx):
                print(f"  {i + 1}/{n}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
