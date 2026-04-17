"""
Task 3: emotion eval from dialog + scenario appraisal_expansion; gold emotion from scenarios.jsonl.

Same prediction target as task 1, but the evaluator prompt includes appraisal_expansion
text from the matching scenario row (all seven dimensions by default, or one dimension with
--only_appraisal_dim for ablation). Output includes only_appraisal_dim (null = full).
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

# Stable order; keys as in scenarios.jsonl "appraisals" object
APPRAISAL_DIM_KEYS: tuple[str, ...] = (
    "attention",
    "certainty",
    "effort",
    "pleasantness",
    "responsibility",
    "control",
    "circumstance",
)

APPRAISAL_DIM_LABELS: dict[str, str] = {
    "attention": "Attention",
    "certainty": "Certainty",
    "effort": "Effort",
    "pleasantness": "Pleasantness",
    "responsibility": "Responsibility",
    "control": "Control",
    "circumstance": "Circumstance",
}


def load_scenarios_emotion_and_appraisals(path: str | Path) -> dict[str, tuple[str, dict]]:
    """Map scenario id -> (emotion string, raw appraisals dict)."""
    path = Path(path)
    out: dict[str, tuple[str, dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", ""))
            em = row.get("emotion")
            appraisals = row.get("appraisals") or {}
            if not sid or em is None:
                continue
            if not isinstance(appraisals, dict):
                appraisals = {}
            out[sid] = (str(em).strip(), appraisals)
    return out


def load_existing_sample_ids(path: str | Path) -> set[str]:
    """Read existing output jsonl and collect sample_id for resume skipping."""
    path = Path(path)
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            done.add(str(sample_id))
    return done


def format_appraisal_expansion_block(
    appraisals: dict,
    only_keys: tuple[str, ...] | None = None,
) -> str:
    """
    Concatenate appraisal_expansion for the prompt.
    If only_keys is set (e.g. one dimension), include only those keys in order.
    """
    keys_iter: tuple[str, ...] = only_keys if only_keys is not None else APPRAISAL_DIM_KEYS
    lines: list[str] = []
    for key in keys_iter:
        block = appraisals.get(key)
        if not isinstance(block, dict):
            continue
        text = block.get("appraisal_expansion")
        if text is None or not str(text).strip():
            continue
        label = APPRAISAL_DIM_LABELS.get(key, key.title())
        lines.append(f"{label}: {str(text).strip()}")
    if lines:
        return "\n".join(lines)
    if only_keys is not None and len(only_keys) == 1:
        return f"(No appraisal_expansion for dimension '{only_keys[0]}' in scenario.)"
    return "(No appraisal_expansion fields in scenario.)"


def normalize_golden_emotion(gold: str) -> str:
    return gold.strip().lower()


def dialog_to_context_text(dialog_record: dict) -> str:
    lines: list[str] = []
    for turn in dialog_record.get("conversation", []):
        for role, content in turn.items():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def parse_predicted_emotion(raw: str) -> str | None:
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
        default="task3/eval_emotion_dialog_plus_appraisal",
        help="Path stem under evaluator/prompt; uses {context_text} and {appraisal_expansion_block}",
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
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N dialogs in dialog_file (0-based). E.g. --offset 10 starts at the 11th line.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to output_file instead of overwriting (use when resuming a partial run).",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--only_appraisal_dim",
        type=str,
        default=None,
        choices=list(APPRAISAL_DIM_KEYS),
        help="Ablation: include only this dimension's appraisal_expansion. Omit to include all dimensions.",
    )
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
    if "{appraisal_expansion_block}" not in prompt_template:
        raise SystemExit(
            f"Prompt {prompt_path} must contain placeholder {{appraisal_expansion_block}}"
        )
    dialogs = load_dialogs_jsonl(dialog_path)
    gold_by_id = load_scenarios_emotion_and_appraisals(scenarios_path)
    if args.offset:
        if args.offset < 0:
            raise SystemExit("--offset must be >= 0")
        if args.offset >= len(dialogs):
            raise SystemExit(f"--offset {args.offset} is >= dialog count {len(dialogs)}")
        dialogs = dialogs[args.offset :]
    if args.limit is not None:
        dialogs = dialogs[: args.limit]

    existing_ids: set[str] = set()
    if args.append:
        existing_ids = load_existing_sample_ids(output_path)

    n = len(dialogs)
    mode = "append" if args.append else "write"
    ablation = args.only_appraisal_dim or "all_dims"
    print(
        f"Loaded {n} dialogs (offset={args.offset}, limit={args.limit}), "
        f"{len(gold_by_id)} scenarios -> {output_path} ({mode}), appraisal={ablation}",
        flush=True,
    )
    if existing_ids:
        print(
            f"Resume mode: found {len(existing_ids)} existing sample_ids, will skip duplicates.",
            flush=True,
        )

    out_mode = "a" if args.append else "w"
    with output_path.open(out_mode, encoding="utf-8", buffering=1) as f:
        for i, record in enumerate(dialogs):
            sample_id = str(record.get("scenario_id", record.get("corpus_id", str(i))))
            if sample_id in existing_ids:
                continue
            row = gold_by_id.get(sample_id)
            if row is None:
                print(f"[warn] no scenario for sample_id={sample_id}, skipping", flush=True)
                continue
            golden, appraisals = row
            only = (args.only_appraisal_dim,) if args.only_appraisal_dim else None
            appraisal_block = format_appraisal_expansion_block(appraisals, only_keys=only)

            prompt = prompt_template.replace("{context_text}", dialog_to_context_text(record))
            prompt = prompt.replace("{appraisal_expansion_block}", appraisal_block)

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
            display_pred = parsed if parsed is not None else (raw or "").strip()

            given = (
                "dialog_and_appraisal_expansion"
                if args.only_appraisal_dim is None
                else f"dialog_and_appraisal_expansion:{args.only_appraisal_dim}"
            )
            out = {
                "sample_id": sample_id,
                "given_information": given,
                "only_appraisal_dim": args.only_appraisal_dim,
                "predicted_emotion": display_pred,
                "golden_emotion": golden,
                "prediction": pred_ok,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            existing_ids.add(sample_id)

            if args.verbose:
                print(
                    f"[{sample_id}] raw={raw!r} -> pred={display_pred} gold={golden} ok={pred_ok}",
                    flush=True,
                )
            global_idx = args.offset + i + 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  batch {i + 1}/{n} (dialog order ~#{global_idx})", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
