#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_EVAL = _REPO_ROOT / "output" / "evaluation"

MODEL_FILES = [
    ("GPT-4o", "gpt-4o"),
    ("DeepSeek", "deepseek-chat"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("Qwen3-4B", "Qwen3-4B"),
]

EMOTION_CANONICAL = {
    "joy": "joy",
    "happiness": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "shame": "shame",
    "guilt": "guilt",
}

EMOTION_ORDER = [
    ("joy", "Happiness"),
    ("sadness", "Sadness"),
    ("anger", "Anger"),
    ("fear", "Fear"),
    ("disgust", "Disgust"),
    ("shame", "Shame"),
    ("guilt", "Guilt"),
]


def normalize_emotion(label: str) -> str:
    if label is None:
        return ""
    return EMOTION_CANONICAL.get(str(label).strip().lower(), str(label).strip().lower())


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_per_emotion_f1(records: List[dict]) -> Dict[str, float]:
    tp = {emo: 0 for emo, _ in EMOTION_ORDER}
    fp = {emo: 0 for emo, _ in EMOTION_ORDER}
    fn = {emo: 0 for emo, _ in EMOTION_ORDER}

    for record in records:
        gold = normalize_emotion(record.get("golden_emotion"))
        pred = normalize_emotion(record.get("predicted_emotion"))
        if gold not in tp:
            continue
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred in fp:
                fp[pred] += 1

    f1 = {}
    for emo, _ in EMOTION_ORDER:
        precision_den = tp[emo] + fp[emo]
        recall_den = tp[emo] + fn[emo]
        precision = (tp[emo] / precision_den) if precision_den > 0 else 0.0
        recall = (tp[emo] / recall_den) if recall_den > 0 else 0.0
        if precision + recall == 0:
            f1[emo] = 0.0
        else:
            f1[emo] = (2 * precision * recall / (precision + recall)) * 100.0
    return f1


def make_random_baseline(records: List[dict]) -> Dict[str, float]:
    # Expected per-class F1 for uniform random prediction.
    num_classes = len(EMOTION_ORDER)
    total = 0
    gold_count = {emo: 0 for emo, _ in EMOTION_ORDER}
    for record in records:
        gold = normalize_emotion(record.get("golden_emotion"))
        if gold in gold_count:
            gold_count[gold] += 1
            total += 1

    if total == 0:
        return {emo: 0.0 for emo, _ in EMOTION_ORDER}

    random_f1: Dict[str, float] = {}
    for emo, _ in EMOTION_ORDER:
        g = gold_count[emo]
        # E[TP]=g/K, E[FP]=(N-g)/K, E[FN]=g*(K-1)/K
        denom = (total + num_classes * g)
        random_f1[emo] = (200.0 * g / denom) if denom > 0 else 0.0
    return random_f1


def format_row(model: str, setting: str, f1: Dict[str, float]) -> Dict[str, str]:
    row = {"Model": model, "Setting": setting}
    vals = []
    for emo, col in EMOTION_ORDER:
        v = f1.get(emo, 0.0)
        vals.append(v)
        row[col] = f"{v:.2f}"
    row["Avg."] = f"{(sum(vals) / len(vals)):.2f}"
    return row


def print_markdown_table(rows: List[Dict[str, str]]) -> None:
    columns = ["Model", "Setting"] + [col for _, col in EMOTION_ORDER] + ["Avg."]
    print("# Table 1 (Task1 Base + Task3 Aug.)")
    print()
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(c, "") for c in columns) + " |")


def build_table_task1_task3(eval_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    reference_records_for_random: List[dict] = []

    for model_name, model_file_stem in MODEL_FILES:
        task1_file = eval_root / "task1" / f"{model_file_stem}.jsonl"
        task3_file = eval_root / "task3" / f"{model_file_stem}.jsonl"
        if not task1_file.exists() or not task3_file.exists():
            continue

        task1_records = load_jsonl(task1_file)
        task3_records = load_jsonl(task3_file)
        if not reference_records_for_random:
            reference_records_for_random = task1_records

        base_f1 = compute_per_emotion_f1(task1_records)
        aug_f1 = compute_per_emotion_f1(task3_records)
        rows.append(format_row(model_name, "Base", base_f1))
        rows.append(format_row(model_name, "Aug.", aug_f1))

    if reference_records_for_random:
        random_f1 = make_random_baseline(reference_records_for_random)
        rows.append(format_row("Random", "--", random_f1))

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-root",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Directory containing task1/, task3/ jsonl (default: <repo>/output/evaluation).",
    )
    args = p.parse_args()
    eval_root = args.eval_root.resolve()
    table1_rows = build_table_task1_task3(eval_root)
    if not table1_rows:
        raise RuntimeError("No valid Task1/Task3 files found for configured models.")
    print_markdown_table(table1_rows)


if __name__ == "__main__":
    main()
