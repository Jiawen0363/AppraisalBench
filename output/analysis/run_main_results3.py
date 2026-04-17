#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_EVAL = _REPO_ROOT / "output" / "evaluation"

MODEL_FOLDERS = [
    ("GPT-4o", "gpt-4o"),
    ("DeepSeek", "deepseek-chat"),
    ("Qwen3-8B", "Qwen3-8B"),
    ("Qwen3-4B", "Qwen3-4B"),
]

# (folder name, table column short name)
APPRAISAL_DIMS = [
    ("attention", "A"),
    ("certainty", "Ce"),
    ("effort", "E"),
    ("pleasantness", "P"),
    ("responsibility", "R"),
    ("control", "Co"),
    ("circumstance", "Ci"),
]


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_macro_f1(records: List[dict]) -> float:
    if not records:
        return 0.0

    labels = set()
    for record in records:
        choice = record.get("choice", {})
        gold = choice.get("gold")
        pred = choice.get("model")
        if isinstance(gold, str) and gold:
            labels.add(gold.strip())
        if isinstance(pred, str) and pred:
            labels.add(pred.strip())
    if not labels:
        return 0.0

    tp = {label: 0 for label in labels}
    fp = {label: 0 for label in labels}
    fn = {label: 0 for label in labels}

    for record in records:
        choice = record.get("choice", {})
        gold = choice.get("gold")
        pred = choice.get("model")
        if not isinstance(gold, str) or not gold.strip():
            continue
        if not isinstance(pred, str) or not pred.strip():
            continue
        gold = gold.strip()
        pred = pred.strip()
        if gold == pred:
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1

    f1_values: List[float] = []
    for label in sorted(labels):
        precision_den = tp[label] + fp[label]
        recall_den = tp[label] + fn[label]
        precision = (tp[label] / precision_den) if precision_den > 0 else 0.0
        recall = (tp[label] / recall_den) if recall_den > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)
    return (sum(f1_values) / len(f1_values)) * 100.0


def build_model_metrics(task5_root: Path, model_folder: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for dim_folder, col_name in APPRAISAL_DIMS:
        file_path = task5_root / model_folder / f"{dim_folder}.jsonl"
        if not file_path.exists():
            metrics[col_name] = 0.0
            continue
        records = load_jsonl(file_path)
        metrics[col_name] = compute_macro_f1(records)
    return metrics


def format_row(model: str, metrics: Dict[str, float]) -> Dict[str, str]:
    row = {"Model": model}
    vals: List[float] = []
    for _, col_name in APPRAISAL_DIMS:
        value = metrics.get(col_name, 0.0)
        vals.append(value)
        row[col_name] = f"{value:.2f}"
    row["Avg."] = f"{(sum(vals) / len(vals)):.2f}"
    return row


def build_table_task5(task5_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_name, model_folder in MODEL_FOLDERS:
        rows.append(format_row(model_name, build_model_metrics(task5_root, model_folder)))

    # Uniform random over 7 choices => expected macro-F1 = 14.29%.
    random_value = round(100.0 / 7.0, 2)
    random_metrics = {col_name: random_value for _, col_name in APPRAISAL_DIMS}
    rows.append(format_row("Random", random_metrics))
    return rows


def print_markdown_table(rows: List[Dict[str, str]]) -> None:
    columns = ["Model"] + [col_name for _, col_name in APPRAISAL_DIMS] + ["Avg."]
    print("# Table 3 (Task5)")
    print()
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(c, "") for c in columns) + " |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-root",
        type=Path,
        default=_DEFAULT_EVAL,
        help="Directory containing task5/ (default: <repo>/output/evaluation).",
    )
    args = p.parse_args()
    task5_root = args.eval_root.resolve() / "task5"
    rows = build_table_task5(task5_root)
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
