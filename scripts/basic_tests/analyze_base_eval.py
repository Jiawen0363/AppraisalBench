#!/usr/bin/env python3
"""
分析 base 评估结果：将 raw_prediction jsonl 与 emotion_appraisal_corpus.tsv 的 golden 对比。

指标：
1. 总体情绪的识别准确率
2. 各类情绪的分别识别的准确率
3. 总体 appraisal 的 MAE
4. 各个 appraisal 维度的 MAE
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_TSV = REPO_ROOT / "emotion_appraisal_corpus.tsv"
APPRAISAL_DIMS = [
    "Attention", "Certainty", "Effort", "Pleasant",
    "Responsibility", "Control", "Circumstance"
]


def load_gold(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_csv(corpus_path, sep="\t")
    for col in APPRAISAL_DIMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def parse_prediction(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
        return {
            "emotion": (obj.get("emotion") or "").strip(),
            "appraisal": obj.get("appraisal") or {}
        }
    except Exception:
        return None


def load_predictions_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id", ""))
            pred = parse_prediction(rec.get("raw_prediction", "{}"))
            if pred:
                out[sid] = pred
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze base eval: emotion accuracy & appraisal MAE")
    parser.add_argument("--pred", type=Path,
                        default=REPO_ROOT / "output/evaluation/base/Qwen3-4B_raw.jsonl",
                        help="Prediction jsonl (sample_id, raw_prediction)")
    parser.add_argument("--corpus", type=Path, default=CORPUS_TSV,
                        help="Gold corpus TSV")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Optional: save CSV reports here")
    args = parser.parse_args()

    gold_df = load_gold(args.corpus)
    preds = load_predictions_jsonl(args.pred)

    # 只分析「gold 与预测都有的样本」
    all_ids = set(gold_df["Sentence_id"].astype(str))
    common_ids = sorted(all_ids & set(preds))
    if not common_ids:
        print("No common sample_id between corpus and prediction file. Exit.")
        return

    gold_sub = gold_df[gold_df["Sentence_id"].astype(str).isin(common_ids)].copy()
    gold_sub["Sentence_id"] = gold_sub["Sentence_id"].astype(str)
    n = len(common_ids)
    print(f"Aligned samples: {n}\n")

    # ---------- 1. 总体情绪准确率 ----------
    gold_emotions = [gold_sub[gold_sub["Sentence_id"] == sid]["Prior_Emotion"].iloc[0] for sid in common_ids]
    pred_emotions = [preds[sid].get("emotion", "") for sid in common_ids]
    # 统一为字符串并 strip，便于比较
    gold_emotions = [str(g).strip() for g in gold_emotions]
    pred_emotions = [str(p).strip() for p in pred_emotions]

    correct = sum(1 for g, p in zip(gold_emotions, pred_emotions) if g == p)
    overall_acc = correct / n if n else 0
    print("=" * 60)
    print("1. 总体情绪识别准确率")
    print("=" * 60)
    print(f"  Accuracy = {correct}/{n} = {overall_acc:.4f}\n")

    # ---------- 2. 各类情绪的分别识别准确率 ----------
    emotion_labels = sorted(set(gold_emotions))
    per_class = []
    for label in emotion_labels:
        indices = [i for i in range(n) if gold_emotions[i] == label]
        if not indices:
            per_class.append({"emotion": label, "support": 0, "correct": 0, "accuracy": 0.0})
            continue
        support = len(indices)
        correct_k = sum(1 for i in indices if pred_emotions[i] == label)
        acc_k = correct_k / support
        per_class.append({
            "emotion": label,
            "support": support,
            "correct": correct_k,
            "accuracy": acc_k
        })

    print("2. 各类情绪的分别识别准确率")
    print("=" * 60)
    for row in per_class:
        print(f"  {row['emotion']:12s}  support={row['support']:4d}  correct={row['correct']:4d}  accuracy={row['accuracy']:.4f}")
    print()

    # ---------- 3 & 4. Appraisal MAE（仅当预测中包含 appraisal 时计算）----------
    has_any_appraisal = any(
        (preds[sid].get("appraisal") or {}) for sid in common_ids
    )
    overall_mae = float("nan")
    dim_mae = {d: float("nan") for d in APPRAISAL_DIMS}

    if has_any_appraisal:
        mae_per_dim = {d: [] for d in APPRAISAL_DIMS}
        for sid in common_ids:
            row = gold_sub[gold_sub["Sentence_id"] == sid].iloc[0]
            app = preds[sid].get("appraisal") or {}
            for d in APPRAISAL_DIMS:
                g = row.get(d)
                if pd.isna(g):
                    continue
                try:
                    p = int(app.get(d, 0))
                except (TypeError, ValueError):
                    p = 0
                mae_per_dim[d].append(abs(int(g) - p))

        all_errors = []
        for d in APPRAISAL_DIMS:
            all_errors.extend(mae_per_dim[d])
        overall_mae = np.mean(all_errors) if all_errors else float("nan")
        dim_mae = {
            d: (np.mean(mae_per_dim[d]) if mae_per_dim[d] else float("nan"))
            for d in APPRAISAL_DIMS
        }
        print("3. 总体 Appraisal MAE")
        print("=" * 60)
        print(f"  Overall MAE (all dimensions, all samples) = {overall_mae:.4f}\n")
        print("4. 各个 Appraisal 维度的 MAE")
        print("=" * 60)
        for d in APPRAISAL_DIMS:
            print(f"  {d:14s}  MAE = {dim_mae[d]:.4f}")
        print()
    else:
        print("3. 总体 Appraisal MAE")
        print("=" * 60)
        print("  (预测中无 appraisal 字段，跳过)\n")
        print("4. 各个 Appraisal 维度的 MAE")
        print("=" * 60)
        print("  (预测中无 appraisal 字段，跳过)\n")

    # ---------- 可选：保存 CSV ----------
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        emotion_df = pd.DataFrame(per_class)
        emotion_df.to_csv(args.out_dir / "emotion_accuracy_by_class.csv", index=False)
        mae_df = pd.DataFrame([
            {"dimension": d, "mae": dim_mae[d]} for d in APPRAISAL_DIMS
        ])
        mae_df.to_csv(args.out_dir / "appraisal_mae_by_dimension.csv", index=False)
        summary_records = [{"metric": "emotion_accuracy", "value": overall_acc}]
        if has_any_appraisal:
            summary_records.append({"metric": "appraisal_mae_overall", "value": overall_mae})
        summary = pd.DataFrame(summary_records)
        summary.to_csv(args.out_dir / "summary.csv", index=False)
        print(f"Saved CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
