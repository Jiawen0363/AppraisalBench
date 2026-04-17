#!/usr/bin/env python3
"""
对比「含 profile」与「不含 profile」的评估结果：
- dialog_first: raw vs with_profile
- dialog_full: raw vs with_profile

指标：Emotion 准确率 / Macro-F1，Appraisal 各维度 MAE（及平均 MAE）。
Gold：emotion_appraisal_corpus.tsv
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

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
        return {"emotion": (obj.get("emotion") or "").strip(), "appraisal": obj.get("appraisal") or {}}
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


def emotion_metrics(gold_emotions: list, pred_emotions: list, labels: list | None = None):
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        return {"accuracy": None, "macro_f1": None}
    gold_emotions = [str(g).strip() for g in gold_emotions]
    pred_emotions = [str(p).strip() for p in pred_emotions]
    acc = accuracy_score(gold_emotions, pred_emotions)
    if labels is None:
        labels = sorted(set(gold_emotions) | set(pred_emotions))
    macro = f1_score(gold_emotions, pred_emotions, labels=labels, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro}


def appraisal_mae(gold_df: pd.DataFrame, preds: dict[str, dict], dims: list[str]) -> dict[str, float]:
    mae_per_dim = {d: [] for d in dims}
    for _, row in gold_df.iterrows():
        sid = str(row["Sentence_id"])
        if sid not in preds:
            continue
        app = preds[sid].get("appraisal") or {}
        for d in dims:
            g = row.get(d)
            if pd.isna(g):
                continue
            try:
                p = int(app.get(d, 0))
            except (TypeError, ValueError):
                p = 0
            mae_per_dim[d].append(abs(int(g) - p))
    return {d: (sum(mae_per_dim[d]) / len(mae_per_dim[d])) if mae_per_dim[d] else float("nan") for d in dims}


def main():
    parser = argparse.ArgumentParser(description="Compare with_profile vs raw (no profile) evaluation")
    parser.add_argument("--corpus", type=Path, default=CORPUS_TSV)
    parser.add_argument("--dialog-first-raw", type=Path,
                       default=REPO_ROOT / "output/evaluation/dialog_first/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl")
    parser.add_argument("--dialog-first-profile", type=Path,
                       default=REPO_ROOT / "output/evaluation/dialog_first/Qwen3-4B_Qwen3-4B_Qwen3-4B_with_profile.jsonl")
    parser.add_argument("--dialog-full-raw", type=Path,
                       default=REPO_ROOT / "output/evaluation/dialog_full/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl")
    parser.add_argument("--dialog-full-profile", type=Path,
                       default=REPO_ROOT / "output/evaluation/dialog_full/Qwen3-4B_Qwen3-4B_Qwen3-4B_with_profile.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    gold_df = load_gold(args.corpus)
    first_raw = load_predictions_jsonl(args.dialog_first_raw)
    first_profile = load_predictions_jsonl(args.dialog_first_profile)
    full_raw = load_predictions_jsonl(args.dialog_full_raw)
    full_profile = load_predictions_jsonl(args.dialog_full_profile)

    # 只对比「gold 与四份预测都有的样本」，保证一致
    all_ids = set(gold_df["Sentence_id"].astype(str))
    common_ids = sorted(all_ids & set(first_raw) & set(first_profile) & set(full_raw) & set(full_profile))
    gold_df_str = gold_df[gold_df["Sentence_id"].astype(str).isin(common_ids)].copy()
    gold_df_str["Sentence_id"] = gold_df_str["Sentence_id"].astype(str)  # 统一 str，后面 lookup 用
    emotion_labels = sorted(gold_df["Prior_Emotion"].astype(str).unique())

    gold_em = [gold_df[gold_df["Sentence_id"].astype(str) == sid]["Prior_Emotion"].iloc[0] for sid in common_ids]

    # ---------- 1. Emotion 准确率与 Macro-F1 ----------
    print("=" * 70)
    print("1. Emotion 识别：准确率 & Macro-F1（含 profile vs 不含 profile）")
    print("=" * 70)

    results = []
    for name, preds in [
        ("dialog_first (raw)", first_raw),
        ("dialog_first (with_profile)", first_profile),
        ("dialog_full (raw)", full_raw),
        ("dialog_full (with_profile)", full_profile),
    ]:
        pred_em = [preds.get(sid, {}).get("emotion", "") for sid in common_ids]
        m = emotion_metrics(gold_em, pred_em, labels=emotion_labels)
        results.append({"mode": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  {name}: accuracy = {m['accuracy']:.4f}, macro_f1 = {m['macro_f1']:.4f}")

    # 对比小结
    print("\n  对比（profile - raw）：")
    acc_first_d = results[1]["accuracy"] - results[0]["accuracy"]
    acc_full_d = results[3]["accuracy"] - results[2]["accuracy"]
    print(f"    dialog_first: accuracy {'+' if acc_first_d >= 0 else ''}{acc_first_d:.4f}, macro_f1 {'+' if (results[1]['macro_f1'] - results[0]['macro_f1']) >= 0 else ''}{results[1]['macro_f1'] - results[0]['macro_f1']:.4f}")
    print(f"    dialog_full:  accuracy {'+' if acc_full_d >= 0 else ''}{acc_full_d:.4f}, macro_f1 {'+' if (results[3]['macro_f1'] - results[2]['macro_f1']) >= 0 else ''}{results[3]['macro_f1'] - results[2]['macro_f1']:.4f}")

    # ---------- 2. Appraisal 各维度 MAE ----------
    print()
    print("=" * 70)
    print("2. Appraisal 识别：各维度 MAE（含 profile vs 不含 profile）")
    print("=" * 70)

    mae_first_raw = appraisal_mae(gold_df, first_raw, APPRAISAL_DIMS)
    mae_first_profile = appraisal_mae(gold_df, first_profile, APPRAISAL_DIMS)
    mae_full_raw = appraisal_mae(gold_df, full_raw, APPRAISAL_DIMS)
    mae_full_profile = appraisal_mae(gold_df, full_profile, APPRAISAL_DIMS)

    mae_table = pd.DataFrame({
        "dimension": APPRAISAL_DIMS,
        "first_raw": [mae_first_raw[d] for d in APPRAISAL_DIMS],
        "first_with_profile": [mae_first_profile[d] for d in APPRAISAL_DIMS],
        "full_raw": [mae_full_raw[d] for d in APPRAISAL_DIMS],
        "full_with_profile": [mae_full_profile[d] for d in APPRAISAL_DIMS],
    })
    print(mae_table.to_string(index=False))

    mean_first_raw = np.nanmean(mae_table["first_raw"])
    mean_first_profile = np.nanmean(mae_table["first_with_profile"])
    mean_full_raw = np.nanmean(mae_table["full_raw"])
    mean_full_profile = np.nanmean(mae_table["full_with_profile"])
    print("\n  平均 MAE:")
    print(f"    dialog_first raw: {mean_first_raw:.4f}  |  with_profile: {mean_first_profile:.4f}  (Δ = {mean_first_profile - mean_first_raw:+.4f})")
    print(f"    dialog_full  raw: {mean_full_raw:.4f}  |  with_profile: {mean_full_profile:.4f}  (Δ = {mean_full_profile - mean_full_raw:+.4f})")

    # ---------- 3. with_profile 下：哪种 emotion 最不准、被错分成了什么 ----------
    def build_wrong_dist(gold_sub, preds, common_ids, emotion_labels):
        # gold_sub 的 Sentence_id 已为 str，与 common_ids 一致
        gold_by_id = gold_sub.set_index("Sentence_id")["Prior_Emotion"].astype(str).to_dict()
        wrong_dist = defaultdict(lambda: defaultdict(int))
        per_emotion_acc = []
        for g_em in emotion_labels:
            ids_em = [sid for sid in common_ids if gold_by_id.get(sid) == g_em]
            if not ids_em:
                continue
            correct = sum(1 for sid in ids_em if (preds.get(sid) or {}).get("emotion") == g_em)
            acc = correct / len(ids_em)
            per_emotion_acc.append((g_em, acc, len(ids_em)))
            for sid in ids_em:
                p_em = (preds.get(sid) or {}).get("emotion", "")
                if g_em != p_em:
                    wrong_dist[g_em][p_em] += 1
        return wrong_dist, per_emotion_acc

    for file_label, preds in [
        ("dialog_first (with_profile)", first_profile),
        ("dialog_full (with_profile)", full_profile),
    ]:
        wrong_dist, per_emotion_acc = build_wrong_dist(gold_df_str, preds, common_ids, emotion_labels)
        # 按准确率排序，最不准的排前面
        per_emotion_acc.sort(key=lambda x: x[1])
        print()
        print("=" * 70)
        print(f"3. {file_label}：各情感识别准确率（从低到高）与误分类成什么")
        print("=" * 70)
        print("\n  各 Prior_Emotion 准确率（越低越难识别）：")
        for g_em, acc, n in per_emotion_acc:
            print(f"    {g_em}: {acc:.2%}  (n={n})")
        worst_em = per_emotion_acc[0][0] if per_emotion_acc else "—"
        print(f"\n  识别最不准的情感: {worst_em}")
        print("\n  误分类分布（当 gold ≠ pred 时，预测成了什么）：")
        for g_em in emotion_labels:
            counts = wrong_dist.get(g_em) or {}
            if not counts:
                print(f"    {g_em}: (无误分类)")
                continue
            parts = [f"{p}={c}" for p, c in sorted(counts.items(), key=lambda x: -x[1])]
            print(f"    {g_em} -> " + ", ".join(parts))
        if args.out_dir:
            rows = []
            for g_em in emotion_labels:
                for p_em, cnt in (wrong_dist.get(g_em) or {}).items():
                    rows.append({"gold_emotion": g_em, "pred_emotion": p_em, "count": cnt})
            if rows:
                suffix = "first" if "first" in file_label else "full"
                pd.DataFrame(rows).to_csv(args.out_dir / f"with_profile_wrong_dist_{suffix}.csv", index=False)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.out_dir / "profile_vs_raw_emotion.csv", index=False)
        mae_table.to_csv(args.out_dir / "profile_vs_raw_appraisal_mae.csv", index=False)
        print("\n[OK] Wrote CSV to", args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
