#!/usr/bin/env python3
"""
分析三种评估模式下的结果：base / dialog_first / dialog_full。

以 emotion_appraisal_corpus.tsv 为 gold，对比：
- output/evaluation/base/Qwen3-4B_raw.jsonl
- output/evaluation/dialog_first/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl
- output/evaluation/dialog_full/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl

分析维度（可做方向）：
1. 情感分类：准确率、Macro-F1、混淆矩阵、按 Prior_Emotion 分组的准确率
2. Appraisal 维度：各维度 MAE、相关系数、精确匹配率，三模式对比
3. 模式间对比：谁更接近 gold、两两预测一致率
4. 按情感细分：每种 Prior_Emotion 下各模式表现，找出难例
5. 错误与一致性：三模式都错 vs 某模式独有错误；与 gold 差异最大的样本
6. 可视化：准确率/MAE 柱状图、混淆矩阵热力图、按情感的表现图
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_TSV = REPO_ROOT / "emotion_appraisal_corpus.tsv"
APPRAISAL_DIMS = [
    "Attention", "Certainty", "Effort", "Pleasant",
    "Responsibility", "Control", "Circumstance"
]


def load_gold(corpus_path: Path) -> pd.DataFrame:
    """加载 gold：Sentence_id, Prior_Emotion, 7 个 appraisal 维度（转为 int）。"""
    df = pd.read_csv(corpus_path, sep="\t")
    for col in ["Attention", "Certainty", "Effort", "Pleasant", "Responsibility", "Control", "Circumstance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def parse_prediction(raw: str) -> dict | None:
    """从 raw_prediction 字符串解析出 emotion 和 appraisal dict。"""
    try:
        obj = json.loads(raw)
        appraisal = obj.get("appraisal") or {}
        emotion = (obj.get("emotion") or "").strip()
        return {"emotion": emotion, "appraisal": appraisal}
    except Exception:
        return None


def load_predictions_jsonl(path: Path) -> dict[str, dict]:
    """加载 jsonl，返回 sample_id -> {emotion, appraisal}。"""
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id", ""))
            raw = rec.get("raw_prediction", "{}")
            pred = parse_prediction(raw)
            if pred:
                out[sid] = pred
    return out


def emotion_metrics(gold_emotions: list, pred_emotions: list, labels: list | None = None):
    """准确率与 Macro-F1（需要 sklearn）。"""
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
    """按维度计算 MAE（只算有 gold 且有 pred 的样本）。"""
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
            g = int(g)
            mae_per_dim[d].append(abs(g - p))
    return {d: (sum(mae_per_dim[d]) / len(mae_per_dim[d])) if mae_per_dim[d] else float("nan") for d in dims}


def main():
    parser = argparse.ArgumentParser(description="Analyze base vs dialog_first vs dialog_full")
    parser.add_argument("--base", type=Path, default=REPO_ROOT / "output/evaluation/base/Qwen3-4B_raw.jsonl")
    parser.add_argument("--dialog-first", type=Path, dest="dialog_first",
                        default=REPO_ROOT / "output/evaluation/dialog_first/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl")
    parser.add_argument("--dialog-full", type=Path, dest="dialog_full",
                        default=REPO_ROOT / "output/evaluation/dialog_full/Qwen3-4B_Qwen3-4B_Qwen3-4B_raw.jsonl")
    parser.add_argument("--corpus", type=Path, default=CORPUS_TSV)
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional: write CSV/figures here")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots (to --out-dir or output/analysis)")
    args = parser.parse_args()

    gold_df = load_gold(args.corpus)
    base_preds = load_predictions_jsonl(args.base)
    first_preds = load_predictions_jsonl(args.dialog_first)
    full_preds = load_predictions_jsonl(args.dialog_full)

    # 对齐：只分析在 gold 中且三个结果都有的 sample_id
    all_ids = set(gold_df["Sentence_id"].astype(str))
    for name, preds in [("base", base_preds), ("dialog_first", first_preds), ("dialog_full", full_preds)]:
        missing = all_ids - set(preds.keys())
        if missing:
            print(f"[WARN] {name}: missing {len(missing)} sample_ids", file=sys.stderr)
    common_ids = sorted(all_ids & set(base_preds) & set(first_preds) & set(full_preds))
    gold_sub = gold_df[gold_df["Sentence_id"].astype(str).isin(common_ids)].copy()
    gold_sub = gold_sub.set_index("Sentence_id")

    print("=" * 60)
    print("1. 情感分类 (Emotion) — 准确率 & Macro-F1")
    print("=" * 60)
    emotion_labels = sorted(gold_df["Prior_Emotion"].astype(str).unique())
    rows = []
    for mode_name, preds in [
        ("base", base_preds),
        ("dialog_first", first_preds),
        ("dialog_full", full_preds),
    ]:
        gold_em = [gold_df[gold_df["Sentence_id"].astype(str) == sid]["Prior_Emotion"].iloc[0] for sid in common_ids]
        pred_em = [preds.get(sid, {}).get("emotion", "") for sid in common_ids]
        m = emotion_metrics(gold_em, pred_em, labels=emotion_labels)
        rows.append({"mode": mode_name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  {mode_name}: accuracy = {m['accuracy']:.4f}, macro_f1 = {m['macro_f1']:.4f}")
    emotion_df = pd.DataFrame(rows)

    print()
    print("=" * 60)
    print("2. Appraisal 各维度 MAE（越小越好）")
    print("=" * 60)
    mae_base = appraisal_mae(gold_df, base_preds, APPRAISAL_DIMS)
    mae_first = appraisal_mae(gold_df, first_preds, APPRAISAL_DIMS)
    mae_full = appraisal_mae(gold_df, full_preds, APPRAISAL_DIMS)
    mae_table = pd.DataFrame({
        "dimension": APPRAISAL_DIMS,
        "base": [mae_base[d] for d in APPRAISAL_DIMS],
        "dialog_first": [mae_first[d] for d in APPRAISAL_DIMS],
        "dialog_full": [mae_full[d] for d in APPRAISAL_DIMS],
    })
    print(mae_table.to_string(index=False))
    mean_mae = {
        "base": mae_table["base"].mean(),
        "dialog_first": mae_table["dialog_first"].mean(),
        "dialog_full": mae_table["dialog_full"].mean(),
    }
    print("\n  平均 MAE: base = {:.4f}, dialog_first = {:.4f}, dialog_full = {:.4f}".format(
        mean_mae["base"], mean_mae["dialog_first"], mean_mae["dialog_full"]))

    print()
    print("=" * 60)
    print("3. 模式间情感预测一致率（两两一致）")
    print("=" * 60)
    def agree_rate(preds_a, preds_b):
        same = sum(1 for sid in common_ids if (preds_a.get(sid, {}).get("emotion") == preds_b.get(sid, {}).get("emotion")))
        return same / len(common_ids) if common_ids else 0
    print("  base vs dialog_first: {:.4f}".format(agree_rate(base_preds, first_preds)))
    print("  base vs dialog_full:  {:.4f}".format(agree_rate(base_preds, full_preds)))
    print("  dialog_first vs dialog_full: {:.4f}".format(agree_rate(first_preds, full_preds)))

    print()
    print("=" * 60)
    print("4. 按 Prior_Emotion 分组的情感准确率（样本数 & 准确率）")
    print("=" * 60)
    gold_df_str = gold_df.copy()
    gold_df_str["Sentence_id"] = gold_df_str["Sentence_id"].astype(str)
    gold_df_str = gold_df_str[gold_df_str["Sentence_id"].isin(common_ids)]
    per_emotion = []
    for emotion in emotion_labels:
        subset = gold_df_str[gold_df_str["Prior_Emotion"] == emotion]
        ids_em = subset["Sentence_id"].tolist()
        if not ids_em:
            continue
        g_em = [emotion] * len(ids_em)
        acc_base = emotion_metrics(g_em, [base_preds.get(i, {}).get("emotion", "") for i in ids_em], labels=emotion_labels)["accuracy"]
        acc_first = emotion_metrics(g_em, [first_preds.get(i, {}).get("emotion", "") for i in ids_em], labels=emotion_labels)["accuracy"]
        acc_full = emotion_metrics(g_em, [full_preds.get(i, {}).get("emotion", "") for i in ids_em], labels=emotion_labels)["accuracy"]
        per_emotion.append({
            "Prior_Emotion": emotion, "n": len(ids_em),
            "base_acc": acc_base, "dialog_first_acc": acc_first, "dialog_full_acc": acc_full,
        })
    per_emotion_df = pd.DataFrame(per_emotion)
    print(per_emotion_df.to_string(index=False))

    # 5. 当某 Prior_Emotion 未被分对时，分别被预测成了什么情绪（各模式）
    print()
    print("=" * 60)
    print("5. 误分类分布：各 Prior_Emotion 被错分时预测成了哪些情绪（计数）")
    print("=" * 60)
    gold_by_id = gold_df_str.set_index("Sentence_id")["Prior_Emotion"].to_dict()
    wrong_dist_all = {}
    for mode_name, preds in [
        ("base", base_preds),
        ("dialog_first", first_preds),
        ("dialog_full", full_preds),
    ]:
        wrong_dist = defaultdict(lambda: defaultdict(int))
        for sid in common_ids:
            g = gold_by_id.get(sid)
            p = (preds.get(sid) or {}).get("emotion", "")
            if g != p and g is not None:
                wrong_dist[g][p] += 1
        wrong_dist_all[mode_name] = dict(wrong_dist)
        print(f"\n  [{mode_name}]")
        for gold_em in emotion_labels:
            counts = wrong_dist[gold_em]
            if not counts:
                print(f"    {gold_em}: (无误分类)")
                continue
            sorted_pred = sorted(counts.items(), key=lambda x: -x[1])
            parts = [f"{pred}={cnt}" for pred, cnt in sorted_pred]
            print(f"    {gold_em} -> " + ", ".join(parts))

    # 6. 当 gold=A 被误分类为 pred=B 时，appraisal 各维度（预测值-真实值）的均值与标准差
    print()
    print("=" * 60)
    print("6. 误分类时 Appraisal 变化：gold->pred 下各维度 (预测-真实) 的均值±标准差")
    print("=" * 60)
    gold_rows_by_id = gold_df.copy()
    gold_rows_by_id["Sentence_id"] = gold_rows_by_id["Sentence_id"].astype(str)
    gold_rows_by_id = gold_rows_by_id.set_index("Sentence_id")

    appraisal_misclass_all = {}
    for mode_name, preds in [
        ("base", base_preds),
        ("dialog_first", first_preds),
        ("dialog_full", full_preds),
    ]:
        pair_diffs = defaultdict(lambda: {d: [] for d in APPRAISAL_DIMS})
        for sid in common_ids:
            if sid not in gold_rows_by_id.index:
                continue
            row = gold_rows_by_id.loc[sid]
            g_em = row["Prior_Emotion"]
            p_em = (preds.get(sid) or {}).get("emotion", "")
            if g_em == p_em:
                continue
            key = (g_em, p_em)
            app = (preds.get(sid) or {}).get("appraisal") or {}
            for d in APPRAISAL_DIMS:
                try:
                    p_val = int(app.get(d, 0))
                except (TypeError, ValueError):
                    p_val = 0
                g_val = int(row.get(d, 0))
                pair_diffs[key][d].append(p_val - g_val)
        appraisal_misclass_all[mode_name] = dict(pair_diffs)

        # 先打整体误分类时的 appraisal 变化（所有 gold!=pred 混在一起）
        all_diffs = {d: [] for d in APPRAISAL_DIMS}
        for key, dim_lists in pair_diffs.items():
            for d in APPRAISAL_DIMS:
                all_diffs[d].extend(dim_lists[d])
        # 存下来供画图用
        appraisal_misclass_all[mode_name + "_overall"] = {}
        for d in APPRAISAL_DIMS:
            arr = np.array(all_diffs[d])
            if len(arr) == 0:
                appraisal_misclass_all[mode_name + "_overall"][d] = {"mean": np.nan, "std": np.nan}
            else:
                appraisal_misclass_all[mode_name + "_overall"][d] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"\n  [{mode_name}] 全体误分类样本上 (预测-真实) 的均值±标准差:")
        for d in APPRAISAL_DIMS:
            arr = np.array(all_diffs[d])
            if len(arr) == 0:
                print(f"    {d}: (无样本)")
            else:
                print(f"    {d}: mean={arr.mean():+.3f}, std={arr.std():.3f}, n={len(arr)}")

        # 再打按 (gold, pred) 对的汇总（表格较多，只打印前几对或写入 CSV）
        rows_detail = []
        for (g_em, p_em), dim_lists in sorted(pair_diffs.items()):
            n = len(dim_lists[APPRAISAL_DIMS[0]]) if dim_lists[APPRAISAL_DIMS[0]] else 0
            if n == 0:
                continue
            row = {"gold_emotion": g_em, "pred_emotion": p_em, "n": n}
            for d in APPRAISAL_DIMS:
                arr = np.array(dim_lists[d])
                row[f"{d}_mean"] = round(arr.mean(), 4)
                row[f"{d}_std"] = round(arr.std(), 4) if len(arr) > 1 else 0.0
            rows_detail.append(row)
        appraisal_detail_df = pd.DataFrame(rows_detail)
        appraisal_misclass_all[mode_name + "_detail_df"] = appraisal_detail_df
        print(f"\n  [{mode_name}] 按 (gold->pred) 的 appraisal 变化详见 CSV（共 {len(rows_detail)} 对）")

    plot_dir = args.out_dir or (REPO_ROOT / "output" / "analysis")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        emotion_df.to_csv(args.out_dir / "emotion_metrics.csv", index=False)
        mae_table.to_csv(args.out_dir / "appraisal_mae.csv", index=False)
        per_emotion_df.to_csv(args.out_dir / "emotion_accuracy_by_prior_emotion.csv", index=False)
        # 误分类分布：每模式一个 CSV (gold_emotion, pred_emotion, count)
        for mode_name, wrong_dist in wrong_dist_all.items():
            if mode_name.endswith("_detail_df"):
                continue
            rows = []
            for g_em in emotion_labels:
                for p_em, cnt in (wrong_dist.get(g_em) or {}).items():
                    rows.append({"gold_emotion": g_em, "pred_emotion": p_em, "count": cnt})
            if rows:
                pd.DataFrame(rows).to_csv(args.out_dir / f"wrong_pred_dist_{mode_name}.csv", index=False)
        # 误分类时 appraisal 变化（按 gold->pred）：每模式一个 CSV
        for mode_name in ["base", "dialog_first", "dialog_full"]:
            df = appraisal_misclass_all.get(mode_name + "_detail_df")
            if df is not None and len(df) > 0:
                df.to_csv(args.out_dir / f"appraisal_misclass_{mode_name}.csv", index=False)
        print("\n[OK] Wrote CSV to", args.out_dir)
        print("  CSVs: emotion_metrics.csv, appraisal_mae.csv, emotion_accuracy_by_prior_emotion.csv,")
        print("        wrong_pred_dist_{base,dialog_first,dialog_full}.csv,")
        print("        appraisal_misclass_{base,dialog_first,dialog_full}.csv")

    if args.plot or args.out_dir:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[WARN] matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        else:
            plot_dir.mkdir(parents=True, exist_ok=True)
            mode_names = ["base", "dialog_first", "dialog_full"]
            colors = ["#2ecc71", "#3498db", "#9b59b6"]

            # 图1: 情感分类 — 准确率 & Macro-F1
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(mode_names))
            w = 0.35
            bars1 = ax.bar(x - w / 2, emotion_df["accuracy"], w, label="Accuracy", color=colors)
            bars2 = ax.bar(x + w / 2, emotion_df["macro_f1"], w, label="Macro-F1", color=colors, alpha=0.7)
            ax.set_ylabel("Score")
            ax.set_title("Emotion classification: Accuracy & Macro-F1 by mode")
            ax.set_xticks(x)
            ax.set_xticklabels(mode_names)
            ax.legend()
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(plot_dir / "emotion_accuracy_f1.png", dpi=150)
            plt.close(fig)
            print("[OK] Saved", plot_dir / "emotion_accuracy_f1.png")

            # 图2: Appraisal 各维度 MAE（分组柱状）
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(APPRAISAL_DIMS))
            w = 0.25
            ax.bar(x - w, mae_table["base"], w, label="base", color=colors[0])
            ax.bar(x, mae_table["dialog_first"], w, label="dialog_first", color=colors[1])
            ax.bar(x + w, mae_table["dialog_full"], w, label="dialog_full", color=colors[2])
            ax.set_ylabel("MAE")
            ax.set_title("Appraisal MAE by dimension (lower is better)")
            ax.set_xticks(x)
            ax.set_xticklabels(APPRAISAL_DIMS, rotation=15, ha="right")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "appraisal_mae_by_dimension.png", dpi=150)
            plt.close(fig)
            print("[OK] Saved", plot_dir / "appraisal_mae_by_dimension.png")

            # 图3: 按 Prior_Emotion 的情感准确率（分组柱状）
            fig, ax = plt.subplots(figsize=(10, 5))
            emotions = per_emotion_df["Prior_Emotion"].tolist()
            x = np.arange(len(emotions))
            w = 0.25
            ax.bar(x - w, per_emotion_df["base_acc"], w, label="base", color=colors[0])
            ax.bar(x, per_emotion_df["dialog_first_acc"], w, label="dialog_first", color=colors[1])
            ax.bar(x + w, per_emotion_df["dialog_full_acc"], w, label="dialog_full", color=colors[2])
            ax.set_ylabel("Accuracy")
            ax.set_title("Emotion accuracy by Prior_Emotion (per emotion)")
            ax.set_xticks(x)
            ax.set_xticklabels(emotions, rotation=20, ha="right")
            ax.legend()
            ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(plot_dir / "emotion_accuracy_by_emotion.png", dpi=150)
            plt.close(fig)
            print("[OK] Saved", plot_dir / "emotion_accuracy_by_emotion.png")

            # 图4: 误分类分布热力图（每模式一张，gold->pred 计数）
            for mode_name, wrong_dist in wrong_dist_all.items():
                if mode_name.endswith("_detail_df"):
                    continue
                mat = np.zeros((len(emotion_labels), len(emotion_labels)))
                for i, g_em in enumerate(emotion_labels):
                    for p_em, cnt in (wrong_dist.get(g_em) or {}).items():
                        if p_em in emotion_labels:
                            j = emotion_labels.index(p_em)
                            mat[i, j] = cnt
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
                ax.set_xticks(np.arange(len(emotion_labels)))
                ax.set_yticks(np.arange(len(emotion_labels)))
                ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
                ax.set_yticklabels(emotion_labels)
                ax.set_xlabel("Predicted emotion")
                ax.set_ylabel("Gold (Prior_Emotion)")
                ax.set_title(f"Wrong prediction distribution [{mode_name}] (count)")
                for i in range(len(emotion_labels)):
                    for j in range(len(emotion_labels)):
                        v = int(mat[i, j])
                        if v > 0:
                            ax.text(j, i, str(v), ha="center", va="center", fontsize=9)
                plt.colorbar(im, ax=ax, label="count")
                fig.tight_layout()
                fig.savefig(plot_dir / f"wrong_pred_heatmap_{mode_name}.png", dpi=150)
                plt.close(fig)
                print("[OK] Saved", plot_dir / f"wrong_pred_heatmap_{mode_name}.png")

            # 图5: 误分类时 Appraisal 各维度 (预测-真实) 的均值±标准差（按模式）
            overall_base = appraisal_misclass_all.get("base_overall") or {}
            overall_first = appraisal_misclass_all.get("dialog_first_overall") or {}
            overall_full = appraisal_misclass_all.get("dialog_full_overall") or {}
            means_b = [overall_base.get(d, {}).get("mean", np.nan) for d in APPRAISAL_DIMS]
            stds_b = [overall_base.get(d, {}).get("std", 0) for d in APPRAISAL_DIMS]
            means_f = [overall_first.get(d, {}).get("mean", np.nan) for d in APPRAISAL_DIMS]
            stds_f = [overall_first.get(d, {}).get("std", 0) for d in APPRAISAL_DIMS]
            means_full = [overall_full.get(d, {}).get("mean", np.nan) for d in APPRAISAL_DIMS]
            stds_full = [overall_full.get(d, {}).get("std", 0) for d in APPRAISAL_DIMS]
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(APPRAISAL_DIMS))
            w = 0.25
            ax.bar(x - w, means_b, w, yerr=stds_b, label="base", color=colors[0], capsize=2)
            ax.bar(x, means_f, w, yerr=stds_f, label="dialog_first", color=colors[1], capsize=2)
            ax.bar(x + w, means_full, w, yerr=stds_full, label="dialog_full", color=colors[2], capsize=2)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_ylabel("Mean (pred − gold)")
            ax.set_title("Appraisal change when misclassified (mean ± std)")
            ax.set_xticks(x)
            ax.set_xticklabels(APPRAISAL_DIMS, rotation=15, ha="right")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / "appraisal_misclass_mean_std.png", dpi=150)
            plt.close(fig)
            print("[OK] Saved", plot_dir / "appraisal_misclass_mean_std.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
