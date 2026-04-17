#!/usr/bin/env python3
"""
Emotion prototype matching: corpus emotion-mean vectors vs. theory prototypes (cosine).

Panels:
  (1) 7×7 similarity matrix; each row highlights argmax (predicted theory).
  (2) Margin = cos(top1) − cos(top2) per corpus emotion (Shame should be smallest).
  (3) Shame-labeled rows only: each episode → 6D vector (global z on raw dims, then
      same alignment as elsewhere); count argmax theory — supports “Shame tag vs. appraisal”.

Default input: full corpus TSV. Use --input for benchmark_included.tsv if needed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

THEORY: Dict[str, List[float]] = {
    "Happiness": [-1.46, 0.09, -0.46, 0.15, -0.33, -0.21],
    "Sadness": [0.87, -0.36, 0.00, -0.21, -0.14, 1.15],
    "Anger": [0.85, -0.94, -0.29, 0.12, 0.53, -0.96],
    "Fear": [0.44, -0.17, 0.73, 0.03, 0.63, 0.59],
    "Disgust": [0.38, -0.50, -0.39, -0.96, 0.06, -0.19],
    "Shame": [0.73, 1.31, 0.21, -0.11, 0.07, -0.07],
    "Guilt": [0.60, 1.31, -0.15, -0.36, 0.00, -0.29],
}

THEORY_ORDER: List[str] = list(THEORY.keys())

ROW_ORDER: List[Tuple[str, str, str]] = [
    ("Happiness", "Joy", "Joy"),
    ("Sadness", "Sadness", "Sadness"),
    ("Anger", "Anger", "Anger"),
    ("Fear", "Fear", "Fear"),
    ("Disgust", "Disgust", "Disgust"),
    ("Shame", "Shame", "Shame"),
    ("Guilt", "Guilt", "Guilt"),
]

RAW_FIELDS = [
    "Attention",
    "Certainty",
    "Effort",
    "Pleasant",
    "Responsibility",
    "Control",
    "Circumstance",
]


def load_means_by_emotion(tsv_path: Path) -> Dict[str, Dict[str, float]]:
    by: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: {k: [] for k in RAW_FIELDS}
    )
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            emo = row["Prior_Emotion"].strip().capitalize()
            for k in RAW_FIELDS:
                by[emo][k].append(int(row[k]))
    return {emo: {k: sum(v[k]) / len(v[k]) for k in RAW_FIELDS} for emo, v in by.items()}


def emotion_mean_z_then_align(
    means: Dict[str, Dict[str, float]],
) -> Dict[str, List[float]]:
    emotions_data = [t[1] for t in ROW_ORDER]
    for e in emotions_data:
        if e not in means:
            raise KeyError(f"Missing emotion {e!r}; have {sorted(means)}")
    mu = {k: float(np.mean([means[e][k] for e in emotions_data])) for k in RAW_FIELDS}
    sd = {}
    for k in RAW_FIELDS:
        vals = [means[e][k] for e in emotions_data]
        s = float(np.std(vals, ddof=0))
        sd[k] = s if s > 1e-12 else 1.0
    z = {
        e: {k: (means[e][k] - mu[k]) / sd[k] for k in RAW_FIELDS}
        for e in emotions_data
    }
    aligned: Dict[str, List[float]] = {}
    for disp, data_emo, _ in ROW_ORDER:
        aligned[disp] = [
            -z[data_emo]["Pleasant"],
            (z[data_emo]["Responsibility"] + z[data_emo]["Control"]) / 2.0,
            -z[data_emo]["Certainty"],
            z[data_emo]["Attention"],
            z[data_emo]["Effort"],
            z[data_emo]["Circumstance"],
        ]
    return aligned


def align_from_raw_z(z: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            -z["Pleasant"],
            (z["Responsibility"] + z["Control"]) / 2.0,
            -z["Certainty"],
            z["Attention"],
            z["Effort"],
            z["Circumstance"],
        ],
        dtype=float,
    )


def global_raw_stats(tsv_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    cols: Dict[str, List[int]] = {k: [] for k in RAW_FIELDS}
    with tsv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for k in RAW_FIELDS:
                cols[k].append(int(row[k]))
    mu = {k: float(np.mean(cols[k])) for k in RAW_FIELDS}
    sd = {k: float(np.std(cols[k], ddof=0)) for k in RAW_FIELDS}
    for k in RAW_FIELDS:
        if sd[k] < 1e-12:
            sd[k] = 1.0
    return mu, sd


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def theory_matrix() -> np.ndarray:
    return np.array([THEORY[k] for k in THEORY_ORDER], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "emotion_appraisal_corpus.tsv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "output/analysis/figure_prototype_matching_top1.png",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO_ROOT / "output/analysis/prototype_matching_top1_summary.csv",
    )
    args = ap.parse_args()

    means = load_means_by_emotion(args.input)
    emp_aligned = emotion_mean_z_then_align(means)
    T = theory_matrix()
    row_disp = [t[0] for t in ROW_ORDER]
    gold_theory = [t[0] for t in ROW_ORDER]
    row_short = [t[2] for t in ROW_ORDER]

    n = len(ROW_ORDER)
    k = len(THEORY_ORDER)
    S = np.zeros((n, k))
    for i, disp in enumerate(row_disp):
        v = np.array(emp_aligned[disp], dtype=float)
        for j, lab in enumerate(THEORY_ORDER):
            S[i, j] = cosine(v, T[j])

    margins = []
    preds = []
    for i in range(n):
        row = S[i].copy()
        order = np.argsort(-row)
        top1, top2 = float(row[order[0]]), float(row[order[1]])
        margins.append(top1 - top2)
        preds.append(THEORY_ORDER[int(order[0])])

    correct = sum(1 for g, p in zip(gold_theory, preds) if g == p)
    acc = correct / n

    # Shame-labeled episodes: global z, then argmax theory
    g_mu, g_sd = global_raw_stats(args.input)
    shame_counts: Counter[str] = Counter()
    n_shame = 0
    guilt_better = 0
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["Prior_Emotion"].strip().capitalize() != "Shame":
                continue
            n_shame += 1
            z = {
                fld: (int(row[fld]) - g_mu[fld]) / g_sd[fld]
                for fld in RAW_FIELDS
            }
            v = align_from_raw_z(z)
            scores = [(lab, cosine(v, T[j])) for j, lab in enumerate(THEORY_ORDER)]
            scores.sort(key=lambda x: -x[1])
            shame_counts[scores[0][0]] += 1
            c_guilt = dict(scores)["Guilt"]
            c_shame = dict(scores)["Shame"]
            if c_guilt > c_shame:
                guilt_better += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_csv.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(
            [
                "corpus_row",
                "gold_theory",
                "pred_top1",
                "correct",
                "margin_top1_minus_top2",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    row_short[i],
                    gold_theory[i],
                    preds[i],
                    int(gold_theory[i] == preds[i]),
                    f"{margins[i]:.4f}",
                ]
            )
        w.writerow([])
        w.writerow(["aggregate_top1_acc", f"{acc:.4f}", f"{correct}/{n}"])
        w.writerow([])
        w.writerow(["shame_labeled_n", n_shame])
        w.writerow(["shame_episodes_cos_guilt_gt_shame", guilt_better, f"{guilt_better/n_shame:.4f}" if n_shame else ""])
        w.writerow([])
        w.writerow(["shame_episode_argmax_theory", "count"])
        for lab in THEORY_ORDER:
            w.writerow([lab, shame_counts.get(lab, 0)])

    # Figure
    fig = plt.figure(figsize=(11.5, 6.2), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.35)
    ax_sim = fig.add_subplot(gs[:, 0])
    ax_mar = fig.add_subplot(gs[0, 1])
    ax_sha = fig.add_subplot(gs[1, 1])

    vmax = max(float(np.max(np.abs(S))), 0.01)
    im = ax_sim.imshow(S, aspect="auto", cmap="viridis", vmin=0, vmax=min(1.0, vmax + 0.05))
    ax_sim.set_xticks(np.arange(k))
    ax_sim.set_yticks(np.arange(n))
    ax_sim.set_xticklabels(THEORY_ORDER, rotation=40, ha="right", fontsize=9)
    ax_sim.set_yticklabels([f"{s} ({d})" for s, d in zip(row_short, row_disp)], fontsize=9)
    ax_sim.set_xlabel("Theory prototype", fontsize=10)
    ax_sim.set_ylabel("Corpus emotion (emotion-mean vector)", fontsize=10)
    ax_sim.set_title(
        "Prototype matching (cosine)\n"
        "Corpus side: z across emotion means / dim → aligned 6D",
        fontsize=10,
    )

    for i in range(n):
        for j in range(k):
            ax_sim.text(
                j,
                i,
                f"{S[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if S[i, j] > 0.55 else "black",
                fontsize=7,
            )
    pred_j = [THEORY_ORDER.index(preds[i]) for i in range(n)]
    for i in range(n):
        rect = mpatches.Rectangle(
            (pred_j[i] - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="cyan" if gold_theory[i] == preds[i] else "magenta",
            linewidth=2.5,
        )
        ax_sim.add_patch(rect)
    leg1 = mpatches.Patch(edgecolor="cyan", facecolor="none", linewidth=2, label="Top-1 = gold")
    leg2 = mpatches.Patch(edgecolor="magenta", facecolor="none", linewidth=2, label="Top-1 ≠ gold")
    ax_sim.legend(handles=[leg1, leg2], loc="upper left", bbox_to_anchor=(0.02, -0.12), fontsize=8, frameon=True)

    plt.colorbar(im, ax=ax_sim, fraction=0.035, pad=0.02, label="Cosine")

    y_pos = np.arange(n)
    ax_mar.barh(y_pos, margins, color=["#c44e52" if row_short[i] == "Shame" else "#4c72b0" for i in range(n)])
    ax_mar.set_yticks(y_pos)
    ax_mar.set_yticklabels(row_short)
    ax_mar.invert_yaxis()
    ax_mar.set_xlabel("cos(top-1) − cos(top-2)", fontsize=10)
    ax_mar.set_title("Separation margin (higher = more decisive)", fontsize=10)
    ax_mar.axvline(0, color="gray", linewidth=0.8)

    if n_shame > 0:
        labs = THEORY_ORDER
        counts = [shame_counts.get(l, 0) for l in labs]
        colors = ["#c44e52" if l == "Guilt" else ("#8172b2" if l == "Shame" else "#888888") for l in labs]
        ax_sha.barh(labs, counts, color=colors)
        ax_sha.invert_yaxis()
        ax_sha.set_xlabel("Count (Shame-labeled rows)", fontsize=10)
        pct_guilt = 100.0 * shame_counts.get("Guilt", 0) / n_shame
        pct_gts = 100.0 * guilt_better / n_shame
        ax_sha.set_title(
            f"Shame-labeled episodes (n={n_shame}): argmax theory\n"
            f"{pct_guilt:.1f}% map to Guilt; {pct_gts:.1f}% have cos(Guilt)>cos(Shame)",
            fontsize=9,
        )
    else:
        ax_sha.text(0.5, 0.5, "No Shame rows", ha="center", va="center", transform=ax_sha.transAxes)

    wrong = [(gold_theory[i], preds[i]) for i in range(n) if gold_theory[i] != preds[i]]
    if len(wrong) == 1:
        miss_txt = f"— only {wrong[0][0]} → {wrong[0][1]}"
    elif wrong:
        miss_txt = "— mismatches: " + ", ".join(f"{g}→{p}" for g, p in wrong)
    else:
        miss_txt = ""
    fig.suptitle(
        f"Theory prototype top-1 match (emotion-mean vectors): {correct}/{n} = {100*acc:.1f}% {miss_txt}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_csv}")
    print(f"top1_acc {acc:.3f} shame_n {n_shame} guilt_argmax_pct {100*shame_counts.get('Guilt',0)/max(n_shame,1):.1f}")


if __name__ == "__main__":
    main()
