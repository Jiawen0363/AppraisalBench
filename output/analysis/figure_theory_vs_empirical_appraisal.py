#!/usr/bin/env python3
"""
Build a clean theory vs. corpus figure: appraisal prototypes aligned to the same
6 dimensions (Unpleasant, Resp./Self, Uncertainty, Attention, Effort, Circumstance).

Corpus side: per-emotion mean on raw 0–3 scales, then z-score each raw dimension
across the 7 emotion means; map Pleasant→Unpleasant, Certainty→Uncertainty,
mean(Responsibility, Control)→Resp./Self.

Theory side: fixed table from Smith & Ellsworth (1985)-style prototype z-scores
(as in the user's reference figure).

Outputs:
  - PNG/PDF heatmap (two panels + optional difference panel)
  - CSV with theory and empirical rows for the same grid
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Theory prototypes (same order as columns below). Source: user's reference table.
THEORY: Dict[str, List[float]] = {
    "Happiness": [-1.46, 0.09, -0.46, 0.15, -0.33, -0.21],
    "Sadness": [0.87, -0.36, 0.00, -0.21, -0.14, 1.15],
    "Anger": [0.85, -0.94, -0.29, 0.12, 0.53, -0.96],
    "Fear": [0.44, -0.17, 0.73, 0.03, 0.63, 0.59],
    "Disgust": [0.38, -0.50, -0.39, -0.96, 0.06, -0.19],
    "Shame": [0.73, 1.31, 0.21, -0.11, 0.07, -0.07],
    "Guilt": [0.60, 1.31, -0.15, -0.36, 0.00, -0.29],
}

COL_LABELS = [
    "Unpleasant",
    "Resp./Self",
    "Uncertainty",
    "Attention",
    "Effort",
    "Circumstance",
]

# Row order: display label (theory) <- corpus Prior_Emotion value
ROW_ORDER: List[Tuple[str, str]] = [
    ("Happiness", "Joy"),
    ("Sadness", "Sadness"),
    ("Anger", "Anger"),
    ("Fear", "Fear"),
    ("Disgust", "Disgust"),
    ("Shame", "Shame"),
    ("Guilt", "Guilt"),
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
    means: Dict[str, Dict[str, float]] = {}
    for emo, cols in by.items():
        means[emo] = {k: sum(cols[k]) / len(cols[k]) for k in RAW_FIELDS}
    return means


def emotion_mean_z_then_align(
    means: Dict[str, Dict[str, float]],
) -> Dict[str, List[float]]:
    """Z-score each raw dimension across the 7 emotions' means; return aligned 6-D vectors."""
    emotions_data = [pair[1] for pair in ROW_ORDER]
    for e in emotions_data:
        if e not in means:
            raise KeyError(f"Missing emotion {e!r} in corpus; have {sorted(means)}")

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
    for disp, data_emo in ROW_ORDER:
        aligned[disp] = [
            -z[data_emo]["Pleasant"],
            (z[data_emo]["Responsibility"] + z[data_emo]["Control"]) / 2.0,
            -z[data_emo]["Certainty"],
            z[data_emo]["Attention"],
            z[data_emo]["Effort"],
            z[data_emo]["Circumstance"],
        ]
    return aligned


def heatmap_panel(
    ax,
    data: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(data[i, j]) > 0.85 else "black",
                fontsize=7,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def write_csv(
    out_csv: Path,
    theory_mat: np.ndarray,
    emp_mat: np.ndarray,
    row_labels: Sequence[str],
) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Emotion"] + list(COL_LABELS))
        for i, lab in enumerate(row_labels):
            w.writerow(["Theory"] + [lab] + [f"{theory_mat[i, j]:.4f}" for j in range(6)])
        for i, lab in enumerate(row_labels):
            w.writerow(["Corpus_z"] + [lab] + [f"{emp_mat[i, j]:.4f}" for j in range(6)])


def main() -> None:
    p = argparse.ArgumentParser(description="Theory vs corpus appraisal heatmaps.")
    p.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "emotion_appraisal_corpus.tsv",
        help="TSV with Prior_Emotion and appraisal columns.",
    )
    p.add_argument(
        "--out-figure",
        type=Path,
        default=_REPO_ROOT / "output/analysis/figure_theory_vs_empirical_appraisal.png",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO_ROOT / "output/analysis/theory_vs_empirical_appraisal_grid.csv",
    )
    p.add_argument(
        "--show-diff",
        action="store_true",
        help="Third panel: corpus minus theory.",
    )
    args = p.parse_args()

    means = load_means_by_emotion(args.input)
    emp_aligned = emotion_mean_z_then_align(means)

    row_labels = [pair[0] for pair in ROW_ORDER]
    theory_mat = np.array([THEORY[lab] for lab in row_labels], dtype=float)
    emp_mat = np.array([emp_aligned[lab] for lab in row_labels], dtype=float)
    diff_mat = emp_mat - theory_mat

    vmax = float(
        max(
            np.max(np.abs(theory_mat)),
            np.max(np.abs(emp_mat)),
            1e-6,
        )
    )
    if args.show_diff:
        vmax_diff = float(max(np.max(np.abs(diff_mat)), 1e-6))
    else:
        vmax_diff = vmax

    ncols = 3 if args.show_diff else 2
    fig_w = 11.0 if args.show_diff else 9.0
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 5.2), constrained_layout=True)

    heatmap_panel(
        axes[0],
        theory_mat,
        row_labels,
        COL_LABELS,
        "Theory (reference z-scores)",
        -vmax,
        vmax,
    )
    heatmap_panel(
        axes[1],
        emp_mat,
        row_labels,
        COL_LABELS,
        "Corpus (emotion-mean z, aligned dims)",
        -vmax,
        vmax,
    )
    if args.show_diff:
        heatmap_panel(
            axes[2],
            diff_mat,
            row_labels,
            COL_LABELS,
            "Difference (corpus − theory)",
            -vmax_diff,
            vmax_diff,
        )

    fig.suptitle(
        "Appraisal prototypes: theory vs. corpus\n"
        "(Corpus: z across emotion means per raw dim; Pleasant→Unpleasant, "
        "Certainty→Uncertainty, mean Resp.+Control→Resp./Self)",
        fontsize=10,
        y=1.02,
    )

    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_figure, dpi=200, bbox_inches="tight")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, theory_mat, emp_mat, row_labels)
    print(f"Wrote {args.out_figure}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
