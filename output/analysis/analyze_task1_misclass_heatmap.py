#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path("/home/jiawen/AppraisalBench/output/evaluation/task1")
OUT_DIR = Path("/home/jiawen/AppraisalBench/output/analysis/task1")

MODEL_FILES: List[Tuple[str, str]] = [
    ("gpt-4o", "gpt-4o.jsonl"),
    ("deepseek-chat", "deepseek-chat.jsonl"),
    ("Qwen3-4B", "Qwen3-4B.jsonl"),
    ("Qwen3-8B", "Qwen3-8B.jsonl"),
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

EMOTION_ORDER = ["joy", "sadness", "anger", "fear", "disgust", "shame", "guilt"]
EMOTION_LABELS = ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Shame", "Guilt"]
EMO2IDX: Dict[str, int] = {emo: idx for idx, emo in enumerate(EMOTION_ORDER)}


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


def build_matrices(records: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(EMOTION_ORDER)
    conf_mat = np.zeros((n, n), dtype=np.int32)
    for record in records:
        gold = normalize_emotion(record.get("golden_emotion"))
        pred = normalize_emotion(record.get("predicted_emotion"))
        if gold not in EMO2IDX or pred not in EMO2IDX:
            continue
        conf_mat[EMO2IDX[gold], EMO2IDX[pred]] += 1

    misclass_count = conf_mat.astype(np.float64)
    np.fill_diagonal(misclass_count, 0.0)

    row_sum = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        misclass_ratio = np.divide(misclass_count, row_sum, out=np.zeros_like(misclass_count), where=row_sum > 0)
    return misclass_count, misclass_ratio


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    fmt: str,
    cbar_label: str,
    cmap: str = "Reds",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(EMOTION_LABELS)))
    ax.set_yticks(np.arange(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(EMOTION_LABELS, fontsize=12)
    ax.set_xlabel("Predicted Emotion", fontsize=15)
    ax.set_ylabel("Golden Emotion", fontsize=15)
    ax.set_title(title, fontsize=16, pad=10)

    threshold = matrix.max() * 0.6 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = format(val, fmt)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val >= threshold else "black",
                fontsize=11,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def combine_ratio_heatmaps(single_paths: List[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, p in zip(axes.ravel(), single_paths):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")
    # Intentionally no suptitle, as requested.
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    ratio_paths: List[Path] = []
    for model_name, filename in MODEL_FILES:
        eval_file = EVAL_ROOT / filename
        if not eval_file.exists():
            print(f"[skip] missing file: {eval_file}")
            continue

        records = load_jsonl(eval_file)
        misclass_count, misclass_ratio = build_matrices(records)

        out_count = OUT_DIR / f"task1_misclass_heatmap_count_{model_name}.png"
        out_ratio = OUT_DIR / f"task1_misclass_heatmap_ratio_{model_name}.png"

        plot_heatmap(
            matrix=misclass_count,
            title=f"Task1 Misclassification Count ({model_name})",
            out_path=out_count,
            fmt=".0f",
            cbar_label="Misclassified Samples (count)",
        )
        plot_heatmap(
            matrix=misclass_ratio,
            title=f"Task1 Misclassification Ratio ({model_name})",
            out_path=out_ratio,
            fmt=".2f",
            cbar_label="Misclassification Ratio (per golden emotion)",
        )

        generated.extend([out_count, out_ratio])
        ratio_paths.append(out_ratio)

    if len(ratio_paths) == 4:
        combined = OUT_DIR / "task1_misclass_heatmap_ratio_combined.png"
        combine_ratio_heatmaps(ratio_paths, combined)
        generated.append(combined)

    print("Generated files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
