#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path("/home/jiawen/AppraisalBench/output/evaluation/task3")
OUT_DIR = Path("/home/jiawen/AppraisalBench/output/analysis/task3")

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


def build_ratio_matrix(records: List[dict]) -> np.ndarray:
    n = len(EMOTION_ORDER)
    conf_mat = np.zeros((n, n), dtype=np.int32)
    for record in records:
        gold = normalize_emotion(record.get("golden_emotion"))
        pred = normalize_emotion(record.get("predicted_emotion"))
        if gold not in EMO2IDX or pred not in EMO2IDX:
            continue
        conf_mat[EMO2IDX[gold], EMO2IDX[pred]] += 1

    misclass = conf_mat.astype(np.float64)
    np.fill_diagonal(misclass, 0.0)
    row_sum = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(misclass, row_sum, out=np.zeros_like(misclass), where=row_sum > 0)
    return ratio


def plot_ratio_heatmap(matrix: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Misclassification Ratio (per golden emotion)")

    ax.set_xticks(np.arange(len(EMOTION_LABELS)))
    ax.set_yticks(np.arange(len(EMOTION_LABELS)))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Golden Emotion")
    ax.set_title(title)

    threshold = matrix.max() * 0.6 if matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val >= threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def combine_ratio_heatmaps(single_paths: List[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, p in zip(axes.ravel(), single_paths):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Task3 Misclassification Ratio Heatmaps", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for model_name, filename in MODEL_FILES:
        eval_file = EVAL_ROOT / filename
        if not eval_file.exists():
            print(f"[skip] missing file: {eval_file}")
            continue
        records = load_jsonl(eval_file)
        ratio = build_ratio_matrix(records)
        out_path = OUT_DIR / f"task3_misclass_heatmap_ratio_{model_name}.png"
        plot_ratio_heatmap(
            matrix=ratio,
            title=f"Task3 Misclassification Ratio ({model_name})",
            out_path=out_path,
        )
        generated.append(out_path)

    if len(generated) == 4:
        combine_ratio_heatmaps(
            generated,
            OUT_DIR / "task3_misclass_heatmap_ratio_combined.png",
        )

    print("Generated files:")
    for p in generated:
        print(f"- {p}")
    combined = OUT_DIR / "task3_misclass_heatmap_ratio_combined.png"
    if combined.exists():
        print(f"- {combined}")


if __name__ == "__main__":
    main()
