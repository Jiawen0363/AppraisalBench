#!/usr/bin/env python3
"""
Filter emotion_appraisal_corpus rows for benchmark construction using z-scored
appraisal vectors vs fixed theory prototypes (cosine similarity).

Typical use: exclude rows labeled Guilt whose profile is closer to Shame (or
vice versa) so the benchmark does not bake in ambiguous shame/guilt labels.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

# Theory prototypes (same as user table). Order per vector:
# Unpleasant, RespSelf, Uncertainty, Attention, Effort, Circumstance
THEORY: dict[str, list[float]] = {
    "Joy": [-1.46, 0.09, -0.46, 0.15, -0.33, -0.21],
    "Sadness": [0.87, -0.36, 0.00, -0.21, -0.14, 1.15],
    "Anger": [0.85, -0.94, -0.29, 0.12, 0.53, -0.96],
    "Fear": [0.44, -0.17, 0.73, 0.03, 0.63, 0.59],
    "Disgust": [0.38, -0.50, -0.39, -0.96, 0.06, -0.19],
    "Shame": [0.73, 1.31, 0.21, -0.11, 0.07, -0.07],
    "Guilt": [0.60, 1.31, -0.15, -0.36, 0.00, -0.29],
}

RAW_KEYS = [
    "Attention",
    "Certainty",
    "Effort",
    "Pleasant",
    "Responsibility",
    "Control",
    "Circumstance",
]


def _mapped_six(row: dict[str, str]) -> list[float]:
    pleasant = float(row["Pleasant"])
    certainty = float(row["Certainty"])
    responsibility = float(row["Responsibility"])
    control = float(row["Control"])
    return [
        3.0 - pleasant,
        (responsibility + control) / 2.0,
        3.0 - certainty,
        float(row["Attention"]),
        float(row["Effort"]),
        float(row["Circumstance"]),
    ]


def _zscore_over_rows(mapped_rows: list[list[float]]) -> list[list[float]]:
    n = len(mapped_rows)
    d = len(mapped_rows[0])
    means = [sum(mapped_rows[i][j] for i in range(n)) / n for j in range(d)]
    stds: list[float] = []
    for j in range(d):
        m = means[j]
        var = sum((mapped_rows[i][j] - m) ** 2 for i in range(n)) / n
        s = math.sqrt(var)
        stds.append(s if s > 1e-12 else 1.0)
    return [[(mapped_rows[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]


def _cosine(a: list[float], b: list[float]) -> float:
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    if den <= 1e-12:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / den


def _best_two(z: list[float]) -> tuple[str, float, str, float]:
    scored = [(emo, _cosine(z, vec)) for emo, vec in THEORY.items()]
    scored.sort(key=lambda t: t[1], reverse=True)
    (e1, s1), (e2, s2) = scored[0], scored[1]
    return e1, s1, e2, s2


def _should_exclude(
    label: str,
    best: str,
    second: str,
    margin: float,
    *,
    exclude_guilt_not_nearest: bool,
    exclude_guilt_nearest_shame: bool,
    exclude_shame_not_nearest: bool,
    exclude_shame_nearest_guilt: bool,
    exclude_pair_ambiguous: bool,
    ambiguity_margin: float,
) -> tuple[bool, str]:
    if exclude_guilt_nearest_shame and label == "Guilt" and best == "Shame":
        return True, "guilt_labeled_but_nearest_shame"
    if exclude_shame_nearest_guilt and label == "Shame" and best == "Guilt":
        return True, "shame_labeled_but_nearest_guilt"

    if exclude_guilt_not_nearest and label == "Guilt" and best != "Guilt":
        return True, f"guilt_labeled_but_nearest_{best.lower()}"

    if exclude_shame_not_nearest and label == "Shame" and best != "Shame":
        return True, f"shame_labeled_but_nearest_{best.lower()}"

    if exclude_pair_ambiguous and label in {"Shame", "Guilt"}:
        pair = {best, second}
        if pair == {"Shame", "Guilt"} and margin < ambiguity_margin:
            return True, f"shame_guilt_ambiguous_margin_{margin:.4f}"

    return False, ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("emotion_appraisal_corpus.tsv"),
        help="Input TSV (repo root relative ok).",
    )
    p.add_argument(
        "--out-included",
        type=Path,
        default=Path("emotion_appraisal_corpus_benchmark_included.tsv"),
    )
    p.add_argument(
        "--out-excluded",
        type=Path,
        default=Path("emotion_appraisal_corpus_benchmark_excluded.tsv"),
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=Path("emotion_appraisal_corpus_benchmark_filter_summary.txt"),
    )
    p.add_argument(
        "--exclude-guilt-nearest-shame",
        action="store_true",
        help="Exclude rows labeled Guilt whose nearest prototype is Shame.",
    )
    p.add_argument(
        "--exclude-shame-nearest-guilt",
        action="store_true",
        help="Exclude rows labeled Shame whose nearest prototype is Guilt.",
    )
    p.add_argument(
        "--exclude-guilt-not-nearest",
        action="store_true",
        help="Exclude any row labeled Guilt where nearest prototype is not Guilt.",
    )
    p.add_argument(
        "--exclude-shame-not-nearest",
        action="store_true",
        help="Exclude any row labeled Shame where nearest prototype is not Shame.",
    )
    p.add_argument(
        "--exclude-shame-guilt-ambiguous",
        action="store_true",
        help="Exclude Shame/Guilt-labeled rows if top-2 are {Shame,Guilt} and margin < threshold.",
    )
    p.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.05,
        help="Margin (top1-top2 cosine) below which Shame/Guilt counts as ambiguous.",
    )
    args = p.parse_args()

    if not any(
        [
            args.exclude_guilt_nearest_shame,
            args.exclude_shame_nearest_guilt,
            args.exclude_guilt_not_nearest,
            args.exclude_shame_not_nearest,
            args.exclude_shame_guilt_ambiguous,
        ]
    ):
        args.exclude_guilt_nearest_shame = True
        args.exclude_shame_nearest_guilt = True

    rows: list[dict[str, str]] = []
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        for row in reader:
            rows.append(row)

    mapped = [_mapped_six(r) for r in rows]
    zs = _zscore_over_rows(mapped)

    extra_cols = [
        "TheoryNearest",
        "TheorySecond",
        "TheoryCosineTop1",
        "TheoryCosineTop2",
        "TheoryMargin",
        "BenchmarkExclude",
        "ExcludeReason",
    ]
    out_fieldnames = list(fieldnames) + extra_cols

    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []

    reasons: Counter[str] = Counter()

    for row, z in zip(rows, zs):
        label = row["Prior_Emotion"].strip().capitalize()
        best, s1, second, s2 = _best_two(z)
        margin = s1 - s2

        excl, reason = _should_exclude(
            label,
            best,
            second,
            margin,
            exclude_guilt_not_nearest=args.exclude_guilt_not_nearest,
            exclude_guilt_nearest_shame=args.exclude_guilt_nearest_shame,
            exclude_shame_not_nearest=args.exclude_shame_not_nearest,
            exclude_shame_nearest_guilt=args.exclude_shame_nearest_guilt,
            exclude_pair_ambiguous=args.exclude_shame_guilt_ambiguous,
            ambiguity_margin=args.ambiguity_margin,
        )

        enriched = dict(row)
        enriched["TheoryNearest"] = best
        enriched["TheorySecond"] = second
        enriched["TheoryCosineTop1"] = f"{s1:.6f}"
        enriched["TheoryCosineTop2"] = f"{s2:.6f}"
        enriched["TheoryMargin"] = f"{margin:.6f}"
        enriched["BenchmarkExclude"] = "1" if excl else "0"
        enriched["ExcludeReason"] = reason if excl else ""

        if excl:
            excluded.append(enriched)
            if reason:
                reasons[reason] += 1
        else:
            included.append(enriched)

    args.out_included.parent.mkdir(parents=True, exist_ok=True)
    with args.out_included.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in included:
            w.writerow(r)

    with args.out_excluded.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in excluded:
            w.writerow(r)

    summary_lines = [
        f"Input: {args.input}",
        f"Total rows: {len(rows)}",
        f"Included: {len(included)} ({len(included)/len(rows):.2%})",
        f"Excluded: {len(excluded)} ({len(excluded)/len(rows):.2%})",
        "",
        "Active exclusion flags:",
        f"  exclude_guilt_nearest_shame={args.exclude_guilt_nearest_shame}",
        f"  exclude_shame_nearest_guilt={args.exclude_shame_nearest_guilt}",
        f"  exclude_guilt_not_nearest={args.exclude_guilt_not_nearest}",
        f"  exclude_shame_not_nearest={args.exclude_shame_not_nearest}",
        f"  exclude_shame_guilt_ambiguous={args.exclude_shame_guilt_ambiguous} (margin<{args.ambiguity_margin})",
        "",
        "Exclude reasons:",
    ]
    for k, v in reasons.most_common():
        summary_lines.append(f"  {k}: {v}")

    args.out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_included} ({len(included)} rows)")
    print(f"Wrote {args.out_excluded} ({len(excluded)} rows)")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
