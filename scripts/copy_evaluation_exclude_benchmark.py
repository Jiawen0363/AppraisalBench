#!/usr/bin/env python3
"""Copy output/evaluation to output/update_evaluation, dropping excluded benchmark rows.

sample_id in jsonl (e.g. sample_0018) is the 0-based row index into emotion_appraisal_corpus.tsv,
not the corpus Sentence_id. We map excluded Sentence_id values from the excluded TSV via that file.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def sentence_ids_to_exclude_sample_ids(
    corpus_path: Path, excluded_tsv: Path
) -> set[str]:
    with corpus_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    sid_to_sample: dict[int, str] = {}
    for i, row in enumerate(rows):
        sid = int(row["Sentence_id"])
        sid_to_sample[sid] = f"sample_{i:04d}"

    bad: set[str] = set()
    with excluded_tsv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = int(row["Sentence_id"])
            sample_id = sid_to_sample.get(sid)
            if sample_id is None:
                raise SystemExit(
                    f"Excluded Sentence_id {sid} not found in {corpus_path}"
                )
            bad.add(sample_id)
    return bad


def scenario_id_from_json_obj(obj: dict) -> str | None:
    sid = obj.get("sample_id")
    if sid is not None:
        return str(sid)
    sample = obj.get("sample")
    if isinstance(sample, dict):
        inner = sample.get("scenario_id")
        if inner is not None:
            return str(inner)
    return None


def filter_jsonl(src: Path, dst: Path, exclude: set[str]) -> tuple[int, int]:
    kept = total = 0
    with src.open(encoding="utf-8") as inf, dst.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            sid = scenario_id_from_json_obj(obj)
            if sid is not None and sid in exclude:
                continue
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    return kept, total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Default: <repo>/emotion_appraisal_corpus.tsv",
    )
    p.add_argument(
        "--excluded",
        type=Path,
        default=None,
        help="Default: <repo>/emotion_appraisal_corpus_benchmark_excluded.tsv",
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Default: <repo>/output/evaluation",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Default: <repo>/output/update_evaluation",
    )
    p.add_argument(
        "--skip_dir",
        action="append",
        default=["old_results"],
        help="Directory names under src to skip (repeatable).",
    )
    args = p.parse_args()
    root = args.repo_root
    corpus = args.corpus or (root / "emotion_appraisal_corpus.tsv")
    excluded = args.excluded or (root / "emotion_appraisal_corpus_benchmark_excluded.tsv")
    src_root = args.src or (root / "output" / "evaluation")
    dst_root = args.dst or (root / "output" / "update_evaluation")
    skip = set(args.skip_dir)

    exclude = sentence_ids_to_exclude_sample_ids(corpus, excluded)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    n_files = n_jsonl = 0
    lines_kept = lines_total = 0
    for path in sorted(src_root.rglob("*")):
        rel = path.relative_to(src_root)
        if any(part in skip for part in rel.parts):
            continue
        if path.is_dir():
            continue
        n_files += 1
        out_path = dst_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".jsonl":
            n_jsonl += 1
            k, t = filter_jsonl(path, out_path, exclude)
            lines_kept += k
            lines_total += t
        else:
            shutil.copy2(path, out_path)

    print(
        f"Excluded {len(exclude)} sample_ids (from {excluded.name}). "
        f"Copied {n_files} files ({n_jsonl} jsonl); "
        f"jsonl lines {lines_kept}/{lines_total} kept."
    )


if __name__ == "__main__":
    main()
