"""
Shared Task 5 MCQA row construction (7-way shuffle).

Used by run_task5_build_7way_mcq.py (export JSONL) and run_task5_appraisal_dimension_mcqa_eval.py (on-the-fly).
The evaluator prompt embeds the task line; no separate question_stem field.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEED_SCENARIO_DIR = _REPO_ROOT / "seed2scenario"
if str(_SEED_SCENARIO_DIR) not in sys.path:
    sys.path.insert(0, str(_SEED_SCENARIO_DIR))

from appraisal_dimensions_static import DIM_META, DIM_ORDER  # noqa: E402

__all__ = [
    "DIM_ORDER",
    "read_jsonl",
    "expansion_for_dim",
    "build_task5_question_row",
    "iter_task5_question_rows",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expansion_for_dim(appraisals: Dict[str, Any], dim: str) -> str:
    block = appraisals.get(dim)
    if not isinstance(block, dict):
        return ""
    return str(block.get("appraisal_expansion", "") or "").strip()


def build_task5_question_row(
    scenario: Dict[str, Any],
    dim: str,
    shuffle_seed: int,
) -> Dict[str, Any] | None:
    """
    One MCQA row for (scenario, target dimension dim). Returns None if row should be skipped.
    """
    if dim not in DIM_ORDER:
        return None

    scenario_id = str(scenario.get("id", "")).strip()
    if not scenario_id:
        return None

    appraisals = scenario.get("appraisals")
    if not isinstance(appraisals, dict):
        return None

    expansions: Dict[str, str] = {}
    for d in DIM_ORDER:
        expansions[d] = expansion_for_dim(appraisals, d)

    missing = [d for d in DIM_ORDER if not expansions[d]]
    if missing:
        return None

    letters = list("ABCDEFG")
    rng = random.Random(f"{shuffle_seed}:{scenario_id}:{dim}")
    order = list(DIM_ORDER)
    rng.shuffle(order)

    options_map: Dict[str, str] = {}
    option_dims_order: List[str] = []
    for i, letter in enumerate(letters):
        dkey = order[i]
        options_map[letter] = expansions[dkey]
        option_dims_order.append(dkey)

    correct_letter = letters[order.index(dim)]
    meta = DIM_META[dim]

    return {
        "scenario_id": scenario_id,
        "appraisal_dimension": dim,
        "appraisal_dimension_display": meta.output_name,
        "seed_event": str(scenario.get("seed_event", "") or "").strip(),
        "emotion": str(scenario.get("emotion", "") or "").strip(),
        "option_dims_order": option_dims_order,
        "option_A": options_map["A"],
        "option_B": options_map["B"],
        "option_C": options_map["C"],
        "option_D": options_map["D"],
        "option_E": options_map["E"],
        "option_F": options_map["F"],
        "option_G": options_map["G"],
        "correct_option": correct_letter,
    }


def iter_task5_question_rows(
    scenarios: List[Dict[str, Any]],
    appraisal_dimension: str,
    shuffle_seed: int,
    limit: int | None = None,
) -> Iterator[Dict[str, Any]]:
    subset = scenarios if limit is None else scenarios[:limit]
    for s in subset:
        row = build_task5_question_row(s, appraisal_dimension, shuffle_seed)
        if row is not None:
            yield row
