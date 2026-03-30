#!/usr/bin/env python3
"""
Debug helper: print the fully-filled Step1 and Step2 prompts for sample_0000.

Writes:
  output/seed2scenario/debug_prompts_sample_0000_step1.txt
  output/seed2scenario/debug_prompts_sample_0000_step2.txt
"""

import json
from pathlib import Path

import pandas as pd

import sys

# Ensure we can import the local modules when running from project root.
SEED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEED_DIR))

import run_seed2scenario as rs
import seed2scenario as st


def expansions_from_saved_appraisals(appraisals: dict) -> dict:
    """Support both legacy JSON (single expansion) and new schema (scenario + appraisal)."""
    out = {}
    for dim_key, v in appraisals.items():
        if "expansion" in v and "scenario_expansion" not in v:
            if dim_key == "pleasantness":
                out[dim_key] = {"appraisal_expansion": v["expansion"]}
            else:
                out[dim_key] = {"scenario_expansion": v["expansion"], "appraisal_expansion": ""}
        elif dim_key == "pleasantness":
            out[dim_key] = {"appraisal_expansion": v.get("appraisal_expansion", "")}
        else:
            out[dim_key] = {
                "scenario_expansion": v.get("scenario_expansion") or "",
                "appraisal_expansion": v.get("appraisal_expansion") or "",
            }
    return out


def main():
    out_jsonl = Path("/home/jiawen/AppraisalBench/output/seed2scenario/scenarios_step1_step2_test.jsonl")
    with out_jsonl.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    seed_event = first["seed_event"]
    expansions = expansions_from_saved_appraisals(first["appraisals"])

    corpus_df = pd.read_csv(rs.CORPUS_PATH, sep="\t")
    row = corpus_df.iloc[0]
    values = {dim: int(row[rs.DIM_META[dim].corpus_col]) for dim in rs.DIM_ORDER}

    anchor_dims = [dim for dim in rs.DIM_ORDER if values[dim] in (0, 3)]
    softer_dims = [dim for dim in rs.DIM_ORDER if values[dim] in (1, 2)]

    state_fns = {
        "attention": st.attention_state,
        "certainty": st.certainty_state,
        "effort": st.effort_state,
        "pleasantness": st.pleasantness_state,
        "responsibility": st.responsibility_state,
        "control": st.self_control_state,
        "circumstance": st.circumstance_state,
    }

    dim_static = {dim: rs.build_dim_static(dim) for dim in rs.DIM_ORDER}

    step1_header = rs.load_prompt("step1_shared_header")
    step2_header = rs.load_prompt("step2_shared_header")

    anchor_blocks = "\n\n".join(
        rs.build_dimension_block(
            dim_key=dim,
            state_text=state_fns[dim](values[dim]),
            dim_static=dim_static[dim],
            dim_meta=rs.DIM_META[dim],
        )
        for dim in anchor_dims
    )
    step1_prompt = step1_header.replace("{EVENT}", seed_event).replace(
        "{DIMENSION_BLOCKS}", anchor_blocks
    )

    expanded_scenario_text = (
        rs.build_expanded_scenario_text(expansions, anchor_dims, seed_event)
        if anchor_dims
        else f"Event:\n{seed_event}"
    )

    soft_blocks = "\n\n".join(
        rs.build_dimension_block(
            dim_key=dim,
            state_text=state_fns[dim](values[dim]),
            dim_static=dim_static[dim],
            dim_meta=rs.DIM_META[dim],
        )
        for dim in softer_dims
    )
    step2_prompt = (
        step2_header.replace("{EXPANDED_SCENARIO}", expanded_scenario_text)
        .replace("{DIMENSION_BLOCKS}", soft_blocks)
    )

    out_dir = Path("/home/jiawen/AppraisalBench/output/seed2scenario")
    out_dir.mkdir(parents=True, exist_ok=True)
    step1_path = out_dir / "debug_prompts_sample_0000_step1.txt"
    step2_path = out_dir / "debug_prompts_sample_0000_step2.txt"
    step1_path.write_text(step1_prompt, encoding="utf-8")
    step2_path.write_text(step2_prompt, encoding="utf-8")

    print("WROTE")
    print(step1_path)
    print(step2_path)


if __name__ == "__main__":
    main()

