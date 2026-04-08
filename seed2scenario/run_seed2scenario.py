#!/usr/bin/env python3
"""
Expand emotion_appraisal_corpus.tsv seeds into full scenario JSON using LLM.

Three-step generation:
1) Step 1 (values 0/3): non-Pleasantness dims → scenario + appraisal lines; Pleasantness → appraisal only
   (labels: Attention_scenario / Attention_appraisal, …, Pleasantness_appraisal).
2) Step 2 (values 1/2): same dual-track labels as Step 1, conditioned on prior scenario text; merged into the same expansion dict.
3) Step 3: summarize and merge seed_event together with six non-Pleasantness scenario_expansion fields into one top-level "scenario" string.
"""

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from openai import OpenAI

# Allow importing seed2scenario (the .py in same dir) when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Per-dimension expansions after Step 1 / Step 2. Pleasantness has only appraisal_expansion.
ExpansionPerDim = Dict[str, str]

# Dimensions whose scenario_expansion feeds Step 3 summary (excludes Pleasantness).
SCENARIO_SUMMARY_DIMS = [
    "attention",
    "certainty",
    "effort",
    "responsibility",
    "control",
    "circumstance",
]


# -------- API config (可改为环境变量 OPENAI_API_KEY / OPENAI_BASE_URL) --------
API_KEY = os.environ.get(
    "OPENAI_API_KEY", "sk-8kgU8Q3tLwhvJ6wtGGVD46z0kIGIaWZElebK6Ag5NIDL18Xe"
)
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://35.164.11.19:3887/v1")
MODEL = "gpt-4o"

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running seed2scenario."
    )

# Retry config
MAX_RETRIES = 5
RETRY_DELAY = 2.0

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = SCRIPT_DIR / "prompt"
CORPUS_PATH = SCRIPT_DIR.parent / "emotion_appraisal_corpus.tsv"
OUTPUT_PATH = SCRIPT_DIR.parent / "output" / "seed2scenario" / "scenarios.jsonl"


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_prompt(name: str) -> str:
    """Load a prompt text file from seed2scenario/prompt."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip()


def call_api_with_retry(client, messages, extract_first_line: bool = False) -> str:
    """Call chat completion with retries. Returns normalized content string."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(messages=messages, model=MODEL)
            content = completion.choices[0].message.content or ""
            content = unicodedata.normalize("NFKC", content).strip()
            if extract_first_line:
                content = content.split("\n")[0].strip()
            return content
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_err


def get_seed_event(client, sentence: str, seed_event_prompt: str) -> str:
    """Convert 'I felt ... when X' to event-only X."""
    user = f"Input:\n{sentence}\n\nOutput:"
    messages = [
        {"role": "system", "content": seed_event_prompt},
        {"role": "user", "content": user},
    ]
    out = call_api_with_retry(client, messages, extract_first_line=True)
    return out if out else sentence


def trim_to_first_sentence(text: str) -> str:
    text = " ".join(text.split())
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    return m.group(1).strip() if m else text.strip()


def parse_dimension_outputs(output_text: str, expected_output_names: List[str]) -> Dict[str, str]:
    """
    Parse model output that follows:
    Label:
    <one sentence>

    Labels may be dimension names (Step 2) or Step 1 labels such as Attention_scenario, Self-Control_appraisal,
    Pleasantness_appraisal.
    """
    expected = set(expected_output_names)
    # Allow dimension names like "Self-Control" and Step 1 suffixes (_scenario, _appraisal).
    label_re = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")

    results: Dict[str, str] = {}
    cur_name: str = ""
    buf: List[str] = []

    for line in output_text.splitlines():
        s = line.strip()
        if not s:
            continue

        m = label_re.match(s)
        if m:
            name = m.group(1)
            rest = m.group(2).strip()
            if name in expected:
                if cur_name:
                    results[cur_name] = trim_to_first_sentence(" ".join(buf))
                cur_name = name
                buf = []
                if rest:
                    buf.append(rest)
                continue

        if cur_name:
            buf.append(s)

    if cur_name:
        results[cur_name] = trim_to_first_sentence(" ".join(buf))

    missing = [n for n in expected_output_names if n not in results]
    if missing:
        raise ValueError(f"Missing outputs for: {missing}. Raw output: {output_text[:500]}")
    return results


def dual_track_expected_output_labels(dim_keys: List[str]) -> List[str]:
    """Output labels for Step 1 or Step 2 when using step(1|2)_shared_header dual-track format."""
    labels: List[str] = []
    for dim in dim_keys:
        if dim == "pleasantness":
            labels.append("Pleasantness_appraisal")
        else:
            out = DIM_META[dim].output_name
            labels.append(f"{out}_scenario")
            labels.append(f"{out}_appraisal")
    return labels


def merge_dual_track_parsed(dim_keys: List[str], parsed: Dict[str, str]) -> Dict[str, ExpansionPerDim]:
    """Turn dual-track flat labels (Attention_scenario, …) into per-dimension expansion dicts."""
    out: Dict[str, ExpansionPerDim] = {}
    for dim in dim_keys:
        if dim == "pleasantness":
            out[dim] = {"appraisal_expansion": parsed["Pleasantness_appraisal"]}
        else:
            on = DIM_META[dim].output_name
            out[dim] = {
                "scenario_expansion": parsed[f"{on}_scenario"],
                "appraisal_expansion": parsed[f"{on}_appraisal"],
            }
    return out


def build_step3_scenario_summary_prompt(
    seed_event: str,
    emotion_lower: str,
    expansions: Dict[str, ExpansionPerDim],
    template: str,
) -> str:
    """Fill Step 3 prompt: merge seed_event + six scenario_expansion lines into one narrative (see step3 template)."""
    parts: List[str] = []
    for dim in SCENARIO_SUMMARY_DIMS:
        scen = (expansions[dim].get("scenario_expansion") or "").strip()
        if not scen:
            raise ValueError(f"Missing scenario_expansion for {dim} before Step 3.")
        out_name = DIM_META[dim].output_name
        parts.append(f"{out_name}:\n{scen}")
    blocks = "\n\n".join(parts)
    return (
        template.replace("{SEED_EVENT}", seed_event)
        .replace("{EMOTION}", emotion_lower)
        .replace("{SCENARIO_EXPANSIONS}", blocks)
    )


def clean_scenario_summary_output(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).strip()
    m = re.match(
        r"^(?:Integrated scenario|Scenario|Narrative|Output)\s*:\s*",
        text,
        flags=re.I,
    )
    if m:
        text = text[m.end() :].strip()
    return text


def integrate_scenario_with_llm(client, prompt_text: str) -> str:
    raw = call_api_with_retry(
        client, messages=[{"role": "user", "content": prompt_text}], extract_first_line=False
    )
    return clean_scenario_summary_output(raw)


def build_scenario(
    row,
    seed_event: str,
    expansions: Dict[str, ExpansionPerDim],
    scenario_id: str,
    integrated_scenario: str,
) -> dict:
    """Build one scenario JSON from row + seed_event + expansions + Step 3 scenario text."""
    emotion = str(row["Prior_Emotion"]).strip().lower()

    def appraisal_block(dim_key: str, col: str) -> dict:
        val = int(row[col])
        ex = expansions[dim_key]
        if dim_key == "pleasantness":
            return {"value": val, "appraisal_expansion": ex["appraisal_expansion"]}
        return {
            "value": val,
            "scenario_expansion": ex.get("scenario_expansion") or "",
            "appraisal_expansion": ex.get("appraisal_expansion") or "",
        }

    appraisals = {
        "attention": appraisal_block("attention", "Attention"),
        "certainty": appraisal_block("certainty", "Certainty"),
        "effort": appraisal_block("effort", "Effort"),
        "pleasantness": appraisal_block("pleasantness", "Pleasant"),
        "responsibility": appraisal_block("responsibility", "Responsibility"),
        "control": appraisal_block("control", "Control"),
        "circumstance": appraisal_block("circumstance", "Circumstance"),
    }
    return {
        "id": scenario_id,
        "emotion": emotion,
        "seed_event": seed_event,
        "scenario": integrated_scenario,
        "appraisals": appraisals,
    }


@dataclass(frozen=True)
class DimMeta:
    json_key: str
    output_name: str  # e.g. "Attention"
    corpus_col: str  # e.g. "Attention"
    placeholder_token: str  # e.g. "{ATTENTION}"


DIM_ORDER = ["attention", "certainty", "effort", "pleasantness", "responsibility", "control", "circumstance"]
DIM_META: Dict[str, DimMeta] = {
    "attention": DimMeta(
        json_key="attention",
        output_name="Attention",
        corpus_col="Attention",
        placeholder_token="{ATTENTION}",
    ),
    "certainty": DimMeta(
        json_key="certainty",
        output_name="Certainty",
        corpus_col="Certainty",
        placeholder_token="{CERTAINTY}",
    ),
    "effort": DimMeta(
        json_key="effort",
        output_name="Effort",
        corpus_col="Effort",
        placeholder_token="{EFFORT}",
    ),
    "pleasantness": DimMeta(
        json_key="pleasantness",
        output_name="Pleasantness",
        corpus_col="Pleasant",
        placeholder_token="{PLEASANTNESS}",
    ),
    "responsibility": DimMeta(
        json_key="responsibility",
        output_name="Responsibility",
        corpus_col="Responsibility",
        placeholder_token="{RESPONSIBILITY}",
    ),
    "control": DimMeta(
        json_key="control",
        output_name="Self-Control",
        corpus_col="Control",
        placeholder_token="{SELF_CONTROL}",
    ),
    "circumstance": DimMeta(
        json_key="circumstance",
        output_name="Circumstance",
        corpus_col="Circumstance",
        placeholder_token="{CIRCUMSTANCE}",
    ),
}


@dataclass(frozen=True)
class DimPromptSource:
    """Per-dimension definition + worked examples (appraisal-only; no corpus emotion label). All hardcoded."""

    definition: str
    example_appraisal_state: str
    example_event: str
    example_scenario_expansion: str  # empty string for pleasantness
    example_appraisal_expansion: str


@dataclass(frozen=True)
class DimStatic:
    definition: str
    example_appraisal_state: str
    example_event: str
    example_scenario_expansion: str
    example_appraisal_expansion: str
    current_state_template: str  # equals DIM_META placeholder_token; replaced with state sentence in blocks


DIMENSION_PROMPT_STATIC: Dict[str, DimPromptSource] = {
    "attention": DimPromptSource(
        definition=(
            "Attention refers to the extent to which the writer wanted to continue attending to the event, "
            "focusing on it, or mentally staying engaged with what was happening."
        ),
        example_appraisal_state="The person strongly wanted to devote further attention to the event.",
        example_event=(
            "The person is waiting to hear whether admission to a preferred university has been granted."
        ),
        example_scenario_expansion=(
            "The person keeps checking email and replaying details of the application process, "
            "unable to stop focusing on the decision."
        ),
        example_appraisal_expansion=(
            "The person kept circling back to the application, as if stepping away entirely "
            "did not feel possible until the answer arrived."
        ),
    ),
    "certainty": DimPromptSource(
        definition=(
            "Certainty refers to the extent to which the writer was certain about what was happening in the "
            "situation, including how well they understood what was going on."
        ),
        example_appraisal_state="The person was very uncertain about what was happening.",
        example_event="The person is waiting for the results of a medical test.",
        example_scenario_expansion=(
            "A doctor had only said that the results would arrive sometime that week, with no specific day "
            "or time and no preview of what the numbers might mean."
        ),
        example_appraisal_expansion=(
            "The person could not pin down what was coming or when anything definitive would land; "
            "the timeline and implications both seemed open-ended."
        ),
    ),
    "effort": DimPromptSource(
        definition=(
            "Effort refers to the extent to which the writer had to expend mental or physical effort to deal "
            "with the situation, respond to it, or cope with what was happening."
        ),
        example_appraisal_state=(
            "The person felt that they needed to expend a great deal of mental or physical effort "
            "to deal with the situation."
        ),
        example_event="The person is preparing for an important final exam.",
        example_scenario_expansion=(
            "The exam covered months of difficult material, and the person spent long nights reviewing notes, "
            "solving practice problems, and drilling weak topics before the test."
        ),
        example_appraisal_expansion=(
            "The person experienced the preparation as heavy and ongoing, with attention pulled back repeatedly "
            "because the workload still did not feel finished."
        ),
    ),
    "pleasantness": DimPromptSource(
        definition="Pleasantness refers to the extent to which the person experienced the event as pleasant.",
        example_appraisal_state="The person felt that the event was very pleasant.",
        example_event="The person receives news of acceptance into a preferred university.",
        example_scenario_expansion="",
        example_appraisal_expansion=(
            "The person read the news as a clear positive outcome after a long wait, not as mixed or neutral."
        ),
    ),
    "responsibility": DimPromptSource(
        definition=(
            "Responsibility refers to the extent to which the writer saw themselves as being responsible for "
            "bringing about the situation or causing what happened."
        ),
        example_appraisal_state="The person felt that they were very responsible for the situation.",
        example_event="The person accidentally sends a confidential email to the wrong recipient.",
        example_scenario_expansion=(
            "The person typed the wrong recipient address while sending the message, which routed the "
            "confidential information to someone who was not supposed to receive it."
        ),
        example_appraisal_expansion=(
            "The person saw the mistake as originating in a slip at the keyboard rather than as something "
            "that happened without personal involvement."
        ),
    ),
    "control": DimPromptSource(
        definition=(
            "Self-Control refers to the extent to which the writer found that he or she was able to influence "
            "or manage what was happening in the situation."
        ),
        example_appraisal_state="The person felt that they were very much in control of the situation.",
        example_event="A false rumor about the person starts spreading in class.",
        example_scenario_expansion=(
            "The person had complete message records on a phone showing how the rumor was fabricated "
            "and who had started it."
        ),
        example_appraisal_expansion=(
            "The person believed those records could be used to show the rumor was false and stop it "
            "from spreading further."
        ),
    ),
    "circumstance": DimPromptSource(
        definition=(
            "Circumstance refers to the extent to which the writer saw the event as determined by circumstances "
            "that could not have been changed, prevented, or influenced by anyone."
        ),
        example_appraisal_state="The person felt that the event could not have been changed or influenced by anyone.",
        example_event="The person’s home is badly damaged during an earthquake.",
        example_scenario_expansion=(
            "The earthquake struck suddenly without warning and caused severe damage to the house within seconds."
        ),
        example_appraisal_expansion=(
            "The person experienced the damage as something no one present could have steered or prevented "
            "in the moment."
        ),
    ),
}


def build_dim_static(dim_key: str) -> DimStatic:
    src = DIMENSION_PROMPT_STATIC[dim_key]
    return DimStatic(
        definition=src.definition,
        example_appraisal_state=src.example_appraisal_state,
        example_event=src.example_event,
        example_scenario_expansion=src.example_scenario_expansion,
        example_appraisal_expansion=src.example_appraisal_expansion,
        current_state_template=DIM_META[dim_key].placeholder_token,
    )


def build_dimension_block(
    dim_key: str,
    state_text: str,
    dim_static: DimStatic,
    dim_meta: DimMeta,
) -> str:
    current_state = dim_static.current_state_template.replace(
        dim_meta.placeholder_token, state_text
    )
    # Appraisal only (no corpus emotion label) in blocks sent to the batch step prompts.
    parts = [
        f"{dim_meta.output_name}\n",
        f"Definition: {dim_static.definition}\n",
        f"Example appraisal state: {dim_static.example_appraisal_state}\n",
        f"Example event: {dim_static.example_event}\n",
    ]
    if dim_key == "pleasantness":
        parts.append(f"Example appraisal expansion: {dim_static.example_appraisal_expansion}\n")
    else:
        parts.append(f"Example scenario expansion: {dim_static.example_scenario_expansion}\n")
        parts.append(f"Example appraisal expansion: {dim_static.example_appraisal_expansion}\n")
    parts.append(f"Current appraisal state: {current_state}")
    return "".join(parts)


def build_expanded_scenario_text(
    expansions: Dict[str, ExpansionPerDim],
    anchor_dims: List[str],
    seed_event: str,
) -> str:
    """
    Text conditioned on in Step 2: only non-Pleasantness anchor scenario_expansion lines
    (Pleasantness never supplies scenario text). If none, use the seed event only.
    """
    parts: List[str] = []
    for dim in anchor_dims:
        if dim == "pleasantness":
            continue
        ex = expansions.get(dim)
        if not ex:
            continue
        scen = (ex.get("scenario_expansion") or "").strip()
        if not scen:
            continue
        out_name = DIM_META[dim].output_name
        parts.append(f"{out_name}:\n{scen}")
    if not parts:
        return f"Event:\n{seed_event}"
    return "\n\n".join(parts).strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Expand seeds to scenario JSON via LLM (three-step).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    parser.add_argument("--start", type=int, default=0, help="Start from row index (0-based)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSONL path")
    parser.add_argument("--corpus", type=str, default=str(CORPUS_PATH), help="Corpus TSV path")
    parser.add_argument(
        "--no-step3",
        action="store_true",
        help="Skip Step 3; write empty string for top-level scenario",
    )
    args = parser.parse_args()

    # Load corpus
    df = pd.read_csv(args.corpus, sep="\t")
    if args.start > 0:
        df = df.iloc[args.start :]
    if args.limit is not None:
        df = df.head(args.limit)

    # Load shared headers
    seed_event_prompt = load_prompt("seed_event")
    step1_header = load_prompt("step1_shared_header")
    step2_header = load_prompt("step2_shared_header")
    step3_header = load_prompt("step3_scenario_summary")

    # Load mapping state functions
    from seed2scenario import (
        attention_state,
        certainty_state,
        effort_state,
        pleasantness_state,
        responsibility_state,
        self_control_state,
        circumstance_state,
    )

    state_fns = {
        "attention": attention_state,
        "certainty": certainty_state,
        "effort": effort_state,
        "pleasantness": pleasantness_state,
        "responsibility": responsibility_state,
        "control": self_control_state,
        "circumstance": circumstance_state,
    }

    dim_static: Dict[str, DimStatic] = {dim: build_dim_static(dim) for dim in DIM_ORDER}

    client = get_client()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        global_i = args.start + i
        scenario_id = f"sample_{global_i:04d}"

        try:
            # 1) Seed event
            sentence = str(row["Sentence"]).strip()
            seed_event = get_seed_event(client, sentence, seed_event_prompt)
            emotion_lower = str(row["Prior_Emotion"]).strip().lower()

            # Read appraisal values
            values = {dim: int(row[DIM_META[dim].corpus_col]) for dim in DIM_ORDER}
            anchor_dims = [dim for dim in DIM_ORDER if values[dim] in (0, 3)]
            softer_dims = [dim for dim in DIM_ORDER if values[dim] in (1, 2)]

            expansions: Dict[str, ExpansionPerDim] = {}

            # Step 1: anchors (0/3) — dual tracks per step1_shared_header.txt
            if anchor_dims:
                dim_blocks = "\n\n".join(
                    build_dimension_block(
                        dim_key=dim,
                        state_text=state_fns[dim](values[dim]),
                        dim_static=dim_static[dim],
                        dim_meta=DIM_META[dim],
                    )
                    for dim in anchor_dims
                )
                prompt = step1_header.replace("{EVENT}", seed_event).replace(
                    "{DIMENSION_BLOCKS}", dim_blocks
                )

                raw = call_api_with_retry(client, messages=[{"role": "user", "content": prompt}])
                expected_labels = dual_track_expected_output_labels(anchor_dims)
                parsed = parse_dimension_outputs(raw, expected_labels)
                expansions.update(merge_dual_track_parsed(anchor_dims, parsed))

            # Step 2: softers (1/2) — same dual-track labels as Step 1
            if softer_dims:
                expanded_scenario_text = build_expanded_scenario_text(
                    expansions, anchor_dims, seed_event
                )

                dim_blocks = "\n\n".join(
                    build_dimension_block(
                        dim_key=dim,
                        state_text=state_fns[dim](values[dim]),
                        dim_static=dim_static[dim],
                        dim_meta=DIM_META[dim],
                    )
                    for dim in softer_dims
                )

                prompt = (
                    step2_header.replace("{EXPANDED_SCENARIO}", expanded_scenario_text)
                    .replace("{DIMENSION_BLOCKS}", dim_blocks)
                )

                raw = call_api_with_retry(client, messages=[{"role": "user", "content": prompt}])
                expected_labels = dual_track_expected_output_labels(softer_dims)
                parsed = parse_dimension_outputs(raw, expected_labels)
                expansions.update(merge_dual_track_parsed(softer_dims, parsed))

            # Sanity: ensure we have all 7 expansions
            missing = [d for d in DIM_ORDER if d not in expansions]
            if missing:
                raise ValueError(f"Missing expansions for {missing}")

            if args.no_step3:
                integrated = ""
            else:
                step3_prompt = build_step3_scenario_summary_prompt(
                    seed_event, emotion_lower, expansions, step3_header
                )
                integrated = integrate_scenario_with_llm(client, step3_prompt)

            scenario = build_scenario(row, seed_event, expansions, scenario_id, integrated)
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(scenario, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{total}] {scenario_id} ok")
        except Exception as e:
            print(f"[{i+1}/{total}] {scenario_id} FAIL: {e}")
            raise

    print(f"Done. Wrote {total} scenarios to {args.output}")


if __name__ == "__main__":
    main()
