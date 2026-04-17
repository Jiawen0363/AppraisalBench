#!/usr/bin/env python3
"""
Build task2 4-option QA files (one file per appraisal dimension).

For each scenario (and matched dialog), this script:
1) takes the gold appraisal_expansion as the correct answer,
2) generates three distractor appraisal interpretations via LLM,
3) emits QA records to 7 JSONL files (one per dimension).
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from evaluator.dialog_roles import line_for_turn
from evaluator.model_utils import call_evaluator


DIMENSIONS = [
    "attention",
    "certainty",
    "effort",
    "pleasantness",
    "responsibility",
    "control",
    "circumstance",
]

DISPLAY_NAME = {
    "attention": "Attention",
    "certainty": "Certainty",
    "effort": "Effort",
    "pleasantness": "Pleasantness",
    "responsibility": "Responsibility",
    "control": "Self-Control",
    "circumstance": "Circumstance",
}

# Reuse-able definitions and examples for the new task2 pipeline.
DIM_DEFINITION = {
    "attention": (
        "Attention refers to the extent to which the person wanted to continue "
        "attending to the event, focusing on it, or mentally staying engaged."
    ),
    "certainty": (
        "Certainty refers to the extent to which the person was certain about "
        "what was happening in the situation."
    ),
    "effort": (
        "Effort refers to the extent to which the person needed mental or "
        "physical effort to deal with the situation."
    ),
    "pleasantness": (
        "Pleasantness refers to the extent to which the person experienced the "
        "event as pleasant."
    ),
    "responsibility": (
        "Responsibility refers to the extent to which the person saw themselves "
        "as responsible for bringing about what happened."
    ),
    "control": (
        "Self-Control refers to the extent to which the person believed they "
        "could influence or manage what was happening."
    ),
    "circumstance": (
        "Circumstance refers to the extent to which the person saw the event as "
        "determined by circumstances that could not have been changed by anyone."
    ),
}

DIM_EXAMPLE = {
    "attention": (
        "Example appraisal state: The person strongly wanted to devote further attention to the event.\n"
        "Example scenario: The person kept checking hospital updates and revisiting the details shared by "
        "the medical team after a child's injury.\n"
        "Example appraisal interpretation: The person was unable to disengage and repeatedly returned to the "
        "situation until every part of it felt mentally accounted for."
    ),
    "certainty": (
        "Example appraisal state: The person was very uncertain about what was happening.\n"
        "Example scenario: The person received partial updates with no clear timeline and no definite "
        "indication of what would happen next.\n"
        "Example appraisal interpretation: The person interpreted the situation as unresolved and ambiguous, "
        "with no stable basis for firm conclusions."
    ),
    "effort": (
        "Example appraisal state: The person needed a great deal of effort to handle the situation.\n"
        "Example scenario: The person coordinated multiple appointments, compared conflicting advice, and "
        "kept adjusting plans under time pressure.\n"
        "Example appraisal interpretation: The person saw the situation as requiring sustained and substantial "
        "exertion rather than a quick or effortless response."
    ),
    "pleasantness": (
        "Example appraisal state: The person felt the event was very pleasant.\n"
        "Example scenario: The person received unexpectedly good news after a long period of waiting.\n"
        "Example appraisal interpretation: The person appraised the event as distinctly positive rather than "
        "neutral, mixed, or aversive."
    ),
    "responsibility": (
        "Example appraisal state: The person felt very responsible for the situation.\n"
        "Example scenario: The person made the final decision that directly led to the outcome.\n"
        "Example appraisal interpretation: The person interpreted the outcome as primarily attributable to their "
        "own actions rather than to external factors."
    ),
    "control": (
        "Example appraisal state: The person felt very much in control of the situation.\n"
        "Example scenario: The person had clear options, access to resources, and the authority to choose "
        "the next steps.\n"
        "Example appraisal interpretation: The person appraised the situation as one they could actively steer "
        "through available actions."
    ),
    "circumstance": (
        "Example appraisal state: The person felt the event could not have been changed by anyone.\n"
        "Example scenario: The critical event unfolded suddenly under conditions no participant could alter "
        "in real time.\n"
        "Example appraisal interpretation: The person interpreted the outcome as determined by unchangeable "
        "circumstances rather than by choices anyone could still make."
    ),
}


def attention_state(value: int) -> str:
    return {
        0: "The person did not want to devote further attention to the event.",
        1: "The person wanted to devote a little attention to the event.",
        2: "The person wanted to devote some attention to the event.",
        3: "The person strongly wanted to devote further attention to the event.",
    }[value]


def certainty_state(value: int) -> str:
    return {
        0: "The person was not certain about what was happening in the situation.",
        1: "The person was only slightly certain about what was happening in the situation.",
        2: "The person was somewhat certain about what was happening in the situation.",
        3: "The person was very certain about what was happening in the situation.",
    }[value]


def effort_state(value: int) -> str:
    return {
        0: "The person did not need to expend effort to deal with the situation.",
        1: "The person needed to expend a little effort to deal with the situation.",
        2: "The person needed to expend some effort to deal with the situation.",
        3: "The person needed to expend a great deal of effort to deal with the situation.",
    }[value]


def pleasantness_state(value: int) -> str:
    return {
        0: "The person felt that the event was not pleasant.",
        1: "The person felt that the event was only slightly pleasant.",
        2: "The person felt that the event was somewhat pleasant.",
        3: "The person felt that the event was very pleasant.",
    }[value]


def responsibility_state(value: int) -> str:
    return {
        0: "The person felt that they were not responsible for the situation.",
        1: "The person felt that they were only slightly responsible for the situation.",
        2: "The person felt that they were somewhat responsible for the situation.",
        3: "The person felt that they were very responsible for the situation.",
    }[value]


def control_state(value: int) -> str:
    return {
        0: "The person felt that they had no ability to influence what was happening in the situation.",
        1: "The person felt that they had little ability to influence what was happening in the situation.",
        2: "The person felt that they had some ability to influence what was happening in the situation.",
        3: "The person felt that they could strongly influence what was happening in the situation.",
    }[value]


def circumstance_state(value: int) -> str:
    return {
        0: "The person felt that the event could have been changed or influenced by someone.",
        1: "The person felt that the event could probably have been changed or influenced by someone.",
        2: "The person felt that the event could not easily have been changed or influenced by anyone.",
        3: "The person felt that the event could not have been changed or influenced by anyone.",
    }[value]


STATE_FN = {
    "attention": attention_state,
    "certainty": certainty_state,
    "effort": effort_state,
    "pleasantness": pleasantness_state,
    "responsibility": responsibility_state,
    "control": control_state,
    "circumstance": circumstance_state,
}


def distractor_values_for(correct_value: int) -> List[int]:
    # Ordered by "harder first" (nearby levels before far levels).
    mapping = {
        0: [1, 2, 3],
        1: [2, 0, 3],
        2: [1, 3, 0],
        3: [2, 1, 0],
    }
    return mapping[correct_value]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_dialog(conversation: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for turn in conversation:
        for role, text in turn.items():
            lines.append(line_for_turn(role, text))
    return "\n".join(lines)


def normalize_one_sentence(text: str) -> str:
    text = " ".join(text.strip().split())
    text = re.sub(
        r"^(Appraisal (expansion|interpretation)|Distractor|Output)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    return text.strip()


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_distractor_prompt(
    template: str,
    scenario: str,
    dim: str,
    target_state: str,
) -> str:
    return (
        template.replace("{SCENARIO}", scenario)
        .replace("{DIMENSION}", DISPLAY_NAME[dim])
        .replace("{DEFINITION}", DIM_DEFINITION[dim])
        .replace("{EXAMPLE}", DIM_EXAMPLE[dim])
        .replace("{TARGET_STATE}", target_state)
    )


def generate_distractor(
    *,
    prompt: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    retries: int,
    sleep_sec: float,
) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            out = call_evaluator(
                prompt=prompt,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=0.2,
                max_tokens=120,
            )
            return normalize_one_sentence(out)
        except Exception as e:  # pragma: no cover
            last_err = e
            if attempt < retries - 1:
                time.sleep(sleep_sec * (attempt + 1))
    raise RuntimeError(f"Failed to generate distractor after retries: {last_err}")


def generate_unique_distractor(
    *,
    prompt: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    retries: int,
    sleep_sec: float,
    existing_answers: List[str],
) -> str:
    """
    Generate one distractor and avoid trivial duplicates.
    """
    seen = {normalize_one_sentence(x).lower() for x in existing_answers}
    for _ in range(3):
        cand = generate_distractor(
            prompt=prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
            retries=retries,
            sleep_sec=sleep_sec,
        )
        key = cand.lower()
        if key and key not in seen:
            return cand
    return cand


def ensure_empty_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dim in DIMENSIONS:
        p = output_dir / f"{dim}.jsonl"
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build task2 4-option QA dataset (7 files).")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("/home/jiawen/AppraisalBench/output/seed2scenario/scenarios.jsonl"),
    )
    parser.add_argument(
        "--dialogs",
        type=Path,
        default=Path("/home/jiawen/AppraisalBench/output/dialog/gpt4o/dialog_advanced.jsonl"),
    )
    parser.add_argument(
        "--question-stems",
        type=Path,
        default=Path("/home/jiawen/AppraisalBench/evaluator/task2_4_quesiont_stem.json"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(
            "/home/jiawen/AppraisalBench/evaluator/prompt/task2/generate_distractor_appraisal_expansion.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jiawen/AppraisalBench/output/evaluation/task2_qa"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N scenarios (0-based).")
    parser.add_argument("--append", action="store_true", help="Append to output files instead of truncating.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=1.5)
    parser.add_argument(
        "--dimensions",
        type=str,
        default="all",
        help="Comma-separated dimensions to run (e.g. attention,certainty) or 'all'.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    scenarios = read_jsonl(args.scenarios)
    dialogs = read_jsonl(args.dialogs)
    stems = json.loads(args.question_stems.read_text(encoding="utf-8"))
    template = load_prompt(args.prompt)

    start_idx = max(0, args.offset)
    if start_idx >= len(scenarios):
        raise ValueError(f"Offset out of range: offset={start_idx}, total={len(scenarios)}")
    scenarios = scenarios[start_idx:]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    if args.dimensions.strip().lower() == "all":
        selected_dims = list(DIMENSIONS)
    else:
        selected_dims = [d.strip().lower() for d in args.dimensions.split(",") if d.strip()]
        invalid = [d for d in selected_dims if d not in DIMENSIONS]
        if invalid:
            raise ValueError(f"Invalid dimensions: {invalid}. Valid: {DIMENSIONS}")
        # Keep deterministic order.
        selected_dims = [d for d in DIMENSIONS if d in set(selected_dims)]

    dialog_by_id: Dict[str, Dict[str, Any]] = {}
    for d in dialogs:
        sid = d.get("scenario_id")
        if sid:
            dialog_by_id[sid] = d

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Clear selected output files only when not appending.
    if not args.append:
        for dim in selected_dims:
            p = args.output_dir / f"{dim}.jsonl"
            if p.exists():
                p.unlink()
    out_files = {
        dim: (args.output_dir / f"{dim}.jsonl").open("a", encoding="utf-8")
        for dim in selected_dims
    }

    try:
        total = len(scenarios)
        for idx, s in enumerate(scenarios, start=start_idx + 1):
            scenario_id = s["id"]
            dialog_rec = dialog_by_id.get(scenario_id)
            if dialog_rec is None:
                print(f"[skip] Missing dialog for scenario_id={scenario_id}")
                continue

            dialog_text = render_dialog(dialog_rec.get("conversation", []))
            emotion = s.get("emotion", "")
            scenario_text = s.get("scenario", "")

            for dim in selected_dims:
                app = s["appraisals"][dim]
                correct_value = int(app["value"])
                correct_answer = (app.get("appraisal_expansion") or "").strip()
                if not correct_answer:
                    raise ValueError(f"Missing correct appraisal_expansion: {scenario_id}/{dim}")

                distractor_values = distractor_values_for(correct_value)
                distractor_answers: List[str] = []
                for dv in distractor_values:
                    target_state = STATE_FN[dim](dv)
                    prompt = build_distractor_prompt(
                        template=template,
                        scenario=scenario_text,
                        dim=dim,
                        target_state=target_state,
                    )
                    da = generate_unique_distractor(
                        prompt=prompt,
                        model=args.model,
                        base_url=args.base_url,
                        api_key=args.api_key,
                        retries=args.retries,
                        sleep_sec=args.retry_sleep,
                        existing_answers=[correct_answer] + distractor_answers,
                    )
                    distractor_answers.append(da)

                all_options = [correct_answer] + distractor_answers
                random.shuffle(all_options)
                option_labels = ["A", "B", "C", "D"]
                options_map = {label: all_options[i] for i, label in enumerate(option_labels)}
                correct_option = next(
                    label for label, text in options_map.items() if text == correct_answer
                )

                record = {
                    "scenario_id": scenario_id,
                    "appraisal_dimension": dim,
                    "question_stem": stems[dim],
                    "scenario": scenario_text,
                    "dialog": dialog_text,
                    "emotion": emotion,
                    "option_A": options_map["A"],
                    "option_B": options_map["B"],
                    "option_C": options_map["C"],
                    "option_D": options_map["D"],
                    "correct_option": correct_option,
                    "correct_value": correct_value,
                    "distractor_values": distractor_values,
                }
                out_files[dim].write(json.dumps(record, ensure_ascii=False) + "\n")
                out_files[dim].flush()

            done_in_batch = idx - start_idx
            print(f"[{done_in_batch}/{total}] global#{idx} {scenario_id} done")
    finally:
        for f in out_files.values():
            f.close()

    print(f"Done. Wrote 7 files under: {args.output_dir}")


if __name__ == "__main__":
    main()
