"""
Build role_desc for User and Assistant from one profile row + one corpus row.
Used by run_dialog.py to fill user_base.txt and assistant_base.txt.
"""
from pathlib import Path

# Corpus TSV columns for appraisal
APPRAISAL_KEYS = [
    "Attention", "Certainty", "Effort", "Pleasant",
    "Responsibility", "Control", "Circumstance",
]

ADV_APPRAISAL_DIM_ORDER = [
    ("attention", "Attention"),
    ("certainty", "Certainty"),
    ("effort", "Effort"),
    ("pleasantness", "Pleasantness"),
    ("responsibility", "Responsibility"),
    ("control", "Control"),
    ("circumstance", "Circumstance"),
]


def _prompt_dir():
    """Interaction prompt folder: interaction/prompt/"""
    return Path(__file__).resolve().parent.parent / "prompt"


def _format_appraisal_values(corpus_row: dict) -> str:
    """Turn one corpus row into a single line: Attention: 3, Certainty: 1, ..."""
    parts = []
    for k in APPRAISAL_KEYS:
        v = corpus_row.get(k, "")
        parts.append(f"{k}: {v}")
    return ", ".join(parts)


def _format_appraisal_expansions(scenario_row: dict) -> str:
    """
    Format all appraisal_expansion fields from a scenarios.jsonl row.
    """
    appraisals = scenario_row.get("appraisals", {})
    if not isinstance(appraisals, dict):
        return ""

    lines = []
    for dim_key, dim_name in ADV_APPRAISAL_DIM_ORDER:
        dim_block = appraisals.get(dim_key, {})
        if not isinstance(dim_block, dict):
            continue
        expansion = str(dim_block.get("appraisal_expansion", "")).strip()
        if expansion:
            lines.append(f"{dim_name}: {expansion}")
    return "\n".join(lines)


def prompt_user(profile_row: dict, corpus_row: dict, template_path: str | Path | None = None) -> str:
    """
    Build User role_desc from one profile row and one corpus row.

    - profile_row: from translated_user_profile.jsonl, must have "player" (persona).
    - corpus_row: from emotion_appraisal_corpus.tsv, must have "Sentence" and the 7 appraisal columns.
    """
    if template_path is None:
        template_path = _prompt_dir() / "user_base.txt"
    template_path = Path(template_path)
    template = template_path.read_text(encoding="utf-8")

    persona = profile_row.get("player", "")
    situation = corpus_row.get("Sentence", "")
    appraisal_values = _format_appraisal_values(corpus_row)

    return template.replace("{persona}", persona).replace(
        "{situation}", situation
    ).replace("{appraisal_values}", appraisal_values)


def prompt_advanced_user(scenario_row: dict, template_path: str | Path | None = None) -> str:
    """
    Build advanced User role_desc from one scenarios.jsonl row.

    Expected keys in scenario_row:
    - "scenario": current event text
    - "emotion": target emotion label
    - "appraisals": dict with per-dimension "appraisal_expansion"
    """
    if template_path is None:
        template_path = _prompt_dir() / "user_advanced.txt"
    template = Path(template_path).read_text(encoding="utf-8")

    scenario = str(scenario_row.get("scenario", "")).strip()
    emotion = str(scenario_row.get("emotion", "")).strip()
    seed_event = str(scenario_row.get("seed_event", "")).strip()
    appraisal = _format_appraisal_expansions(scenario_row)

    return (
        template
        .replace("{scenario}", scenario)
        .replace("{appraisal}", appraisal)
        .replace("{emotion}", emotion)
        .replace("{seed_event}", seed_event)
    )


def prompt_assistant(template_path: str | Path | None = None) -> str:
    """
    Read Assistant role_desc from a fixed prompt file (e.g. assistant_base.txt).
    Path is normally set from shell via run_dialog.py (--assistant_prompt).
    """
    if template_path is None:
        template_path = _prompt_dir() / "assistant_base.txt"
    return Path(template_path).read_text(encoding="utf-8").strip()


def build_user_and_assistant_descs(
    profile_row: dict,
    corpus_row: dict,
    user_prompt_path: str | Path | None = None,
    assistant_prompt_path: str | Path | None = None,
):
    """
    Return (user_desc, assistant_desc) for one (profile, corpus) pair.
    user_prompt_path and assistant_prompt_path can be set from shell (e.g. run_dialog.sh).
    """
    user_desc = prompt_user(profile_row, corpus_row, template_path=user_prompt_path)
    assistant_desc = prompt_assistant(template_path=assistant_prompt_path)
    return user_desc, assistant_desc
