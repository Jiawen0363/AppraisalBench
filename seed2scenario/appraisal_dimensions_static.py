"""
Appraisal dimension display names and prompt definitions (no heavy deps).

Shared by seed2scenario generation and evaluator Task 5 question building.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DimMeta:
    json_key: str
    output_name: str  # e.g. "Attention"
    corpus_col: str  # e.g. "Attention"
    placeholder_token: str  # e.g. "{ATTENTION}"


DIM_ORDER = [
    "attention",
    "certainty",
    "effort",
    "pleasantness",
    "responsibility",
    "control",
    "circumstance",
]

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
