"""
Data utilities: DataBuilder for corpus, prompt, gold, prediction, and record.
"""
import csv
import json
from pathlib import Path


def load_corpus_tsv(path: str | Path) -> list[dict]:
    """Load TSV corpus; return list of row dicts (keys = column names)."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def load_dialogs_jsonl(path: str | Path) -> list[dict]:
    """Load dialogs.jsonl; return list of records (profile_id, corpus_id, conversation)."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dialog_eval_prompt(prompt_template: str, dialog_record: dict, mode: str = "dialog_full") -> str:
    """
    Build evaluator prompt for one dialog.
    - dialog_first: use only the first User message as context_text (evaluator infers emotion from that).
    - dialog_full: use full conversation as context_text.
    dialog_record must have "conversation": [{"User": "..."}, {"Assistant": "..."}, ...].
    """
    conv = dialog_record.get("conversation", [])
    if mode == "dialog_first":
        # First User turn only
        for turn in conv:
            if "User" in turn:
                context_text = turn["User"]
                break
        else:
            context_text = ""
    else:
        lines = []
        for turn in conv:
            for role, content in turn.items():
                lines.append(f"{role}: {content}")
        context_text = "\n".join(lines)
    return prompt_template.replace("{context_text}", context_text)


class DataBuilder:
    def __init__(self, prompt_template, event_row: dict):
        self.prompt_template = prompt_template
        self.event_row = event_row
        self.event_id = None

    # 构建evaluator在只评估event文本时候的prompt。无需dialog交互。
    def build_eval_base_prompt(self) -> str:
        context_text = f"Sentence: {self.event_row['Sentence']}"
        return self.prompt_template.replace("{context_text}", context_text)

    # 为了方便debug，要构建一个gold，用于评估时与prediction进行对比。
    def build_gold(self) -> dict:
        return {
            "emotion": self.event_row["Prior_Emotion"],
            "appraisal": {
                "Attention": self.event_row["Attention"],
                "Certainty": self.event_row["Certainty"],
                "Effort": self.event_row["Effort"],
                "Pleasant": self.event_row["Pleasant"],
                "Responsibility": self.event_row["Responsibility"],
                "Control": self.event_row["Control"],
                "Circumstance": self.event_row["Circumstance"]
            }
        }

    # 构建prediction
    def build_prediction(self) -> dict:
        # 从已经生成的raw_prediction文件中读取。

        return None

    
    def _get_event_id(self):
        self.event_id = self.event_row["Sentence_id"]
