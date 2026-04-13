"""Display names for dialog turns (corpus roles -> prompt labels)."""

from __future__ import annotations

# Keys match role strings in output/dialog JSONL "conversation" turns.
ROLE_IN_PROMPT: dict[str, str] = {
    "User": "Experiencer",
    "Assistant": "Responder",
}


def line_for_turn(role: str, content: str) -> str:
    label = ROLE_IN_PROMPT.get(role, role)
    return f"{label}: {content}"


def relabel_dialog_block(text: str) -> str:
    """Rewrite User:/Assistant: prefixes in already-rendered dialog strings."""
    out: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("User:"):
            out.append("Experiencer:" + line.removeprefix("User:"))
        elif line.startswith("Assistant:"):
            out.append("Responder:" + line.removeprefix("Assistant:"))
        else:
            out.append(line)
    return "\n".join(out)
