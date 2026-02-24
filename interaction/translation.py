"""
Translate player field (Chinese -> English) in orig_user_profile.jsonl.
Reads id, player, main_cha, cha_group; sends player to LLM; writes same structure with translated player.
"""
import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


def call_llm(prompt: str, model: str, base_url: str, max_tokens: int = 4096, timeout: float = 120.0) -> str:
    client = OpenAI(api_key="dummy", base_url=base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content
    return (content or "").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, default="orig_user_profile.jsonl")
    p.add_argument("--output_file", type=str, default="interaction/translated_user_profile.jsonl")
    p.add_argument("--prompt_file", type=str, default="interaction/prompt/translation_prompt.txt")
    p.add_argument("--model", type=str, default=os.environ.get("VLLM_MODEL", "/data/models/Qwen3-8B"))
    p.add_argument("--base_url", type=str, default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    input_path = repo_root / args.input_file
    output_path = repo_root / args.output_file
    prompt_path = repo_root / args.prompt_file

    template = prompt_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))

    if args.limit is not None:
        lines = lines[: args.limit]

    n = len(lines)
    print(f"Translating {n} profiles, output to {output_path}", flush=True)

    with output_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(lines):
            row_id = rec.get("id", "")
            player_cn = rec.get("player", "")
            main_cha = rec.get("main_cha", "")
            cha_group = rec.get("cha_group", [])

            prompt = template.replace("{player_text}", player_cn)
            player_en = call_llm(prompt, model=args.model, base_url=args.base_url)

            out_rec = {
                "id": row_id,
                "player": player_en,
                "main_cha": main_cha,
                "cha_group": cha_group,
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1}/{n}", flush=True)

    print(f"Done. Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
