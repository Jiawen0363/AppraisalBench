# -*- coding: utf-8 -*-
"""
Run User–Assistant dialog: load profile + corpus, pair them, build prompts,
run Conversation (User speaks first), save to jsonl.

Usage: run from repo root. Start vLLM with scripts/run_engine_user.sh (port 8000)
and scripts/run_engine_assistant.sh (port 8001), then run scripts/run_dialog.sh.
"""
import argparse
import json
import csv
import random
import os
import sys
from pathlib import Path

# Run from repo root; add interaction/ so that chatarena and utils resolve
_INTERACTION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_INTERACTION_DIR))

from chatarena.agent import Player
from chatarena.agent_assistant import Assistant
from chatarena.backends.openai_vllm import VLLMChat
from chatarena.environments.conversation import Conversation
from chatarena.arena import Arena
from utils.make_prompt import build_user_and_assistant_descs, prompt_advanced_user, prompt_assistant


def parse_args():
    parser = argparse.ArgumentParser(description="Run User–Assistant dialog simulation.")
    parser.add_argument("--user_prompt", type=str, required=True,
                        help="Path to user prompt template (e.g. interaction/prompt/user_base.txt).")
    parser.add_argument("--assistant_prompt", type=str, required=True,
                        help="Path to assistant prompt template (e.g. interaction/prompt/assistant_base.txt).")
    parser.add_argument("--profile_file", type=str, default="",
                        help="Path to translated_user_profile.jsonl.")
    parser.add_argument("--corpus_file", type=str, default="",
                        help="Path to emotion_appraisal_corpus.tsv.")
    parser.add_argument("--scenario_file", type=str, default="",
                        help="Path to scenarios.jsonl used in advanced dialog mode.")
    parser.add_argument("--dialog_mode", type=str, default="base", choices=["base", "advanced"],
                        help="Prompt construction mode: base (profile+corpus) or advanced (scenario+appraisal_expansion).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output path for dialogs (jsonl).")
    parser.add_argument("--user_model", type=str, default="/data/models/Qwen3-4B",
                        help="Model name/path for User.")
    parser.add_argument("--assistant_model", type=str, default="/data/models/Qwen3-4B",
                        help="Model name/path for Assistant.")
    parser.add_argument("--vllm_endpoint_user", type=str, default="http://localhost:8000/v1",
                        help="API base URL for User backend (vLLM or OpenAI-compatible endpoint).")
    parser.add_argument("--vllm_endpoint_assistant", type=str, default="http://localhost:8001/v1",
                        help="API base URL for Assistant backend (vLLM or OpenAI-compatible endpoint).")
    parser.add_argument("--user_api_key", type=str, default="",
                        help="API key for User backend. If empty, use OPENAI_API_KEY; fallback to EMPTY.")
    parser.add_argument("--assistant_api_key", type=str, default="",
                        help="API key for Assistant backend. If empty, use ASSISTANT_OPENAI_API_KEY or OPENAI_API_KEY; fallback to EMPTY.")
    parser.add_argument("--max_rounds", type=int, default=1,
                        help="Max dialog rounds (one round = User + Assistant each one message).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of (profile, corpus) pairs to run (default: all).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for pairing.")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Max tokens per response.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    return parser.parse_args()


def load_profiles(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_corpus(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out.append(row)
    return out


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main(args):
    random.seed(args.random_seed)
    user_api_key = args.user_api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
    assistant_api_key = (
        args.assistant_api_key
        or os.getenv("ASSISTANT_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "EMPTY")
    )

    # Paths relative to repo root (cwd when run via run_dialog.sh)
    root = Path.cwd()
    output_path = root / args.output_file
    user_prompt_path = root / args.user_prompt
    assistant_prompt_path = root / args.assistant_prompt

    if args.dialog_mode == "advanced":
        scenario_path = root / args.scenario_file
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        scenarios = load_jsonl(scenario_path)
        n_pairs = len(scenarios) if args.limit is None else min(len(scenarios), args.limit)
        if n_pairs == 0:
            print("No scenario data. Exit.")
            return
    else:
        profile_path = root / args.profile_file
        corpus_path = root / args.corpus_file
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_path}")
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        profiles = load_profiles(profile_path)
        corpus = load_corpus(corpus_path)
        # Event (corpus): use in order from top to bottom. Profile: random sample per dialog.
        n_pairs = len(corpus) if args.limit is None else min(len(corpus), args.limit)
        if n_pairs == 0 or len(profiles) == 0:
            print("No profile or corpus data. Exit.")
            return

        pairs = [(random.choice(profiles), corpus[i]) for i in range(n_pairs)]

    # One backend per agent (reused across dialogs)
    user_backend = VLLMChat(
        vllm_api_key=user_api_key,
        vllm_endpoint=args.vllm_endpoint_user,
        model_name_or_path=args.user_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_latest_messages=-1,
    )
    assistant_backend = VLLMChat(
        vllm_api_key=assistant_api_key,
        vllm_endpoint=args.vllm_endpoint_assistant,
        model_name_or_path=args.assistant_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_latest_messages=-1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_steps = args.max_rounds * 2  # User and Assistant each speak per round

    with output_path.open("w", encoding="utf-8") as fw:
        if args.dialog_mode == "advanced":
            items = scenarios[:n_pairs]
        else:
            items = pairs

        for idx, item in enumerate(items):
            if args.dialog_mode == "advanced":
                scenario_row = item
                print(
                    f"\n========== Dialog {idx + 1}/{len(items)} (scenario_id={scenario_row.get('id', '')}, emotion={scenario_row.get('emotion', '')}) ==========",
                    flush=True,
                )
                user_desc = prompt_advanced_user(scenario_row, template_path=user_prompt_path)
                assistant_desc = prompt_assistant(template_path=assistant_prompt_path)
            else:
                profile_row, corpus_row = item
                print(
                    f"\n========== Dialog {idx + 1}/{len(items)} (profile_id={profile_row.get('id', '')[:8]}..., corpus_id={corpus_row.get('Sentence_id', '')}) ==========",
                    flush=True,
                )
                user_desc, assistant_desc = build_user_and_assistant_descs(
                    profile_row,
                    corpus_row,
                    user_prompt_path=user_prompt_path,
                    assistant_prompt_path=assistant_prompt_path,
                )

            user_agent = Player(name="User", role_desc=user_desc, backend=user_backend)
            assistant_agent = Assistant(role_desc=assistant_desc, backend=assistant_backend)
            env = Conversation(player_names=["User", "Assistant"])
            arena = Arena(players=[user_agent, assistant_agent], environment=env)
            arena.launch_cli(
                max_steps=num_steps,
                interactive=False,
                show_description=False,
                show_message=True,
            )

            messages = env.message_pool.get_all_messages()
            conv = [{"User" if m.agent_name == "User" else "Assistant": m.content} for m in messages]
            if args.dialog_mode == "advanced":
                record = {
                    "scenario_id": scenario_row.get("id", ""),
                    "emotion": scenario_row.get("emotion", ""),
                    "conversation": conv,
                }
            else:
                record = {
                    "profile_id": profile_row.get("id", ""),
                    "corpus_id": corpus_row.get("Sentence_id", ""),
                    "conversation": conv,
                }
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")
            fw.flush()
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  {idx + 1}/{len(items)}", flush=True)

    print(f"Done. Wrote {n_pairs} dialogs to {output_path}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
