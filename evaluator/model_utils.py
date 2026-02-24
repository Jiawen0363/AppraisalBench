"""
Call evaluator model (e.g. vLLM / OpenAI-compatible API). Returns raw response string.
"""
import os
from openai import OpenAI


def call_evaluator(
    prompt: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str = "dummy",
    temperature: float = 0.0,
    max_tokens: int = 512,
    **kwargs,
) -> str:
    """
    Send prompt to the evaluator model (OpenAI-compatible, e.g. vLLM), return raw response text.
    """
    base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    content = resp.choices[0].message.content
    return (content or "").strip()
