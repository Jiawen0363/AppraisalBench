"""
Call evaluator model (e.g. vLLM / OpenAI-compatible API). Returns raw response string.
"""
import os
import unicodedata
from openai import OpenAI


def call_evaluator(
    prompt: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 512,
    **kwargs,
) -> str:
    """
    Send prompt to the evaluator model (OpenAI-compatible, e.g. vLLM), return raw response text.

    max_tokens: pass None to omit (matches minimal OpenAI relay calls); default 512 for JSON-heavy evals.
    Response text is normalized with unicodedata NFKC (full-width / compatibility forms).
    """
    if base_url is not None and not str(base_url).strip():
        base_url = None
    base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
    )
    if not base_url:
        raise ValueError(
            "Missing evaluator base URL. Set OPENAI_BASE_URL/VLLM_BASE_URL "
            "or pass --vllm_endpoint from your run script."
        )
    # vLLM's OpenAI-compatible server often uses "EMPTY" as a placeholder API key.
    # If the server is started with --api-key, the client must send a matching key.
    api_key = (
        api_key
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    create_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        use_mct = os.environ.get("OPENAI_USE_MAX_COMPLETION_TOKENS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if use_mct:
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**create_kwargs, **kwargs)
    content = resp.choices[0].message.content
    text = (content or "").strip()
    return unicodedata.normalize("NFKC", text)
