"""One client for every LLM call, over the OpenAI-compatible API (Groq, Ollama, Gemini...)."""

from __future__ import annotations

import time
from functools import lru_cache

from openai import APIStatusError, OpenAI, RateLimitError

from company_intel.config import Settings, get_settings


@lru_cache
def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key or "not-needed")


def get_client(settings: Settings | None = None) -> OpenAI:
    s = settings or get_settings()
    return _client(s.llm_base_url, s.llm_api_key.get_secret_value())


def get_judge_client(settings: Settings | None = None) -> OpenAI:
    s = settings or get_settings()
    key = s.eval_judge_api_key or s.llm_api_key
    return _client(s.eval_judge_base_url, key.get_secret_value())


def _retry_after_seconds(err: RateLimitError) -> float:
    try:
        return min(float(err.response.headers.get("retry-after", 0)), 90.0)
    except (TypeError, ValueError, AttributeError):
        return 0.0


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    retries: int = 5,
) -> str:
    """Single chat completion with backoff on free-tier rate limits."""
    s = get_settings()
    model = model or s.llm_model
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if s.llm_reasoning_effort:
        kwargs["reasoning_effort"] = s.llm_reasoning_effort
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = get_client(s).chat.completions.create(
                model=model,
                max_tokens=max_tokens or s.llm_max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
            content = resp.choices[0].message.content or ""
            if not content.strip() and resp.choices[0].finish_reason == "length":
                raise RuntimeError(
                    f"{model} used all {max_tokens or s.llm_max_tokens} tokens before answering "
                    "(likely on reasoning). Raise LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT."
                )
            return content
        except RateLimitError as e:
            if attempt == retries - 1:
                raise
            # Free tiers limit tokens per minute; wait as long as the server asks.
            retry_after = _retry_after_seconds(e)
            time.sleep(max(delay, retry_after))
            delay *= 2
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")
