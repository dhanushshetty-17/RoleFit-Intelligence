from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime
from typing import Any

from anthropic import APIError, AsyncAnthropic, RateLimitError
from dotenv import load_dotenv


_client: AsyncAnthropic | None = None


def load_env() -> str:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is missing. Add it to .env before running the pipeline.")
    return api_key


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=load_env())
    return _client


async def call_claude(prompt: str, max_tokens: int = 1500) -> str:
    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts: list[str] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            return "".join(text_parts).strip()
        except RateLimitError as exc:
            last_error = exc
        except APIError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < 3:
            await asyncio.sleep((2**attempt) + random.random())
    if last_error is not None:
        raise last_error
    raise RuntimeError("Claude call failed without an error object.")


def parse_json_response(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = ValueError("Parsed JSON is not an object.")
    raise ValueError(f"Could not parse JSON response. Last error: {last_error}. Preview: {text[:500]!r}")


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    clamped = max(min_val, min(max_val, value))
    return (clamped - min_val) / (max_val - min_val)


_COLOR_MAP = {
    "step": "\033[96m",
    "info": "\033[92m",
    "warn": "\033[93m",
    "error": "\033[91m",
}
_RESET = "\033[0m"


def log(step: str, message: str, data: Any | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = _COLOR_MAP.get(step.lower(), "\033[97m")
    payload = f" {json.dumps(data, ensure_ascii=False, default=str)}" if data is not None else ""
    print(f"{color}[{timestamp}] [{step.upper()}] {message}{payload}{_RESET}")
