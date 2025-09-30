from __future__ import annotations
import os
from typing import Any, Dict, List
from openai import AsyncOpenAI

from .logging_basic import info, debug
from .utils import extract_json, retry_async

_clients: dict[str, AsyncOpenAI] = {}

def model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

def client_for(model: str | None) -> AsyncOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama") or "ollama"
    key = f"{base_url}|{api_key}|{model or ''}"
    if key not in _clients:
        _clients[key] = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=180)
    return _clients[key]

@retry_async()
async def chat_json(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    model: str | None = None,
) -> Any:
    mn = model or model_name()
    debug("llm.chat_json.begin", model=mn, temperature=temperature)
    resp = await client_for(mn).chat.completions.create(
        model=mn,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or ""
    try:
        return extract_json(content)
    except Exception:
        # fallback: try text mode
        resp2 = await client_for(mn).chat.completions.create(
            model=mn,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content2 = resp2.choices[0].message.content or ""
        return extract_json(content2)

@retry_async()
async def chat_text(
    messages: List[Dict[str, Any]],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    model: str | None = None,
) -> str:
    mn = model or model_name()
    resp = await client_for(mn).chat.completions.create(
        model=mn,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
