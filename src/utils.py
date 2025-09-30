from __future__ import annotations
import asyncio
import json
import math
import random
import re
from typing import Any, Callable, Coroutine, List

from .logging_basic import debug

JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}$")

def extract_json(text: str) -> Any:
    """
    Try hard to extract a single JSON object from a model response.
    """
    if not text:
        raise ValueError("empty response")
    text = text.strip()
    # Already JSON?
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find last JSON-looking block
    m = JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(0))
    # Fallback: code fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("{") and p.endswith("}"):
                return json.loads(p)
    raise ValueError(f"Could not parse JSON from: {text[:200]}...")

def grid_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    return a == b

def grid_similarity(a: List[List[int]], b: List[List[int]]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return 0.0
    total = len(a) * len(a[0])
    same = sum(1 for i in range(len(a)) for j in range(len(a[0])) if a[i][j] == b[i][j])
    return same / total if total else 0.0

def backoff_times(n: int, base: float = 0.5, cap: float = 8.0):
    t = base
    for _ in range(n):
        yield min(t * (1 + random.random()), cap)
        t *= 2

def retry_async(retries: int = 2, exceptions: tuple[type[Exception], ...] = (Exception,)):
    def deco(fn: Callable[..., Coroutine[Any, Any, Any]]):
        async def wrapper(*a, **kw):
            last: Exception | None = None
            for i, delay in enumerate([0.0, *backoff_times(retries)], 0):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    return await fn(*a, **kw)
                except exceptions as e:
                    last = e
                    debug("retry", attempt=i, err=str(e))
            assert last is not None
            raise last
        return wrapper
    return deco
