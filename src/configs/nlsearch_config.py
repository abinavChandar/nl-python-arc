# src/configs/nlsearch_config.py
from __future__ import annotations
import os
from dataclasses import dataclass

def _b(s: str, default: bool) -> bool:
    v = os.getenv(s)
    if v is None: 
        return default
    return v.lower() in ("1", "true", "yes", "on")

def _i(s: str, default: int) -> int:
    v = os.getenv(s)
    return int(v) if v is not None else default

def _f(s: str, default: float) -> float:
    v = os.getenv(s)
    return float(v) if v is not None else default

@dataclass
class NLSearchConfig:
    # Evolution knobs
    max_generations: int         # how many evolutionary generations
    proposals_per_gen: int       # how many fresh instruction drafts per generation
    top_k: int                   # keep the best K per generation
    revisions_per_top_k: int     # how many revision rounds for each kept draft
    final_follow_times: int      # attempts per test input (ARC policy allows up to 2)

    # Execution knobs
    max_concurrency: int         # async parallelism across LLM calls
    patience_gen: int            # stop after N gens without improvement
    stop_on_perfect: bool        # immediately finalize when train score hits 1.0
    max_min_per_task: float      # wall-clock hard limit per task (minutes)

    # UI / logging
    live_progress: bool
    show_tty_progress: bool
    log_level: str

    # Dataset / model (also exposed via CLI; env is default)
    dataset: str
    root: str
    split: str
    ollama_model: str

def from_env() -> NLSearchConfig:
    return NLSearchConfig(
        max_generations=_i("MAX_GENERATIONS", 4),
        proposals_per_gen=_i("PROPOSALS_PER_GEN", 6),
        top_k=_i("TOP_K", 3),
        revisions_per_top_k=_i("REVISIONS_PER_TOPK", 2),
        final_follow_times=_i("FINAL_FOLLOW_TIMES", 2),

        max_concurrency=_i("MAX_CONCURRENCY", 2),
        patience_gen=_i("PATIENCE_GEN", 2),
        stop_on_perfect=_b("STOP_ON_PERFECT", True),
        max_min_per_task=_f("MAX_MIN_PER_TASK", 12.0),

        live_progress=_b("LIVE_PROGRESS", True),
        show_tty_progress=_b("SHOW_TTY_PROGRESS", True),
        log_level=os.getenv("LOG_LEVEL", "INFO"),

        dataset=os.getenv("DATASET", "arc2"),
        root=os.getenv("ARC2_ROOT", ""),
        split=os.getenv("ARC2_SPLIT", "evaluation"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    )
