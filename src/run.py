# src/run.py
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# --- optional .env support ----------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    for candidate in (
        Path(".env"),
        Path("src/configs/.env"),
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)
except Exception:
    pass  # dotenv is optional; continue with real environment

# --- minimal JSONL telemetry ---------------------------------------------------
def jlog(level: str, event: str, **fields: Any) -> None:
    rec = {
        "ts": dt.datetime.utcnow().isoformat(timespec="microseconds") + "Z",
        "level": level.upper(),
        "event": event,
        **fields,
    }
    print(json.dumps(rec, ensure_ascii=False), flush=True)


# --- config -------------------------------------------------------------------
@dataclasses.dataclass
class Config:
    root: str = os.getenv("ARC2_ROOT", "").strip()
    split: str = os.getenv("ARC2_SPLIT", "evaluation").strip()  # "evaluation" | "training"

    # Execution knobs (optional)
    max_tasks: int | None = None                # limit how many tasks to load
    live_progress: bool = os.getenv("LIVE_PROGRESS", "1") == "1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Model / engine fields passed through to your solver (optional)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    # Evolution knobs (only used if your solver reads them from cfg)
    max_generations: int = int(os.getenv("MAX_GENERATIONS", "4"))
    proposals_per_gen: int = int(os.getenv("PROPOSALS_PER_GEN", "6"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    revisions_per_topk: int = int(os.getenv("REVISIONS_PER_TOPK", "2"))
    final_follow_times: int = int(os.getenv("FINAL_FOLLOW_TIMES", "2"))
    stop_on_perfect: bool = os.getenv("STOP_ON_PERFECT", "1") == "1"
    patience_gen: int = int(os.getenv("PATIENCE_GEN", "2"))
    max_min_per_task: int = int(os.getenv("MAX_MIN_PER_TASK", "12"))

    @classmethod
    def from_args(cls) -> "Config":
        p = argparse.ArgumentParser(description="Run ARC-AGI-2 pipeline")
        p.add_argument("--root", help="Path to ARC-AGI-2 repo root or its data/ folder "
                                      "(can also set ARC2_ROOT in .env)")
        p.add_argument("--split", default=None, help="evaluation | training (default from ARC2_SPLIT)")
        p.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks loaded")
        args = p.parse_args()

        cfg = cls()
        if args.root:
            cfg.root = args.root
        if args.split:
            cfg.split = args.split
        if args.max_tasks is not None:
            cfg.max_tasks = args.max_tasks
        return cfg


# --- ARC-AGI-2 loader ---------------------------------------------------------
def _resolve_arc2_data_root(root: str) -> Path:
    """
    Accept either the ARC-AGI-2 repo root or its data dir; return the data dir.
    Raise with clear guidance if anything is wrong.
    """
    if not root:
        raise ValueError(
            "ARC2_ROOT is empty. Set it to the ARC-AGI-2/data directory in your .env.\n"
            "Example:\n"
            "  ARC2_ROOT=/Users/you/datasets/ARC-AGI-2/data\n"
        )
    p = Path(root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"ARC2_ROOT does not exist: {p}")

    # If they passed the repo root (contains /data but no task JSONs), adjust to /data
    if (p / "data").is_dir() and not any(p.glob("*.json")):
        p = (p / "data").resolve()

    # At this point p should be the data dir with 'training' and/or 'evaluation'
    if not (p / "training").is_dir() and not (p / "evaluation").is_dir():
        raise FileNotFoundError(
            f"ARC2_ROOT={p} does not contain 'training/' or 'evaluation/' folders.\n"
            "Point ARC2_ROOT to the ARC-AGI-2/data directory."
        )
    return p


def load_arc2_tasks(root: str, split: str, max_tasks: int | None = None) -> list[dict]:
    data_root = _resolve_arc2_data_root(root)
    split_dir = (data_root / split).resolve()

    # Diagnostic info
    print(f"[ARC2] Using split dir: {split_dir}")
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Split directory does not exist: {split_dir}  "
            f"(ARC2_ROOT={data_root}, split={split})"
        )

    files = sorted(split_dir.glob("*.json"))
    print(f"[ARC2] Found {len(files)} JSON files in {split_dir}")
    if max_tasks:
        files = files[:max_tasks]

    tasks: list[dict] = []
    for path in files:
        try:
            obj = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse {path}: {e}") from e

        if not all(k in obj for k in ("train", "test")):
            raise ValueError(f"{path} missing 'train' or 'test' keys.")
        tasks.append({
            "task_id": path.stem,
            "train": obj["train"],
            "test": obj["test"],
        })
    return tasks


# --- Optional solver hook ------------------------------------------------------
# If you have your own solver, expose it as src/pipeline.py::solve_task(task:dict, cfg:Config) -> dict
_SOLVER = None
try:
    # Your own pipeline (if present)
    from .pipeline import solve_task  # type: ignore
    _SOLVER = solve_task
except Exception:
    # No solver available is fine for data loading diagnostics
    pass


def _run_solver_if_available(task: dict, cfg: Config) -> None:
    if _SOLVER is None:
        # No pipeline imported; just show the task id as a placeholder
        jlog("INFO", "task.loaded", task_id=task["task_id"])
        return

    # Call your solver and log a summary
    jlog("INFO", "task.begin", task_id=task["task_id"], model=cfg.ollama_model)
    try:
        result = _SOLVER(task=task, cfg=cfg)  # your function should handle cfg knobs
        # result is expected to be a dict; log a quick summary if it contains scores
        if isinstance(result, dict):
            jlog("INFO", "task.end",
                 task_id=task["task_id"],
                 status=result.get("status", "ok"),
                 best_score=result.get("best_score"),
                 generations=result.get("generations"),
                 duration_ms=result.get("duration_ms"))
        else:
            jlog("INFO", "task.end", task_id=task["task_id"], status="ok")
    except KeyboardInterrupt:
        jlog("WARN", "task.interrupted", task_id=task["task_id"])
        raise
    except Exception as e:  # noqa: BLE001
        jlog("ERROR", "task.error", task_id=task["task_id"], error=str(e))


# --- main ---------------------------------------------------------------------
def main() -> None:
    cfg = Config.from_args()

    # Echo what we loaded from env/args
    jlog("INFO", "run.config",
         ARC2_ROOT=cfg.root,
         ARC2_SPLIT=cfg.split,
         OLLAMA_MODEL=cfg.ollama_model,
         max_tasks=cfg.max_tasks)

    # Load tasks
    tasks = load_arc2_tasks(cfg.root, cfg.split, max_tasks=cfg.max_tasks or None)
    print(f"Loaded {len(tasks)} {cfg.split} tasks from { _resolve_arc2_data_root(cfg.root) }")

    # Fail fast (the message below is exactly what you asked for)
    if not tasks:
        raise SystemExit(
            "No tasks found.\n"
            f"- ARC2_ROOT currently resolves to: {_resolve_arc2_data_root(cfg.root)}\n"
            f"- Split: {cfg.split}\n"
            "Check that ARC-AGI-2 was cloned and ARC2_ROOT points to its *data* dir, e.g.\n"
            "  ARC2_ROOT=/Users/you/datasets/ARC-AGI-2/data\n"
            "Also verify files exist, e.g.\n"
            "  ls $ARC2_ROOT/evaluation/*.json\n"
        )

    # Run (optionally) per task
    jlog("INFO", "run.begin", total_tasks=len(tasks))
    try:
        for i, task in enumerate(tasks, 1):
            jlog("INFO", "progress", current=i, total=len(tasks),
                 percent=round(100.0 * i / len(tasks), 2))
            _run_solver_if_available(task, cfg)
    finally:
        jlog("INFO", "run.end")


if __name__ == "__main__":
    # Allow `python3 -m src.run`
    # Ensure this file is executed as a module within the project (src is a package)
    # If your PYTHONPATH isn't set, launch from repo root: python3 -m src.run
    try:
        main()
    except SystemExit as e:
        # Show clean message without a messy traceback
        if str(e):
            print(str(e), file=sys.stderr)
        sys.exit(e.code if isinstance(e.code, int) else 1)
