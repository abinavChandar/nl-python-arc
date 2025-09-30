# src/arc2_runner.py
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .datasets.arc2_loader import iter_arc2_tasks


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _log(event: str, **kw: Any) -> None:
    rec = {"ts": _ts(), "level": "INFO", "event": event}
    rec.update(kw)
    print(json.dumps(rec), flush=True)


def _progress(i: int, total: int) -> None:
    pct = (i / total * 100.0) if total else 0.0
    _log("progress", current=i, total=total, percent=round(pct, 2))


# ---- locate your existing solver (Jeremy-style evolution) -------------------

def _find_solver():
    """
    We try a few common places / names and return (callable, adapter).
    The adapter wraps different signatures to a unified call:
        result = adapter(solver_callable, task_dict)
    Expected result dict contains at least:
        {"test_outputs": [grid, ...]}  # your predicted test grids
    Optionally:
        "best_score": float
        "generations": int
        "revisions": int
        "telemetry": dict
    """
    candidates = []
    # 1) src/pipeline.py: solve_task(task, cfg=...)
    try:
        from .pipeline import solve_task as _fn
        candidates.append(("pipeline.solve_task", _fn))
    except Exception:
        pass

    # 2) src/run.py: solve_challenge(c, config=...)  (common in your earlier code)
    try:
        from .run import solve_challenge as _fn  # type: ignore
        candidates.append(("run.solve_challenge", _fn))
    except Exception:
        pass

    # 3) src/main.py: solve_challenge(...)
    try:
        from .main import solve_challenge as _fn  # type: ignore
        candidates.append(("main.solve_challenge", _fn))
    except Exception:
        pass

    if not candidates:
        raise RuntimeError(
            "Could not find your solver. Expose one of:\n"
            "  - src/pipeline.py::solve_task(task: dict, cfg=None)\n"
            "  - src/run.py::solve_challenge(c: dict, config=None)\n"
            "  - src/main.py::solve_challenge(c: dict, config=None)\n"
            "…or edit arc2_runner.py to import your solver directly."
        )

    name, fn = candidates[0]
    sig = inspect.signature(fn)

    def _adapter(task: Dict[str, Any]) -> Dict[str, Any]:
        # Try “task” param
        try:
            if "task" in sig.parameters:
                return fn(task=task, cfg=None)  # pipeline style
        except Exception:
            pass
        # Try “c” param
        try:
            if "c" in sig.parameters or "challenge" in sig.parameters:
                # Many solvers accept (c=<task>, config=None, …)
                kwargs = {}
                if "c" in sig.parameters:
                    kwargs["c"] = task
                if "challenge" in sig.parameters:
                    kwargs["challenge"] = task
                if "config" in sig.parameters:
                    kwargs["config"] = None
                return fn(**kwargs)
        except Exception:
            pass
        # Try positional single-arg call
        try:
            return fn(task)
        except Exception as e:
            raise RuntimeError(f"Cannot adapt solver call for {name}: {e}") from e

    return name, _adapter


# ---- attempt persistence ----------------------------------------------------

def _write_attempt(out_dir: Path, task_id: str, result: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task_id}.json"
    with p.open("w") as f:
        json.dump(result, f)
    _log("attempt.saved", task_id=task_id, path=str(p))


# ---- scoring (only when ground truth outputs exist in JSON) -----------------

def _score_if_possible(task: Dict[str, Any], result: Dict[str, Any]) -> Optional[float]:
    """
    If the task JSON includes test outputs (some 'training' sets do),
    compute exact-match accuracy. Otherwise return None.
    """
    preds = result.get("test_outputs")
    if preds is None:
        return None

    # Only score if ground truth outputs are present
    gt_pairs = [t for t in task.get("test", []) if "output" in t]
    if not gt_pairs or len(gt_pairs) != len(preds):
        return None

    correct = 0
    for t, yhat in zip(gt_pairs, preds):
        if t["output"] == yhat:
            correct += 1
    return correct / len(preds) if preds else 0.0


# ---- CLI --------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("ARC2_ROOT", ""),
                    help="Path to ARC-AGI-2/data (e.g., /path/to/ARC-AGI-2/data)")
    ap.add_argument("--split", default=os.getenv("ARC2_SPLIT", "evaluation"),
                    choices=["evaluation", "training"], help="Which split")
    ap.add_argument("--year", default=os.getenv("ARC2_YEAR", ""),
                    help="Optional year folder (e.g., 2025). Leave empty to scan all.")
    ap.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")))
    ap.add_argument("--offset", type=int, default=int(os.getenv("OFFSET", "0")))
    ap.add_argument("--max-tasks", type=int, default=int(os.getenv("MAX_TASKS", "0")))
    ap.add_argument("--attempts-dir", default=os.getenv("ATTEMPTS_DIR", "attempts/arc2"),
                    help="Where to write per-task results JSON")
    args = ap.parse_args(argv)

    if not args.root:
        print("ARC2_ROOT not set and --root not given.", file=sys.stderr)
        return 2

    # Find your solver once
    solver_name, solver_adapter = _find_solver()
    _log("runner.solver", using=solver_name)

    # Load tasks
    year = args.year or None
    tasks = list(iter_arc2_tasks(args.root, split=args.split, year=year))
    total = len(tasks)
    _log("tasks.loaded", split=args.split, count=total, root=args.root, year=year or "all")

    if total == 0:
        print("Loaded 0 tasks. Check ARC2_ROOT / structure.", file=sys.stderr)
        return 1

    # Slice window
    offset = max(args.offset, 0)
    if args.limit and args.limit > 0:
        tasks = tasks[offset : offset + args.limit]
    else:
        tasks = tasks[offset:]

    if args.max_tasks and args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    attempts_dir = Path(args.attempts_dir) / (year or "all") / args.split
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        tid = task.get("task_id", f"task_{i}")
        _progress(i=i, total=len(tasks))
        _log("task.begin", task_id=tid)

        start = time.time()
        try:
            # Call your existing Jeremy-style evolutionary solver
            result = solver_adapter(task)
        except KeyboardInterrupt:
            _log("task.interrupt", task_id=tid)
            return 130
        except Exception as e:
            _log("task.error", task_id=tid, error=str(e))
            continue

        # Ensure at least {"test_outputs": [...]} exists
        if "test_outputs" not in result:
            # Try to be helpful: if solver returned "grids" or similar, adapt.
            if "grids" in result:
                result["test_outputs"] = result["grids"]

        # Optional exact-match scoring (only if GT present in JSON)
        score = _score_if_possible(task, result)
        if score is not None:
            result["exact_match"] = score

        # Telemetry passthrough if your solver returns it
        gens = result.get("generations")
        revs = result.get("revisions")
        best = result.get("best_score")

        dur = (time.time() - start) * 1000.0
        _log("task.end", task_id=tid, ms=round(dur, 2),
             generations=gens, revisions=revs, best_score=best, exact_match=score)

        # Persist attempts/results
        _write_attempt(attempts_dir, tid, result)

    dt = time.time() - t0
    _log("run.end", tasks=len(tasks), seconds=round(dt, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
