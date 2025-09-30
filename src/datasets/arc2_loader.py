# src/datasets/arc2_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _read_task_json(path: Path) -> Dict:
    with path.open("r") as f:
        data = json.load(f)
    # Some ARC2 repos don’t include task_id inside each JSON file
    data.setdefault("task_id", path.stem)
    # Normalize structure to {task_id, train:[{input,output},…], test:[{input},…]}
    if "train" not in data or "test" not in data:
        raise ValueError(f"Malformed ARC task at {path}")
    return data


def iter_arc2_tasks(
    root: str | Path,
    split: str = "evaluation",   # or "training"
    year: Optional[str] = None,  # e.g. "2025" or "2024"; if None, scan all year dirs
) -> Iterable[Dict]:
    """
    Walks ARC-AGI-2 layout:

      <root>/
        2025/
          evaluation/**.json
          training/**.json
        2024/
          evaluation/**.json
          training/**.json

    Yields normalized task dicts.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ARC2 root not found: {root}")

    split = split.lower().strip()
    if split not in {"evaluation", "training"}:
        raise ValueError("split must be 'evaluation' or 'training'")

    years: List[str]
    if year:
        years = [year]
    else:
        years = [d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
        years.sort()

    for y in years:
        base = root / y / split
        if not base.exists():
            continue
        for jf in sorted(base.rglob("*.json")):
            try:
                yield _read_task_json(jf)
            except Exception as e:
                print(f"[warn] could not read {jf}: {e}")
