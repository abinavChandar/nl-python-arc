from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from pydantic import TypeAdapter
from .arc_types import Challenge, Example, Grid

def load_challenges(path: Path) -> Dict[str, Challenge]:
    raw = json.loads(path.read_text())
    out: Dict[str, Challenge] = {}
    for task_id, obj in raw.items():
        train = [Example.model_validate(e) for e in obj["train"]]
        test = obj["test"]
        out[task_id] = Challenge(task_id=task_id, train=train, test=test)
    return out

def load_solutions(path: Path) -> Dict[str, List[Grid]]:
    if not path.exists():
        return {}
    ta = TypeAdapter(dict[str, list[list[list[int]]]])
    return ta.validate_json(path.read_text())
