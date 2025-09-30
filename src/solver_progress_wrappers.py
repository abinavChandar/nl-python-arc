from __future__ import annotations
from typing import List, Any, Callable, Iterable
from .telemetry import progress_iter, emit_task_overall, phase_total, phase_done

def instrument_sync_loop(
    items: Iterable[Any],
    *,
    label: str,
    event: str,
    phase: str,
    gen_idx: int,
    gen_count: int,
    body_fn: Callable[[int, Any], None],
    note_fn: Callable[[int, Any], str] | None = None,
) -> None:
    """Run a sync loop with progress updates per item."""
    count = len(items) if hasattr(items, "__len__") else None
    if count is None:
        items = list(items)
        count = len(items)
    phase_total(count, label=label, phase_name=phase, gen_idx=gen_idx, gen_count=gen_count)
    for i, obj in enumerate(progress_iter(
        items,
        label=label,
        event_name=event,
        phase_name=phase,
        gen_idx=gen_idx,
        gen_count=gen_count,
        note_fn=note_fn,
    )):
        body_fn(i, obj)
    phase_done(label=label, phase_name=phase)
