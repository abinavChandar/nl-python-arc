# src/telemetry.py
from __future__ import annotations
import json, sys, time
from dataclasses import dataclass, field

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1e6):06d}Z"

@dataclass
class Phase:
    name: str
    total: int = 0
    done: int = 0

@dataclass
class Progress:
    run_id: str
    task_id: str
    phases: dict[str, Phase] = field(default_factory=dict)
    tty: bool = True
    json_log: bool = True

    def set_total(self, phase: str, total: int, extra: dict | None = None):
        self.phases.setdefault(phase, Phase(phase)).total = total
        self.emit(phase, "phase.total", extra)

    def inc_done(self, phase: str, n: int = 1, extra: dict | None = None):
        p = self.phases.setdefault(phase, Phase(phase))
        p.done += n
        self.emit(phase, "phase.progress", extra)

    def emit(self, phase: str, event: str, extra: dict | None = None):
        p = self.phases.get(phase, Phase(phase))
        percent = (p.done / p.total * 100) if p.total else 0.0
        payload = {
            "ts": _now_iso(),
            "level": "INFO",
            "event": event,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "phase": phase,
            "done": p.done,
            "total": p.total,
            "percent": round(percent, 1),
        }
        if extra:
            payload.update(extra)
        # JSON line
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        # TTY one-line bar
        if self.tty:
            bar = f"[{phase}] {p.done}/{p.total} {percent:5.1f}%"
            sys.stderr.write("\r" + bar)
            sys.stderr.flush()

    def end_phase(self, phase: str, extra: dict | None = None):
        self.emit(phase, "phase.end", extra)
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()
