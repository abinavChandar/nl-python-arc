from __future__ import annotations
import os
import sys
import json
import time
import uuid
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict

# ---------- config ----------
LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TTY_PROGRESS = os.getenv("SHOW_TTY_PROGRESS", "0") == "1"   # draw single-line ticker on stdout
LIVE_PROGRESS = os.getenv("LIVE_PROGRESS", "1") == "1"      # emit heartbeat once/sec (default on)
MAX_TTY_WIDTH = int(os.getenv("TTY_WIDTH", "160"))

# ---------- shared context ----------
_CTX: Dict[str, Any] = {}

def set_ctx(**fields: Any) -> None:
    _CTX.update({k: v for k, v in fields.items() if v is not None})

def clear_ctx(*keys: str) -> None:
    for k in keys:
        _CTX.pop(k, None)

# ---------- emitters ----------
def _emit(level: str, event: str, **kw: Any) -> None:
    msg = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
       "event": event,
        **_CTX,
        **kw,
    }
    print(json.dumps(msg, ensure_ascii=False), file=sys.stderr, flush=True)

def info(event: str, **kw: Any) -> None:
    if LEVEL in ("INFO", "DEBUG"):
        _emit("INFO", event, **kw)

def debug(event: str, **kw: Any) -> None:
    if LEVEL == "DEBUG":
        _emit("DEBUG", event, **kw)

def warn(event: str, **kw: Any) -> None:
    _emit("WARN", event, **kw)

def error(event: str, **kw: Any) -> None:
    _emit("ERROR", event, **kw)

# ---------- live progress state & ticker ----------
_LIVE_LOCK = threading.Lock()
_LIVE_STATE: Dict[str, Any] = {
    "label": None,    # e.g., "evo:follow"
    "current": 0,
    "total": 0,
    "note": "",
    "active": False,
}
_LIVE_THREAD: threading.Thread | None = None
_LIVE_STOP = threading.Event()

def _pct(current: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * current / total, 2)

def _tty_line(label: str, current: int, total: int, extra: str = "") -> None:
    pct = _pct(current, total)
    s = f"\r[{label}] {current}/{total}  {pct:5.1f}% {extra}".rstrip()
    s = s[:MAX_TTY_WIDTH]
    try:
        sys.stdout.write(s)
        sys.stdout.flush()
    except Exception:
        pass

def _live_loop():
    last_snapshot = None
    while not _LIVE_STOP.is_set():
        with _LIVE_LOCK:
            active = _LIVE_STATE["active"]
            snap = dict(_LIVE_STATE)
        if active and snap["label"]:
            if TTY_PROGRESS:
                _tty_line(snap["label"], snap["current"], snap["total"], extra=snap["note"])
            # emit a JSON heartbeat (once/sec) so logs show % even if TTY is buffered
            if LIVE_PROGRESS:
                if last_snapshot != (snap["label"], snap["current"], snap["total"], snap["note"]):
                    info("progress.live",
                         label=snap["label"],
                         current=snap["current"],
                         total=snap["total"],
                         percent=_pct(snap["current"], snap["total"]),
                         note=snap["note"])
                    last_snapshot = (snap["label"], snap["current"], snap["total"], snap["note"])
        time.sleep(1.0)

def start_live_progress() -> None:
    if not LIVE_PROGRESS and not TTY_PROGRESS:
        return
    global _LIVE_THREAD
    if _LIVE_THREAD and _LIVE_THREAD.is_alive():
        return
    _LIVE_STOP.clear()
    _LIVE_THREAD = threading.Thread(target=_live_loop, daemon=True)
    _LIVE_THREAD.start()

def stop_live_progress() -> None:
    _LIVE_STOP.set()
    if TTY_PROGRESS:
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass

def progress(event: str, *, current: int, total: int, label: str | None = None, **kw: Any) -> None:
    """Emit one progress event AND update the live ticker state."""
    percent = _pct(current, total)
    info(event, current=current, total=total, percent=percent, **kw)
    if label:
        with _LIVE_LOCK:
            _LIVE_STATE["label"] = label
            _LIVE_STATE["current"] = current
            _LIVE_STATE["total"] = total
            _LIVE_STATE["note"] = kw.get("note", "")
            _LIVE_STATE["active"] = True
        if TTY_PROGRESS:
            _tty_line(label, current, total, extra=kw.get("note", ""))

def progress_end(label: str | None = None) -> None:
    with _LIVE_LOCK:
        lab = label or _LIVE_STATE.get("label") or "phase"
        cur = int(_LIVE_STATE.get("current", 1))
        tot = int(_LIVE_STATE.get("total", 1))
        _LIVE_STATE["active"] = False
    if TTY_PROGRESS:
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass
    info("progress.end", label=lab, current=cur, total=tot, percent=_pct(cur, tot))

# ---------- span helper ----------
@contextmanager
def span(event: str, **kw: Any):
    sid = str(uuid.uuid4())[:8]
    info(f"{event}.begin", span_id=sid, **kw)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        info(f"{event}.end", span_id=sid, duration_ms=dur_ms, **kw)
