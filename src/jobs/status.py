from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

from pydantic import BaseModel

from src.jobs.artifacts import RunLayout


StatusValue = Literal["running", "succeeded", "failed"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(BaseModel):
    run_id: str
    status: StatusValue

    created_at: str
    updated_at: str

    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    pid: Optional[int] = None

    error: Optional[str] = None


def status_path(layout: RunLayout) -> Path:
    return layout.run_dir / "status.json"


def save_status(layout: RunLayout, s: RunStatus) -> None:
    s.updated_at = utc_now_iso()
    status_path(layout).write_text(json.dumps(s.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_status(layout: RunLayout) -> RunStatus:
    p = status_path(layout)
    if not p.exists():
        raise FileNotFoundError(f"status.json not found for run_id={layout.run_id}")
    return RunStatus.model_validate_json(p.read_text(encoding="utf-8"))


def set_status_running(layout: RunLayout, pid: Optional[int] = None) -> None:
    now = utc_now_iso()
    s = RunStatus(
        run_id=layout.run_id,
        status="running",
        created_at=now,
        updated_at=now,
        started_at=now,
        last_heartbeat_at=now,
        pid=pid,
    )
    save_status(layout, s)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def touch_heartbeat(layout: RunLayout) -> None:
    s = load_status(layout)
    if s.status != "running":
        return
    s.last_heartbeat_at = utc_now_iso()
    save_status(layout, s)


def mark_status_failed(layout: RunLayout, error: str) -> RunStatus:
    s = load_status(layout)
    s.status = "failed"
    s.error = error
    s.finished_at = utc_now_iso()
    save_status(layout, s)
    return s


def reconcile_running_status(layout: RunLayout, stale_after_seconds: int = 900) -> RunStatus:
    s = load_status(layout)
    if s.status != "running":
        return s

    if s.pid is not None and not is_pid_alive(s.pid):
        return mark_status_failed(layout, f"Worker process is not alive (pid={s.pid}).")

    now = datetime.now(timezone.utc)
    heartbeat_dt = _parse_iso(s.last_heartbeat_at) or _parse_iso(s.updated_at) or _parse_iso(s.started_at)
    if heartbeat_dt is not None:
        age_seconds = (now - heartbeat_dt).total_seconds()
        if age_seconds > stale_after_seconds:
            return mark_status_failed(
                layout,
                f"Run heartbeat stale for {int(age_seconds)}s (> {stale_after_seconds}s).",
            )

    return s