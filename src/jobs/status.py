from __future__ import annotations

import json
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
        pid=pid,
    )
    save_status(layout, s)