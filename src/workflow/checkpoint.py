"""
Checkpoint (state.json): recoverable workflow foundation.

能力点：
- 可恢复（recoverable/resumable）：程序挂了/你 Ctrl+C 后能从 checkpoint 继续
- 状态机（state machine）：把“当前在哪一步”变成可持久化的数据

文件：runs/<run_id>/state.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


StateName = Literal["Init", "Explore", "Preprocess", "Train", "Summarize", "End"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkflowCheckpoint(BaseModel):
    run_id: str
    current_state: StateName = "Init"
    # Train 状态用于控制最多训练几次（你现在 main.py 是 2 次）
    train_trials_done: int = 0

    # attempts 用于防止卡死：比如 Explore 连续失败 20 次就应该终止
    attempts: Dict[StateName, int] = Field(default_factory=dict)

    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    def bump_attempt(self, state: StateName) -> None:
        self.attempts[state] = int(self.attempts.get(state, 0)) + 1
        self.updated_at = utc_now_iso()


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def load_or_init_checkpoint(run_id: str, run_dir: Path) -> WorkflowCheckpoint:
    p = checkpoint_path(run_dir)
    if p.exists():
        return WorkflowCheckpoint.model_validate_json(p.read_text(encoding="utf-8"))
    ckpt = WorkflowCheckpoint(run_id=run_id)
    save_checkpoint(run_dir, ckpt)
    return ckpt


def save_checkpoint(run_dir: Path, ckpt: WorkflowCheckpoint) -> None:
    p = checkpoint_path(run_dir)
    ckpt.updated_at = utc_now_iso()
    p.write_text(json.dumps(ckpt.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")