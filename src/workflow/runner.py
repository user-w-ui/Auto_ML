"""
Workflow runner process (background job).

能力点：
- run lifecycle：running -> succeeded/failed
- structured logs: events.jsonl
- checkpoint 更新：current_state（目前先粗粒度）

实现策略（先稳后优雅）：
- 第一版先用 subprocess 调用现有 main.py
- 这样你不用立刻重构 main.py
- 下一版再把 main.py 拆成函数，做到 state 粒度的 checkpoint/resume
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.config.load import load_config
from src.jobs.artifacts import RunLayout
from src.jobs.events import EventLogger
from src.jobs.status import (
    load_status,
    save_status,
    RunStatus,
)
from src.workflow.checkpoint import load_or_init_checkpoint, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_id = args.run_id
    config_path = Path(args.config)

    layout = RunLayout.from_run_id(run_id)
    layout.ensure_dirs()

    events = EventLogger(layout.logs_dir / "events.jsonl")
    cfg = load_config(config_path)

    # init checkpoint
    ckpt = load_or_init_checkpoint(run_id, layout.run_dir)
    events.emit("run_started", run_id=run_id)

    # update status with started_at
    s = load_status(layout)
    s.updated_at = s.updated_at  # keep
    # add started_at/finished_at/pid later in status model if you want
    save_status(layout, s)

    # 我们先把 checkpoint 标到 Explore（粗粒度）
    ckpt.current_state = "Explore"
    save_checkpoint(layout.run_dir, ckpt)
    events.emit("state_entered", run_id=run_id, state=ckpt.current_state)

    # Run the existing workflow script (main.py) as-is.
    # 这样最少改动就能 async 化。
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(Path(__file__).resolve().parents[2]),  # repo root
            check=True,
        )
    except subprocess.CalledProcessError as e:
        events.emit("run_failed", run_id=run_id, error=f"workflow exited with {e.returncode}")
        s.status = "failed"
        s.updated_at = s.updated_at
        s.error = f"workflow exited with {e.returncode}"
        save_status(layout, s)
        ckpt.current_state = "End"
        save_checkpoint(layout.run_dir, ckpt)
        return 1

    # succeeded
    events.emit("run_succeeded", run_id=run_id)
    s.status = "succeeded"
    s.error = None
    save_status(layout, s)
    ckpt.current_state = "End"
    save_checkpoint(layout.run_dir, ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())