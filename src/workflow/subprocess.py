"""
Spawn background process (async job runner).

能力点：
- 作业管理（job management）：run 是一个独立进程
- Windows 兼容：使用 CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS

我们采用 “python -m src.workflow.runner --run-id ... --config ...”
这样不会依赖当前 working dir 的 import 细节（更稳）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def spawn_workflow_process(run_id: str, config_path: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "src.workflow.runner",
        "--run-id",
        run_id,
        "--config",
        str(config_path),
    ]

    kwargs = {}

    if os.name == "nt":
        # Detach child process from the current console so CLI can return immediately.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    # Redirect output to DEVNULL; we rely on structured events logs instead.
    # 你也可以改成写入 runs/<run_id>/logs/stdout.log
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=(os.name != "nt"),
        **kwargs,
    )