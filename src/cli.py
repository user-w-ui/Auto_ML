"""
CLI entry for job management (作业管理).

Commands (命令总览):
- run:
    创建一次新运行，生成 run_id、目录结构、config snapshot，并把状态置为 running。
    Example:
    python -m src.cli run --config configs/example.yaml

- status:
    查询某个 run_id 的 status.json。
    Example:
    python -m src.cli status --run-id 20260322_135226_01fee3ff

- list:
    列出 runs/ 下所有历史运行（按名称倒序，通常也是时间倒序）。
    Example:
    python -m src.cli list

Quick start:
1) python -m src.cli run --config configs/example.yaml
2) python -m src.cli list
3) python -m src.cli status --run-id <your_run_id>
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer

from src.config.load import load_config
from src.jobs.artifacts import RunLayout
from src.jobs.status import (
    load_status,
    mark_status_failed,
    reconcile_running_status,
    set_status_running,
)
from src.workflow.subprocess import spawn_workflow_process

# 注册 CLI 命令
app = typer.Typer(add_completion=False, help="Auto_ML job runner (AG2 pipeline)")

RUNS_DIR = Path("runs")


# 为每次执行生成一个唯一的 run_id
def make_run_id() -> str:
    """run_id = timestamp + short uuid ."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:8]
    return f"{ts}_{short}"


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional: provide your own run_id"),
    foreground: bool = typer.Option(False, "--foreground", help="Run workflow in foreground and stream logs."),
):
    """
    Create a new run folder, snapshot config, set status=running.

    这一步是 Job Management 的 MVP：
    - 不关心你跑的是什么任务（workflow 还没接入）
    - 先把“运行容器（run folder）”和“可复现记录（config snapshot）”建立起来
    """
    # 1) 读取配置
    cfg = load_config(config)

    # 2) 创建 run 目录
    RUNS_DIR.mkdir(exist_ok=True)
    rid = run_id or make_run_id()

    # 3) 构建目录结构
    layout = RunLayout.from_run_id(rid)
    layout.ensure_dirs()

    # 4) 保存 config snapshot（可复现记录）
    (layout.run_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if foreground:
        # 5) 前台运行：容器/终端内可直接看到实时输出
        set_status_running(layout, pid=None)
        typer.echo(f"Submitted run: {rid}")
        typer.echo("Mode: foreground")
        typer.echo(f"Run dir: {layout.run_dir.resolve()}")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.workflow.runner",
            "--run-id",
            rid,
            "--config",
            str(config),
        ]
        rc = subprocess.call(cmd)
        # 如果 runner 进程异常退出且未能写回最终状态，这里做一次兜底收口。
        if rc != 0:
            s_after = load_status(layout)
            if s_after.status == "running":
                mark_status_failed(layout, f"Foreground runner exited with code {rc} before final status update.")
        s = load_status(layout)
        typer.echo(f"Final status: {s.status}")
        if s.error:
            typer.echo(f"Error: {s.error}")
        raise typer.Exit(code=rc)
    else:
        # 5) 异步运行：CLI 立即返回
        proc = spawn_workflow_process(rid, config)
        set_status_running(layout, pid=proc.pid)

        typer.echo(f"Submitted run: {rid}")
        typer.echo(f"PID: {proc.pid}")
        typer.echo(f"Run dir: {layout.run_dir.resolve()}")
        typer.echo("Tip: check progress via:")
        typer.echo(f"  python -m src.cli status --run-id {rid}")


@app.command()
def status(
    run_id: str = typer.Option(..., "--run-id", "-r"),
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Auto-reconcile stale running status."),
    stale_after: int = typer.Option(900, "--stale-after", help="Heartbeat timeout seconds for stale running runs."),
):
    layout = RunLayout.from_run_id(run_id)
    if refresh:
        s = reconcile_running_status(layout, stale_after_seconds=stale_after)
    else:
        s = load_status(layout)
    typer.echo(json.dumps(s.model_dump(), ensure_ascii=False, indent=2))


@app.command("list")
def list_runs():
    if not RUNS_DIR.exists():
        typer.echo("No runs/ directory yet.")
        raise typer.Exit(code=0)

    run_dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name, reverse=True)

    if not run_dirs:
        typer.echo("No runs found.")
        raise typer.Exit(code=0)

    for p in run_dirs:
        typer.echo(p.name)


if __name__ == "__main__":
    app()
