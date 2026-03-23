from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config.load import load_config
from src.jobs.artifacts import RunLayout
from src.jobs.events import EventLogger
from src.jobs.status import load_status, save_status
from src.workflow.checkpoint import load_or_init_checkpoint, save_checkpoint
from src.workflow.app import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_id = args.run_id
    config_path = Path(args.config)

    layout = RunLayout.from_run_id(run_id)
    layout.ensure_dirs()

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    cfg = load_config(config_path)
    events = EventLogger(layout.logs_dir / "events.jsonl")

    ckpt = load_or_init_checkpoint(run_id, layout.run_dir)
    s = load_status(layout)

    try:
        events.emit("run_started", run_id=run_id, resume=True, current_state=ckpt.current_state)

        summary = run_workflow(cfg, run_dir=layout.run_dir, resume=True)

        events.emit("run_succeeded", run_id=run_id, summary=summary)
        s.status = "succeeded"
        s.error = None
        save_status(layout, s)
        return 0

    except Exception as e:
        latest_ckpt = load_or_init_checkpoint(run_id, layout.run_dir)
        events.emit("run_failed", run_id=run_id, error=str(e), current_state=latest_ckpt.current_state)
        s.status = "failed"
        s.error = str(e)
        save_status(layout, s)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())