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

    # load .env once in the worker process
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    events = EventLogger(layout.logs_dir / "events.jsonl")
    cfg = load_config(config_path)

    ckpt = load_or_init_checkpoint(run_id, layout.run_dir)
    events.emit("run_started", run_id=run_id)

    s = load_status(layout)

    try:
        ckpt.current_state = "Explore"
        save_checkpoint(layout.run_dir, ckpt)
        events.emit("state_entered", run_id=run_id, state="Explore")

        summary = run_workflow(cfg, run_dir=layout.run_dir)

        events.emit("run_succeeded", run_id=run_id, summary=summary)
        s.status = "succeeded"
        s.error = None
        save_status(layout, s)

        ckpt.current_state = "End"
        save_checkpoint(layout.run_dir, ckpt)
        return 0

    except Exception as e:
        events.emit("run_failed", run_id=run_id, error=str(e))
        s.status = "failed"
        s.error = str(e)
        save_status(layout, s)

        ckpt.current_state = "End"
        save_checkpoint(layout.run_dir, ckpt)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())