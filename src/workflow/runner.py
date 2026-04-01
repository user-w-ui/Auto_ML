from __future__ import annotations

import argparse
import threading
from pathlib import Path

from dotenv import load_dotenv

from src.config.load import load_config
from src.jobs.artifacts import RunLayout
from src.jobs.events import EventLogger
from src.jobs.status import load_status, save_status, touch_heartbeat, utc_now_iso
from src.workflow.app import run_workflow


def _start_heartbeat(layout: RunLayout, events: EventLogger, interval_seconds: float = 20.0):
    stop_event = threading.Event()

    def _beat() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                touch_heartbeat(layout)
                events.emit("run_heartbeat", run_id=layout.run_id)
            except Exception:
                # Heartbeat failure should not crash the workflow process.
                pass

    t = threading.Thread(target=_beat, name=f"heartbeat-{layout.run_id}", daemon=True)
    t.start()
    return stop_event


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

    s = load_status(layout)
    heartbeat_stop = _start_heartbeat(layout, events)

    try:
        events.emit("run_started", run_id=run_id)

        summary = run_workflow(cfg, run_dir=layout.run_dir)

        events.emit("run_succeeded", run_id=run_id, summary=summary)
        s.status = "succeeded"
        s.error = None
        s.finished_at = utc_now_iso()
        save_status(layout, s)
        return 0

    except Exception as e:
        events.emit("run_failed", run_id=run_id, error=str(e))
        s.status = "failed"
        s.error = str(e)
        s.finished_at = utc_now_iso()
        save_status(layout, s)
        return 1
    finally:
        heartbeat_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())