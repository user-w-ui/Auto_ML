from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from src.config.load import load_config
from src.workflow.app import run_workflow


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))

    cfg_path = Path("configs/example.yaml")
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    # manual run output (kept separate from async runs/)
    run_dir = Path("runs") / "manual_run"
    run_workflow(cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()