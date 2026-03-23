from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from src.config.load import load_config
from src.workflow.app import run_workflow

load_dotenv(Path(__file__).with_name(".env"))

if __name__ == "__main__":
    # 兼容：没有 config 就跑默认
    cfg_path = Path("configs/example.yaml")
    cfg = load_config(cfg_path) if cfg_path.exists() else {}
    run_workflow(cfg, run_dir=Path("runs") / "manual_run")