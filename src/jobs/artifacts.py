"""
Artifacts layout.

把“run 产物结构”抽象出来的价值：
- 后续 workflow 不用到处拼路径，统一通过 layout 获取路径
- Web UI/下载 artifacts 时也更容易
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 把一次 run 需要的关键目录集中封装成一个对象
@dataclass(frozen=True)
class RunLayout:
    run_id: str
    run_dir: Path
    logs_dir: Path

    @staticmethod
    def from_run_id(run_id: str) -> "RunLayout":
        run_dir = Path("runs") / run_id
        return RunLayout(
            run_id=run_id,
            run_dir=run_dir,
            logs_dir=run_dir / "logs",
        )

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)