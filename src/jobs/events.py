"""
Structured events (JSONL).

能力点：
- 可观测性（observability）：每一步发生了什么、什么时候、在哪个 state
- 为 resume/调试提供证据：你可以回放 run 的历史

文件：runs/<run_id>/logs/events.jsonl
每行都是一个 JSON object，天然适合后续接 ELK/Datadog（面试加分点）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EventLogger:
    path: Path

    def emit(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"ts": utc_now_iso(), "event": event, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")