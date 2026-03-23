from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunMemory:
    """
    Minimal run memory stored as JSONL.

    Why JSONL:
    - append-only (safe)
    - easy to inspect
    - easy to retrieve "last N events"
    """
    path: Path

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": utc_now_iso(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def tail(self, n: int = 20) -> str:
        """
        Return last n lines as a single string for prompt injection.
        (We won't inject yet, but having this helps debug & future optimization.)
        """
        if not self.path.exists():
            return ""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-n:]
        return "\n".join(tail_lines)