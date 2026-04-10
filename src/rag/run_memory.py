from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunMemory:
    """
    运行期短期记忆（JSONL 形式）。

    选择 JSONL 的原因：
    - 追加写入，天然适合事件流
    - 人类可读，便于排障
    - 易于做后续聚合/压缩
    """
    path: Path

    def append(self, record: Dict[str, Any]) -> None:
        """追加一条事件记录，并自动补上时间戳。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": utc_now_iso(), **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_records(self) -> List[Dict[str, Any]]:
        """读取并解析全部 JSONL 记录，忽略空行和坏行。"""
        if not self.path.exists():
            return []

        records: List[Dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                records.append(data)
        return records

    @staticmethod
    def _normalize_text(value: Any, max_len: int = 200) -> str:
        """规整文本：压缩空白并限制最大长度，防止 prompt 过长。"""
        text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_len:
            return text[: max_len - 3] + "..."
        return text

    def build_prompt_context(
        self,
        current_state: Optional[str] = None,
        max_failures: int = 3,
        max_successes: int = 2,
        max_decisions: int = 2,
    ) -> str:
        """
        构建注入给 Agent 的记忆摘要文本。

        逻辑：
        - 从事件流中提取失败模式/成功模式/决策记录
        - 对失败与成功做去重聚合，并统计出现次数
        - 按当前状态优先 + 频次优先输出有限条目
        """
        records = self._load_records()
        if not records:
            return ""

        failures: Dict[str, Dict[str, Any]] = {}
        successes: Dict[str, Dict[str, Any]] = {}
        decisions: List[Dict[str, Any]] = []

        for record in records:
            rtype = str(record.get("type", ""))
            state = str(record.get("state", "UNKNOWN"))
            ts = str(record.get("ts", ""))

            # 失败模式：按 state + error_type + error_message 去重。
            if rtype == "state_exit" and record.get("ok") is False:
                failure = record.get("failure", {})
                if not isinstance(failure, dict):
                    failure = {}

                err_type = self._normalize_text(failure.get("error_type") or "RuntimeError", max_len=80)
                err_msg = self._normalize_text(failure.get("error_message") or "Unknown error", max_len=220)
                signature = f"{state}|{err_type}|{err_msg.lower()}"

                item = failures.get(signature)
                if item is None:
                    failures[signature] = {
                        "state": state,
                        "error_type": err_type,
                        "error_message": err_msg,
                        "count": 1,
                        "last_ts": ts,
                    }
                else:
                    item["count"] += 1
                    if ts >= str(item.get("last_ts", "")):
                        item["last_ts"] = ts

            # 成功模式：按 state 去重并累积次数。
            if rtype == "state_exit" and record.get("ok") is True:
                signature = state
                item = successes.get(signature)
                if item is None:
                    successes[signature] = {
                        "state": state,
                        "count": 1,
                        "last_ts": ts,
                    }
                else:
                    item["count"] += 1
                    if ts >= str(item.get("last_ts", "")):
                        item["last_ts"] = ts

            if rtype == "decision":
                decisions.append(record)

        def _state_rank(state: str) -> int:
            if not current_state:
                return 0
            return 0 if state == current_state else 1

        sorted_failures = sorted(
            failures.values(),
            key=lambda x: (_state_rank(str(x.get("state", ""))), -int(x.get("count", 0)), str(x.get("last_ts", ""))),
        )[: max(0, max_failures)]

        sorted_successes = sorted(
            successes.values(),
            key=lambda x: (_state_rank(str(x.get("state", ""))), -int(x.get("count", 0)), str(x.get("last_ts", ""))),
        )[: max(0, max_successes)]

        sorted_decisions = sorted(decisions, key=lambda x: str(x.get("ts", "")), reverse=True)[: max(0, max_decisions)]

        lines: List[str] = [
            "Recent Run Memory (deduped patterns):",
            "Use these patterns to avoid repeating known failures and to reuse successful steps.",
        ]

        if sorted_failures:
            lines.append("Failure Patterns:")
            for item in sorted_failures:
                lines.append(
                    "- "
                    + f"[{item['state']}] {item['error_type']}: {item['error_message']} "
                    + f"(seen {item['count']}x, last={item['last_ts']})"
                )

        if sorted_successes:
            lines.append("Success Patterns:")
            for item in sorted_successes:
                lines.append(
                    "- "
                    + f"[{item['state']}] completed successfully "
                    + f"(seen {item['count']}x, last={item['last_ts']})"
                )

        if sorted_decisions:
            lines.append("Recent Decisions:")
            for item in sorted_decisions:
                action = self._normalize_text(item.get("action", ""), max_len=120)
                target = self._normalize_text(item.get("target", ""), max_len=60)
                summary = self._normalize_text(item.get("summary", ""), max_len=160)
                lines.append(
                    "- "
                    + f"state={item.get('state', 'UNKNOWN')}"
                    + (f", target={target}" if target else "")
                    + (f", action={action}" if action else "")
                    + (f", summary={summary}" if summary else "")
                )

        if len(lines) <= 2:
            return ""
        return "\n".join(lines) + "\n"