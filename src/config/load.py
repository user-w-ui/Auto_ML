"""
Config loader.

- 配置文件可版本控制（configs/example.yaml）
- run 时保存 snapshot（run_config.json） => 可复现
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    return data