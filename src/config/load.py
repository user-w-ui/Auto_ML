"""
Config loader.

- 配置文件可版本控制（configs/example.yaml）
- run 时保存 snapshot（run_config.json） => 可复现
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_in_string(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} expressions in config strings."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        resolved = os.environ.get(key)
        if resolved is not None:
            return resolved
        if default is not None:
            return default
        raise ValueError(f"Missing required environment variable in config: {key}")

    return _ENV_PATTERN.sub(_replace, value)


def _expand_env(obj: Any) -> Any:
    """Recursively expand environment placeholders in YAML-loaded objects."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_in_string(obj)
    return obj


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping (dict).")
    expanded = _expand_env(data)
    if not isinstance(expanded, dict):
        raise ValueError("Expanded config must remain a YAML mapping (dict).")
    return expanded