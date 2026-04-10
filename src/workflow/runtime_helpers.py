from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.rag.run_memory import RunMemory


def ensure_windows_scripts_on_path() -> None:
    """在 Windows 下将当前 Python 的 Scripts 目录提前加入 PATH。"""
    if os.name != "nt":
        return

    scripts_dir = Path(sys.executable).parent / "Scripts"
    if scripts_dir.exists():
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{current_path}"


def build_llm_config_from_env() -> Dict[str, Any]:
    """从环境变量构建 LLM 配置，并校验必要字段。"""
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if not deepseek_api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY. Please set it in .env or environment variables.")

    return {
        "api_type": "openai",
        "model": deepseek_model,
        "api_key": deepseek_api_key,
        "base_url": deepseek_base_url,
    }


def memory_context(mem: RunMemory) -> str:
    """构建用于提示词注入的去重记忆摘要。"""
    return mem.build_prompt_context(
        current_state=None,
        max_failures=3,
        max_successes=2,
        max_decisions=2,
    )


def progress(message: str) -> None:
    """统一工作流日志输出格式，便于终端观察运行过程。"""
    print(f"[workflow] {message}", flush=True)


def organize_generated_files(run_dir: Path, plot_dir: Path, data_dir: Path) -> None:
    """按扩展名把产物归档到 plot/data 子目录。"""
    image_exts = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".pdf"}
    data_exts = {".csv", ".parquet", ".json", ".xlsx", ".pkl", ".joblib"}
    reserved_dirs = {"plot", "data", "coding", "logs"}
    reserved_files = {"status.json", "run_config.json", "run_memory.jsonl"}

    for p in run_dir.iterdir():
        if p.is_dir() or p.name in reserved_files:
            continue
        if p.parent == run_dir and p.name in reserved_dirs:
            continue

        ext = p.suffix.lower()
        dest = None
        if ext in image_exts:
            dest = plot_dir / p.name
        elif ext in data_exts:
            dest = data_dir / p.name

        if dest is None:
            continue

        if dest.exists():
            # 避免重名覆盖：自动追加递增后缀。
            stem = p.stem
            suffix = p.suffix
            idx = 1
            while True:
                candidate = dest.with_name(f"{stem}_{idx}{suffix}")
                if not candidate.exists():
                    dest = candidate
                    break
                idx += 1

        p.replace(dest)


def load_explorer_profile(profile_path: Path) -> Optional[Dict[str, Any]]:
    """Load and validate explorer_dataset_profile.json contract."""
    if not profile_path.exists():
        return None
    try:
        obj = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    required_keys = {
        "task_type",
        "target_type",
        "sample_size_bucket",
        "feature_type_profile",
        "signal_hypothesis",
        "compute_budget",
    }
    if any(k not in obj for k in required_keys):
        return None

    task_type_value = str(obj.get("task_type", "")).strip().lower()
    if task_type_value not in {"supervised", "unsupervised"}:
        return None

    # Keep profile source-pure; retrieval query is derived later by trainer.
    if "recommended_labels" in obj:
        return None

    return obj
