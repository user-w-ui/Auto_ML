from __future__ import annotations

import re


def extract_python_code_blocks(content: str) -> str:
    """Extract and merge all ```python``` code blocks from markdown text."""
    text = content or ""
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        return ""
    return "\n\n".join(b.strip() for b in blocks if b.strip())


def sanitize_python_code(code: str) -> str:
    """Remove risky or unsupported notebook patterns before execution."""
    lines = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if re.search(r"^exec\s*\(\s*open\s*\(.*\)\s*\.read\s*\(\s*\)\s*\)\s*$", s):
            continue
        if re.search(r"^\s*%run\s+", s):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def state_script_name(step_name: str) -> str:
    """Map workflow step/state names to stable script file names."""
    mapping = {
        "Explore": "exploration.py",
        "Preprocess": "preprocess.py",
        "Train": "train.py",
        "Summarize": "summary.py",
        "EXPLORE": "exploration.py",
        "PREPROCESS": "preprocess.py",
        "TRAIN": "train.py",
        "EVALUATE": "evaluate.py",
        "REPLAN": "replan.py",
        "HUMAN_REVIEW": "human_review.md",
        "DONE": "summary.py",
        "FAILED": "failure_report.md",
    }
    return mapping.get(step_name, f"{step_name.lower()}.py")
