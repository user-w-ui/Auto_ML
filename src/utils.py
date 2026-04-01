import json
import re


def _extract_first_json_obj(text: str):
    """Extract and parse the first JSON object found in text."""
    s = (text or "").strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*?\}", s)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def is_ready_for_train(groupchat, client):
    messages = [
        {
            "role": "system",
            "content": """Based on the dataset exploration and preprocessing, determine whether the data is ready for model training.

Return ONLY one JSON object and nothing else:
{"summary":"<short summary>","decision":"ready_for_training|need_more_processing"}
""",
        }
    ] + groupchat.messages

    response = client.create(messages=messages)
    response_str = client.extract_text_or_completion_object(response)[0]

    print("-" * 50)
    print(response_str)
    print("-" * 50)

    parsed = _extract_first_json_obj(response_str)
    if not parsed:
        return False

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision in {"ready_for_training", "ready for training"}:
        return True
    if decision in {"need_more_processing", "need more processing"}:
        return False
    return False


def did_code_execution_fail(executor_output: str) -> bool:
    """Best-effort failure detection for notebook executor outputs."""
    text = str(executor_output or "")
    lowered = text.lower()

    # Prefer explicit exit code if present.
    m = re.search(r"exitcode\s*:\s*(-?\d+)", lowered)
    if m:
        try:
            return int(m.group(1)) != 0
        except Exception:
            pass

    # Common traceback signatures.
    if "traceback (most recent call last):" in lowered:
        return True

    # Typical Python error/exception suffixes shown by notebook outputs.
    if re.search(r"\b[a-z_]*error\s*:", lowered):
        return True
    if re.search(r"\b[a-z_]*exception\s*:", lowered):
        return True

    # Fallback markers often emitted by execution wrappers.
    markers = [
        "execution failed",
        "failed to execute",
        "cell execution error",
        "kernel error",
    ]
    return any(k in lowered for k in markers)


def analyze_code_execution_output(executor_output: str) -> dict:
    """Return structured diagnostics for code execution output."""
    text = str(executor_output or "")
    lowered = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    exit_code = None
    m = re.search(r"exitcode\s*:\s*(-?\d+)", lowered)
    if m:
        try:
            exit_code = int(m.group(1))
        except Exception:
            exit_code = None

    error_line = ""
    for ln in lines:
        if re.search(r"\b[a-z_]*(error|exception)\s*:", ln, flags=re.IGNORECASE):
            error_line = ln
            break

    error_type = "unknown"
    traceback_present = "traceback (most recent call last):" in lowered
    if error_line:
        m_type = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", error_line)
        if m_type:
            error_type = m_type.group(1)
    elif traceback_present:
        error_type = "Traceback"
    elif exit_code is not None and exit_code != 0:
        error_type = f"ExitCode{exit_code}"

    failed = did_code_execution_fail(text)

    if not error_line and failed and lines:
        # Keep memory compact: record only one concise clue line.
        error_line = lines[-1][:300]

    return {
        "failed": bool(failed),
        "exit_code": exit_code,
        "error_type": error_type,
        "error_message": error_line,
        "traceback": traceback_present,
    }


def count_train_trials(groupchat):
    messages = groupchat.messages

    tcount = 0
    for i, message in enumerate(messages):
        name = message.get("name")
        if name == "Model_Trainer":
            tcount += 1
        elif (
            i > 0
            and name == "Code_Executor"
            and did_code_execution_fail(message.get("content", ""))
            and messages[i - 1].get("name") == "Model_Trainer"
        ):
            tcount -= 1

    return tcount
