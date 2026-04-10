import json
import re


def _extract_first_json_obj(text: str):
    """从文本中提取并解析第一个 JSON 对象（仅接受 dict）。"""
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


def is_ready_for_train(groupchat, client) -> dict:
    """让模型判断数据是否已满足训练条件，返回结构化判定结果。"""
    messages = [
        {
            "role": "system",
            "content": """Based on the dataset exploration and preprocessing, determine whether the data is ready for model training.

Return ONLY one JSON object and nothing else:
{"summary":"<short summary>","decision":"ready_for_training|need_more_processing"}
""",
        }
    ] + groupchat.messages

    try:
        response = client.create(messages=messages, response_format={"type": "json_object"})
    except TypeError:
        # Some OpenAI-compatible wrappers/backends may not support response_format.
        response = client.create(messages=messages)
    response_str = client.extract_text_or_completion_object(response)[0]

    print("-" * 50)
    print(response_str)
    print("-" * 50)

    parsed = _extract_first_json_obj(response_str)
    if not parsed:
        return {
            "ready": False,
            "summary": "Model response is not valid JSON.",
        }

    decision_raw = str(parsed.get("decision", "")).strip().lower()
    summary = str(parsed.get("summary", "")).strip()

    if decision_raw in {"ready_for_training", "ready for training"}:
        return {
            "ready": True,
            "summary": summary,
        }

    return {
        "ready": False,
        "summary": summary,
    }


def decide_train_next_action(groupchat, client) -> dict:
    """Ask model to decide next train action from latest training evidence."""
    messages = [
        {
            "role": "system",
            "content": """You are evaluating model-training progress.

Decide one action based on recent improvement and metric stability:
- continue_next_candidate: move to next model candidate in queue
- retune_same_candidate: keep current model and tune hyperparameters again
- finish_training: stop training loop and finalize

Return ONLY one JSON object and nothing else:
{"summary":"<short summary>","decision":"continue_next_candidate|retune_same_candidate|finish_training"}
""",
        }
    ] + groupchat.messages

    try:
        response = client.create(messages=messages, response_format={"type": "json_object"})
    except TypeError:
        response = client.create(messages=messages)
    response_str = client.extract_text_or_completion_object(response)[0]

    parsed = _extract_first_json_obj(response_str)
    if not parsed:
        return {
            "decision": "continue_next_candidate",
            "summary": "Model response is not valid JSON.",
        }

    decision_raw = str(parsed.get("decision", "")).strip().lower()
    summary = str(parsed.get("summary", "")).strip()
    if decision_raw not in {"continue_next_candidate", "retune_same_candidate", "finish_training"}:
        decision_raw = "continue_next_candidate"

    return {
        "decision": decision_raw,
        "summary": summary,
    }


def did_code_execution_fail(executor_output: str) -> bool:
    """基于执行输出做尽力失败检测，判断代码执行是否失败。"""
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
    """分析执行输出并返回结构化诊断结果（失败、错误类型、错误信息等）。"""
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
    """统计有效训练轮次：训练发言计数，紧随其后的失败执行会回退一次计数。"""
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
