from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import autogen
from autogen import OpenAIWrapper
from autogen.coding.jupyter import DockerJupyterServer, JupyterCodeExecutor, LocalJupyterServer

from src.utils import (
    analyze_code_execution_output,
    count_train_trials,
    decide_train_next_action,
    did_code_execution_fail,
    is_ready_for_train,
)
from src.agent.definition import create_initializer, create_workflow_agents
from src.agent.tools import create_rag_tool_registry
from src.rag.run_memory import RunMemory
from src.workflow.code_helpers import extract_python_code_blocks, sanitize_python_code, state_script_name
from src.workflow.runtime_helpers import (
    build_llm_config_from_env,
    ensure_windows_scripts_on_path,
    load_explorer_profile,
    memory_context,
    organize_generated_files,
    progress,
)
from src.workflow.prompts import CODE_EXECUTOR_SYSTEM_MESSAGE, build_task_prompt


class WorkflowState(str, Enum):
    EXPLORE = "EXPLORE"
    PREPROCESS = "PREPROCESS"
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    REPLAN = "REPLAN"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    DONE = "DONE"
    FAILED = "FAILED"


DEFAULT_STATE_ATTEMPT_LIMITS: Dict[str, int] = {
    WorkflowState.EXPLORE.value: 4,
    WorkflowState.PREPROCESS.value: 4,
    WorkflowState.TRAIN.value: 8,
    WorkflowState.EVALUATE.value: 8,
    WorkflowState.REPLAN.value: 3,
    WorkflowState.HUMAN_REVIEW.value: 1,
    WorkflowState.DONE.value: 1,
    WorkflowState.FAILED.value: 1,
}


def run_workflow(config: Dict[str, Any], run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """执行 AutoML 多 Agent 主流程：Explore -> Preprocess -> Train -> Summarize。"""
    ensure_windows_scripts_on_path()
    os.environ.setdefault("KG_WS_PING_INTERVAL_SECS", "0")

    run_dir = Path(run_dir or ".").resolve()
    progress(f"run started | run_dir={run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 准备本次运行所需目录结构。
    plot_dir = run_dir / "plot"
    data_dir = run_dir / "data"
    coding_dir = run_dir / "coding"
    logs_dir = run_dir / "logs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    coding_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    explorer_profile_path = data_dir / "explorer_dataset_profile.json"

    mem = RunMemory(run_dir / "run_memory.jsonl")

    # 读取并规范化配置项，保证缺省值可用。
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    workflow_cfg = config.get("workflow", {}) if isinstance(config.get("workflow", {}), dict) else {}
    exec_cfg = config.get("execution", {}) if isinstance(config.get("execution", {}), dict) else {}

    data_path = str(data_cfg.get("path", "./house_prices_train.csv"))
    target = str(data_cfg.get("target", "SalePrice"))
    task_type = str(data_cfg.get("task_type", "regression"))

    train_trials = int(workflow_cfg.get("train_trials", 2))
    max_rounds = int(workflow_cfg.get("max_rounds", 20))

    executor_backend = str(exec_cfg.get("code_executor_backend", os.environ.get("CODE_EXECUTOR_BACKEND", "local-jupyter"))).strip().lower()
    docker_image = exec_cfg.get("docker_jupyter_image") or os.environ.get("DOCKER_JUPYTER_IMAGE") or None

    llm_config = build_llm_config_from_env()
    client = OpenAIWrapper(config_list=[llm_config])

    # 组装历史记忆上下文，并创建可调用的 RAG 工具注册表。
    rag_cfg = config.get("rag", {}) if isinstance(config.get("rag", {}), dict) else {}
    kb_dir = Path(rag_cfg.get("kb_dir", "kb"))
    tool_registry = create_rag_tool_registry(kb_dir=kb_dir)
    mem_ctx = memory_context(mem)

    # 创建初始化器与各阶段 Agent。
    initializer = create_initializer()
    workflow_agents = create_workflow_agents(
        llm_config=llm_config,
        mem_ctx=mem_ctx,
        tool_registry=tool_registry,
    )
    data_explorer = workflow_agents["data_explorer"]
    data_processer = workflow_agents["data_processer"]
    model_trainer = workflow_agents["model_trainer"]
    summarizer = workflow_agents["summarizer"]
    evaluator = workflow_agents["evaluator"]

    # 根据配置选择代码执行后端（本地 Jupyter 或 Docker Jupyter）。
    if executor_backend == "docker-jupyter":
        server = DockerJupyterServer(custom_image_name=docker_image)
    else:
        server = LocalJupyterServer()

    code_executor = autogen.UserProxyAgent(
        name="Code_Executor",
        system_message=CODE_EXECUTOR_SYSTEM_MESSAGE,
        human_input_mode="NEVER",
        code_execution_config={"executor": JupyterCodeExecutor(server, output_dir=coding_dir)},
    )

    task_prompt = build_task_prompt(
        target=target,
        task_type=task_type,
        data_path=data_path,
        plot_dir=str(plot_dir),
        data_dir=str(data_dir),
    )

    exec_seq = 0  # Counter for generated code files
    preprocess_attempts_done = 0
    train_attempts_done = 0
    user_limits_raw = workflow_cfg.get("state_attempt_limits", {})
    user_limits = user_limits_raw if isinstance(user_limits_raw, dict) else {}
    state_attempt_limits: Dict[WorkflowState, int] = {
        s: int(user_limits.get(s.value, DEFAULT_STATE_ATTEMPT_LIMITS[s.value])) for s in WorkflowState
    }
    state_attempt_counts: Dict[WorkflowState, int] = {s: 0 for s in WorkflowState}
    current_state = WorkflowState.EXPLORE
    evaluate_target: Optional[WorkflowState] = None

    def save_step_code(step_name: str, content: str) -> None:
        """从 Agent 消息中提取代码并保存（编号文件 + 状态固定文件）。"""
        nonlocal exec_seq
        code_text = extract_python_code_blocks(content)
        if code_text:
            code_text = sanitize_python_code(code_text)
            if code_text:
                exec_seq += 1
                # 保存编号脚本，便于按时序回放。
                code_path = coding_dir / f"{exec_seq:03d}_{step_name}.py"
                code_path.write_text(code_text + "\n", encoding="utf-8")
                # 保存状态固定脚本，便于快速定位最新版本。
                state_code_path = coding_dir / state_script_name(step_name)
                state_code_path.write_text(code_text + "\n", encoding="utf-8")

    state_to_agent = {
        WorkflowState.EXPLORE: data_explorer,
        WorkflowState.PREPROCESS: data_processer,
        WorkflowState.TRAIN: model_trainer,
        WorkflowState.EVALUATE: evaluator,
        WorkflowState.HUMAN_REVIEW: summarizer,
        WorkflowState.DONE: summarizer,
        WorkflowState.FAILED: summarizer,
    }

    def _record_state_enter(state: WorkflowState, **fields: Any) -> bool:
        """Record state enter and return whether state attempt is within limit."""
        nonlocal current_state
        current_state = state
        state_attempt_counts[state] += 1
        payload = {
            "type": "state_enter",
            "state": state.value,
            "attempt": state_attempt_counts[state],
            "max_attempts": state_attempt_limits[state],
        }
        payload.update(fields)
        mem.append(payload)
        return state_attempt_counts[state] <= state_attempt_limits[state]

    def _schedule_state(state: WorkflowState, groupchat, **fields: Any):
        """Schedule next state; if limit exceeded, hand off to Evaluator."""
        nonlocal evaluate_target, current_state

        within_limit = _record_state_enter(state, **fields)
        if state == WorkflowState.REPLAN and within_limit:
            reason = str(fields.get("reason", "replan_required"))
            summary = str(fields.get("summary", ""))
            groupchat.messages.append(
                {
                    "role": "system",
                    "name": "Workflow_Controller",
                    "content": (
                        "Replan required before next step. "
                        f"Reason: {reason}. Summary: {summary}\n"
                        "Adjust exploration/preprocessing strategy to resolve this issue before training."
                    ),
                }
            )
            progress("state=REPLAN -> EXPLORE")
            return _schedule_state(WorkflowState.EXPLORE, groupchat, via=WorkflowState.REPLAN.value)

        if within_limit:
            return state_to_agent[state]

        mem.append(
            {
                "type": "decision",
                "state": state.value,
                "action": "limit_exceeded",
                "route": WorkflowState.EVALUATE.value,
            }
        )

        if state == WorkflowState.EVALUATE:
            progress("state=EVALUATE exceeded max loops -> HUMAN_REVIEW")
            _record_state_enter(WorkflowState.HUMAN_REVIEW, reason="evaluate_max_loops_exceeded")
            return summarizer

        groupchat.messages.append(
            {
                "role": "system",
                "name": "Workflow_Controller",
                "content": (
                    f"State {state.value} exceeded max loops ({state_attempt_limits[state]}). "
                    "Evaluator must decide next action."
                ),
            }
        )
        evaluate_target = state
        progress(f"state={state.value} exceeded max loops -> EVALUATE")
        ok = _record_state_enter(WorkflowState.EVALUATE, target=state.value, reason="max_loops_exceeded")
        if ok:
            return evaluator
        _record_state_enter(WorkflowState.HUMAN_REVIEW, reason="evaluate_max_loops_exceeded")
        return summarizer

    def _goto_replan(groupchat, reason: str, summary: str):
        return _schedule_state(WorkflowState.REPLAN, groupchat, reason=reason, summary=summary)

    def select_next_speaker(last_speaker, groupchat):
        """基于显式状态机选择下一位发言者。"""
        nonlocal preprocess_attempts_done, train_attempts_done, current_state, evaluate_target
        messages = groupchat.messages

        if last_speaker is initializer:
            progress("state=EXPLORE")
            return _schedule_state(WorkflowState.EXPLORE, groupchat)

        if last_speaker in [data_explorer, data_processer, model_trainer]:
            agent_content = messages[-1].get("content", "") if messages else ""
            if agent_content:
                if last_speaker is data_explorer:
                    save_step_code(WorkflowState.EXPLORE.value, agent_content)
                elif last_speaker is data_processer:
                    save_step_code(WorkflowState.PREPROCESS.value, agent_content)
                elif last_speaker is model_trainer:
                    save_step_code(WorkflowState.TRAIN.value, agent_content)
            return code_executor

        if last_speaker is code_executor:
            if len(messages) < 2:
                return _schedule_state(WorkflowState.EXPLORE, groupchat)

            last_worker = messages[-2].get("name")
            executor_output = messages[-1].get("content", "") or ""

            if last_worker == "Data_Processer":
                preprocess_attempts_done += 1
                mem.append(
                    {
                        "type": "attempt",
                        "state": WorkflowState.PREPROCESS.value,
                        "attempt": preprocess_attempts_done,
                        "max_attempts": state_attempt_limits[WorkflowState.PREPROCESS],
                    }
                )
            elif last_worker == "Model_Trainer":
                train_attempts_done += 1
                mem.append(
                    {
                        "type": "attempt",
                        "state": WorkflowState.TRAIN.value,
                        "attempt": train_attempts_done,
                        "max_attempts": state_attempt_limits[WorkflowState.TRAIN],
                    }
                )

            if did_code_execution_fail(executor_output):
                failure_info = analyze_code_execution_output(executor_output)
                mem.append(
                    {
                        "type": "state_exit",
                        "state": current_state.value,
                        "ok": False,
                        "failure": {
                            "exit_code": failure_info.get("exit_code"),
                            "error_type": failure_info.get("error_type"),
                            "error_message": failure_info.get("error_message"),
                            "traceback": failure_info.get("traceback"),
                        },
                    }
                )

                if last_worker == "Data_Explorer":
                    progress("state=EXPLORE failed -> retry")
                    return _schedule_state(WorkflowState.EXPLORE, groupchat, via="retry")
                if last_worker == "Data_Processer":
                    progress("state=PREPROCESS failed -> retry")
                    return _schedule_state(WorkflowState.PREPROCESS, groupchat, via="retry")
                if last_worker == "Model_Trainer":
                    progress("state=TRAIN failed -> retry")
                    return _schedule_state(WorkflowState.TRAIN, groupchat, via="retry")
                return _schedule_state(WorkflowState.EXPLORE, groupchat)

            if last_worker == "Data_Explorer":
                mem.append({"type": "state_exit", "state": WorkflowState.EXPLORE.value, "ok": True})
                profile = load_explorer_profile(explorer_profile_path)
                if not profile:
                    mem.append(
                        {
                            "type": "decision",
                            "state": WorkflowState.EXPLORE.value,
                            "action": "retry_missing_or_invalid_profile",
                            "profile_path": str(explorer_profile_path),
                        }
                    )
                    groupchat.messages.append(
                        {
                            "role": "system",
                            "name": "Workflow_Controller",
                            "content": (
                                "Explore output contract not satisfied. "
                                f"Write a valid profile JSON at {explorer_profile_path} and retry."
                            ),
                        }
                    )
                    progress("state=EXPLORE contract invalid -> retry")
                    return _schedule_state(WorkflowState.EXPLORE, groupchat, via="contract_retry")
                progress("state=EXPLORE done -> PREPROCESS")
                return _schedule_state(WorkflowState.PREPROCESS, groupchat)

            if last_worker == "Data_Processer":
                mem.append({"type": "state_exit", "state": WorkflowState.PREPROCESS.value, "ok": True})
                evaluate_target = WorkflowState.PREPROCESS
                progress("state=PREPROCESS done -> EVALUATE")
                return _schedule_state(WorkflowState.EVALUATE, groupchat, target=evaluate_target.value)

            if last_worker == "Model_Trainer":
                mem.append({"type": "state_exit", "state": WorkflowState.TRAIN.value, "ok": True})
                evaluate_target = WorkflowState.TRAIN
                progress("state=TRAIN done -> EVALUATE")
                return _schedule_state(WorkflowState.EVALUATE, groupchat, target=evaluate_target.value)

        if last_speaker is evaluator:
            if current_state != WorkflowState.EVALUATE or evaluate_target is None:
                return data_explorer

            if evaluate_target not in {WorkflowState.PREPROCESS, WorkflowState.TRAIN}:
                mem.append(
                    {
                        "type": "decision",
                        "state": WorkflowState.EVALUATE.value,
                        "target": evaluate_target.value,
                        "action": "max_loops_exceeded",
                    }
                )
                return _goto_replan(
                    groupchat,
                    f"{evaluate_target.value.lower()}_max_loops_exceeded",
                    "state loop limit exceeded",
                )

            if evaluate_target == WorkflowState.PREPROCESS:
                readiness = is_ready_for_train(groupchat=groupchat, client=client)
                ready = bool(readiness.get("ready", False))
                summary = str(readiness.get("summary", "")).strip()
                mem.append(
                    {
                        "type": "decision",
                        "state": WorkflowState.EVALUATE.value,
                        "target": WorkflowState.PREPROCESS.value,
                        "ready_for_train": ready,
                        "summary": summary,
                    }
                )
                if ready:
                    progress("state=EVALUATE(PREPROCESS) passed -> TRAIN")
                    return _schedule_state(WorkflowState.TRAIN, groupchat)

                groupchat.messages.append(
                    {
                        "role": "system",
                        "name": "Workflow_Controller",
                        "content": (
                            "Preprocess quality gate not passed. "
                            f"Reason summary: {summary or 'insufficient preprocessing quality'}\n"
                            "Please revise the next exploration/preprocess step to address this reason."
                        ),
                    }
                )
                progress("state=EVALUATE(PREPROCESS) failed -> EXPLORE")
                return _schedule_state(WorkflowState.EXPLORE, groupchat, via=WorkflowState.EVALUATE.value)

            if evaluate_target == WorkflowState.TRAIN:
                trials_done = count_train_trials(groupchat)
                decision_obj = decide_train_next_action(groupchat=groupchat, client=client)
                decision = str(decision_obj.get("decision", "continue_next_candidate")).strip().lower()
                summary = str(decision_obj.get("summary", "")).strip()

                if trials_done >= train_trials:
                    decision = "finish_training"
                    summary = summary or "train_trials limit reached"

                mem.append(
                    {
                        "type": "decision",
                        "state": WorkflowState.EVALUATE.value,
                        "target": WorkflowState.TRAIN.value,
                        "train_trials_done": trials_done,
                        "train_trials_target": train_trials,
                        "train_next_action": decision,
                        "summary": summary,
                    }
                )

                if decision == "finish_training":
                    progress("state=EVALUATE(TRAIN) passed -> DONE")
                    return _schedule_state(WorkflowState.DONE, groupchat)

                if decision == "retune_same_candidate":
                    groupchat.messages.append(
                        {
                            "role": "system",
                            "name": "Workflow_Controller",
                            "content": (
                                "Evaluator decision: retune_same_candidate. "
                                f"Reason: {summary or 'need stability/metric improvement'}"
                            ),
                        }
                    )
                    progress("state=EVALUATE(TRAIN) retune -> TRAIN")
                    return _schedule_state(WorkflowState.TRAIN, groupchat, via=WorkflowState.EVALUATE.value)

                groupchat.messages.append(
                    {
                        "role": "system",
                        "name": "Workflow_Controller",
                        "content": (
                            "Evaluator decision: continue_next_candidate. "
                            f"Reason: {summary or 'continue search'}"
                        ),
                    }
                )
                progress("state=EVALUATE(TRAIN) continue next -> TRAIN")
                return _schedule_state(WorkflowState.TRAIN, groupchat, via=WorkflowState.EVALUATE.value)

        if last_speaker is summarizer:
            if messages:
                summarizer_content = messages[-1].get("content", "")
                if summarizer_content:
                    save_step_code("Summarize", summarizer_content)

            if current_state == WorkflowState.HUMAN_REVIEW:
                current_state = WorkflowState.FAILED
                mem.append({"type": "state_enter", "state": WorkflowState.FAILED.value, "reason": "needs_human_review"})
                progress("state=FAILED | handoff for human review")
                return None

            current_state = WorkflowState.DONE
            mem.append({"type": "state_exit", "state": WorkflowState.DONE.value, "ok": True})
            progress("state=DONE | workflow completed")
            return None

        return None

    groupchat = autogen.GroupChat(
        agents=[initializer, data_explorer, data_processer, model_trainer, evaluator, summarizer, code_executor],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=select_next_speaker,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=None)

    chat_result = None
    try:
        # 启动多 Agent 对话主循环。
        chat_result = initializer.initiate_chat(manager, message=task_prompt)
        # 对输出产物做归档整理。
        organize_generated_files(run_dir=run_dir, plot_dir=plot_dir, data_dir=data_dir)

        saved_train_file = None
        if chat_result and chat_result.chat_history:
            final_content = chat_result.chat_history[-1].get("content", "")
            code = extract_python_code_blocks(final_content)
            if code:
                saved_train_file = coding_dir / "train_file_by_agent.py"
                saved_train_file.write_text(code + "\n", encoding="utf-8")

            (coding_dir / "chat_history.json").write_text(
                json.dumps(chat_result.chat_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "data_path": data_path,
            "target": target,
            "task_type": task_type,
            "executor_backend": executor_backend,
            "run_memory": str(mem.path),
            "train_trials_done": count_train_trials(groupchat),
            "preprocess_attempts_done": preprocess_attempts_done,
            "train_attempts_done": train_attempts_done,
            "state_attempt_limits": {s.value: state_attempt_limits[s] for s in WorkflowState},
            "saved_train_file": str(saved_train_file) if saved_train_file else None,
        }
    finally:
        # 无论流程是否报错，都尝试优雅关闭执行服务器。
        try:
            server.stop()
        except Exception:
            pass
